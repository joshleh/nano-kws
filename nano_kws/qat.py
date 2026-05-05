"""Quantization-aware training (QAT) for the bundled DS-CNN.

Why QAT?
--------
The post-training quantization (PTQ) pipeline in
:mod:`nano_kws.quantize` calibrates a fp32 model to INT8 by collecting
activation histograms on a held-out batch and rounding weights once.
That works well when the fp32 weight distribution already lies on a
symmetric, low-dynamic-range grid; for DS-CNN at width 0.5 it costs
~7 pp top-1 (see the README TL;DR), which is the headline argument that
PTQ alone is leaving accuracy on the table.

QAT instead simulates INT8 numerics in the forward pass *during
training*: each weight and activation is rounded to the nearest INT8
grid point on the way out, with a straight-through estimator (STE)
passing the gradient unchanged on the way back. The model learns
weights whose distribution is robust to quantization noise, and the
INT8 accuracy gap typically collapses to <2 pp.

What this module does
---------------------
1. Load the existing 30-epoch fp32 checkpoint.
2. Replace every ``Conv2d`` / ``Linear`` with a wrapper that applies
   per-tensor symmetric INT8 fake-quant on inputs and per-output-channel
   symmetric INT8 fake-quant on weights.
3. Fine-tune for a few epochs at a low LR (1e-4 by default), feeding
   the same training pipeline used in :mod:`nano_kws.train`
   (background-noise mixing + SpecAugment + log-mel features).
4. After ``--freeze-observers-after`` epochs, freeze the activation
   observers (standard QAT trick) so the running scales stop drifting
   while the weights converge to them.
5. Strip the wrappers — the model is now structurally identical to the
   original DS-CNN, but its weights have been trained against the
   quantization grid. Save the result as a regular fp32 ``.pt`` so the
   existing :mod:`nano_kws.export_onnx` and :mod:`nano_kws.quantize`
   modules can consume it without modification.

The downstream PTQ pipeline then re-quantises the QAT weights to INT8
ONNX. Calibration is still required (it's how the activation scales
end up in the ONNX graph), but with QAT-trained weights the resulting
INT8 model recovers most of the fp32 accuracy.

Why not torch.ao.quantization.quantize_fx?
------------------------------------------
The FX path traces the model into a ``GraphModule`` and renames
parameters, which makes it awkward to reuse the existing
:mod:`nano_kws.export_onnx` pipeline (which expects a vanilla
:class:`~nano_kws.models.ds_cnn.DSCNN`). A custom STE wrapper is more
transparent for portfolio purposes — it's easy to point at exactly
what's being simulated and how the STE backward works — and integrates
cleanly with the rest of the pipeline.

Usage
-----
::

    # 1. Fine-tune from the existing fp32 checkpoint.
    python -m nano_kws.qat \\
        --fp32-checkpoint assets/ds_cnn_w0p5.pt \\
        --output assets/ds_cnn_w0p5_qat.pt \\
        --epochs 5

    # 2. Export and quantise via the standard pipeline.
    python -m nano_kws.quantize \\
        --checkpoint assets/ds_cnn_w0p5_qat.pt \\
        --output assets/ds_cnn_small_qat_int8.onnx \\
        --fp32-output assets/ds_cnn_small_qat_fp32.onnx

    # 3. Refresh the README benchmark table with the QAT row.
    python -m nano_kws.benchmark \\
        --checkpoint assets/ds_cnn_w0p5.pt \\
        --fp32 assets/ds_cnn_small_fp32.onnx \\
        --int8 assets/ds_cnn_small_int8.onnx \\
        --int8-qat assets/ds_cnn_small_qat_int8.onnx \\
        --update-readme

Or run all three steps in one shot with ``--auto-pipeline``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nano_kws import config
from nano_kws.data.augment import BackgroundNoiseMixer, SpecAugment
from nano_kws.data.features import LogMelSpectrogram
from nano_kws.data.speech_commands import SpeechCommandsKWS
from nano_kws.evaluate import load_checkpoint
from nano_kws.models.ds_cnn import count_macs, count_parameters
from nano_kws.train import _evaluate, _pick_device, _train_one_epoch

logger = logging.getLogger("nano_kws.qat")

# ---------------------------------------------------------------------------
# Fake-quant primitives.
# ---------------------------------------------------------------------------

QMIN_INT8: int = -128
QMAX_INT8: int = 127

# ─── Interview note: why a custom STE instead of torch.ao.quantization? ──────
# Two reasons. (1) torch.ao.quantization.quantize_fx traces the model into a
# GraphModule and rewrites parameter names, which breaks our existing fp32
# ONNX export pipeline — we'd need a separate export path just for QAT
# checkpoints. (2) A 30-line custom STE is much more transparent to walk
# through in an interview: forward is `round((x/scale) + zero_point) * scale -
# zero_point*scale` clamped to the INT8 grid, backward is identity. There's
# no magic. The architecture stays a vanilla DSCNN, the QAT wrappers are
# stripped after training, and the resulting .pt is byte-compatible with the
# original PTQ pipeline. A production team would absolutely use the official
# AO toolkit (or even Brevitas) — that's the right scaling answer — but for
# a portfolio piece "I implemented STE myself and can explain every line"
# is a stronger signal than "I called .prepare_qat_fx and trusted the magic".
# ─────────────────────────────────────────────────────────────────────────────


class _FakeQuantSTE(torch.autograd.Function):
    """Symmetric INT8 fake-quant with a straight-through estimator backward.

    Forward: round to the nearest INT8 grid point at the supplied scale,
    clamped to ``[qmin, qmax]``, then dequantise back to float so the
    output stays in the fp domain (only its *values* are constrained to
    the INT8 grid).

    Backward: pass the upstream gradient through unchanged. This is the
    "STE" of Bengio, Léonard, Courville 2013 — round() has zero
    derivative almost everywhere, so we pretend it's identity for the
    purposes of computing the weight update. Empirically this works
    extremely well for INT8 QAT.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int
    ) -> torch.Tensor:
        scale = scale.clamp(min=1e-9)
        q = torch.clamp(torch.round(x / scale), float(qmin), float(qmax))
        return q * scale

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        return grad_output, None, None, None


def fake_quantize_symmetric(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply symmetric INT8 fake-quant via :class:`_FakeQuantSTE`."""
    return _FakeQuantSTE.apply(x, scale, QMIN_INT8, QMAX_INT8)


class _FakeQuantSTEAsymmetric(torch.autograd.Function):
    """Asymmetric INT8 fake-quant with STE backward.

    Maps fp values to ``round(x / scale + zero_point)`` clamped to the
    INT8 grid, then dequantises back. This matches the asymmetric scheme
    onnxruntime's :func:`quantize_static` picks for activations by
    default (a single ``QuantizeLinear``/``DequantizeLinear`` pair with
    a non-zero zero-point), so the QAT-trained model sees the same
    quantisation behaviour at training time that it will see at
    deployment time. Critical for ReLU outputs, which lie entirely on
    one side of zero — symmetric quant would waste half the INT8 grid.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        scale = scale.clamp(min=1e-9)
        q = torch.clamp(torch.round(x / scale + zero_point), float(qmin), float(qmax))
        return (q - zero_point) * scale

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None]:
        return grad_output, None, None, None, None


def fake_quantize_asymmetric(
    x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Apply asymmetric INT8 fake-quant via :class:`_FakeQuantSTEAsymmetric`."""
    return _FakeQuantSTEAsymmetric.apply(x, scale, zero_point, QMIN_INT8, QMAX_INT8)


def per_channel_weight_fake_quant(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel INT8 fake-quant for Conv2d / Linear weights.

    Conv2d weight shape is ``(out_channels, in_channels // groups, kH, kW)``;
    Linear weight shape is ``(out_features, in_features)``. In both cases
    we reduce along all dims except the first to derive a per-output
    channel scale. Per-channel weight quantization is the convention the
    ONNX quantizer uses too (see ``per_channel=True`` in
    :func:`nano_kws.quantize.quantize_onnx`); matching that convention
    here keeps the simulated grid aligned with the eventual deployment
    grid.
    """
    dims = list(range(1, weight.dim()))
    abs_max = weight.detach().abs().amax(dim=dims, keepdim=True)
    scale = (abs_max / float(QMAX_INT8)).clamp(min=1e-9)
    return fake_quantize_symmetric(weight, scale)


# ─── Interview note: why this choice cost us 5+ pp in the first run ─────────
# The initial QAT implementation used a *symmetric* activation observer, and
# it actually regressed accuracy by ~5 pp vs PTQ. The fix that recovered
# the gain was switching to asymmetric here (matching ORT's deployment
# scheme), plus three other small alignment changes:
#   * Disable augmentation during fine-tune (training distribution should
#     match the calibration distribution PTQ sees).
#   * Lower the LR to 5e-5 (we're tuning around the existing fp32 weights,
#     not training from scratch).
#   * Save the *last* epoch's checkpoint, not the best-val one — early best
#     val happens before observers freeze, so the weights haven't yet
#     adapted to stable scales.
# The lesson worth rehearsing: QAT only helps when training-time numerics
# match deployment-time numerics. A subtly different fake-quant scheme is
# worse than no QAT at all because the model learns to compensate for
# noise it'll never see in production.
# ─────────────────────────────────────────────────────────────────────────────


class ActivationObserver(nn.Module):
    """EMA observer for asymmetric per-tensor INT8 activation quantization.

    Tracks the running min and max of activations seen during training
    and derives the affine ``(scale, zero_point)`` pair that maps the
    ``[running_min, running_max]`` range linearly into the INT8 grid
    ``[qmin, qmax]``. Fake-quant in the forward uses that affine map;
    once :meth:`freeze` is called the running stats stop updating
    (standard "freeze observers, fine-tune weights" QAT phase).

    Why asymmetric and not symmetric?
        DS-CNN activations are post-ReLU and therefore non-negative.
        A symmetric scheme (``scale = max/127``, no zero-point) wastes
        half the INT8 grid on values that never occur. The deployment
        pipeline (:mod:`nano_kws.quantize`) calls
        :func:`onnxruntime.quantization.quantize_static` with
        ``activation_type=QuantType.QInt8`` and MinMax calibration,
        which picks an asymmetric ``(scale, zero_point)`` per activation
        tensor. Matching that scheme during training is what lets QAT
        actually help: train against the same numerics that will run on
        the chip.
    """

    def __init__(self, momentum: float = 0.05) -> None:
        super().__init__()
        if not 0.0 < momentum < 1.0:
            raise ValueError(f"momentum must be in (0, 1), got {momentum}.")
        self.momentum = momentum
        self.register_buffer("running_min", torch.tensor(0.0))
        self.register_buffer("running_max", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("frozen", torch.tensor(False))

    def freeze(self) -> None:
        """Stop the EMA from updating on subsequent forward passes."""
        self.frozen.fill_(True)

    def _scale_zero_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Affine (scale, zero_point) such that
        ``round(x / scale + zero_point)`` lands in ``[qmin, qmax]``."""
        rng = (self.running_max - self.running_min).clamp(min=1e-9)
        scale = rng / float(QMAX_INT8 - QMIN_INT8)
        zero_point = torch.round(QMIN_INT8 - self.running_min / scale).clamp(
            float(QMIN_INT8), float(QMAX_INT8)
        )
        return scale, zero_point

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not bool(self.frozen.item()):
            with torch.no_grad():
                cur_min = x.detach().amin()
                cur_max = x.detach().amax()
                if not bool(self.initialized.item()):
                    # Initialise on the first batch so the EMA isn't pulled
                    # toward zero by a useless prior.
                    self.running_min.copy_(cur_min)
                    self.running_max.copy_(cur_max)
                    self.initialized.fill_(True)
                else:
                    self.running_min.mul_(1 - self.momentum).add_(self.momentum * cur_min)
                    self.running_max.mul_(1 - self.momentum).add_(self.momentum * cur_max)
        # If the observer has never seen data (e.g. eval-only path on a
        # freshly constructed module), passing through unmodified is the
        # safe choice; the downstream PTQ will calibrate the real scale.
        if not bool(self.initialized.item()):
            return x
        scale, zero_point = self._scale_zero_point()
        return fake_quantize_asymmetric(x, scale, zero_point)


# ---------------------------------------------------------------------------
# QAT wrappers around Conv2d / Linear.
# ---------------------------------------------------------------------------


class QatConv2d(nn.Module):
    """Wraps a :class:`nn.Conv2d` with input + per-channel weight fake-quant."""

    def __init__(self, conv: nn.Conv2d) -> None:
        super().__init__()
        self.conv = conv
        self.input_obs = ActivationObserver()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.input_obs(x)
        w_q = per_channel_weight_fake_quant(self.conv.weight)
        return F.conv2d(
            x_q,
            w_q,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class QatLinear(nn.Module):
    """Wraps a :class:`nn.Linear` with input + per-channel weight fake-quant."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear
        self.input_obs = ActivationObserver()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.input_obs(x)
        w_q = per_channel_weight_fake_quant(self.linear.weight)
        return F.linear(x_q, w_q, self.linear.bias)


# ---------------------------------------------------------------------------
# prepare / strip / freeze helpers.
# ---------------------------------------------------------------------------


def prepare_qat(model: nn.Module) -> nn.Module:
    """In-place: replace every Conv2d / Linear with the QAT wrapper.

    Returns the (mutated) ``model`` for chaining. The wrapped modules
    share parameter tensors with the originals — there is no copy — so
    weights trained through the wrapped graph are reflected in the inner
    ``Conv2d`` / ``Linear`` directly, and :func:`strip_qat` can
    therefore reverse the wrapping without touching state.
    """

    def _walk(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, (QatConv2d, QatLinear)):
                continue
            if isinstance(child, nn.Conv2d):
                setattr(parent, name, QatConv2d(child))
            elif isinstance(child, nn.Linear):
                setattr(parent, name, QatLinear(child))
            else:
                _walk(child)

    _walk(model)
    return model


def strip_qat(model: nn.Module) -> nn.Module:
    """In-place inverse of :func:`prepare_qat`.

    Replaces every :class:`QatConv2d` / :class:`QatLinear` with its
    inner module, so the resulting state_dict is identical in structure
    to the vanilla DS-CNN and can be loaded by
    :func:`nano_kws.evaluate.load_checkpoint` without modification.
    """

    def _walk(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, QatConv2d):
                setattr(parent, name, child.conv)
            elif isinstance(child, QatLinear):
                setattr(parent, name, child.linear)
            else:
                _walk(child)

    _walk(model)
    return model


def freeze_observers(model: nn.Module) -> int:
    """Stop the EMA from updating in every :class:`ActivationObserver`.

    Returns the number of observers frozen. The standard QAT recipe is
    to leave observers floating for the first few epochs (so they
    converge on a sensible activation range) then freeze them so the
    weights can fine-tune against a fixed grid for the remaining
    epochs. Without freezing, the moving target makes the last-mile
    convergence noisier.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, ActivationObserver):
            module.freeze()
            n += 1
    return n


def count_qat_observers(model: nn.Module) -> int:
    """Number of :class:`ActivationObserver` instances inside ``model``."""
    return sum(1 for m in model.modules() if isinstance(m, ActivationObserver))


# ---------------------------------------------------------------------------
# QAT training run.
# ---------------------------------------------------------------------------


@dataclass
class QatConfig:
    fp32_checkpoint: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    freeze_observers_after: int
    silence_per_class_ratio: float
    unknown_per_class_ratio: float
    snr_db_low: float
    snr_db_high: float
    bg_mix_prob: float
    spec_freq_mask: int
    spec_time_mask: int
    spec_n_freq_masks: int
    spec_n_time_masks: int
    device: str
    num_workers: int


@dataclass
class QatEpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    observers_frozen: bool
    seconds: float


@dataclass
class QatHistory:
    config: QatConfig
    fp32_baseline_val_acc: float
    parameters: int
    macs: int
    best_epoch: int = -1
    best_val_acc: float = 0.0
    history: list[QatEpochResult] = field(default_factory=list)


def _save_qat_checkpoint(
    *,
    model: nn.Module,
    fp32_ckpt: dict[str, Any],
    qat_cfg: QatConfig,
    epoch: int,
    val_acc: float,
    parameters: int,
    macs: int,
    output_path: Path,
) -> None:
    """Persist a stripped (vanilla-DS-CNN-shaped) checkpoint to ``output_path``."""
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": fp32_ckpt["model_config"],
        "train_config": fp32_ckpt.get("train_config", {}),
        "qat_config": asdict(qat_cfg),
        "epoch": epoch,
        "val_acc": val_acc,
        "parameters": parameters,
        "macs": macs,
        "qat": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def train_qat(
    *,
    fp32_checkpoint: Path,
    output_path: Path,
    epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    seed: int = 0,
    freeze_observers_after: int = 2,
    num_workers: int = 4,
    device_str: str = "auto",
    data_root: Path | None = None,
    silence_per_class_ratio: float = 1.0,
    unknown_per_class_ratio: float = 1.0,
    bg_mix_prob: float = 0.8,
    snr_db_low: float = 5.0,
    snr_db_high: float = 20.0,
    spec_freq_mask: int = 8,
    spec_time_mask: int = 16,
    spec_n_freq_masks: int = 2,
    spec_n_time_masks: int = 2,
    no_bg_mixer: bool = False,
    no_spec_aug: bool = False,
    save_last: bool = False,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    history_path: Path | None = None,
) -> QatHistory:
    """Fine-tune ``fp32_checkpoint`` with QAT for ``epochs`` epochs.

    The best-val checkpoint is written to ``output_path`` as a regular
    fp32 ``.pt`` (the QAT wrappers are stripped before save), so the
    existing export + quantize pipeline can consume it directly.

    Returns the run history.
    """
    torch.manual_seed(seed)
    device = _pick_device(device_str)
    logger.info("Device: %s", device)

    fp32_model, fp32_ckpt = load_checkpoint(fp32_checkpoint, device=device)
    fp32_baseline_val_acc = float(fp32_ckpt.get("val_acc", float("nan")))
    n_params = count_parameters(fp32_model)
    n_macs = count_macs(fp32_model)
    logger.info(
        "Loaded fp32 checkpoint %s | width=%s | val_acc=%.4f | params=%d | MACs=%d",
        fp32_checkpoint,
        fp32_ckpt["model_config"]["width_multiplier"],
        fp32_baseline_val_acc,
        n_params,
        n_macs,
    )

    model = prepare_qat(fp32_model)
    n_observers = count_qat_observers(model)
    logger.info("Inserted fake-quant on %d activations + every Conv2d/Linear weight.", n_observers)

    train_ds = SpeechCommandsKWS(
        root=data_root,
        subset="training",
        unknown_per_class_ratio=unknown_per_class_ratio,
        silence_per_class_ratio=silence_per_class_ratio,
        seed=seed,
    )
    val_ds = SpeechCommandsKWS(
        root=data_root,
        subset="validation",
        unknown_per_class_ratio=unknown_per_class_ratio,
        silence_per_class_ratio=silence_per_class_ratio,
        seed=seed,
    )
    logger.info("QAT train: %d clips | val: %d clips", len(train_ds), len(val_ds))

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    featurizer = LogMelSpectrogram().to(device)

    bg_mixer: nn.Module | None = None
    if not no_bg_mixer and train_ds._bg_clips:
        bg_mixer = BackgroundNoiseMixer(
            train_ds._bg_clips,
            p=bg_mix_prob,
            snr_db_range=(snr_db_low, snr_db_high),
        ).to(device)

    spec_aug: nn.Module | None = None
    if not no_spec_aug:
        spec_aug = SpecAugment(
            freq_mask_param=spec_freq_mask,
            time_mask_param=spec_time_mask,
            n_freq_masks=spec_n_freq_masks,
            n_time_masks=spec_n_time_masks,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    qat_cfg = QatConfig(
        fp32_checkpoint=str(fp32_checkpoint),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        freeze_observers_after=freeze_observers_after,
        silence_per_class_ratio=silence_per_class_ratio,
        unknown_per_class_ratio=unknown_per_class_ratio,
        snr_db_low=snr_db_low,
        snr_db_high=snr_db_high,
        bg_mix_prob=bg_mix_prob if bg_mixer is not None else 0.0,
        spec_freq_mask=spec_freq_mask if spec_aug is not None else 0,
        spec_time_mask=spec_time_mask if spec_aug is not None else 0,
        spec_n_freq_masks=spec_n_freq_masks if spec_aug is not None else 0,
        spec_n_time_masks=spec_n_time_masks if spec_aug is not None else 0,
        device=str(device),
        num_workers=num_workers,
    )

    run = QatHistory(
        config=qat_cfg,
        fp32_baseline_val_acc=fp32_baseline_val_acc,
        parameters=n_params,
        macs=n_macs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = history_path or output_path.with_suffix(".history.json")
    observers_frozen = False

    for epoch in range(1, epochs + 1):
        if not observers_frozen and epoch > freeze_observers_after:
            n_frozen = freeze_observers(model)
            observers_frozen = True
            logger.info("Froze %d activation observers ahead of epoch %d.", n_frozen, epoch)

        t0 = time.perf_counter()
        train_loss, train_acc = _train_one_epoch(
            model=model,
            featurizer=featurizer,
            bg_mixer=bg_mixer,
            spec_aug=spec_aug,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=max_train_batches,
        )
        val_loss, val_acc = _evaluate(
            model=model,
            featurizer=featurizer,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0

        logger.info(
            "QAT epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | observers_frozen=%s | %.1fs",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            observers_frozen,
            elapsed,
        )

        run.history.append(
            QatEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                observers_frozen=observers_frozen,
                seconds=elapsed,
            )
        )

        # ``save_last``: only persist the final epoch's weights. Best-val
        # under fake-quant can be misleading because the observer scales
        # at the moment of save differ from what the deployment-time PTQ
        # picks (per-tensor MinMax over a clean calibration sample, vs
        # the EMA over augmented training batches). The last epoch is
        # the most settled state of the weights against frozen
        # observers, and is the closest analogue to a "production
        # checkpoint" you'd want to deploy.
        save_now = (epoch == epochs) if save_last else (val_acc > run.best_val_acc)
        if save_now:
            run.best_val_acc = val_acc
            run.best_epoch = epoch
            # Strip on a *copy* of the model so the in-progress training
            # graph isn't mutated mid-loop. Cheapest copy: build a fresh
            # vanilla model from the same config and load the (post-strip)
            # state_dict into it.
            from nano_kws.models.ds_cnn import build_ds_cnn

            cfg = fp32_ckpt["model_config"]
            snapshot = build_ds_cnn(
                num_classes=cfg.get("num_classes", config.NUM_CLASSES),
                width_multiplier=cfg["width_multiplier"],
                n_blocks=cfg.get("n_blocks", 4),
                base_channels=cfg.get("base_channels", 224),
            )
            # The shared-parameter wiring in prepare_qat means the inner
            # Conv2d / Linear weights *are* the trained weights, so we can
            # read them out via the snapshot model's own state_dict by
            # copying name-by-name. Build a name map: Qat wrappers nest
            # the original under `.conv` / `.linear`, so e.g. the trained
            # weight at "stem.0.conv.weight" maps to "stem.0.weight".
            trained_state = {}
            for k, v in model.state_dict().items():
                if ".input_obs." in k:
                    continue
                stripped = k.replace(".conv.weight", ".weight")
                stripped = stripped.replace(".conv.bias", ".bias")
                stripped = stripped.replace(".linear.weight", ".weight")
                stripped = stripped.replace(".linear.bias", ".bias")
                trained_state[stripped] = v.detach().cpu().clone()
            snapshot.load_state_dict(trained_state)
            _save_qat_checkpoint(
                model=snapshot,
                fp32_ckpt=fp32_ckpt,
                qat_cfg=qat_cfg,
                epoch=epoch,
                val_acc=val_acc,
                parameters=n_params,
                macs=n_macs,
                output_path=output_path,
            )
            logger.info(
                "  ↑ new best (val_acc=%.4f) — QAT checkpoint saved to %s", val_acc, output_path
            )

        history_path.write_text(
            json.dumps(
                {
                    "config": asdict(run.config),
                    "fp32_baseline_val_acc": run.fp32_baseline_val_acc,
                    "parameters": run.parameters,
                    "macs": run.macs,
                    "best_epoch": run.best_epoch,
                    "best_val_acc": run.best_val_acc,
                    "history": [asdict(h) for h in run.history],
                },
                indent=2,
            )
        )

    delta_pp = (run.best_val_acc - fp32_baseline_val_acc) * 100
    logger.info(
        "QAT complete. Best val_acc=%.4f (fp32 baseline=%.4f, Δ=%+.2f pp). Checkpoint: %s",
        run.best_val_acc,
        fp32_baseline_val_acc,
        delta_pp,
        output_path,
    )
    return run


# ---------------------------------------------------------------------------
# Auto-pipeline: train → export → quantize → benchmark in one shot.
# ---------------------------------------------------------------------------


def run_auto_pipeline(
    *,
    qat_checkpoint: Path,
    fp32_pt_checkpoint: Path,
    output_int8: Path,
    output_fp32_onnx: Path | None = None,
    calibration_batches: int = 50,
    calibration_batch_size: int = 16,
    data_root: Path | None = None,
    seed: int = 0,
    update_readme: bool = False,
    fp32_int8_path: Path | None = None,
    fp32_onnx_baseline: Path | None = None,
) -> dict[str, Path]:
    """Export → quantize → (optionally) benchmark the QAT checkpoint.

    Returns a dict with the resolved output paths.
    """
    from nano_kws import benchmark as bench
    from nano_kws.export_onnx import export_to_onnx, write_label_map
    from nano_kws.quantize import (
        quantize_onnx,
        real_calibration_batches,
        synthetic_calibration_batches,
        verify_int8_argmax_agreement,
    )

    output_int8 = Path(output_int8)
    output_fp32_onnx = (
        Path(output_fp32_onnx)
        if output_fp32_onnx
        else output_int8.with_name(output_int8.stem.replace("int8", "fp32") + ".onnx")
    )
    if output_fp32_onnx == output_int8:
        output_fp32_onnx = output_int8.with_name(f"{output_int8.stem}_fp32.onnx")

    logger.info("Auto-pipeline step 1/3: export QAT checkpoint to fp32 ONNX")
    qat_model, _ = load_checkpoint(qat_checkpoint, device="cpu")
    export_to_onnx(model=qat_model, output_path=output_fp32_onnx)
    write_label_map(output_fp32_onnx.with_suffix(".label_map.json"))

    logger.info(
        "Auto-pipeline step 2/3: PTQ-quantize the QAT-trained ONNX (this is the QAT+PTQ INT8 model)"
    )
    archive = (data_root or config.DATA_DIR) / "SpeechCommands" / "speech_commands_v0.02"
    if archive.is_dir():
        batches = real_calibration_batches(
            data_root=data_root,
            batch_size=calibration_batch_size,
            n_batches=calibration_batches,
            seed=seed,
        )
    else:
        logger.warning(
            "Speech Commands not cached at %s — falling back to synthetic calibration. "
            "INT8 accuracy will be degraded.",
            archive,
        )
        batches = synthetic_calibration_batches(
            n_batches=calibration_batches, batch_size=calibration_batch_size, seed=seed
        )
    quantize_onnx(
        fp32_path=output_fp32_onnx,
        int8_path=output_int8,
        calibration_batches=batches,
    )
    write_label_map(output_int8.with_suffix(".label_map.json"))
    agreement = verify_int8_argmax_agreement(
        int8_path=output_int8, fp32_model=qat_model, batches=batches
    )
    logger.info("QAT INT8 vs QAT fp32 argmax agreement on calibration: %.4f", agreement)

    if update_readme:
        logger.info(
            "Auto-pipeline step 3/3: refresh benchmark table with the QAT row (--update-readme)"
        )
        argv = [
            "--checkpoint",
            str(fp32_pt_checkpoint),
        ]
        if fp32_onnx_baseline is not None:
            argv.extend(["--fp32", str(fp32_onnx_baseline)])
        if fp32_int8_path is not None:
            argv.extend(["--int8", str(fp32_int8_path)])
        argv.extend(
            [
                "--int8-qat",
                str(output_int8),
                "--output",
                str(config.ASSETS_DIR / "benchmark_table.md"),
                "--update-readme",
            ]
        )
        bench.main(argv)
    else:
        logger.info(
            "Auto-pipeline step 3/3: skipped (--no-update-readme). Run benchmark manually to refresh."
        )

    return {"fp32_onnx": output_fp32_onnx, "int8_onnx": output_int8}


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fp32-checkpoint",
        default="assets/ds_cnn_w0p5.pt",
        help="fp32 .pt to fine-tune from (default: assets/ds_cnn_w0p5.pt).",
    )
    parser.add_argument(
        "--output",
        default="assets/ds_cnn_w0p5_qat.pt",
        help="Where to write the QAT-trained .pt (vanilla-DS-CNN-shaped state_dict).",
    )
    # Training hparams
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--freeze-observers-after",
        type=int,
        default=2,
        help="Freeze the activation EMA observers after this many epochs.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--silence-per-class-ratio", type=float, default=1.0)
    parser.add_argument("--unknown-per-class-ratio", type=float, default=1.0)
    # Augmentation
    parser.add_argument("--no-bg-mixer", action="store_true")
    parser.add_argument("--no-spec-aug", action="store_true")
    parser.add_argument("--bg-mix-prob", type=float, default=0.8)
    parser.add_argument("--snr-db-low", type=float, default=5.0)
    parser.add_argument("--snr-db-high", type=float, default=20.0)
    parser.add_argument("--spec-freq-mask", type=int, default=8)
    parser.add_argument("--spec-time-mask", type=int, default=16)
    parser.add_argument("--spec-n-freq-masks", type=int, default=2)
    parser.add_argument("--spec-n-time-masks", type=int, default=2)
    parser.add_argument(
        "--save-last",
        action="store_true",
        help=(
            "Save the final epoch's checkpoint instead of best-val. With "
            "fake-quant active, val_acc can swing wildly while observers "
            "settle, so 'best-val' may pick an early lucky checkpoint "
            "with poorly-converged scales. Last-epoch is the most "
            "settled production-shaped weight set."
        ),
    )
    # Smoke / debug
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    # Auto-pipeline
    parser.add_argument(
        "--auto-pipeline",
        action="store_true",
        help=(
            "After training, also export QAT → fp32 ONNX → INT8 ONNX (PTQ on QAT weights). "
            "Pair with --update-readme to also refresh the README benchmark table."
        ),
    )
    parser.add_argument(
        "--int8-output",
        default="assets/ds_cnn_small_qat_int8.onnx",
        help="Auto-pipeline: where to write the QAT INT8 ONNX.",
    )
    parser.add_argument(
        "--fp32-onnx-output",
        default=None,
        help="Auto-pipeline: where to write the QAT fp32 ONNX (default: alongside --int8-output).",
    )
    parser.add_argument("--calibration-batches", type=int, default=50)
    parser.add_argument("--calibration-batch-size", type=int, default=16)
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Auto-pipeline: also refresh the README TL;DR table with the QAT row.",
    )
    parser.add_argument(
        "--baseline-int8",
        default="assets/ds_cnn_small_int8.onnx",
        help="Auto-pipeline: PTQ-only INT8 ONNX, for the side-by-side row.",
    )
    parser.add_argument(
        "--baseline-fp32-onnx",
        default="assets/ds_cnn_small_fp32.onnx",
        help="Auto-pipeline: fp32 ONNX baseline, for the side-by-side row.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    fp32_ckpt = Path(args.fp32_checkpoint)
    output = Path(args.output)
    if not fp32_ckpt.is_file():
        raise SystemExit(
            f"fp32 checkpoint not found: {fp32_ckpt}. Run `make train` first or pass "
            f"--fp32-checkpoint."
        )

    train_qat(
        fp32_checkpoint=fp32_ckpt,
        output_path=output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
        freeze_observers_after=args.freeze_observers_after,
        num_workers=args.num_workers,
        device_str=args.device,
        data_root=Path(args.data_root) if args.data_root else None,
        silence_per_class_ratio=args.silence_per_class_ratio,
        unknown_per_class_ratio=args.unknown_per_class_ratio,
        bg_mix_prob=args.bg_mix_prob,
        snr_db_low=args.snr_db_low,
        snr_db_high=args.snr_db_high,
        spec_freq_mask=args.spec_freq_mask,
        spec_time_mask=args.spec_time_mask,
        spec_n_freq_masks=args.spec_n_freq_masks,
        spec_n_time_masks=args.spec_n_time_masks,
        no_bg_mixer=args.no_bg_mixer,
        no_spec_aug=args.no_spec_aug,
        save_last=args.save_last,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    if args.auto_pipeline:
        run_auto_pipeline(
            qat_checkpoint=output,
            fp32_pt_checkpoint=fp32_ckpt,
            output_int8=Path(args.int8_output),
            output_fp32_onnx=Path(args.fp32_onnx_output) if args.fp32_onnx_output else None,
            calibration_batches=args.calibration_batches,
            calibration_batch_size=args.calibration_batch_size,
            data_root=Path(args.data_root) if args.data_root else None,
            seed=args.seed,
            update_readme=args.update_readme,
            fp32_int8_path=Path(args.baseline_int8) if args.baseline_int8 else None,
            fp32_onnx_baseline=Path(args.baseline_fp32_onnx) if args.baseline_fp32_onnx else None,
        )


if __name__ == "__main__":
    main()


__all__ = [
    "QMAX_INT8",
    "QMIN_INT8",
    "ActivationObserver",
    "QatConfig",
    "QatConv2d",
    "QatEpochResult",
    "QatHistory",
    "QatLinear",
    "count_qat_observers",
    "fake_quantize_asymmetric",
    "fake_quantize_symmetric",
    "freeze_observers",
    "main",
    "per_channel_weight_fake_quant",
    "prepare_qat",
    "run_auto_pipeline",
    "strip_qat",
    "train_qat",
]
