"""Train a single DS-CNN configuration on Speech Commands.

Usage::

    python -m nano_kws.train                              # defaults: w=0.5, 30 epochs
    python -m nano_kws.train --width 0.25 --epochs 30
    python -m nano_kws.train --epochs 3 --max-train-batches 50  # quick smoke

Pipeline per step::

    waveform batch (CPU, from DataLoader workers)
        --> .to(device)                       (batch is small: 256 * 16000 * 4 = 16 MB)
        --> [optional] BackgroundNoiseMixer   (train-mode only)
        --> LogMelSpectrogram                 (on device — fast on GPU, OK on CPU)
        --> [optional] SpecAugment            (train-mode only)
        --> DS-CNN
        --> CrossEntropyLoss

Saves the best-validation checkpoint to ``--output`` (default
``assets/ds_cnn_small.pt``) along with all the metadata downstream phases
(quantization, evaluation, benchmark) need to reconstruct the model.

Implemented in Phase 2.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nano_kws import config
from nano_kws.data.augment import BackgroundNoiseMixer, SpecAugment
from nano_kws.data.features import LogMelSpectrogram
from nano_kws.data.speech_commands import SpeechCommandsKWS
from nano_kws.models.ds_cnn import build_ds_cnn, count_macs, count_parameters

logger = logging.getLogger("nano_kws.train")


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Run config / history bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    width: float
    n_blocks: int
    base_channels: int
    num_classes: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
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
class EpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    seconds: float


@dataclass
class RunHistory:
    config: TrainConfig
    parameters: int
    macs: int
    best_epoch: int = -1
    best_val_acc: float = 0.0
    history: list[EpochResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Train / evaluate single epoch
# ---------------------------------------------------------------------------


def _train_one_epoch(
    *,
    model: nn.Module,
    featurizer: nn.Module,
    bg_mixer: nn.Module | None,
    spec_aug: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    log_every: int = 50,
) -> tuple[float, float]:
    model.train()
    if bg_mixer is not None:
        bg_mixer.train()
    if spec_aug is not None:
        spec_aug.train()

    n_seen = 0
    n_correct = 0
    loss_sum = 0.0

    for step, (waveforms, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if bg_mixer is not None:
            waveforms = bg_mixer(waveforms)
        features = featurizer(waveforms)
        if spec_aug is not None:
            features = spec_aug(features)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_seen += labels.size(0)
            loss_sum += loss.item() * labels.size(0)

        if (step + 1) % log_every == 0:
            logger.info(
                "  step %4d | loss %.4f | acc %.4f", step + 1, loss_sum / n_seen, n_correct / n_seen
            )

    return loss_sum / max(1, n_seen), n_correct / max(1, n_seen)


@torch.no_grad()
def _evaluate(
    *,
    model: nn.Module,
    featurizer: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    featurizer.eval()
    n_seen = 0
    n_correct = 0
    loss_sum = 0.0

    for step, (waveforms, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = featurizer(waveforms)
        logits = model(features)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        n_correct += (preds == labels).sum().item()
        n_seen += labels.size(0)
        loss_sum += loss.item() * labels.size(0)

    return loss_sum / max(1, n_seen), n_correct / max(1, n_seen)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Model
    parser.add_argument("--width", type=float, default=0.5, help="DS-CNN width multiplier.")
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=224)
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    # Dataset
    parser.add_argument("--data-root", default=None, help="Override config.DATA_DIR.")
    parser.add_argument("--silence-per-class-ratio", type=float, default=1.0)
    parser.add_argument("--unknown-per-class-ratio", type=float, default=1.0)
    # Augmentation
    parser.add_argument(
        "--no-bg-mixer", action="store_true", help="Disable background-noise mixing."
    )
    parser.add_argument("--no-spec-aug", action="store_true", help="Disable SpecAugment.")
    parser.add_argument("--bg-mix-prob", type=float, default=0.8)
    parser.add_argument("--snr-db-low", type=float, default=5.0)
    parser.add_argument("--snr-db-high", type=float, default=20.0)
    parser.add_argument("--spec-freq-mask", type=int, default=8)
    parser.add_argument("--spec-time-mask", type=int, default=16)
    parser.add_argument("--spec-n-freq-masks", type=int, default=2)
    parser.add_argument("--spec-n-time-masks", type=int, default=2)
    # Output
    parser.add_argument(
        "--output", default=None, help="Path to checkpoint .pt (default depends on width)."
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Path to JSON training history (default: alongside checkpoint).",
    )
    # Smoke / debug
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Cap the number of training batches per epoch (for quick smoke runs).",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Cap the number of validation batches per epoch.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    device = _pick_device(args.device)
    logger.info("Device: %s", device)

    # ----- Data -----
    data_root = Path(args.data_root) if args.data_root else None
    train_ds = SpeechCommandsKWS(
        root=data_root,
        subset="training",
        unknown_per_class_ratio=args.unknown_per_class_ratio,
        silence_per_class_ratio=args.silence_per_class_ratio,
        seed=args.seed,
    )
    val_ds = SpeechCommandsKWS(
        root=data_root,
        subset="validation",
        unknown_per_class_ratio=args.unknown_per_class_ratio,
        silence_per_class_ratio=args.silence_per_class_ratio,
        seed=args.seed,
    )
    logger.info("Train: %d clips | Val: %d clips", len(train_ds), len(val_ds))

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    # ----- Model + featurizer + augmentation -----
    model = build_ds_cnn(
        num_classes=config.NUM_CLASSES,
        width_multiplier=args.width,
        n_blocks=args.n_blocks,
        base_channels=args.base_channels,
    ).to(device)
    n_params = count_parameters(model)
    n_macs = count_macs(model)
    logger.info(
        "Model: width=%.2f channels=%d  params=%d  MACs=%d",
        args.width,
        model.channels,
        n_params,
        n_macs,
    )

    featurizer = LogMelSpectrogram().to(device)

    bg_mixer: nn.Module | None = None
    if not args.no_bg_mixer and train_ds._bg_clips:
        bg_mixer = BackgroundNoiseMixer(
            train_ds._bg_clips,
            p=args.bg_mix_prob,
            snr_db_range=(args.snr_db_low, args.snr_db_high),
        ).to(device)

    spec_aug: nn.Module | None = None
    if not args.no_spec_aug:
        spec_aug = SpecAugment(
            freq_mask_param=args.spec_freq_mask,
            time_mask_param=args.spec_time_mask,
            n_freq_masks=args.spec_n_freq_masks,
            n_time_masks=args.spec_n_time_masks,
        ).to(device)

    # ----- Optimizer + schedule -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ----- Output paths -----
    width_tag = f"w{args.width:g}".replace(".", "p")
    default_ckpt = config.ASSETS_DIR / f"ds_cnn_{width_tag}.pt"
    ckpt_path = Path(args.output) if args.output else default_ckpt
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history) if args.history else ckpt_path.with_suffix(".history.json")

    train_cfg = TrainConfig(
        width=args.width,
        n_blocks=args.n_blocks,
        base_channels=args.base_channels,
        num_classes=config.NUM_CLASSES,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        silence_per_class_ratio=args.silence_per_class_ratio,
        unknown_per_class_ratio=args.unknown_per_class_ratio,
        snr_db_low=args.snr_db_low,
        snr_db_high=args.snr_db_high,
        bg_mix_prob=args.bg_mix_prob if bg_mixer is not None else 0.0,
        spec_freq_mask=args.spec_freq_mask if spec_aug is not None else 0,
        spec_time_mask=args.spec_time_mask if spec_aug is not None else 0,
        spec_n_freq_masks=args.spec_n_freq_masks if spec_aug is not None else 0,
        spec_n_time_masks=args.spec_n_time_masks if spec_aug is not None else 0,
        device=str(device),
        num_workers=args.num_workers,
    )
    run = RunHistory(config=train_cfg, parameters=n_params, macs=n_macs)

    # ----- Loop -----
    for epoch in range(1, args.epochs + 1):
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
            max_batches=args.max_train_batches,
        )
        val_loss, val_acc = _evaluate(
            model=model,
            featurizer=featurizer,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            seconds=elapsed,
        )
        run.history.append(result)
        logger.info(
            "Epoch %3d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | %.1fs",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            elapsed,
        )

        if val_acc > run.best_val_acc:
            run.best_val_acc = val_acc
            run.best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "num_classes": config.NUM_CLASSES,
                        "width_multiplier": args.width,
                        "n_blocks": args.n_blocks,
                        "base_channels": args.base_channels,
                    },
                    "train_config": asdict(train_cfg),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "parameters": n_params,
                    "macs": n_macs,
                },
                ckpt_path,
            )
            logger.info("  ↑ new best (val_acc=%.4f) — checkpoint saved to %s", val_acc, ckpt_path)

        history_path.write_text(
            json.dumps(
                {
                    "config": asdict(run.config),
                    "parameters": run.parameters,
                    "macs": run.macs,
                    "best_epoch": run.best_epoch,
                    "best_val_acc": run.best_val_acc,
                    "history": [asdict(h) for h in run.history],
                },
                indent=2,
            )
        )

    logger.info(
        "Training complete. Best val_acc=%.4f at epoch %d. Checkpoint: %s",
        run.best_val_acc,
        run.best_epoch,
        ckpt_path,
    )


if __name__ == "__main__":
    main()
