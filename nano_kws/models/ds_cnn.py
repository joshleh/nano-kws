"""DS-CNN — depthwise-separable convolutional KWS network.

Architecture follows Zhang et al. 2017 ("Hello Edge: Keyword Spotting on
Microcontrollers")::

    Conv2d(1 -> C, kernel 10x4, stride 2x2)  --> BN --> ReLU
    [
      DepthwiseConv2d(C, kernel 3x3, padding 1) --> BN --> ReLU
      Conv2d(C -> C, kernel 1x1)                --> BN --> ReLU
    ] x n_blocks
    AdaptiveAvgPool2d(1) --> Flatten --> Linear(C, num_classes)

The ``width_multiplier`` scales ``base_channels`` so the same code can
produce three points on the accuracy / parameter / MAC trade-off curve
for the Phase 5 sweep (~18 K / 62 K / 224 K params at w = 0.25 / 0.5 / 1.0).

We expose three helpers used everywhere downstream:

* :func:`build_ds_cnn`   — factory; this is the only constructor callers
  should use.
* :func:`count_parameters` — trainable parameter count.
* :func:`count_macs`     — multiply-accumulates per forward pass, computed
  by hooking Conv2d / Linear modules during a single forward, so the
  number is exact for the architecture as built (no third-party MAC
  library needed).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_kws import config

DEFAULT_BASE_CHANNELS: int = 224
DEFAULT_N_BLOCKS: int = 4
INITIAL_KERNEL: tuple[int, int] = (10, 4)
INITIAL_STRIDE: tuple[int, int] = (2, 2)

# ─── Design note: why DS-CNN over peer architectures? ────────────────────
# DS-CNN is the canonical "edge KWS baseline" in the literature (Zhang et al.
# "Hello Edge", 2017) and a fair midpoint among the alternatives:
#   * MobileNet — same depthwise-separable trick but designed for ImageNet
#     scale; overkill at our 12-class budget.
#   * TC-ResNet — temporal-only convs, slightly better acc/MAC at the cost
#     of a more bespoke architecture.
#   * BC-ResNet — broadcasting frequency/time blocks; SOTA in 2021 KWS work
#     but more architecturally bespoke than is justified at this scope.
# DS-CNN wins on (a) being a well-known, citable baseline, (b) plain Conv2d
# blocks that quantize and SIMD-vectorize cleanly, and (c) a single width
# multiplier knob that gives the whole accuracy-vs-compute curve.
# ────────────────────────────────────────────────────────────────────────────


def _scaled_channels(base: int, multiplier: float, divisor: int = 8) -> int:
    """Scale ``base`` by ``multiplier`` and round to a multiple of ``divisor``.

    Rounding to a multiple of 8 keeps the channel counts friendly for SIMD
    / vectorised kernels and matches the convention used by MobileNet et al.
    The result is clamped to be at least ``divisor``.
    """
    n = max(divisor, round(base * multiplier / divisor) * divisor)
    return n


class DepthwiseSeparableBlock(nn.Module):
    """3x3 depthwise + 1x1 pointwise, each followed by BN and ReLU."""

    # ─── Design note: depthwise-separable vs standard conv ──────────────
    # A standard 3x3 conv with C in/out channels does 3*3*C*C MACs per
    # output spatial position. Depthwise-separable splits that into:
    #   * depthwise:  3*3*C   MACs  (one filter per channel, no mixing)
    #   * pointwise:  1*1*C*C MACs  (channel mixing only)
    # Total: (9 + C) * C, vs 9*C*C for standard. At C=56 that's a ~8.6x
    # reduction in compute. The (small) accuracy cost is well-studied and
    # near-zero at our scale. The trade-off in deployment: depthwise has
    # very low arithmetic intensity (one FMA per load), so it's typically
    # memory-bound — the AVX2 microbench in cpp/microbench/ shows this in
    # action. Pointwise dominates the FLOP count and is what wins from
    # SIMD/GEMM kernels. bias=False because every conv is followed by BN,
    # which absorbs any constant offset.
    # ───────────────────────────────────────────────────────────────────────

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    """DS-CNN keyword spotter.

    Parameters
    ----------
    num_classes
        Output logit dimension. Default is 12 (the project's class count).
    width_multiplier
        Scales ``base_channels``; rounded to a multiple of 8.
    n_blocks
        Number of depthwise-separable blocks after the initial conv.
    base_channels
        Channel count when ``width_multiplier == 1.0``.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        width_multiplier: float = 1.0,
        n_blocks: int = DEFAULT_N_BLOCKS,
        base_channels: int = DEFAULT_BASE_CHANNELS,
    ) -> None:
        super().__init__()
        if width_multiplier <= 0:
            raise ValueError(f"width_multiplier must be > 0, got {width_multiplier}.")
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}.")

        self.width_multiplier = width_multiplier
        self.n_blocks = n_blocks
        self.base_channels = base_channels
        self.channels = _scaled_channels(base_channels, width_multiplier)

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.channels,
                kernel_size=INITIAL_KERNEL,
                stride=INITIAL_STRIDE,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *(DepthwiseSeparableBlock(self.channels) for _ in range(n_blocks))
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.channels, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_ds_cnn(
    num_classes: int = config.NUM_CLASSES,
    width_multiplier: float = 1.0,
    n_blocks: int = DEFAULT_N_BLOCKS,
    base_channels: int = DEFAULT_BASE_CHANNELS,
) -> DSCNN:
    """Construct a DS-CNN. The canonical entry point for callers."""
    return DSCNN(
        num_classes=num_classes,
        width_multiplier=width_multiplier,
        n_blocks=n_blocks,
        base_channels=base_channels,
    )


# ---------------------------------------------------------------------------
# Parameter and MAC accounting.
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Design note: why count MACs ourselves instead of using ptflops/fvcore? ──
# Two reasons. First, removing a third-party MAC counter is one less dependency
# that could go stale or report something subtly different from the next one.
# Second, writing the hook is a 20-line exercise that directly forces you to
# think through the per-layer formula — exactly the back-of-the-envelope
# arithmetic that comes up whenever you sanity-check an edge-AI deployment
# budget. ("How many MACs does a 3x3 depthwise conv on a 16x47 feature map
# with 56 channels do?" → 3*3 * 1 * 56 *
# 16 * 47 = 379,008.) The hook handles depthwise via `groups`, BN/ReLU/pool
# are intentionally not counted (negligible compared to convs/linears, and the
# reporting convention for edge-AI MAC budgets is conv+linear only).
# ───────────────────────────────────────────────────────────────────────────────


def count_macs(model: nn.Module, input_shape: tuple[int, ...] = config.INPUT_SHAPE) -> int:
    """Estimate multiply-accumulates per forward pass for one input.

    Implementation: register forward hooks on ``Conv2d`` and ``Linear``
    layers, run a single forward pass with a dummy input of shape
    ``(1, *input_shape)``, then sum up per-layer contributions:

    * Conv2d (incl. depthwise via ``groups``):
      ``out_channels * (in_channels / groups) * k_h * k_w * out_h * out_w``
    * Linear:
      ``in_features * out_features``

    Multiplies and adds are folded together into a single MAC count, which
    is the standard reporting convention for edge-AI accelerators (one MAC
    = one fused multiply-and-accumulate, ~= 2 FLOPs).
    """
    macs = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(module: nn.Conv2d, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal macs
        out_h, out_w = out.shape[-2:]
        kh, kw = module.kernel_size
        in_per_group = module.in_channels // module.groups
        macs += module.out_channels * in_per_group * kh * kw * out_h * out_w

    def linear_hook(module: nn.Linear, _inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        nonlocal macs
        macs += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            model(dummy)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    return macs


__all__ = [
    "DEFAULT_BASE_CHANNELS",
    "DEFAULT_N_BLOCKS",
    "DSCNN",
    "DepthwiseSeparableBlock",
    "build_ds_cnn",
    "count_macs",
    "count_parameters",
]
