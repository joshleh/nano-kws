"""DS-CNN — depthwise-separable convolutional KWS network.

Architecture follows Zhang et al. 2017 ("Hello Edge: Keyword Spotting on
Microcontrollers"):

  Conv (5x1, stride 2x2)
    -> N x [DepthwiseConv (3x3) -> PointwiseConv (1x1)]
    -> AvgPool (global)
    -> Linear (NUM_CLASSES)

The width multiplier scales the per-block channel count so we can sweep
{0.25, 0.5, 1.0} and report accuracy vs. parameter count vs. MACs.

Implemented in Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


def build_ds_cnn(
    num_classes: int,
    width_multiplier: float = 0.5,
    n_blocks: int = 4,
) -> nn.Module:
    """Construct a DS-CNN with the given width multiplier.

    Implemented in Phase 2.
    """
    raise NotImplementedError("Phase 2: implement DS-CNN.")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    raise NotImplementedError("Phase 2: implement parameter counter.")


def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Estimate multiply-accumulate operations per forward pass.

    Implemented in Phase 2 via a manual walk over Conv2d / Linear layers
    (avoids a heavy third-party MAC-counter dependency).
    """
    raise NotImplementedError("Phase 2: implement MAC counter.")
