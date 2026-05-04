"""Phase 2 — DS-CNN architecture, parameter and MAC accounting."""

from __future__ import annotations

import pytest
import torch

from nano_kws import config
from nano_kws.models.ds_cnn import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_N_BLOCKS,
    DSCNN,
    build_ds_cnn,
    count_macs,
    count_parameters,
)


@pytest.fixture(scope="module")
def small_model() -> DSCNN:
    return build_ds_cnn(width_multiplier=0.5)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def test_forward_pass_shape(small_model: DSCNN) -> None:
    x = torch.zeros(4, *config.INPUT_SHAPE)
    y = small_model(x)
    assert y.shape == (4, config.NUM_CLASSES)
    assert y.dtype == torch.float32


def test_forward_pass_works_with_single_example(small_model: DSCNN) -> None:
    x = torch.zeros(1, *config.INPUT_SHAPE)
    y = small_model(x)
    assert y.shape == (1, config.NUM_CLASSES)


def test_forward_pass_supports_backprop(small_model: DSCNN) -> None:
    x = torch.randn(2, *config.INPUT_SHAPE, requires_grad=True)
    y = small_model(x).sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# Parameter / channel scaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "expected_channels", "min_params", "max_params"),
    [
        (0.25, 56, 12_000, 25_000),
        (0.50, 112, 50_000, 80_000),
        (1.00, 224, 200_000, 260_000),
    ],
)
def test_parameter_count_budget_per_width(
    width: float, expected_channels: int, min_params: int, max_params: int
) -> None:
    model = build_ds_cnn(width_multiplier=width)
    assert model.channels == expected_channels
    n_params = count_parameters(model)
    assert min_params <= n_params <= max_params, (
        f"width={width}: expected {min_params} <= params <= {max_params}, got {n_params}"
    )


def test_parameter_count_scales_monotonically_with_width() -> None:
    counts = [count_parameters(build_ds_cnn(width_multiplier=w)) for w in (0.25, 0.5, 1.0)]
    assert counts[0] < counts[1] < counts[2]


def test_channel_count_is_multiple_of_eight() -> None:
    for w in (0.1, 0.25, 0.5, 0.75, 1.0):
        model = build_ds_cnn(width_multiplier=w)
        assert model.channels % 8 == 0, f"channels={model.channels} for w={w} not divisible by 8"
        assert model.channels >= 8


# ---------------------------------------------------------------------------
# MAC counter
# ---------------------------------------------------------------------------


def test_mac_count_is_positive_and_scales_with_width() -> None:
    macs = [count_macs(build_ds_cnn(width_multiplier=w)) for w in (0.25, 0.5, 1.0)]
    assert all(m > 0 for m in macs)
    assert macs[0] < macs[1] < macs[2]


def test_mac_count_matches_manual_estimate_for_initial_conv() -> None:
    """Lower bound: MACs >= initial conv contribution.

    Initial conv: 1 -> C, kernel 10x4, stride 2x2 over (40, 97).
    Output spatial: (16, 47).
    MACs_initial = 1 * C * 10 * 4 * 16 * 47 = 30080 * C
    """
    model = build_ds_cnn(width_multiplier=0.5)  # C = 112
    macs = count_macs(model)
    initial_macs = 1 * model.channels * 10 * 4 * 16 * 47
    assert macs >= initial_macs
    # Sanity upper bound — we shouldn't be off by more than 100x.
    assert macs < initial_macs * 100


def test_mac_count_is_deterministic() -> None:
    model = build_ds_cnn(width_multiplier=0.5)
    assert count_macs(model) == count_macs(model)


def test_mac_counter_does_not_leave_hooks_attached() -> None:
    model = build_ds_cnn(width_multiplier=0.5)
    count_macs(model)
    for m in model.modules():
        assert not m._forward_hooks, f"{type(m).__name__} still has forward hooks attached"


# ---------------------------------------------------------------------------
# Defaults / argument validation
# ---------------------------------------------------------------------------


def test_defaults_match_module_constants() -> None:
    model = build_ds_cnn()
    assert model.n_blocks == DEFAULT_N_BLOCKS
    assert model.base_channels == DEFAULT_BASE_CHANNELS
    assert model.channels == DEFAULT_BASE_CHANNELS  # width_multiplier=1.0 default


def test_invalid_width_raises() -> None:
    with pytest.raises(ValueError, match="width_multiplier"):
        build_ds_cnn(width_multiplier=0.0)


def test_invalid_n_blocks_raises() -> None:
    with pytest.raises(ValueError, match="n_blocks"):
        build_ds_cnn(n_blocks=0)
