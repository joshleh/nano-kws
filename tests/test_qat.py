"""Tests for the QAT module: fake-quant primitives + prepare/strip round-trip.

The training loop itself is exercised end-to-end (one batch) to make
sure the wired-together pipeline doesn't blow up with shape mismatches
or autograd issues. The test suite never downloads Speech Commands; we
synthesise tiny inputs and confirm that a single prepared forward +
backward updates the weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_kws import config
from nano_kws.models.ds_cnn import build_ds_cnn
from nano_kws.qat import (
    QMAX_INT8,
    QMIN_INT8,
    ActivationObserver,
    QatConv2d,
    QatLinear,
    count_qat_observers,
    fake_quantize_asymmetric,
    fake_quantize_symmetric,
    freeze_observers,
    per_channel_weight_fake_quant,
    prepare_qat,
    strip_qat,
)

# ---------------------------------------------------------------------------
# fake_quantize_symmetric: forward maps to the INT8 grid, STE backward = identity.
# ---------------------------------------------------------------------------


def test_fake_quantize_forward_lies_on_int8_grid() -> None:
    """Output values should be exactly k * scale for k in [-128, 127]."""
    x = torch.linspace(-1.0, 1.0, 1000)
    scale = torch.tensor(0.01)
    y = fake_quantize_symmetric(x, scale)
    # Every output should be representable as an integer multiple of scale.
    ratios = y / scale
    diffs = (ratios - ratios.round()).abs().max().item()
    assert diffs < 1e-4
    assert int(ratios.min().item()) >= QMIN_INT8
    assert int(ratios.max().item()) <= QMAX_INT8


def test_fake_quantize_backward_is_identity_ste() -> None:
    """STE: gradient w.r.t. x is passed through unchanged."""
    x = torch.randn(8, requires_grad=True)
    scale = torch.tensor(0.05)
    y = fake_quantize_symmetric(x, scale)
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_fake_quantize_clamp_saturates_at_qmax() -> None:
    """Values beyond qmax * scale clamp to qmax * scale."""
    scale = torch.tensor(1.0)
    x = torch.tensor([1000.0, -1000.0, 0.0])
    y = fake_quantize_symmetric(x, scale)
    assert y[0].item() == float(QMAX_INT8)
    assert y[1].item() == float(QMIN_INT8)
    assert y[2].item() == 0.0


# ---------------------------------------------------------------------------
# per_channel_weight_fake_quant.
# ---------------------------------------------------------------------------


def test_per_channel_weight_fake_quant_uses_independent_scales() -> None:
    """Each output channel should use its own scale: a wildly larger
    channel shouldn't drag the smaller channels' precision down."""
    w = torch.zeros(3, 2, 3, 3)
    w[0] = 0.01  # tiny channel
    w[1] = 1.0  # medium channel
    w[2] = 100.0  # huge channel
    q = per_channel_weight_fake_quant(w)
    # Every channel's output should equal the input within its own
    # per-channel scale's grid resolution (each channel is constant, so
    # quantization is exact).
    for i in range(3):
        assert torch.allclose(q[i], w[i], atol=w[i].abs().max().item() / float(QMAX_INT8) + 1e-9)


def test_per_channel_weight_fake_quant_handles_zero_channel() -> None:
    """A zero-valued output channel should pass through without NaN."""
    w = torch.zeros(2, 3)
    w[0] = 1.0
    q = per_channel_weight_fake_quant(w)
    assert torch.isfinite(q).all()
    assert q[1].abs().sum().item() == 0.0


def test_per_channel_weight_fake_quant_supports_linear_2d_shape() -> None:
    """Linear weights are 2D; the per-channel reduction must still work."""
    w = torch.randn(7, 13)
    q = per_channel_weight_fake_quant(w)
    assert q.shape == w.shape
    assert torch.isfinite(q).all()


# ---------------------------------------------------------------------------
# ActivationObserver: EMA initialisation + freezing semantics.
# ---------------------------------------------------------------------------


def test_activation_observer_initialises_on_first_train_batch() -> None:
    obs = ActivationObserver()
    assert not bool(obs.initialized.item())
    obs.train()
    x = torch.randn(4, 8) * 5.0
    obs(x)
    assert bool(obs.initialized.item())
    assert obs.running_max.item() > obs.running_min.item()


def test_activation_observer_eval_mode_does_not_update_ema() -> None:
    obs = ActivationObserver()
    obs.train()
    obs(torch.ones(2) * 3.0)
    initial_min = obs.running_min.item()
    initial_max = obs.running_max.item()
    obs.eval()
    obs(torch.ones(2) * 100.0)
    assert obs.running_min.item() == initial_min
    assert obs.running_max.item() == initial_max


def test_activation_observer_freeze_stops_ema_in_train_mode() -> None:
    obs = ActivationObserver(momentum=0.5)
    obs.train()
    obs(torch.ones(2) * 1.0)
    initial_min = obs.running_min.item()
    initial_max = obs.running_max.item()
    obs.freeze()
    obs(torch.ones(2) * 100.0)
    assert obs.running_min.item() == initial_min
    assert obs.running_max.item() == initial_max


def test_activation_observer_passes_through_when_uninitialised() -> None:
    """In eval mode before any train-mode forward, the observer is a no-op."""
    obs = ActivationObserver()
    obs.eval()
    x = torch.randn(4, 8)
    out = obs(x)
    assert torch.allclose(out, x)


def test_activation_observer_uses_asymmetric_grid_for_one_sided_inputs() -> None:
    """For all-positive inputs (think post-ReLU), the asymmetric scheme
    should pick zero_point = qmin so the full INT8 range maps to the
    actual data range — symmetric would waste the negative half."""
    obs = ActivationObserver()
    obs.train()
    x = torch.linspace(0.0, 4.0, 1000)
    obs(x)
    scale, zero_point = obs._scale_zero_point()
    assert torch.isclose(zero_point, torch.tensor(float(QMIN_INT8)), atol=1.0)
    expected_scale = 4.0 / float(QMAX_INT8 - QMIN_INT8)
    assert abs(scale.item() - expected_scale) < 1e-6


def test_fake_quantize_asymmetric_round_trips_endpoints() -> None:
    """The endpoints of the calibration range should fake-quant back to
    themselves (modulo a single grid step), proving the affine map is
    set up correctly."""
    scale = torch.tensor(0.02)
    zero_point = torch.tensor(-128.0)
    x = torch.tensor([0.0, 5.10])
    y = fake_quantize_asymmetric(x, scale, zero_point)
    assert abs(y[0].item()) < scale.item()
    assert abs(y[1].item() - x[1].item()) < scale.item()


# ---------------------------------------------------------------------------
# QatConv2d / QatLinear forward shape + parameter aliasing.
# ---------------------------------------------------------------------------


def test_qat_conv2d_preserves_output_shape() -> None:
    conv = nn.Conv2d(4, 8, kernel_size=3, padding=1)
    qat = QatConv2d(conv)
    qat.train()
    x = torch.randn(2, 4, 16, 16)
    y = qat(x)
    assert y.shape == (2, 8, 16, 16)


def test_qat_linear_preserves_output_shape() -> None:
    linear = nn.Linear(16, 32)
    qat = QatLinear(linear)
    qat.train()
    x = torch.randn(5, 16)
    y = qat(x)
    assert y.shape == (5, 32)


def test_qat_wrappers_share_parameter_storage_with_inner_module() -> None:
    """The wrapped Conv2d's weight tensor *is* the same object as the
    QatConv2d's; training the wrapper updates the inner module."""
    conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    qat = QatConv2d(conv)
    assert qat.conv.weight.data_ptr() == conv.weight.data_ptr()


# ---------------------------------------------------------------------------
# prepare_qat / strip_qat round-trip on a real DS-CNN.
# ---------------------------------------------------------------------------


def test_prepare_qat_replaces_every_conv_and_linear() -> None:
    model = build_ds_cnn(width_multiplier=0.25)
    n_conv_before = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_linear_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    model = prepare_qat(model)
    n_qat_conv = sum(1 for m in model.modules() if isinstance(m, QatConv2d))
    n_qat_linear = sum(1 for m in model.modules() if isinstance(m, QatLinear))
    assert n_qat_conv == n_conv_before
    assert n_qat_linear == n_linear_before
    assert count_qat_observers(model) == n_qat_conv + n_qat_linear


def test_prepare_qat_then_strip_qat_round_trip_preserves_output() -> None:
    """Strip after prepare → forward should match the original (modulo
    the activation observer never having seen data, in which case it's a
    pass-through). We disable train mode so no quantization noise is
    injected; the equality is exact."""
    torch.manual_seed(0)
    model = build_ds_cnn(width_multiplier=0.25).eval()
    x = torch.randn(2, *config.INPUT_SHAPE)
    with torch.no_grad():
        y_before = model(x).clone()

    prepare_qat(model)
    # In eval mode with uninitialised observers, the QAT path collapses to
    # input passthrough + per-channel weight fake-quant. Rather than
    # asserting exact equality (which weight quantization breaks), we
    # check that strip restores numerical identity.
    strip_qat(model)
    with torch.no_grad():
        y_after = model(x)
    assert torch.allclose(y_before, y_after, atol=1e-6)


def test_qat_model_forward_backward_updates_weights() -> None:
    """End-to-end: a single batch through prepare_qat + criterion + step
    should produce non-zero gradients on every Conv2d / Linear weight,
    proving the STE backward is wired correctly."""
    torch.manual_seed(0)
    model = build_ds_cnn(width_multiplier=0.25)
    prepare_qat(model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, *config.INPUT_SHAPE)
    y = torch.randint(0, config.NUM_CLASSES, (4,))
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    # Every Conv2d / Linear inside the wrappers should have a finite,
    # non-zero gradient.
    n_checked = 0
    for module in model.modules():
        if isinstance(module, (QatConv2d, QatLinear)):
            inner = module.conv if isinstance(module, QatConv2d) else module.linear
            assert inner.weight.grad is not None
            assert torch.isfinite(inner.weight.grad).all()
            assert inner.weight.grad.abs().sum().item() > 0
            n_checked += 1
    assert n_checked > 0
    optimizer.step()  # smoke: optimiser.step on QAT params should not raise


def test_freeze_observers_returns_count_and_freezes_all() -> None:
    model = build_ds_cnn(width_multiplier=0.25)
    prepare_qat(model)
    n_obs = count_qat_observers(model)
    n_frozen = freeze_observers(model)
    assert n_frozen == n_obs
    for module in model.modules():
        if isinstance(module, ActivationObserver):
            assert bool(module.frozen.item())


def test_strip_qat_state_dict_loadable_into_vanilla_dscnn() -> None:
    """The whole point of QAT-then-strip: the resulting state_dict must
    drop into a freshly constructed DS-CNN with no name surgery."""
    torch.manual_seed(0)
    width = 0.25
    model = build_ds_cnn(width_multiplier=width)
    prepare_qat(model)
    model.train()
    # Run a forward to make sure observers are initialised (so any
    # observer-related buffers exist).
    model(torch.randn(2, *config.INPUT_SHAPE))
    strip_qat(model)

    fresh = build_ds_cnn(width_multiplier=width)
    fresh.load_state_dict(model.state_dict())
    # And vice versa: the param keys should match exactly.
    assert set(fresh.state_dict().keys()) == set(model.state_dict().keys())
