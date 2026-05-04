"""Phase 1 — augmentation transforms.

Smoke tests with synthetic inputs: shape preservation, training/eval
gating, and a sanity check that a noise mix-in actually changes the
waveform when applied.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nano_kws import config
from nano_kws.data.augment import BackgroundNoiseMixer, SpecAugment


@pytest.fixture
def fake_background(rng: np.random.Generator) -> list[torch.Tensor]:
    """A list of fake background-noise clips, each ~5 s long."""
    return [
        torch.from_numpy(rng.normal(0, 0.1, size=5 * config.SAMPLE_RATE).astype(np.float32))
        for _ in range(3)
    ]


def test_background_noise_mixer_changes_waveform_in_train_mode(
    fake_background: list[torch.Tensor],
    sine_waveform: np.ndarray,
) -> None:
    mixer = BackgroundNoiseMixer(fake_background, p=1.0).train()
    original = torch.from_numpy(sine_waveform)
    out = mixer(original)
    assert out.shape == original.shape
    assert not torch.allclose(out, original)


def test_background_noise_mixer_no_op_in_eval_mode(
    fake_background: list[torch.Tensor],
    sine_waveform: np.ndarray,
) -> None:
    mixer = BackgroundNoiseMixer(fake_background, p=1.0).eval()
    original = torch.from_numpy(sine_waveform)
    out = mixer(original)
    torch.testing.assert_close(out, original)


def test_specaugment_preserves_shape(synthetic_logmel: np.ndarray) -> None:
    aug = SpecAugment(freq_mask_param=8, time_mask_param=16, n_freq_masks=2, n_time_masks=2).train()
    spec = torch.from_numpy(synthetic_logmel)
    out = aug(spec)
    assert out.shape == spec.shape


def test_specaugment_no_op_in_eval_mode(synthetic_logmel: np.ndarray) -> None:
    aug = SpecAugment().eval()
    spec = torch.from_numpy(synthetic_logmel)
    out = aug(spec)
    torch.testing.assert_close(out, spec)


def test_specaugment_validates_param_bounds() -> None:
    with pytest.raises(ValueError, match="freq_mask_param"):
        SpecAugment(freq_mask_param=config.N_MELS + 1)
    with pytest.raises(ValueError, match="time_mask_param"):
        SpecAugment(time_mask_param=config.N_FRAMES + 1)


def test_background_noise_mixer_requires_clips() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BackgroundNoiseMixer([])
