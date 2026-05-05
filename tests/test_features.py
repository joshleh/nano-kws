"""Phase 1 — log-mel featurizer.

These tests pin the contract that every other phase relies on: the
featurizer is deterministic, accepts numpy and torch input, and produces
a tensor of exactly ``config.INPUT_SHAPE`` for a 1-second clip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nano_kws import config
from nano_kws.data.features import (
    LOG_EPS,
    LogMelSpectrogram,
    pad_or_crop,
    waveform_to_logmel,
)


def test_logmel_output_shape_single(silence_waveform: np.ndarray) -> None:
    out = waveform_to_logmel(silence_waveform)
    assert out.shape == config.INPUT_SHAPE
    assert out.dtype == torch.float32


def test_logmel_output_shape_batched(rng: np.random.Generator) -> None:
    batch = rng.standard_normal((4, config.CLIP_SAMPLES)).astype(np.float32)
    out = waveform_to_logmel(batch)
    assert out.shape == (4, *config.INPUT_SHAPE)
    assert out.dtype == torch.float32


def test_logmel_accepts_numpy_and_torch(sine_waveform: np.ndarray) -> None:
    from_numpy = waveform_to_logmel(sine_waveform)
    from_torch = waveform_to_logmel(torch.from_numpy(sine_waveform))
    torch.testing.assert_close(from_numpy, from_torch)


def test_logmel_is_deterministic(sine_waveform: np.ndarray) -> None:
    a = waveform_to_logmel(sine_waveform)
    b = waveform_to_logmel(sine_waveform)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_logmel_silence_floor(silence_waveform: np.ndarray) -> None:
    """Pure silence should saturate the log floor at log(LOG_EPS)."""
    out = waveform_to_logmel(silence_waveform)
    expected = float(np.log(LOG_EPS))
    assert torch.allclose(out, torch.full_like(out, expected))


def test_logmel_speech_has_more_energy_than_silence(
    silence_waveform: np.ndarray, sine_waveform: np.ndarray
) -> None:
    silent = waveform_to_logmel(silence_waveform)
    voiced = waveform_to_logmel(sine_waveform)
    assert voiced.mean().item() > silent.mean().item() + 5.0


def test_logmel_rejects_wrong_length(rng: np.random.Generator) -> None:
    short = rng.standard_normal(config.CLIP_SAMPLES // 2).astype(np.float32)
    with pytest.raises(RuntimeError, match="Run `pad_or_crop`"):
        waveform_to_logmel(short)


def test_logmel_module_can_be_called_directly(sine_waveform: np.ndarray) -> None:
    """The nn.Module form is what training loops will use; smoke-test it."""
    featurizer = LogMelSpectrogram().eval()
    with torch.no_grad():
        out = featurizer(torch.from_numpy(sine_waveform))
    assert out.shape == config.INPUT_SHAPE


def test_pad_short_waveform_returns_clip_samples(rng: np.random.Generator) -> None:
    short = rng.standard_normal(config.CLIP_SAMPLES // 2).astype(np.float32)
    out = pad_or_crop(short)
    assert out.shape == (config.CLIP_SAMPLES,)
    n_zeros = (out == 0.0).sum().item()
    n_pad = config.CLIP_SAMPLES - len(short)
    assert n_zeros >= n_pad


def test_crop_long_waveform_returns_clip_samples(rng: np.random.Generator) -> None:
    long = rng.standard_normal(config.CLIP_SAMPLES * 2).astype(np.float32)
    out = pad_or_crop(long)
    assert out.shape == (config.CLIP_SAMPLES,)
    expected = torch.from_numpy(
        long[config.CLIP_SAMPLES // 2 : config.CLIP_SAMPLES // 2 + config.CLIP_SAMPLES]
    )
    torch.testing.assert_close(out, expected)


def test_pad_or_crop_idempotent_on_correct_length(sine_waveform: np.ndarray) -> None:
    out = pad_or_crop(sine_waveform)
    assert out.shape == (config.CLIP_SAMPLES,)
    torch.testing.assert_close(out, torch.from_numpy(sine_waveform))


def test_pad_or_crop_preserves_batch_dim(rng: np.random.Generator) -> None:
    batch = rng.standard_normal((3, config.CLIP_SAMPLES // 2)).astype(np.float32)
    out = pad_or_crop(batch)
    assert out.shape == (3, config.CLIP_SAMPLES)


def test_pad_or_crop_always_returns_torch_float32_tensor(rng: np.random.Generator) -> None:
    """Locks the return-type contract `pad_or_crop` advertises in its
    docstring. Callers (e.g. `app/streamlit_app.py`) rely on this and
    will break if the function ever silently switches to returning the
    input type — see the `.astype()` regression hit during local
    Streamlit testing."""
    for inp in (
        rng.standard_normal(config.CLIP_SAMPLES // 2).astype(np.float32),
        rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32),
        rng.standard_normal(config.CLIP_SAMPLES * 2).astype(np.float32),
        torch.from_numpy(rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)),
        torch.from_numpy(
            rng.standard_normal(config.CLIP_SAMPLES // 3).astype(np.float64)
        ),
    ):
        out = pad_or_crop(inp)
        assert isinstance(out, torch.Tensor)
        assert out.dtype == torch.float32
