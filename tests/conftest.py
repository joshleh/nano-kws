"""Shared pytest fixtures.

CI never downloads Speech Commands (2.4 GB). All fixtures here are
synthetic so that the test suite runs in seconds on a fresh clone.
"""

from __future__ import annotations

import numpy as np
import pytest

from nano_kws import config


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic numpy RNG for reproducible synthetic audio."""
    return np.random.default_rng(seed=0)


@pytest.fixture
def silence_waveform() -> np.ndarray:
    """1 second of silence at SAMPLE_RATE, float32 in [-1, 1]."""
    return np.zeros(config.CLIP_SAMPLES, dtype=np.float32)


@pytest.fixture
def sine_waveform(rng: np.random.Generator) -> np.ndarray:
    """1 second of a 440 Hz sine wave with a touch of noise."""
    t = np.arange(config.CLIP_SAMPLES, dtype=np.float32) / config.SAMPLE_RATE
    wave = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    wave += rng.normal(0, 0.01, size=wave.shape).astype(np.float32)
    return wave


@pytest.fixture
def white_noise_waveform(rng: np.random.Generator) -> np.ndarray:
    """1 second of white noise scaled to roughly [-0.5, 0.5]."""
    return rng.normal(0, 0.15, size=config.CLIP_SAMPLES).astype(np.float32)


@pytest.fixture
def synthetic_logmel(rng: np.random.Generator) -> np.ndarray:
    """A fake log-mel tensor with the model's expected input shape.

    Lets quantization / inference / benchmark tests run before the real
    featurizer is implemented.
    """
    return rng.standard_normal(config.INPUT_SHAPE).astype(np.float32)
