"""Shared pytest fixtures.

CI never downloads Speech Commands (2.4 GB). All fixtures here are
synthetic so that the test suite runs in seconds on a fresh clone.

Phase 3 fixtures (``tiny_model``, ``fp32_onnx``, ``int8_onnx``) build a
freshly-initialised DS-CNN and run it through the full export +
quantization pipeline once per session, so the export/quantize/infer
test files can share the artefacts without paying the construction cost
each test.
"""

from __future__ import annotations

from pathlib import Path

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
    """A fake log-mel tensor with the model's expected input shape."""
    return rng.standard_normal(config.INPUT_SHAPE).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 3 fixtures: a tiny DS-CNN built and exported once per test session.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model():
    """Smallest DS-CNN we sweep over (w=0.25). Float, eval mode, on CPU."""
    import torch

    from nano_kws.models.ds_cnn import build_ds_cnn

    torch.manual_seed(0)
    model = build_ds_cnn(width_multiplier=0.25).cpu().eval()
    return model


@pytest.fixture(scope="session")
def fp32_onnx(tmp_path_factory: pytest.TempPathFactory, tiny_model) -> Path:
    """Tiny DS-CNN exported to fp32 ONNX (built once per test session)."""
    from nano_kws.export_onnx import export_to_onnx

    out = tmp_path_factory.mktemp("nano_kws_p3") / "tiny_fp32.onnx"
    return export_to_onnx(model=tiny_model, output_path=out)


@pytest.fixture(scope="session")
def int8_onnx(tmp_path_factory: pytest.TempPathFactory, tiny_model, fp32_onnx) -> Path:
    """Tiny DS-CNN quantised to INT8 ONNX (synthetic calibration)."""
    from nano_kws.quantize import quantize_onnx, synthetic_calibration_batches

    out = tmp_path_factory.mktemp("nano_kws_p3_int8") / "tiny_int8.onnx"
    batches = synthetic_calibration_batches(n_batches=8, batch_size=4)
    return quantize_onnx(fp32_path=fp32_onnx, int8_path=out, calibration_batches=batches)
