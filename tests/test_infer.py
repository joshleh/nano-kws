"""Phase 3 — single-clip inference helper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nano_kws import config
from nano_kws.infer import KwsInferencer


def test_inferencer_loads_fp32_model(fp32_onnx: Path) -> None:
    inf = KwsInferencer(fp32_onnx)
    assert len(inf.labels) == config.NUM_CLASSES


def test_inferencer_loads_int8_model(int8_onnx: Path) -> None:
    inf = KwsInferencer(int8_onnx)
    assert len(inf.labels) == config.NUM_CLASSES


def test_predict_returns_valid_probability_vector(
    fp32_onnx: Path, sine_waveform: np.ndarray
) -> None:
    inf = KwsInferencer(fp32_onnx)
    probs = inf.predict(sine_waveform)
    assert probs.shape == (config.NUM_CLASSES,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert pytest.approx(probs.sum(), abs=1e-5) == 1.0


def test_predict_label_returns_label_and_confidence(
    fp32_onnx: Path, sine_waveform: np.ndarray
) -> None:
    inf = KwsInferencer(fp32_onnx)
    label, conf = inf.predict_label(sine_waveform)
    assert label in config.LABELS
    assert 0.0 <= conf <= 1.0


def test_predict_handles_short_and_long_waveforms(
    fp32_onnx: Path, rng: np.random.Generator
) -> None:
    """Inputs are pad/cropped before featurisation, so any length should work."""
    inf = KwsInferencer(fp32_onnx)
    short = rng.standard_normal(config.CLIP_SAMPLES // 2).astype(np.float32)
    long = rng.standard_normal(config.CLIP_SAMPLES * 3).astype(np.float32)
    for w in (short, long):
        probs = inf.predict(w)
        assert probs.shape == (config.NUM_CLASSES,)


def test_predict_batch_returns_batched_probs(fp32_onnx: Path, rng: np.random.Generator) -> None:
    inf = KwsInferencer(fp32_onnx)
    waveforms = rng.standard_normal((4, config.CLIP_SAMPLES)).astype(np.float32)
    import torch

    probs = inf.predict_batch(torch.from_numpy(waveforms))
    assert probs.shape == (4, config.NUM_CLASSES)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_inferencer_loads_label_map_sidecar(fp32_onnx: Path, tmp_path: Path) -> None:
    """When a .label_map.json sits next to the .onnx, it should be picked up."""
    custom_labels = ["custom_" + lbl for lbl in config.LABELS]
    sidecar = fp32_onnx.with_suffix(".label_map.json")
    sidecar.write_text(
        json.dumps(
            {
                "labels": custom_labels,
                "num_classes": len(custom_labels),
                "sample_rate": config.SAMPLE_RATE,
                "input_shape": list(config.INPUT_SHAPE),
                "input_name": "input",
                "output_name": "logits",
            }
        )
    )
    try:
        inf = KwsInferencer(fp32_onnx)
        assert inf.labels == tuple(custom_labels)
    finally:
        sidecar.unlink()


def test_inferencer_raises_when_model_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        KwsInferencer(tmp_path / "does_not_exist.onnx")
