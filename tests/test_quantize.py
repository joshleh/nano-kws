"""Phase 3 — static PTQ + INT8 ONNX export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from nano_kws import config
from nano_kws.export_onnx import INPUT_NAME, OUTPUT_NAME
from nano_kws.quantize import (
    FeatureCalibrationReader,
    synthetic_calibration_batches,
    verify_int8_argmax_agreement,
)


def test_synthetic_calibration_batches_have_expected_shape() -> None:
    batches = synthetic_calibration_batches(n_batches=3, batch_size=4)
    assert len(batches) == 3
    for b in batches:
        assert b.shape == (4, *config.INPUT_SHAPE)
        assert b.dtype == np.float32


def test_calibration_reader_returns_dicts_then_none() -> None:
    batches = synthetic_calibration_batches(n_batches=2, batch_size=2)
    reader = FeatureCalibrationReader(batches)
    first = reader.get_next()
    assert first is not None
    assert INPUT_NAME in first
    assert first[INPUT_NAME].shape == (2, *config.INPUT_SHAPE)
    assert reader.get_next() is not None
    assert reader.get_next() is None  # exhausted


def test_int8_onnx_loads_in_onnxruntime(int8_onnx: Path) -> None:
    session = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == INPUT_NAME
    assert session.get_outputs()[0].name == OUTPUT_NAME


def test_int8_onnx_runs_forward_with_correct_shape(int8_onnx: Path) -> None:
    session = ort.InferenceSession(str(int8_onnx), providers=["CPUExecutionProvider"])
    x = np.zeros((2, *config.INPUT_SHAPE), dtype=np.float32)
    out = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
    assert out.shape == (2, config.NUM_CLASSES)


def test_int8_model_is_smaller_than_fp32(fp32_onnx: Path, int8_onnx: Path) -> None:
    """Loose bound: per-channel QDQ + the pre-processing bias-initializer
    expansion gives roughly 50-70% of fp32 size on tiny models, not the
    textbook 4x. The compression ratio improves with model size — a real
    100K-param checkpoint typically lands closer to 30-40% of fp32."""
    fp32_size = fp32_onnx.stat().st_size
    int8_size = int8_onnx.stat().st_size
    assert int8_size < fp32_size * 0.75, (
        f"INT8 should be noticeably smaller than fp32; "
        f"got int8={int8_size}B fp32={fp32_size}B (ratio={int8_size / fp32_size:.2%})"
    )


def test_int8_argmax_agrees_with_fp32_on_calibration_inputs(
    tiny_model: torch.nn.Module, int8_onnx: Path
) -> None:
    """Soft check: on the same inputs used for calibration, INT8 argmax should
    agree with fp32 argmax most of the time. We expect well over 50% even on
    a freshly-initialised (untrained) model — full agreement is not required."""
    batches = synthetic_calibration_batches(n_batches=8, batch_size=4)
    agreement = verify_int8_argmax_agreement(
        int8_path=int8_onnx, fp32_model=tiny_model, batches=batches
    )
    assert 0.0 <= agreement <= 1.0
    assert agreement > 0.5, f"expected >50% argmax agreement, got {agreement:.4f}"
