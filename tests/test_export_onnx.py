"""Phase 3 — fp32 ONNX export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from nano_kws import config
from nano_kws.export_onnx import (
    INPUT_NAME,
    OUTPUT_NAME,
    export_to_onnx,
    write_label_map,
)


def test_fp32_onnx_loads_in_onnxruntime(fp32_onnx: Path) -> None:
    session = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    assert len(inputs) == 1 and len(outputs) == 1
    assert inputs[0].name == INPUT_NAME
    assert outputs[0].name == OUTPUT_NAME


def test_fp32_onnx_supports_dynamic_batch(fp32_onnx: Path) -> None:
    session = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    for batch in (1, 8):
        x = np.zeros((batch, *config.INPUT_SHAPE), dtype=np.float32)
        out = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
        assert out.shape == (batch, config.NUM_CLASSES)
        assert out.dtype == np.float32


def test_fp32_onnx_predictions_match_pytorch(
    tiny_model: torch.nn.Module, fp32_onnx: Path, rng: np.random.Generator
) -> None:
    """Round-trip equivalence: ONNX outputs must equal PyTorch within 1e-4."""
    session = ort.InferenceSession(str(fp32_onnx), providers=["CPUExecutionProvider"])
    x = rng.standard_normal((4, *config.INPUT_SHAPE)).astype(np.float32)
    onnx_out = session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
    with torch.no_grad():
        torch_out = tiny_model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(onnx_out, torch_out, rtol=1e-4, atol=1e-4)


def test_export_raises_when_pytorch_and_onnx_disagree(
    tmp_path: Path, tiny_model: torch.nn.Module
) -> None:
    """Smoke check: passing a tighter tolerance than the round-trip can hit fails."""
    out = tmp_path / "should_fail.onnx"
    # Force a tolerance so tight no float export will satisfy it.
    try:
        export_to_onnx(model=tiny_model, output_path=out, verify_tolerance=0.0)
    except RuntimeError as e:
        assert "max abs diff" in str(e)
        return
    # If the tolerance-zero export succeeded, the underlying op set is just
    # that deterministic — fine, no error to raise. (Skip rather than fail.)
    import pytest

    pytest.skip("Export was bit-exact; cannot induce verify failure.")


def test_write_label_map_roundtrip(tmp_path: Path) -> None:
    path = write_label_map(tmp_path / "label_map.json")
    payload = json.loads(path.read_text())
    assert payload["labels"] == list(config.LABELS)
    assert payload["num_classes"] == config.NUM_CLASSES
    assert payload["sample_rate"] == config.SAMPLE_RATE
    assert tuple(payload["input_shape"]) == config.INPUT_SHAPE
    assert payload["input_name"] == INPUT_NAME
    assert payload["output_name"] == OUTPUT_NAME
