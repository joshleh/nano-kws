"""Phase 3 — static PTQ + INT8 ONNX export round-trip."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 3: implement nano_kws.quantize")
def test_int8_onnx_loads_in_onnxruntime() -> None:
    """Quantized export must load in onnxruntime InferenceSession."""


@pytest.mark.skip(reason="Phase 3: implement nano_kws.quantize")
def test_int8_predictions_agree_with_torch_quantized_model() -> None:
    """argmax(onnxruntime INT8) == argmax(torch quantized) on calibration batch."""


@pytest.mark.skip(reason="Phase 3: implement nano_kws.quantize")
def test_int8_model_is_smaller_than_fp32() -> None:
    """INT8 ONNX should be roughly 1/4 the size of fp32 ONNX."""
