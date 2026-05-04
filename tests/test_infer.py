"""Phase 3 — single-clip inference helper."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 3: implement nano_kws.infer.KwsInferencer")
def test_predict_on_silence_returns_silence_class() -> None:
    """Pure-zero waveform should be classified as _silence_ (with the
    bundled INT8 model). Loose check; intended as a smoke signal."""


@pytest.mark.skip(reason="Phase 3: implement nano_kws.infer.KwsInferencer")
def test_predict_returns_valid_probability_vector() -> None:
    """Output must be length NUM_CLASSES, sum to ~1.0, all entries in [0, 1]."""
