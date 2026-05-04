"""Phase 1 — log-mel featurizer.

Currently skipped pending implementation. The intended assertions are
documented inline so the contract is unambiguous when work begins.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 1: implement nano_kws.data.features.waveform_to_logmel")
def test_logmel_output_shape() -> None:
    """Output must equal config.INPUT_SHAPE for a 1-second clip."""


@pytest.mark.skip(reason="Phase 1: implement nano_kws.data.features.waveform_to_logmel")
def test_logmel_is_deterministic() -> None:
    """Same input -> bit-exact same output across calls."""


@pytest.mark.skip(reason="Phase 1: implement nano_kws.data.features.pad_or_crop")
def test_pad_or_crop_short_and_long_inputs() -> None:
    """Short -> zero-padded to CLIP_SAMPLES; long -> center-cropped."""
