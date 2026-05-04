"""Phase 2 — DS-CNN architecture."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 2: implement nano_kws.models.ds_cnn.build_ds_cnn")
def test_forward_pass_shape() -> None:
    """Forward on (B, *INPUT_SHAPE) -> (B, NUM_CLASSES) logits."""


@pytest.mark.skip(reason="Phase 2: implement parameter / MAC counters")
def test_parameter_count_budget_per_width() -> None:
    """w=0.25 ~10K params; w=0.5 ~60K params; w=1.0 ~240K params (within +/-50%)."""


@pytest.mark.skip(reason="Phase 2: implement MAC counter")
def test_mac_count_is_positive_and_scales_with_width() -> None:
    """MACs(w=1.0) > MACs(w=0.5) > MACs(w=0.25)."""
