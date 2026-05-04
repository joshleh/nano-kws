"""Phase 4 — benchmark end-to-end smoke test."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Phase 4: implement nano_kws.benchmark")
def test_benchmark_runs_latency_only_on_bundled_int8_model() -> None:
    """`nano_kws.benchmark --int8 assets/... --skip-accuracy` should run end
    to end and write the Markdown table."""
