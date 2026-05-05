"""Smoke tests for the conv microbench Python plumbing.

The C library may or may not be built on a given machine — these tests
intentionally exercise only the parts that don't need the .dll, plus
graceful skip behaviour for the ones that do. CI runs without building
the kernels, so the suite has to pass either way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.conv_microbench import (
    DEFAULT_C,
    DEFAULT_H,
    DEFAULT_W,
    Result,
    aten_depthwise_3x3,
    aten_pointwise,
    benchmark_op,
    c_depthwise_3x3,
    c_pointwise,
    find_kernel_lib,
    load_c_kernels,
    numpy_depthwise_3x3,
    numpy_pointwise,
    render_markdown,
    update_readme_section,
)

# ---------------------------------------------------------------------------
# Reference correctness: NumPy and ATen agree on tiny inputs.
# ---------------------------------------------------------------------------


def _make_inputs(c: int = 4, h: int = 3, w: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    inp = rng.standard_normal((c, h, w)).astype(np.float32)
    pw_w = rng.standard_normal((c, c)).astype(np.float32)
    dw_w = rng.standard_normal((c, 9)).astype(np.float32)
    return inp, pw_w, dw_w


def test_numpy_pointwise_agrees_with_aten() -> None:
    inp, pw_w, _ = _make_inputs()
    a = numpy_pointwise(inp, pw_w)
    b = aten_pointwise(inp, pw_w)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) < 1e-4


def test_numpy_depthwise_agrees_with_aten() -> None:
    inp, _, dw_w = _make_inputs()
    a = numpy_depthwise_3x3(inp, dw_w)
    b = aten_depthwise_3x3(inp, dw_w)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) < 1e-5


def test_aten_pointwise_output_shape() -> None:
    inp, pw_w, _ = _make_inputs(c=DEFAULT_C, h=DEFAULT_H, w=DEFAULT_W)
    out = aten_pointwise(inp, pw_w)
    assert out.shape == (DEFAULT_C, DEFAULT_H, DEFAULT_W)


def test_aten_depthwise_output_shape() -> None:
    inp, _, dw_w = _make_inputs(c=DEFAULT_C, h=DEFAULT_H, w=DEFAULT_W)
    out = aten_depthwise_3x3(inp, dw_w)
    assert out.shape == (DEFAULT_C, DEFAULT_H, DEFAULT_W)


# ---------------------------------------------------------------------------
# Markdown rendering: well-formed table, includes both ops, computes speedup.
# ---------------------------------------------------------------------------


def _fake_results(op: str) -> list[Result]:
    return [
        Result(
            op=op,
            impl="ATen (reference)",
            mean_ms=0.10,
            p50_ms=0.09,
            p95_ms=0.15,
            correct=None,
            max_abs_err=None,
        ),
        Result(
            op=op,
            impl="NumPy einsum",
            mean_ms=0.50,
            p50_ms=0.48,
            p95_ms=0.70,
            correct=True,
            max_abs_err=1e-6,
        ),
        Result(
            op=op,
            impl="C naive",
            mean_ms=1.00,
            p50_ms=0.98,
            p95_ms=1.20,
            correct=True,
            max_abs_err=1e-6,
        ),
        Result(
            op=op,
            impl="C AVX2",
            mean_ms=0.20,
            p50_ms=0.19,
            p95_ms=0.30,
            correct=True,
            max_abs_err=1e-6,
        ),
    ]


def test_render_markdown_includes_both_ops_and_speedup_column() -> None:
    md = render_markdown(
        _fake_results("pointwise"), _fake_results("depthwise_3x3"), c=8, h=4, w=5, aten_threads=1
    )
    assert "Pointwise (1x1)" in md
    assert "Depthwise 3x3" in md
    assert "Speedup vs C naive" in md
    # ATen at 0.1 ms vs C naive at 1.0 ms -> 10.00x
    assert "10.00x" in md
    # AVX2 at 0.2 ms vs C naive at 1.0 ms -> 5.00x
    assert "5.00x" in md
    # ATen reference row marked as ref
    assert "ref" in md
    # Single-thread context line
    assert "torch.set_num_threads(1)" in md


def test_render_markdown_explains_missing_c_naive_baseline() -> None:
    """When the C kernels aren't built, the rendered table should explain
    why the *Speedup vs C naive* column is blank instead of just showing
    a column full of em-dashes / "n/a"s."""
    aten_only = [
        Result(
            op="pointwise",
            impl="ATen (reference)",
            mean_ms=0.10,
            p50_ms=0.09,
            p95_ms=0.15,
            correct=None,
            max_abs_err=None,
        ),
        Result(
            op="pointwise",
            impl="NumPy einsum",
            mean_ms=0.50,
            p50_ms=0.48,
            p95_ms=0.70,
            correct=True,
            max_abs_err=1e-6,
        ),
    ]
    md = render_markdown(aten_only, aten_only, c=8, h=4, w=5, aten_threads=1)
    # Legend explains both jargon terms.
    assert "Legend." in md
    assert "Speedup vs C naive" in md
    assert "`ref`" in md
    # Explicit "build to populate" note is present.
    assert "make microbench-build" in md
    assert "n/a" in md  # Blank-baseline cell shouldn't be a bare em-dash.


def test_render_markdown_omits_missing_baseline_note_when_c_naive_present() -> None:
    """Sanity-check the inverse: when C naive *is* present, we don't
    spam the build instructions."""
    md = render_markdown(
        _fake_results("pointwise"),
        _fake_results("depthwise_3x3"),
        c=8,
        h=4,
        w=5,
        aten_threads=1,
    )
    assert "make microbench-build" not in md


def test_render_markdown_handles_failed_correctness() -> None:
    bad = _fake_results("pointwise")
    bad[2] = Result(
        op="pointwise",
        impl="C naive",
        mean_ms=1.0,
        p50_ms=1.0,
        p95_ms=1.0,
        correct=False,
        max_abs_err=0.5,
    )
    md = render_markdown(bad, _fake_results("depthwise_3x3"), c=4, h=3, w=5, aten_threads=1)
    assert "NO" in md  # marked as incorrect


# ---------------------------------------------------------------------------
# README stamping behaviour.
# ---------------------------------------------------------------------------


def test_update_readme_section_replaces_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# proj\n\n## Microbench\n\n<!-- BEGIN_MICROBENCH_TABLE -->\nold\n"
        "<!-- END_MICROBENCH_TABLE -->\n\n## Other\n",
        encoding="utf-8",
    )
    assert update_readme_section(readme, "| new |\n| - |\n| 1 |\n") is True
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text
    assert "| new |" in text
    assert "## Microbench" in text
    assert "## Other" in text


def test_update_readme_section_returns_false_when_markers_missing(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# no markers here\n", encoding="utf-8")
    assert update_readme_section(readme, "| x |") is False


# ---------------------------------------------------------------------------
# benchmark_op: smoke through the timing harness with synthetic ops.
# ---------------------------------------------------------------------------


def test_benchmark_op_returns_reference_plus_each_impl() -> None:
    """Drive the timing harness with trivial callables so we can assert
    on its bookkeeping without depending on real conv kernels."""
    arr = np.zeros((2, 3), dtype=np.float32)
    inputs = {
        "reference": ("ref", lambda: arr.copy()),
        "impls": [
            ("good", lambda: arr.copy()),
            ("bad", lambda: arr.copy() + 1.0),
            ("skipped", None),
        ],
    }
    results = benchmark_op("noop", inputs, warmup=1, iters=3, atol=1e-9)
    impls = [r.impl for r in results]
    assert impls == ["ref", "good", "bad"]  # "skipped" filtered out
    # First row is the reference (no correctness check).
    assert results[0].correct is None and results[0].max_abs_err is None
    # "good" matches the reference exactly.
    assert results[1].correct is True
    # "bad" is off by 1.0.
    assert results[2].correct is False
    assert results[2].max_abs_err == pytest.approx(1.0)
    for r in results:
        assert r.mean_ms >= 0.0


# ---------------------------------------------------------------------------
# Optional C-kernel tests — only run if the build artefacts exist.
# ---------------------------------------------------------------------------


_LIB = find_kernel_lib()
_NEED_C = pytest.mark.skipif(_LIB is None, reason="C kernel library not built")


@_NEED_C
def test_c_pointwise_matches_aten() -> None:
    kernels = load_c_kernels(_LIB)
    inp, pw_w, _ = _make_inputs(c=DEFAULT_C, h=DEFAULT_H, w=DEFAULT_W)
    out_aten = aten_pointwise(inp, pw_w)
    out_naive = c_pointwise(kernels.pointwise_naive, inp, pw_w)
    out_avx2 = c_pointwise(kernels.pointwise_avx2, inp, pw_w)
    assert np.max(np.abs(out_aten - out_naive)) < 1e-3
    assert np.max(np.abs(out_aten - out_avx2)) < 1e-3


@_NEED_C
def test_c_depthwise_3x3_matches_aten() -> None:
    kernels = load_c_kernels(_LIB)
    inp, _, dw_w = _make_inputs(c=DEFAULT_C, h=DEFAULT_H, w=DEFAULT_W)
    out_aten = aten_depthwise_3x3(inp, dw_w)
    out_naive = c_depthwise_3x3(kernels.depthwise_3x3_naive, inp, dw_w)
    out_avx2 = c_depthwise_3x3(kernels.depthwise_3x3_avx2, inp, dw_w)
    assert np.max(np.abs(out_aten - out_naive)) < 1e-4
    assert np.max(np.abs(out_aten - out_avx2)) < 1e-4
