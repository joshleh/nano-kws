"""Phase 5 — sweep orchestration unit tests.

These cover the *pure* parts of ``scripts/sweep_sizes`` (table / JSON
rendering, README marker handling, the per-width vs end-of-run split)
without invoking training, export, quantization, or benchmarking — those
are exercised end-to-end by the overnight sweep itself.

The split-tests in particular are the regression tests for the bug
where ``_persist_outputs`` was called both inside and after the
per-width loop and triggered a false-positive
``BEGIN_SWEEP_TABLE markers not found`` warning on the second call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.sweep_sizes import (
    SWEEP_BEGIN,
    SWEEP_END,
    SweepRow,
    _render_final_outputs,
    _write_table_files,
    render_sweep_table,
    update_readme_section,
)


def _make_rows() -> list[SweepRow]:
    return [
        SweepRow(
            width=0.25,
            parameters=18_492,
            macs=12_634_272,
            fp32_size_bytes=76_147,
            int8_size_bytes=44_503,
            fp32_top1=0.7357,
            int8_top1=0.6162,
            fp32_latency_ms=0.330,
            int8_latency_ms=0.306,
            int8_vs_fp32_argmax_agreement=None,
        ),
        SweepRow(
            width=0.5,
            parameters=62_060,
            macs=44_134_720,
            fp32_size_bytes=248_413,
            int8_size_bytes=95_649,
            fp32_top1=0.7932,
            int8_top1=0.7220,
            fp32_latency_ms=0.976,
            int8_latency_ms=0.459,
            int8_vs_fp32_argmax_agreement=None,
        ),
    ]


def _args(tmp_path: Path, *, update_readme: bool = False) -> argparse.Namespace:
    """Minimal argparse Namespace matching what sweep_sizes.main() builds."""
    return argparse.Namespace(
        output=str(tmp_path / "sweep_table.md"),
        plot=str(tmp_path / "sweep_plot.png"),
        readme=str(tmp_path / "README.md"),
        no_plot=True,
        update_readme=update_readme,
    )


# ---------------------------------------------------------------------------
# render_sweep_table
# ---------------------------------------------------------------------------


def test_render_sweep_table_includes_all_widths_and_delta() -> None:
    md = render_sweep_table(_make_rows())
    assert "0.25" in md
    assert "0.5" in md
    assert "73.57%" in md
    assert "61.62%" in md
    # Negative delta surfaced with explicit sign.
    assert "-11.95 pp" in md or "-11.95\u00a0pp" in md
    # Headers we promise the README about.
    for header in ("Width", "Params", "MACs", "fp32 top-1", "INT8 top-1", "Δ acc"):
        assert header in md


def test_render_sweep_table_handles_no_accuracy() -> None:
    rows = _make_rows()
    rows[0].fp32_top1 = None
    rows[0].int8_top1 = None
    md = render_sweep_table(rows)
    assert "—" in md  # Δ acc + accuracies render as em-dash


def test_render_sweep_table_returns_placeholder_when_empty() -> None:
    md = render_sweep_table([])
    assert "no sweep results" in md


# ---------------------------------------------------------------------------
# Per-width incremental writes (regression: was bundled into _persist_outputs)
# ---------------------------------------------------------------------------


def test_write_table_files_writes_md_and_json(tmp_path: Path) -> None:
    args = _args(tmp_path)
    rows = _make_rows()

    table_md = _write_table_files(rows, args)

    md_path = Path(args.output)
    json_path = md_path.with_suffix(".json")
    assert md_path.is_file()
    assert json_path.is_file()
    assert "0.25" in md_path.read_text(encoding="utf-8")
    assert table_md == md_path.read_text(encoding="utf-8")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [r["width"] for r in payload] == [0.25, 0.5]  # sorted ascending
    assert payload[1]["parameters"] == 62_060


def test_write_table_files_does_not_touch_readme(tmp_path: Path) -> None:
    """Regression: the per-width incremental call must not stamp README,
    otherwise the bug from yesterday's overnight run reappears (false-
    positive 'markers not found' warning, and N redundant writes).
    """
    args = _args(tmp_path, update_readme=True)
    readme = Path(args.readme)
    readme.write_text(
        f"# repo\n\n{SWEEP_BEGIN}\nplaceholder\n{SWEEP_END}\n",
        encoding="utf-8",
    )

    _write_table_files(_make_rows(), args)

    # README is untouched: still has the placeholder, no actual table data.
    text = readme.read_text(encoding="utf-8")
    assert "placeholder" in text
    assert "0.25" not in text


# ---------------------------------------------------------------------------
# End-of-run rendering
# ---------------------------------------------------------------------------


def test_render_final_outputs_stamps_readme_when_markers_present(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, update_readme=True)
    readme = Path(args.readme)
    readme.write_text(
        f"# repo\n\n## Sweep\n\n{SWEEP_BEGIN}\nplaceholder\n{SWEEP_END}\n\n## Tail\n",
        encoding="utf-8",
    )

    _render_final_outputs(_make_rows(), args)

    text = readme.read_text(encoding="utf-8")
    assert "placeholder" not in text
    assert "0.25" in text
    assert "0.5" in text
    # Surrounding markdown is preserved.
    assert "## Sweep" in text
    assert "## Tail" in text


def test_render_final_outputs_warns_then_continues_when_markers_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    args = _args(tmp_path, update_readme=True)
    Path(args.readme).write_text("# repo\n\nNo markers.\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        _render_final_outputs(_make_rows(), args)

    # Table file was still written even though README didn't take.
    assert Path(args.output).is_file()
    assert any("markers" in rec.message.lower() for rec in caplog.records)


def test_render_final_outputs_no_op_on_readme_when_flag_off(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, update_readme=False)
    readme = Path(args.readme)
    readme.write_text(
        f"# repo\n\n{SWEEP_BEGIN}\nplaceholder\n{SWEEP_END}\n",
        encoding="utf-8",
    )

    _render_final_outputs(_make_rows(), args)

    assert "placeholder" in readme.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# update_readme_section helper directly
# ---------------------------------------------------------------------------


def test_update_readme_section_replaces_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"head\n{SWEEP_BEGIN}\nold\n{SWEEP_END}\ntail\n", encoding="utf-8")
    assert (
        update_readme_section(
            readme, table_md="| new |\n| --- |\n", begin=SWEEP_BEGIN, end=SWEEP_END
        )
        is True
    )
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text
    assert "| new |" in text
    assert "head" in text
    assert "tail" in text


def test_update_readme_section_returns_false_when_markers_absent(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers", encoding="utf-8")
    assert (
        update_readme_section(readme, table_md="| x |\n", begin=SWEEP_BEGIN, end=SWEEP_END) is False
    )
