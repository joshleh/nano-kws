"""Phase 4 — benchmark end-to-end smoke."""

from __future__ import annotations

from pathlib import Path

import torch

from nano_kws.benchmark import (
    README_BEGIN,
    README_END,
    benchmark_onnx,
    benchmark_pytorch,
    render_markdown,
    update_readme_table,
)
from nano_kws.models.ds_cnn import count_macs, count_parameters


def test_benchmark_pytorch_returns_complete_result(tiny_model: torch.nn.Module) -> None:
    res = benchmark_pytorch(
        model=tiny_model,
        name="tiny",
        accuracy_loader=None,
        warmup=2,
        iters=5,
    )
    assert res.runtime.startswith("PyTorch")
    assert res.parameters > 0
    assert res.macs > 0
    assert res.file_size_bytes is None
    assert res.top1 is None
    assert res.latency_mean_ms > 0
    assert res.latency_p95_ms >= res.latency_p50_ms


def test_benchmark_onnx_returns_complete_result(
    tiny_model: torch.nn.Module, fp32_onnx: Path
) -> None:
    res = benchmark_onnx(
        onnx_path=fp32_onnx,
        name="tiny fp32",
        parameters=count_parameters(tiny_model),
        macs=count_macs(tiny_model),
        accuracy_loader=None,
        warmup=2,
        iters=5,
    )
    assert res.runtime.startswith("ONNX")
    assert res.file_size_bytes == fp32_onnx.stat().st_size
    assert res.latency_mean_ms > 0


def test_render_markdown_includes_all_variants(
    tiny_model: torch.nn.Module, fp32_onnx: Path, int8_onnx: Path
) -> None:
    n_params = count_parameters(tiny_model)
    n_macs = count_macs(tiny_model)
    results = [
        benchmark_pytorch(
            model=tiny_model, name="DS-CNN tiny fp32", accuracy_loader=None, warmup=2, iters=5
        ),
        benchmark_onnx(
            onnx_path=fp32_onnx,
            name="DS-CNN tiny fp32",
            parameters=n_params,
            macs=n_macs,
            accuracy_loader=None,
            warmup=2,
            iters=5,
        ),
        benchmark_onnx(
            onnx_path=int8_onnx,
            name="DS-CNN tiny INT8",
            parameters=n_params,
            macs=n_macs,
            accuracy_loader=None,
            warmup=2,
            iters=5,
        ),
    ]
    md = render_markdown(results)
    assert "DS-CNN tiny INT8" in md
    assert "DS-CNN tiny fp32" in md
    assert "INT8 (PTQ) vs fp32" in md
    assert "Latency mean" in md
    assert "Size on disk" in md


def test_render_markdown_includes_qat_row_when_present(
    tiny_model: torch.nn.Module, fp32_onnx: Path, int8_onnx: Path
) -> None:
    """When both PTQ-only and QAT INT8 rows are present, the table grows
    a second summary block calling out the QAT vs PTQ delta."""
    n_params = count_parameters(tiny_model)
    n_macs = count_macs(tiny_model)
    results = [
        benchmark_onnx(
            onnx_path=fp32_onnx,
            name="DS-CNN tiny fp32",
            parameters=n_params,
            macs=n_macs,
            accuracy_loader=None,
            warmup=2,
            iters=5,
        ),
        benchmark_onnx(
            onnx_path=int8_onnx,
            name="DS-CNN tiny INT8 (PTQ)",
            parameters=n_params,
            macs=n_macs,
            accuracy_loader=None,
            warmup=2,
            iters=5,
        ),
        # Reuse the same INT8 file as a stand-in for "QAT INT8" so the
        # smoke test doesn't depend on actually running QAT.
        benchmark_onnx(
            onnx_path=int8_onnx,
            name="DS-CNN tiny INT8 (QAT)",
            parameters=n_params,
            macs=n_macs,
            accuracy_loader=None,
            warmup=2,
            iters=5,
        ),
    ]
    md = render_markdown(results)
    assert "INT8 (PTQ)" in md
    assert "INT8 (QAT)" in md
    assert "INT8 (QAT) vs fp32" in md


def test_update_readme_table_replaces_section(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Project\n\n## TL;DR\n\n{README_BEGIN}\nold table\n{README_END}\n\n## Other\n",
        encoding="utf-8",
    )
    updated = update_readme_table(readme, "| new | table |\n| --- | --- |\n| 1 | 2 |\n")
    assert updated is True
    text = readme.read_text(encoding="utf-8")
    assert "old table" not in text
    assert "| new | table |" in text
    # Surrounding markdown is preserved.
    assert "## Other" in text
    assert "## TL;DR" in text


def test_update_readme_table_returns_false_when_markers_missing(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Project\n\nNo markers here.\n", encoding="utf-8")
    assert update_readme_table(readme, "| x |\n| - |\n") is False
    assert "No markers here" in readme.read_text(encoding="utf-8")


def test_update_readme_table_handles_missing_file(tmp_path: Path) -> None:
    assert update_readme_table(tmp_path / "nope.md", "| x |\n| - |\n") is False
