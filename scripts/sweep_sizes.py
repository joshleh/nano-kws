"""Phase 5 — train DS-CNN at multiple widths, render the trade-off table.

Trains a DS-CNN at each requested ``--widths`` value, exports + quantizes
each, benchmarks the resulting fp32 / INT8 ONNX pair on the test split,
and renders a Markdown comparison table to ``--output`` (default
``assets/sweep_table.md``).

The headline question this answers — and the reason it exists in this
repo — is the *hardware-aware ML* trade-off:

* How much does top-1 accuracy fall as we shrink the model?
* How does that scale with parameter count and MACs / inference (the
  numbers that matter on an NDP)?
* Does INT8 PTQ behave consistently across the whole curve, or does
  the smallest variant break first?

Sweep artefacts (per-width checkpoints + ONNX) live in ``--workdir``
(default ``runs/sweep``, gitignored). Only the rendered table — and,
for the canonical width, the refreshed assets in ``assets/`` — are
committed.

Usage::

    # Full overnight sweep (3 widths * 30 epochs each, ~8h on CPU).
    python -m scripts.sweep_sizes --epochs 30

    # Smoke (1 epoch, 1 batch each — finishes in ~30 s).
    python -m scripts.sweep_sizes --epochs 1 --max-train-batches 1 --skip-accuracy
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import DataLoader

import nano_kws.train as train_mod
from nano_kws import config
from nano_kws.benchmark import (
    VariantResult,
    benchmark_onnx,
)
from nano_kws.evaluate import load_checkpoint
from nano_kws.export_onnx import export_to_onnx, write_label_map
from nano_kws.models.ds_cnn import build_ds_cnn, count_macs, count_parameters
from nano_kws.quantize import (
    quantize_onnx,
    real_calibration_batches,
    synthetic_calibration_batches,
)

logger = logging.getLogger("nano_kws.sweep")

# Sentinel markers used in the README to locate the auto-rendered section.
SWEEP_BEGIN = "<!-- BEGIN_SWEEP_TABLE -->"
SWEEP_END = "<!-- END_SWEEP_TABLE -->"

# The width whose artefacts we publish to the canonical assets/ directory.
CANONICAL_WIDTH = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _width_tag(width: float) -> str:
    """0.25 -> 'w0p25', 1.0 -> 'w1p0' (matches train.py's convention)."""
    return f"w{width:g}".replace(".", "p")


def _format_int(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def _format_size_kb(n_bytes: int | None) -> str:
    return "n/a" if n_bytes is None else f"{n_bytes / 1024:.1f} KB"


def _format_acc(top1: float | None) -> str:
    return "—" if top1 is None else f"{top1 * 100:.2f}%"


# ---------------------------------------------------------------------------
# Per-width run
# ---------------------------------------------------------------------------


@dataclass
class SweepRow:
    width: float
    parameters: int
    macs: int
    fp32_size_bytes: int
    int8_size_bytes: int
    fp32_top1: float | None
    int8_top1: float | None
    fp32_latency_ms: float
    int8_latency_ms: float
    int8_vs_fp32_argmax_agreement: float | None


def _train_width(
    *,
    width: float,
    epochs: int,
    workdir: Path,
    extra_train_args: list[str],
    skip_existing: bool,
) -> Path:
    """Train one width via nano_kws.train.main(argv); return checkpoint path."""
    tag = _width_tag(width)
    width_dir = workdir / tag
    width_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = width_dir / f"ds_cnn_{tag}.pt"

    if skip_existing and ckpt_path.is_file():
        logger.info("[%s] reusing existing checkpoint %s", tag, ckpt_path)
        return ckpt_path

    argv = [
        "--width",
        str(width),
        "--epochs",
        str(epochs),
        "--output",
        str(ckpt_path),
        *extra_train_args,
    ]
    logger.info("[%s] training: python -m nano_kws.train %s", tag, " ".join(argv))
    t0 = time.perf_counter()
    train_mod.main(argv)
    logger.info("[%s] training done in %.1f min", tag, (time.perf_counter() - t0) / 60)
    return ckpt_path


def _export_and_quantize(
    *,
    width: float,
    ckpt_path: Path,
    workdir: Path,
    calibration_batches: int,
    calibration_batch_size: int,
    use_synthetic_calibration: bool,
    seed: int,
) -> tuple[Path, Path]:
    tag = _width_tag(width)
    width_dir = workdir / tag
    fp32_path = width_dir / f"ds_cnn_{tag}_fp32.onnx"
    int8_path = width_dir / f"ds_cnn_{tag}_int8.onnx"

    model, ckpt = load_checkpoint(ckpt_path, device="cpu")
    n_params = count_parameters(model)
    n_macs = count_macs(model)
    logger.info(
        "[%s] loaded ckpt val_acc=%.4f params=%d MACs=%d",
        tag,
        ckpt.get("val_acc", float("nan")),
        n_params,
        n_macs,
    )

    export_to_onnx(model=model, output_path=fp32_path, opset=17)
    write_label_map(fp32_path.with_suffix(".label_map.json"))

    if use_synthetic_calibration:
        batches = synthetic_calibration_batches(
            n_batches=calibration_batches, batch_size=calibration_batch_size, seed=seed
        )
    else:
        batches = real_calibration_batches(
            data_root=None,
            batch_size=calibration_batch_size,
            n_batches=calibration_batches,
            seed=seed,
        )

    quantize_onnx(
        fp32_path=fp32_path,
        int8_path=int8_path,
        calibration_batches=batches,
    )
    write_label_map(int8_path.with_suffix(".label_map.json"))
    return fp32_path, int8_path


def _benchmark_pair(
    *,
    width: float,
    fp32_path: Path,
    int8_path: Path,
    accuracy_loader: DataLoader | None,
    warmup: int,
    iters: int,
) -> SweepRow:
    model = build_ds_cnn(width_multiplier=width)  # for params + MACs only
    n_params = count_parameters(model)
    n_macs = count_macs(model)

    fp32_res: VariantResult = benchmark_onnx(
        onnx_path=fp32_path,
        name=f"DS-CNN w={width:g} fp32",
        parameters=n_params,
        macs=n_macs,
        accuracy_loader=accuracy_loader,
        warmup=warmup,
        iters=iters,
    )
    int8_res: VariantResult = benchmark_onnx(
        onnx_path=int8_path,
        name=f"DS-CNN w={width:g} INT8",
        parameters=n_params,
        macs=n_macs,
        accuracy_loader=accuracy_loader,
        warmup=warmup,
        iters=iters,
    )

    return SweepRow(
        width=width,
        parameters=n_params,
        macs=n_macs,
        fp32_size_bytes=fp32_res.file_size_bytes or 0,
        int8_size_bytes=int8_res.file_size_bytes or 0,
        fp32_top1=fp32_res.top1,
        int8_top1=int8_res.top1,
        fp32_latency_ms=fp32_res.latency_mean_ms,
        int8_latency_ms=int8_res.latency_mean_ms,
        int8_vs_fp32_argmax_agreement=None,  # populated by quantize.py log; not tracked here
    )


# ---------------------------------------------------------------------------
# Table + plot rendering
# ---------------------------------------------------------------------------


def render_sweep_table(rows: list[SweepRow]) -> str:
    if not rows:
        return "_(no sweep results)_\n"
    lines: list[str] = []
    lines.append(
        "| Width | Params | MACs | fp32 top-1 | INT8 top-1 | Δ acc | "
        "fp32 size | INT8 size | Size ratio | fp32 latency | INT8 latency |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in sorted(rows, key=lambda x: x.width):
        if r.fp32_top1 is None or r.int8_top1 is None:
            delta = "—"
        else:
            delta_pp = (r.int8_top1 - r.fp32_top1) * 100
            sign = "+" if delta_pp >= 0 else ""
            delta = f"{sign}{delta_pp:.2f} pp"
        size_ratio = f"{r.int8_size_bytes / r.fp32_size_bytes:.1%}" if r.fp32_size_bytes else "—"
        lines.append(
            f"| {r.width:g} | {_format_int(r.parameters)} | {_format_int(r.macs)} | "
            f"{_format_acc(r.fp32_top1)} | {_format_acc(r.int8_top1)} | {delta} | "
            f"{_format_size_kb(r.fp32_size_bytes)} | {_format_size_kb(r.int8_size_bytes)} | "
            f"{size_ratio} | {r.fp32_latency_ms:.3f} ms | {r.int8_latency_ms:.3f} ms |"
        )
    return "\n".join(lines) + "\n"


def maybe_render_plot(rows: list[SweepRow], plot_path: Path) -> bool:
    """Render accuracy-vs-MACs scatter (fp32 + INT8). Returns True on success."""
    if not rows or all(r.fp32_top1 is None and r.int8_top1 is None for r in rows):
        logger.info("No accuracy data; skipping plot.")
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot.")
        return False

    rows_sorted = sorted(rows, key=lambda x: x.macs)
    macs = [r.macs for r in rows_sorted]
    fp32 = [None if r.fp32_top1 is None else r.fp32_top1 * 100 for r in rows_sorted]
    int8 = [None if r.int8_top1 is None else r.int8_top1 * 100 for r in rows_sorted]
    widths = [r.width for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.plot(macs, fp32, marker="o", label="fp32 ONNX", linewidth=2)
    ax.plot(macs, int8, marker="s", label="INT8 ONNX", linewidth=2, linestyle="--")
    for x, y_fp, w in zip(macs, fp32, widths, strict=False):
        if y_fp is not None:
            ax.annotate(
                f"w={w:g}", (x, y_fp), textcoords="offset points", xytext=(8, 6), fontsize=9
            )
    ax.set_xscale("log")
    ax.set_xlabel("MACs / inference (log scale)")
    ax.set_ylabel("Top-1 accuracy (%, 12-class)")
    ax.set_title("nano-kws — accuracy vs MACs across DS-CNN widths")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    logger.info("Wrote sweep plot to %s", plot_path)
    return True


def update_readme_section(readme_path: Path, *, table_md: str, begin: str, end: str) -> bool:
    """Same contract as benchmark.update_readme_table, parameterised on markers."""
    import re as _re

    if not readme_path.is_file():
        return False
    text = readme_path.read_text(encoding="utf-8")
    pattern = _re.compile(
        rf"({_re.escape(begin)})(.*?)({_re.escape(end)})",
        flags=_re.DOTALL,
    )
    if not pattern.search(text):
        return False
    new_section = f"{begin}\n\n{table_md}\n{end}"
    new_text = pattern.sub(new_section, text)
    if new_text != text:
        readme_path.write_text(new_text, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--widths",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="DS-CNN width multipliers to sweep.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workdir", default="runs/sweep")
    parser.add_argument("--output", default="assets/sweep_table.md")
    parser.add_argument("--plot", default="assets/sweep_plot.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse {workdir}/{tag}/ds_cnn_{tag}.pt if it exists (skip training).",
    )
    parser.add_argument(
        "--publish-canonical",
        action="store_true",
        help=(
            "After the sweep, copy the width=0.5 artefacts to assets/ and "
            "refresh assets/benchmark_table.md + the README TL;DR block."
        ),
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--calibration-batches", type=int, default=50)
    parser.add_argument("--calibration-batch-size", type=int, default=16)
    parser.add_argument("--synthetic-calibration", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Pass-through to train.py for smoke testing.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Pass-through to train.py for smoke testing.",
    )
    parser.add_argument(
        "--num-workers-train",
        type=int,
        default=2,
        help="DataLoader workers used during training (separate knob from sweep eval).",
    )
    return parser.parse_args(argv)


def _train_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = ["--num-workers", str(args.num_workers_train)]
    if args.max_train_batches is not None:
        extra += ["--max-train-batches", str(args.max_train_batches)]
    if args.max_val_batches is not None:
        extra += ["--max-val-batches", str(args.max_val_batches)]
    extra += ["--seed", str(args.seed)]
    return extra


def _publish_canonical(
    *,
    workdir: Path,
    fp32_path: Path,
    int8_path: Path,
    args: argparse.Namespace,
) -> None:
    """Copy width=0.5 sweep artefacts into assets/ AND refresh the TL;DR table.

    The file copy mirrors the previous behaviour (so the canonical
    bundled artefacts move forward to the new run). On top of that we
    chain :func:`nano_kws.benchmark.main` to re-render
    ``assets/benchmark_table.md`` and re-stamp the README's
    ``BEGIN_BENCHMARK_TABLE`` block from the freshly-copied files —
    otherwise the TL;DR table on the README silently goes stale relative
    to the artefacts under it.
    """
    tag = _width_tag(CANONICAL_WIDTH)
    src_ckpt = workdir / tag / f"ds_cnn_{tag}.pt"
    src_history = src_ckpt.with_suffix(".history.json")
    dst_ckpt = config.ASSETS_DIR / f"ds_cnn_{tag}.pt"
    dst_history = config.ASSETS_DIR / f"ds_cnn_{tag}.history.json"
    dst_fp32 = config.ASSETS_DIR / "ds_cnn_small_fp32.onnx"
    dst_int8 = config.ASSETS_DIR / "ds_cnn_small_int8.onnx"

    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if src_ckpt.is_file():
        shutil.copy2(src_ckpt, dst_ckpt)
        logger.info("Refreshed canonical %s", dst_ckpt)
    if src_history.is_file():
        shutil.copy2(src_history, dst_history)
    shutil.copy2(fp32_path, dst_fp32)
    shutil.copy2(int8_path, dst_int8)
    shutil.copy2(
        fp32_path.with_suffix(".label_map.json"),
        dst_fp32.with_suffix(".label_map.json"),
    )
    shutil.copy2(
        int8_path.with_suffix(".label_map.json"),
        dst_int8.with_suffix(".label_map.json"),
    )
    logger.info("Refreshed canonical fp32/INT8 ONNX in %s", config.ASSETS_DIR)

    # Chain a clean isolated benchmark run so the README TL;DR block (and
    # assets/benchmark_table.md) move forward in lock-step with the new
    # canonical .pt / ONNX files.
    from nano_kws.benchmark import main as benchmark_main

    benchmark_argv = [
        "--checkpoint",
        str(dst_ckpt),
        "--fp32",
        str(dst_fp32),
        "--int8",
        str(dst_int8),
        "--output",
        str(config.ASSETS_DIR / "benchmark_table.md"),
        "--update-readme",
        "--readme",
        str(args.readme),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.skip_accuracy:
        benchmark_argv.append("--skip-accuracy")
    logger.info("Refreshing canonical benchmark table via nano_kws.benchmark ...")
    benchmark_main(benchmark_argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # Build the accuracy loader once and share across all widths.
    accuracy_loader = None
    if not args.skip_accuracy:
        from nano_kws.benchmark import _maybe_build_accuracy_loader  # lazy import

        loader_args = argparse.Namespace(
            skip_accuracy=False,
            data_root=None,
            subset="testing",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        accuracy_loader = _maybe_build_accuracy_loader(loader_args)

    rows: list[SweepRow] = []
    canonical_paths: tuple[Path, Path] | None = None
    extra_train = _train_extra_args(args)

    for width in args.widths:
        tag = _width_tag(width)
        try:
            ckpt = _train_width(
                width=width,
                epochs=args.epochs,
                workdir=workdir,
                extra_train_args=extra_train,
                skip_existing=args.skip_existing,
            )
            fp32, int8 = _export_and_quantize(
                width=width,
                ckpt_path=ckpt,
                workdir=workdir,
                calibration_batches=args.calibration_batches,
                calibration_batch_size=args.calibration_batch_size,
                use_synthetic_calibration=args.synthetic_calibration,
                seed=args.seed,
            )
            row = _benchmark_pair(
                width=width,
                fp32_path=fp32,
                int8_path=int8,
                accuracy_loader=accuracy_loader,
                warmup=args.warmup,
                iters=args.iters,
            )
            rows.append(row)
            if width == CANONICAL_WIDTH:
                canonical_paths = (fp32, int8)
            # Cheap incremental persistence (table .md + .json only) so a
            # crash on width N still leaves a usable table for widths < N.
            # Plot rendering and README stamping are deferred to the end
            # to avoid redundant work and false-positive marker warnings.
            _write_table_files(rows, args)
        except Exception:
            logger.exception("[%s] failed; continuing with the next width.", tag)

    _render_final_outputs(rows, args)

    if args.publish_canonical and canonical_paths is not None:
        _publish_canonical(
            workdir=workdir,
            fp32_path=canonical_paths[0],
            int8_path=canonical_paths[1],
            args=args,
        )

    return 0 if rows else 1


def _write_table_files(rows: list[SweepRow], args: argparse.Namespace) -> str:
    """Persist sweep_table.md + sweep_table.json. Returns the rendered table.

    Called once per successful width so a crash on width N still leaves
    behind a valid table for widths < N. Crucially this does NOT touch
    the README or render the plot — those are end-of-run-only side
    effects (see :func:`_render_final_outputs`), which keeps the per-width
    cost low and prevents the false-positive "markers not found" warnings
    that fire when the readme stamp is invoked repeatedly.
    """
    output_md = Path(args.output)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    table_md = render_sweep_table(rows)
    output_md.write_text(table_md, encoding="utf-8")

    output_json = output_md.with_suffix(".json")
    output_json.write_text(
        json.dumps([asdict(r) for r in sorted(rows, key=lambda x: x.width)], indent=2),
        encoding="utf-8",
    )
    return table_md


def _render_final_outputs(rows: list[SweepRow], args: argparse.Namespace) -> None:
    """Render the plot and stamp the README — call exactly once at end of run."""
    table_md = _write_table_files(rows, args)

    if not args.no_plot:
        maybe_render_plot(rows, Path(args.plot))

    if args.update_readme:
        if update_readme_section(
            Path(args.readme), table_md=table_md, begin=SWEEP_BEGIN, end=SWEEP_END
        ):
            logger.info("Stamped sweep table into %s", args.readme)
        else:
            logger.warning(
                "Could not find %s / %s markers in %s; README sweep section not updated.",
                SWEEP_BEGIN,
                SWEEP_END,
                args.readme,
            )


if __name__ == "__main__":
    raise SystemExit(main())
