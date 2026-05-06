"""Benchmark fp32 vs INT8 inference; emit the README TL;DR table.

Measures three variants on the host CPU:

* DS-CNN fp32 in **PyTorch** (eval mode, no_grad).
* DS-CNN fp32 in **ONNX Runtime** (CPU EP).
* DS-CNN INT8 in **ONNX Runtime** (CPU EP) — the artefact a real edge
  deployment would consume.

For each variant we report:

* Top-1 accuracy on the chosen Speech Commands split (test by default).
* Single-inference latency on the host CPU: mean, p50, p95, over many
  runs after warmup. Latency on a laptop CPU is *not* a deployment claim
  — the headline edge-AI metrics are MACs and parameter count, which are
  hardware-independent and reported alongside. The CPU latency is here
  as a sanity check ("INT8 is not slower than fp32 on the same runtime")
  and as a number readers can reproduce locally.
* On-disk model size.

Output is rendered to ``--output`` as a Markdown table and (optionally,
``--update-readme``) stamped into the project README between
``<!-- BEGIN_BENCHMARK_TABLE -->`` / ``<!-- END_BENCHMARK_TABLE -->``
markers so the headline numbers cannot drift from the latest run.

Usage::

    # Latency only (no dataset required)
    python -m nano_kws.benchmark \\
        --fp32 assets/ds_cnn_small_fp32.onnx \\
        --int8 assets/ds_cnn_small_int8.onnx \\
        --skip-accuracy

    # Full table (needs the dataset cached)
    python -m nano_kws.benchmark \\
        --checkpoint assets/ds_cnn_w0p5.pt \\
        --fp32 assets/ds_cnn_small_fp32.onnx \\
        --int8 assets/ds_cnn_small_int8.onnx \\
        --output assets/benchmark_table.md \\
        --update-readme
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nano_kws import config
from nano_kws.data.features import LogMelSpectrogram
from nano_kws.data.speech_commands import SpeechCommandsKWS
from nano_kws.evaluate import evaluate_dataset, load_checkpoint
from nano_kws.export_onnx import INPUT_NAME, OUTPUT_NAME
from nano_kws.models.ds_cnn import count_macs, count_parameters

logger = logging.getLogger("nano_kws.benchmark")

README_BEGIN = "<!-- BEGIN_BENCHMARK_TABLE -->"
README_END = "<!-- END_BENCHMARK_TABLE -->"


# ---------------------------------------------------------------------------
# Result struct + helpers
# ---------------------------------------------------------------------------


@dataclass
class VariantResult:
    name: str
    runtime: str
    parameters: int
    macs: int
    file_size_bytes: int | None  # None for in-memory PyTorch
    top1: float | None  # None if accuracy was skipped
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    iters: int


def _format_size(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "n/a"
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / 1024 / 1024:.2f} MB"


def _format_macs(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def _format_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def _format_acc(top1: float | None) -> str:
    return "—" if top1 is None else f"{top1 * 100:.2f}%"


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def _percentile(arr: np.ndarray, q: float) -> float:
    """np.percentile but always returns a Python float."""
    return float(np.percentile(arr, q))


# ─── Design note: the latency-measurement methodology in one place ─────────
# Easy to do badly. Three things this function gets right that a naive
# `time.time()` loop wouldn't:
#   1. Warmup runs that are NOT timed. The first few runs pay JIT/cache
#      misses, ORT graph compilation, allocator warmup, etc. — including
#      them inflates the mean and skews the percentiles.
#   2. perf_counter_ns(), not time.time() / time.perf_counter(). The former
#      has nanosecond resolution and is monotonic; the latter is system-
#      time-based and can jitter under NTP adjustments.
#   3. Reporting mean + p50 + p95 (not just mean). Latency distributions in
#      production are usually skewed right (occasional context switches,
#      GC pauses). Mean alone hides tail behaviour; p50 says "what does
#      a typical request feel like"; p95 says "what does a slow request
#      feel like". Edge-AI deployments often spec on p99 or p99.9 for
#      hard-real-time wake-word use cases.
# Caveats worth flagging:
#   * This measures host-CPU latency, NOT dedicated-edge-accelerator
#     latency. Hardware vendors quote latency on their silicon at a
#     specified clock; ours is a loose proxy. The hardware-independent
#     metrics (parameters, MACs) are the headline; latency is a sanity
#     check on top of those.
#   * Inputs are pre-staged outside the timed region so we measure model +
#     runtime, not numpy memcpy.
#   * Single-thread because the rest of the report is CPU-EP single-thread.
# ───────────────────────────────────────────────────────────────────────────


def _measure_latency(infer_once, *, warmup: int, iters: int) -> tuple[float, float, float]:
    """Run ``infer_once`` ``warmup`` + ``iters`` times; return (mean, p50, p95) ms.

    ``infer_once`` is a zero-arg callable that performs exactly one
    single-inference forward pass; the caller is responsible for staging
    the input tensor so this loop measures only the model.
    """
    for _ in range(warmup):
        infer_once()

    times_ns = np.empty(iters, dtype=np.int64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        infer_once()
        times_ns[i] = time.perf_counter_ns() - t0

    times_ms = times_ns.astype(np.float64) / 1e6
    return float(times_ms.mean()), _percentile(times_ms, 50), _percentile(times_ms, 95)


# ---------------------------------------------------------------------------
# Per-variant benchmark drivers
# ---------------------------------------------------------------------------


def benchmark_pytorch(
    *,
    model: nn.Module,
    name: str,
    accuracy_loader: DataLoader | None,
    warmup: int,
    iters: int,
) -> VariantResult:
    model = model.cpu().eval()
    n_params = count_parameters(model)
    n_macs = count_macs(model)

    # Latency: feed a single (1, 1, N_MELS, N_FRAMES) tensor through the model.
    dummy = torch.zeros(1, *config.INPUT_SHAPE, dtype=torch.float32)

    @torch.no_grad()
    def infer_once() -> None:
        model(dummy)

    mean_ms, p50_ms, p95_ms = _measure_latency(infer_once, warmup=warmup, iters=iters)

    top1 = None
    if accuracy_loader is not None:
        result = evaluate_dataset(model=model, loader=accuracy_loader, device=torch.device("cpu"))
        top1 = result["top1_accuracy"]

    return VariantResult(
        name=name,
        runtime="PyTorch (CPU)",
        parameters=n_params,
        macs=n_macs,
        file_size_bytes=None,
        top1=top1,
        latency_mean_ms=mean_ms,
        latency_p50_ms=p50_ms,
        latency_p95_ms=p95_ms,
        iters=iters,
    )


def benchmark_onnx(
    *,
    onnx_path: Path,
    name: str,
    parameters: int,
    macs: int,
    accuracy_loader: DataLoader | None,
    warmup: int,
    iters: int,
) -> VariantResult:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    dummy = np.zeros((1, *config.INPUT_SHAPE), dtype=np.float32)
    feed = {INPUT_NAME: dummy}
    output_names = [OUTPUT_NAME]

    def infer_once() -> None:
        session.run(output_names, feed)

    mean_ms, p50_ms, p95_ms = _measure_latency(infer_once, warmup=warmup, iters=iters)

    top1 = None
    if accuracy_loader is not None:
        top1 = _onnx_accuracy(session, accuracy_loader)

    return VariantResult(
        name=name,
        runtime="ONNX Runtime (CPU)",
        parameters=parameters,
        macs=macs,
        file_size_bytes=onnx_path.stat().st_size,
        top1=top1,
        latency_mean_ms=mean_ms,
        latency_p50_ms=p50_ms,
        latency_p95_ms=p95_ms,
        iters=iters,
    )


def _onnx_accuracy(session: ort.InferenceSession, loader: DataLoader) -> float:
    """Top-1 over an ONNX session, featurising on CPU through LogMelSpectrogram."""
    featurizer = LogMelSpectrogram().eval()
    n_total = 0
    n_correct = 0
    output_names = [OUTPUT_NAME]
    with torch.no_grad():
        for waveforms, labels in loader:
            features = featurizer(waveforms).numpy()
            logits = session.run(output_names, {INPUT_NAME: features})[0]
            preds = np.argmax(logits, axis=1)
            n_total += labels.shape[0]
            n_correct += int(np.sum(preds == labels.numpy()))
    return n_correct / max(1, n_total)


# ---------------------------------------------------------------------------
# Markdown rendering + README stamping
# ---------------------------------------------------------------------------


def render_markdown(results: list[VariantResult]) -> str:
    lines: list[str] = []
    lines.append(
        "| Variant | Runtime | Params | MACs | Top-1 acc | Size on disk | "
        "Latency mean | Latency p50 | Latency p95 |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        lines.append(
            f"| {r.name} | {r.runtime} | {_format_params(r.parameters)} | {_format_macs(r.macs)} | "
            f"{_format_acc(r.top1)} | {_format_size(r.file_size_bytes)} | "
            f"{r.latency_mean_ms:.3f} ms | {r.latency_p50_ms:.3f} ms | {r.latency_p95_ms:.3f} ms |"
        )

    # Speedup / size ratio summary, only if both ONNX variants are present.
    fp32_onnx = next(
        (r for r in results if "INT8" not in r.name and r.file_size_bytes is not None), None
    )
    int8_ptq = next(
        (r for r in results if "INT8" in r.name and "QAT" not in r.name),
        None,
    )
    int8_qat = next((r for r in results if "INT8" in r.name and "QAT" in r.name), None)
    if fp32_onnx and int8_ptq:
        size_ratio = int8_ptq.file_size_bytes / fp32_onnx.file_size_bytes
        speedup = fp32_onnx.latency_mean_ms / max(int8_ptq.latency_mean_ms, 1e-9)
        acc_delta = (
            None
            if fp32_onnx.top1 is None or int8_ptq.top1 is None
            else (int8_ptq.top1 - fp32_onnx.top1) * 100
        )
        lines.append("")
        lines.append("**INT8 (PTQ) vs fp32 (ONNX Runtime):**")
        lines.append(
            f"- Size: {size_ratio:.1%} of fp32 ({_format_size(fp32_onnx.file_size_bytes)} -> "
            f"{_format_size(int8_ptq.file_size_bytes)})"
        )
        lines.append(
            f"- Latency: {speedup:.2f}x ({fp32_onnx.latency_mean_ms:.3f} ms -> "
            f"{int8_ptq.latency_mean_ms:.3f} ms mean)"
        )
        if acc_delta is not None:
            sign = "+" if acc_delta >= 0 else ""
            lines.append(f"- Top-1 accuracy: {sign}{acc_delta:.2f} pp")
    if fp32_onnx and int8_qat:
        qat_acc_delta_vs_fp32 = (
            None
            if fp32_onnx.top1 is None or int8_qat.top1 is None
            else (int8_qat.top1 - fp32_onnx.top1) * 100
        )
        lines.append("")
        lines.append("**INT8 (QAT) vs fp32 (ONNX Runtime):**")
        if qat_acc_delta_vs_fp32 is not None:
            sign = "+" if qat_acc_delta_vs_fp32 >= 0 else ""
            lines.append(f"- Top-1 accuracy: {sign}{qat_acc_delta_vs_fp32:.2f} pp")
        if int8_ptq and int8_ptq.top1 is not None and int8_qat.top1 is not None:
            recovered = (int8_qat.top1 - int8_ptq.top1) * 100
            sign = "+" if recovered >= 0 else ""
            lines.append(f"- vs PTQ-only INT8: {sign}{recovered:.2f} pp top-1 recovered by QAT.")
    return "\n".join(lines) + "\n"


def update_readme_table(readme_path: Path, table_md: str) -> bool:
    """Stamp ``table_md`` between BEGIN/END markers in ``readme_path``.

    Returns ``True`` if the README was modified, ``False`` if the markers
    weren't found.
    """
    if not readme_path.is_file():
        return False
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"({re.escape(README_BEGIN)})(.*?)({re.escape(README_END)})",
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        return False
    new_section = f"{README_BEGIN}\n\n{table_md}\n{README_END}"
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
        "--checkpoint", default=None, help="PyTorch .pt for the fp32 PyTorch row + accuracy."
    )
    parser.add_argument("--fp32", default=None, help="Path to fp32 ONNX model.")
    parser.add_argument("--int8", default=None, help="Path to INT8 ONNX model (PTQ-only).")
    parser.add_argument(
        "--int8-qat",
        default=None,
        help="Optional path to a second INT8 ONNX model produced via QAT + PTQ. "
        "When supplied, an extra row is added to the benchmark table.",
    )
    parser.add_argument(
        "--output",
        default="assets/benchmark_table.md",
        help="Where to write the rendered Markdown table.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Also stamp the table between BEGIN/END markers in README.md.",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="README file to update (only used with --update-readme).",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size used for accuracy eval."
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", default=None, help="Override config.DATA_DIR.")
    parser.add_argument(
        "--subset",
        choices=("validation", "testing"),
        default="testing",
        help="Which Speech Commands split to evaluate accuracy on.",
    )
    parser.add_argument("--skip-accuracy", action="store_true", help="Skip accuracy measurement.")
    return parser.parse_args(argv)


def _maybe_build_accuracy_loader(args: argparse.Namespace) -> DataLoader | None:
    if args.skip_accuracy:
        logger.info("Accuracy measurement skipped (--skip-accuracy).")
        return None
    data_root = Path(args.data_root) if args.data_root else config.DATA_DIR
    archive = data_root / "SpeechCommands" / "speech_commands_v0.02"
    if not archive.is_dir():
        logger.warning(
            "Dataset not cached at %s — accuracy will be reported as '—'. "
            "Run `make download-data` to enable accuracy measurement.",
            archive,
        )
        return None
    dataset = SpeechCommandsKWS(
        root=Path(args.data_root) if args.data_root else None,
        subset=args.subset,
        unknown_per_class_ratio=1.0,
        silence_per_class_ratio=1.0,
        seed=0,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    if not (args.checkpoint or args.fp32 or args.int8 or args.int8_qat):
        raise SystemExit("Pass at least one of --checkpoint / --fp32 / --int8 / --int8-qat.")

    accuracy_loader = _maybe_build_accuracy_loader(args)

    # Resolve model metadata. We need params + MACs for the ONNX rows; the
    # cleanest source is the original PyTorch checkpoint (when supplied) or
    # a default-built model otherwise (only for synthetic / smoke runs).
    pt_model: nn.Module | None = None
    if args.checkpoint:
        pt_model, ckpt = load_checkpoint(args.checkpoint, device="cpu")
        logger.info(
            "Loaded checkpoint %s (width=%s, val_acc=%.4f)",
            args.checkpoint,
            ckpt["model_config"]["width_multiplier"],
            ckpt.get("val_acc", float("nan")),
        )
        n_params = count_parameters(pt_model)
        n_macs = count_macs(pt_model)
    else:
        from nano_kws.models.ds_cnn import build_ds_cnn

        pt_model = build_ds_cnn(width_multiplier=0.5)
        n_params = count_parameters(pt_model)
        n_macs = count_macs(pt_model)
        logger.warning(
            "No --checkpoint provided; reporting params/MACs for a freshly-initialised "
            "DS-CNN at width=0.5 (n_params=%d, MACs=%d). Pass --checkpoint to honour the "
            "actual exported architecture.",
            n_params,
            n_macs,
        )

    results: list[VariantResult] = []

    if args.checkpoint:
        results.append(
            benchmark_pytorch(
                model=pt_model,
                name="DS-CNN small fp32",
                accuracy_loader=accuracy_loader,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    if args.fp32:
        results.append(
            benchmark_onnx(
                onnx_path=Path(args.fp32),
                name="DS-CNN small fp32",
                parameters=n_params,
                macs=n_macs,
                accuracy_loader=accuracy_loader,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    if args.int8:
        results.append(
            benchmark_onnx(
                onnx_path=Path(args.int8),
                name="DS-CNN small INT8 (PTQ)",
                parameters=n_params,
                macs=n_macs,
                accuracy_loader=accuracy_loader,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    if args.int8_qat:
        results.append(
            benchmark_onnx(
                onnx_path=Path(args.int8_qat),
                name="DS-CNN small INT8 (QAT)",
                parameters=n_params,
                macs=n_macs,
                accuracy_loader=accuracy_loader,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    table_md = render_markdown(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_md, encoding="utf-8")
    logger.info("Wrote benchmark table to %s", output_path)
    print("\n" + table_md)

    if args.update_readme:
        readme_path = Path(args.readme)
        if update_readme_table(readme_path, table_md):
            logger.info("Stamped table into %s between BEGIN/END markers.", readme_path)
        else:
            logger.warning(
                "Could not find %s / %s markers in %s; README not updated. "
                "Add the markers to the README to enable auto-update.",
                README_BEGIN,
                README_END,
                readme_path,
            )


if __name__ == "__main__":
    main()
