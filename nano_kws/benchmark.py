"""Benchmark fp32 vs INT8 inference and emit the README TL;DR table.

Phase 4 implements:

* Wall-clock latency (mean + p95) for single-inference, post warmup, on the
  host CPU; both fp32 ONNX and INT8 ONNX through ONNX Runtime.
* Top-1 accuracy on the Speech Commands v0.02 test split (12-class).
* On-disk model size.

Results are written to a Markdown table that ``README.md`` references.
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", help="Path to fp32 .onnx model.")
    parser.add_argument("--int8", help="Path to INT8 .onnx model.")
    parser.add_argument(
        "--output",
        default="assets/benchmark_table.md",
        help="Where to write the Markdown results table.",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip accuracy measurement (latency-only; useful when no dataset is present).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    raise NotImplementedError("Phase 4: implement benchmark.")


if __name__ == "__main__":
    main()
