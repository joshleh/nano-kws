"""Microbench: hand-written depthwise-separable conv vs PyTorch ATen.

The two ops the DS-CNN spends almost all of its compute on:

* **pointwise (1x1) conv** — channel mixing; equivalent to a matrix-
  vector product applied at every spatial position. Vectorises
  cleanly: the inner loop is a chain of fused multiply-adds along the
  spatial axis, with the weight broadcast hoisted outside.
* **depthwise 3x3 conv** — independent per-channel spatial filtering.
  Smaller compute footprint than pointwise but more memory accesses
  per FLOP, and the boundary handling (padding) is annoying for SIMD.

For each op we measure four implementations on the same fp32 input:

1. **NumPy reference** (``np.einsum`` for pointwise, explicit Python
   loops for depthwise) — the "I wrote this in 5 minutes" baseline.
2. **C scalar** — straight nested loops, compiler auto-vectorisation
   off-by-default for these sizes; close to a textbook C implementation.
3. **C AVX2** — hand-written 256-bit / FMA intrinsics (see
   ``cpp/microbench/conv_kernels.c``).
4. **PyTorch ATen** — what ``torch.nn.functional.conv2d`` dispatches
   to on this CPU; that's vendor-tuned MKL-DNN under the hood.

Why this exists for a Syntiant-shaped portfolio
-----------------------------------------------
Edge inference engines (CMSIS-NN, TFLite Micro, vendor-specific stacks
like Syntiant's) are essentially hand-written kernel libraries
targeting tight resource budgets. The thing that distinguishes a
hireable MLE in this space from someone who only knows the
PyTorch-import-and-train layer is being able to (a) reason about
throughput and memory traffic, (b) write the SIMD path themselves at
some level, and (c) honestly benchmark hand-rolled kernels against
vendor-tuned ones and reason about the gap. This script is a focused
exercise in exactly that.

We do not expect to beat ATen — MKL-DNN has years of cache-blocking,
register-tiling and CPU-dispatch work behind it. The point is to
*quantify* the gap and have an articulate answer to "what would you
do to close it?" (im2col + GEMM, register tiling, L1-blocking,
prefetching, and so on).

Usage
-----
::

    # 1. Build the C shared library (one-time).
    make microbench-build

    # 2. Run the benchmark + emit the table.
    make microbench

    # Or directly:
    python -m scripts.conv_microbench --update-readme
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nano_kws import config

logger = logging.getLogger("nano_kws.microbench")


# Default microbench dimensions — match the interior block of the
# bundled DS-CNN at width=0.5: after the stem (kernel 10x4, stride 2x2)
# the spatial dims are
#   floor((40 - 10) / 2) + 1 = 16  mel bins
#   floor((97 - 4)  / 2) + 1 = 47  frames
# with 56 channels (channel count rounded to multiple of 8 by the
# width-multiplier helper).
DEFAULT_C: int = 56
DEFAULT_H: int = 16
DEFAULT_W: int = 47

CPP_BUILD_DIR: Path = config.REPO_ROOT / "cpp" / "microbench" / "build"

README_BEGIN: str = "<!-- BEGIN_MICROBENCH_TABLE -->"
README_END: str = "<!-- END_MICROBENCH_TABLE -->"


# ---------------------------------------------------------------------------
# Locate + load the compiled C kernel library.
# ---------------------------------------------------------------------------


def _candidate_lib_paths() -> list[Path]:
    """Where the CMake build might have dropped the shared library."""
    if platform.system() == "Windows":
        return [
            CPP_BUILD_DIR / "Release" / "conv_kernels.dll",
            CPP_BUILD_DIR / "Debug" / "conv_kernels.dll",
            CPP_BUILD_DIR / "conv_kernels.dll",
        ]
    if platform.system() == "Darwin":
        return [
            CPP_BUILD_DIR / "conv_kernels.dylib",
            CPP_BUILD_DIR / "libconv_kernels.dylib",
        ]
    return [
        CPP_BUILD_DIR / "conv_kernels.so",
        CPP_BUILD_DIR / "libconv_kernels.so",
    ]


def find_kernel_lib() -> Path | None:
    for path in _candidate_lib_paths():
        if path.is_file():
            return path
    return None


@dataclass
class _CKernels:
    """Wrapper around the four C entry points."""

    lib: ctypes.CDLL
    pointwise_naive: object
    pointwise_avx2: object
    depthwise_3x3_naive: object
    depthwise_3x3_avx2: object


def load_c_kernels(path: Path) -> _CKernels:
    lib = ctypes.CDLL(str(path))
    fp = ctypes.POINTER(ctypes.c_float)
    sig_pw = [fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    sig_dw = [fp, fp, fp, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    for name, sig in (
        ("conv_pointwise_naive", sig_pw),
        ("conv_pointwise_avx2", sig_pw),
        ("conv_depthwise_3x3_naive", sig_dw),
        ("conv_depthwise_3x3_avx2", sig_dw),
    ):
        fn = getattr(lib, name)
        fn.argtypes = sig
        fn.restype = None

    return _CKernels(
        lib=lib,
        pointwise_naive=lib.conv_pointwise_naive,
        pointwise_avx2=lib.conv_pointwise_avx2,
        depthwise_3x3_naive=lib.conv_depthwise_3x3_naive,
        depthwise_3x3_avx2=lib.conv_depthwise_3x3_avx2,
    )


def _as_ctypes(arr: np.ndarray) -> object:
    """Return a ctypes pointer that refers to ``arr``'s buffer.

    The array must be a contiguous fp32. We rely on the caller keeping
    ``arr`` alive for the duration of the C call (the pointer does not
    own the memory).
    """
    if arr.dtype != np.float32:
        raise TypeError(f"expected float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ---------------------------------------------------------------------------
# Reference + driver implementations.
# ---------------------------------------------------------------------------


def numpy_pointwise(input_chw: np.ndarray, weight_oi: np.ndarray) -> np.ndarray:
    """Pointwise (1x1) conv via :func:`numpy.einsum`.

    Shapes:
        input_chw : (C_in, H, W)  fp32
        weight_oi : (C_out, C_in) fp32
    Returns:
        (C_out, H, W) fp32
    """
    return np.einsum("oi,ihw->ohw", weight_oi, input_chw)


def numpy_depthwise_3x3(input_chw: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Depthwise 3x3 conv (stride=1, padding=1) via PyTorch on CPU.

    Implementing the boundary-padded sliding window in pure NumPy is
    a one-liner with ``np.pad`` + nine slices, but we'd be timing the
    Python overhead more than the compute. Instead we delegate to
    :func:`torch.nn.functional.conv2d` purely as a *correctness*
    reference and never time this path.
    """
    c = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)  # 1, C, H, W
    wt = torch.from_numpy(weight).reshape(c, 1, 3, 3)
    y = F.conv2d(x, wt, stride=1, padding=1, groups=c)
    return y.squeeze(0).numpy()


def aten_pointwise(input_chw: np.ndarray, weight_oi: np.ndarray) -> np.ndarray:
    c_in = input_chw.shape[0]
    c_out = weight_oi.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    wt = torch.from_numpy(weight_oi).reshape(c_out, c_in, 1, 1)
    y = F.conv2d(x, wt, stride=1, padding=0)
    return y.squeeze(0).numpy()


def aten_depthwise_3x3(input_chw: np.ndarray, weight: np.ndarray) -> np.ndarray:
    c = input_chw.shape[0]
    x = torch.from_numpy(input_chw).unsqueeze(0)
    wt = torch.from_numpy(weight).reshape(c, 1, 3, 3)
    y = F.conv2d(x, wt, stride=1, padding=1, groups=c)
    return y.squeeze(0).numpy()


def c_pointwise(
    fn: object,
    input_chw: np.ndarray,
    weight_oi: np.ndarray,
) -> np.ndarray:
    c_in, h, w = input_chw.shape
    c_out = weight_oi.shape[0]
    inp_flat = np.ascontiguousarray(input_chw.reshape(c_in, h * w))
    out_flat = np.empty((c_out, h * w), dtype=np.float32)
    fn(_as_ctypes(inp_flat), _as_ctypes(weight_oi), _as_ctypes(out_flat), c_in, c_out, h * w)
    return out_flat.reshape(c_out, h, w)


def c_depthwise_3x3(
    fn: object,
    input_chw: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    c, h, w = input_chw.shape
    out = np.empty_like(input_chw)
    fn(_as_ctypes(input_chw), _as_ctypes(weight), _as_ctypes(out), c, h, w)
    return out


# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------


def _time_callable(fn, *, warmup: int, iters: int) -> tuple[float, float, float]:
    """Run ``fn`` ``warmup`` + ``iters`` times; return (mean, p50, p95) ms."""
    for _ in range(warmup):
        fn()
    times_ns = np.empty(iters, dtype=np.int64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        times_ns[i] = time.perf_counter_ns() - t0
    times_ms = times_ns.astype(np.float64) / 1e6
    return (
        float(times_ms.mean()),
        float(np.percentile(times_ms, 50)),
        float(np.percentile(times_ms, 95)),
    )


@dataclass
class Result:
    op: str
    impl: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    correct: bool | None  # None for the reference; True/False for everyone else
    max_abs_err: float | None


def _check_correctness(
    reference: np.ndarray, candidate: np.ndarray, *, atol: float
) -> tuple[bool, float]:
    err = float(np.max(np.abs(reference - candidate)))
    return err <= atol, err


def benchmark_op(
    op: str,
    inputs: dict[str, object],
    warmup: int,
    iters: int,
    atol: float,
) -> list[Result]:
    """Benchmark every available implementation of ``op`` and check results."""
    impls = inputs["impls"]  # list of (impl_name, callable_returning_output_arr)
    reference_name, reference_fn = inputs["reference"]
    ref_out = reference_fn()

    results: list[Result] = []
    # Reference timing too — useful to compare against.
    mean, p50, p95 = _time_callable(reference_fn, warmup=warmup, iters=iters)
    results.append(
        Result(
            op=op,
            impl=reference_name,
            mean_ms=mean,
            p50_ms=p50,
            p95_ms=p95,
            correct=None,
            max_abs_err=None,
        )
    )

    for impl_name, impl_fn in impls:
        if impl_fn is None:
            continue
        out = impl_fn()
        ok, err = _check_correctness(ref_out, out, atol=atol)
        mean, p50, p95 = _time_callable(impl_fn, warmup=warmup, iters=iters)
        results.append(
            Result(
                op=op,
                impl=impl_name,
                mean_ms=mean,
                p50_ms=p50,
                p95_ms=p95,
                correct=ok,
                max_abs_err=err,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Markdown rendering + README stamping.
# ---------------------------------------------------------------------------


def render_markdown(
    pointwise_results: list[Result],
    depthwise_results: list[Result],
    *,
    c: int,
    h: int,
    w: int,
    aten_threads: int,
) -> str:
    lines: list[str] = []
    lines.append(
        f"Inputs: `(C, H, W) = ({c}, {h}, {w})`, fp32, single-thread "
        f"(`torch.set_num_threads({aten_threads})`). All times wall-clock from "
        "`perf_counter_ns`."
    )
    lines.append("")

    for op_name, results in (
        ("Pointwise (1x1)", pointwise_results),
        ("Depthwise 3x3", depthwise_results),
    ):
        lines.append(f"### {op_name}")
        lines.append("")
        lines.append(
            "| Implementation | Mean (ms) | p50 (ms) | p95 (ms) | Speedup vs C naive | Correct? |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | :---: |")

        c_naive = next((r for r in results if r.impl == "C naive"), None)
        for r in results:
            if c_naive and r.impl != "C naive":
                speedup = c_naive.mean_ms / max(r.mean_ms, 1e-9)
                speedup_str = f"{speedup:.2f}x"
            elif r.impl == "C naive":
                speedup_str = "1.00x"
            else:
                speedup_str = "—"
            if r.correct is None:
                correct_str = "ref"
            elif r.correct:
                correct_str = f"yes (err {r.max_abs_err:.1e})"
            else:
                correct_str = f"NO (err {r.max_abs_err:.1e})"
            lines.append(
                f"| {r.impl} | {r.mean_ms:.4f} | {r.p50_ms:.4f} | {r.p95_ms:.4f} | "
                f"{speedup_str} | {correct_str} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def update_readme_section(readme_path: Path, table_md: str) -> bool:
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
# CLI / main.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--channels", type=int, default=DEFAULT_C)
    parser.add_argument("--height", type=int, default=DEFAULT_H)
    parser.add_argument("--width", type=int, default=DEFAULT_W)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for the per-implementation correctness check vs ATen reference.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch.set_num_threads(N) — keep at 1 for a fair comparison to single-threaded C.",
    )
    parser.add_argument(
        "--output",
        default="assets/microbench_table.md",
        help="Where to write the rendered markdown table.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Also stamp the table between BEGIN/END markers in README.md.",
    )
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    rng = np.random.default_rng(args.seed)
    c, h, w = args.channels, args.height, args.width
    inp = rng.standard_normal((c, h, w)).astype(np.float32)
    pw_weight = rng.standard_normal((c, c)).astype(np.float32)
    dw_weight = rng.standard_normal((c, 9)).astype(np.float32)

    lib_path = find_kernel_lib()
    c_kernels: _CKernels | None = None
    if lib_path is None:
        logger.warning(
            "C kernel library not found in %s. Build it with `make microbench-build`. "
            "C rows will be skipped.",
            CPP_BUILD_DIR,
        )
    else:
        logger.info("Loaded C kernels from %s", lib_path)
        c_kernels = load_c_kernels(lib_path)

    # Pointwise inputs.
    pointwise_inputs: dict[str, object] = {
        "reference": ("ATen (reference)", lambda: aten_pointwise(inp, pw_weight)),
        "impls": [
            ("NumPy einsum", lambda: numpy_pointwise(inp, pw_weight)),
            (
                "C naive",
                (lambda: c_pointwise(c_kernels.pointwise_naive, inp, pw_weight))
                if c_kernels
                else None,
            ),
            (
                "C AVX2",
                (lambda: c_pointwise(c_kernels.pointwise_avx2, inp, pw_weight))
                if c_kernels
                else None,
            ),
        ],
    }

    # Depthwise inputs.
    depthwise_inputs: dict[str, object] = {
        "reference": ("ATen (reference)", lambda: aten_depthwise_3x3(inp, dw_weight)),
        "impls": [
            (
                "C naive",
                (lambda: c_depthwise_3x3(c_kernels.depthwise_3x3_naive, inp, dw_weight))
                if c_kernels
                else None,
            ),
            (
                "C AVX2",
                (lambda: c_depthwise_3x3(c_kernels.depthwise_3x3_avx2, inp, dw_weight))
                if c_kernels
                else None,
            ),
        ],
    }

    pw = benchmark_op("pointwise", pointwise_inputs, args.warmup, args.iters, args.atol)
    dw = benchmark_op("depthwise_3x3", depthwise_inputs, args.warmup, args.iters, args.atol)

    table_md = render_markdown(pw, dw, c=c, h=h, w=w, aten_threads=args.threads)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_md, encoding="utf-8")
    logger.info("Wrote microbench table to %s", output_path)
    print("\n" + table_md)

    if args.update_readme:
        readme_path = Path(args.readme)
        if update_readme_section(readme_path, table_md):
            logger.info("Stamped table into %s between BEGIN/END markers.", readme_path)
        else:
            logger.warning(
                "Could not find %s / %s markers in %s; README not updated.",
                README_BEGIN,
                README_END,
                readme_path,
            )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_C",
    "DEFAULT_H",
    "DEFAULT_W",
    "Result",
    "aten_depthwise_3x3",
    "aten_pointwise",
    "benchmark_op",
    "c_depthwise_3x3",
    "c_pointwise",
    "find_kernel_lib",
    "load_c_kernels",
    "main",
    "numpy_depthwise_3x3",
    "numpy_pointwise",
    "render_markdown",
    "update_readme_section",
]
