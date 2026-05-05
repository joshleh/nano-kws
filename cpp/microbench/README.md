# Hand-written conv kernels — microbenchmark

This directory contains a from-scratch C implementation of the two ops
that DS-CNN spends almost all its compute on, with a Python harness
that pits each implementation against PyTorch's MKL-DNN-backed
`torch.nn.functional.conv2d`.

The point isn't to *beat* ATen — it's to **quantify the gap** and have
something concrete to point at when an interviewer asks "OK, but could
you write the kernel yourself?". For an MLE role on a vendor-kernel
team (CMSIS-NN, TFLite Micro, Syntiant's NDP toolchain, …) being
articulate about that gap and what closes it is core competency.

## Contents

- `conv_kernels.h` / `conv_kernels.c` — four functions, plain C with
  AVX2 + FMA intrinsics:
  - `conv_pointwise_naive`        — 1×1 conv, triple nested loop
  - `conv_pointwise_avx2`         — 1×1 conv, 256-bit FMA along H·W
  - `conv_depthwise_3x3_naive`    — 3×3 depthwise, scalar
  - `conv_depthwise_3x3_avx2`     — 3×3 depthwise, 256-bit FMA along W
- `CMakeLists.txt` — builds `conv_kernels.{dll,so,dylib}` with `/O2 /arch:AVX2`
  on MSVC and `-O3 -mavx2 -mfma -ffast-math` on GCC/Clang.
- `../../scripts/conv_microbench.py` — Python orchestrator: loads the
  shared library via `ctypes`, generates fp32 inputs, runs every
  implementation, checks correctness against `torch.nn.functional.conv2d`,
  reports mean / p50 / p95 latency, and renders a markdown table.

## Build

The library has zero external dependencies — just CMake and a C
compiler with AVX2 support (any x86-64 CPU from ~2013 onward).

```bash
make microbench-build
# or, manually:
cmake -S cpp/microbench -B cpp/microbench/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/microbench/build --config Release -j
```

The build drops `cpp/microbench/build/[Release/]conv_kernels.dll`
(Windows) or `cpp/microbench/build/conv_kernels.so` (Linux).

If you don't build the library, the Python harness skips the C rows
with a warning and reports only NumPy / ATen — useful for sanity
checking the harness itself on a machine with no C toolchain.

## Run

```bash
make microbench
# or:
python -m scripts.conv_microbench --update-readme
```

The default input shape `(C=56, H=16, W=47)` is the interior
DepthwiseSeparableBlock activation shape of the bundled DS-CNN at
width 0.5 — the same compute your INT8 deployment would actually run.

## What to look at in the table

Three things are interesting:

1. **Correctness column** — every implementation must match ATen
   within a tight tolerance. If it's flagged as incorrect, the kernel
   is broken regardless of how fast it is.
2. **C naive vs C AVX2 speedup** — for the pointwise kernel this
   should be in the 4–8× range; the inner loop is ~all FMA. For the
   depthwise it's smaller because boundary handling forces the SIMD
   path to leave a one-pixel scalar border around the output.
3. **C AVX2 vs ATen** — this is the "how far am I from production"
   number. ATen on x86 dispatches to MKL-DNN, which adds register
   tiling, L1 blocking, prefetch, and im2col-then-GEMM where it
   helps. We expect to land at 0.2–0.5× ATen on pointwise and
   somewhere similar on depthwise.

## What I'd reach for next

Two-line summary of what's missing from the AVX2 path, in priority
order — this is the answer to "OK, you're 3× slower than ATen, what
do you do?":

1. **Pointwise** is essentially a `(C_out, H·W)` x `(C_in)` matmul.
   The standard production move is to *reshape it to a GEMM* — call
   into BLAS (or a hand-rolled `cblas_sgemm` lookalike with register
   tiling and L1 cache blocking) instead of writing the loops by
   hand. That alone usually closes most of the gap to ATen.
2. **Depthwise** is harder to GEMM-ify (each channel has its own
   filter) but benefits from a NHWC layout, register-blocked inner
   loops that compute multiple `oh` rows at once to amortise the
   padded-row reload, and (on bigger CPUs) AVX-512 — same FMA count,
   double the lane width.

These are deliberately out of scope here — the goal is the
microbench, not a CMSIS-NN clone.
