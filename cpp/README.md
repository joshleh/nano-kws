# C++ inference harness (Stretch)

A ~150-line reference that loads `assets/ds_cnn_small_int8.onnx` via the
**ONNX Runtime C++ API** and times the *pure model forward pass* outside
the Python interpreter. The point: prove the same artefact the Python
pipeline ships also runs from a systems-language harness, and report
host-CPU latency without `torch` / `torchaudio` / `numpy` / Python
overhead in the way.

## Layout

```
cpp/
├── CMakeLists.txt   # locate ONNX Runtime, build a single binary
├── infer.cpp        # ~150 lines: load model, time N runs, print stats
└── README.md        # this file
```

## Why a synthetic input?

The harness intentionally does **not** depend on libsndfile / dr_wav /
DSP libraries. It generates a distribution-plausible synthetic log-mel
spectrogram of the right shape (`1 x 1 x 40 x 97`) and feeds it to the
model. This isolates **model latency** from audio I/O and featurisation
overhead, and keeps the dependency footprint to *just* ONNX Runtime —
which is the only piece you'd realistically port to an embedded target.

For real-audio inference, run `python -m nano_kws.infer` instead — it
shares the bundled INT8 ONNX with this harness so the numerical
behaviour is identical.

## Build

You need an ONNX Runtime release. The easiest path:

1. Download a prebuilt release for your platform from
   <https://github.com/microsoft/onnxruntime/releases> (e.g.
   `onnxruntime-linux-x64-1.18.0.tgz`, `onnxruntime-win-x64-1.18.0.zip`,
   `onnxruntime-osx-arm64-1.18.0.tgz`).
2. Extract somewhere; note the path. It should contain `include/` and
   `lib/`.
3. Configure + build:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=/path/to/onnxruntime-<platform>-1.18.0
cmake --build cpp/build --config Release
```

On Windows the binary lands at `cpp/build/Release/nano_kws_infer.exe`;
on Linux/macOS at `cpp/build/nano_kws_infer`. Make sure the runtime
shared library (`onnxruntime.dll` / `libonnxruntime.so` /
`libonnxruntime.dylib`) is on your loader path.

## Run

```bash
./cpp/build/nano_kws_infer assets/ds_cnn_small_int8.onnx --iters 1000 --warmup 100
```

Expected output (numbers will vary by host):

```
Model: assets/ds_cnn_small_int8.onnx
Input: input [1, 1, 40, 97]
Threads: 1 intra-op (single-thread latency)
Warmup iters: 100 | Timed iters: 1000

First-iter top-1 (synthetic input): _silence_ (logit=4.21)

Latency (ms) — mean 0.34 | p50 0.33 | p95 0.42 | p99 0.51
```

## What this proves

The Python benchmark (`python -m nano_kws.benchmark`) reports ~0.38 ms
mean for the same model on the same machine. The C++ harness should be
within noise of that (or slightly faster — no Python C-extension
boundary) which validates that **the Python timing isn't artificially
inflated by the wrapper**. For an actual edge port, the next step is to
swap `Ort::Session` for the C API on a static-linked, threadless build,
or to replace ONNX Runtime entirely with a microcontroller runtime
(TFLite Micro / CMSIS-NN / a vendor SDK such as Syntiant's NDP toolchain)
— that's the natural follow-on to this harness.
