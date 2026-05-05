# nano-kws

**A tiny "Hey Alexa"-style keyword spotter that runs in under 100 KB.**

Trained in PyTorch, compressed to 8-bit integer math, exported for any
edge device — the same engineering pipeline that ships voice models to
earbuds, doorbells, and smart-home hubs.

[**Try the live demo →**](https://nano-kws.streamlit.app/) &nbsp;·&nbsp;
[Skills demonstrated](#skills-demonstrated) &nbsp;·&nbsp;
[Quickstart](#quickstart) &nbsp;·&nbsp;
[Results](#results)

[![Live demo](https://img.shields.io/badge/live%20demo-nano--kws.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://nano-kws.streamlit.app/)
[![CI](https://github.com/joshualee/nano-kws/actions/workflows/ci.yml/badge.svg)](https://github.com/joshualee/nano-kws/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[![nano-kws live demo screenshot](assets/streamlit_demo.png)](https://nano-kws.streamlit.app/)
<sub>↑ The live demo at [nano-kws.streamlit.app](https://nano-kws.streamlit.app/). Drop in a 16 kHz WAV in the **Upload WAV** tab, or upload a longer recording in **Continuous** to watch the model fire keyword detections over time.</sub>

---

## What this project is, in plain English

Modern voice interfaces — your earbuds saying "Hey Alexa", a doorbell
listening for "ring", a hearing aid filtering background noise — run a
neural network on a tiny chip with about as much memory as a 1990s
floppy disk and a power budget smaller than an LED's. Getting a model
onto that hardware is its own engineering discipline: you have to train
something small enough to fit, compress it to integer math without
losing accuracy, and benchmark every step honestly.

**`nano-kws` is a focused walk through that whole pipeline** for a
10-keyword recognizer (`yes / no / up / down / left / right / on / off /
stop / go`). The final model is **under 100 KB on disk** and classifies
a 1-second audio clip in **under half a millisecond on a single CPU
thread**. You can try it yourself in 30 seconds at the
[live demo](https://nano-kws.streamlit.app/) — drop in a WAV of yourself
saying any of the keywords and watch the model decide.

The repo is intentionally small and readable: one model family, one
dataset, one quantization path, real numbers in every table, every
design decision explained in the design-decisions table below.

## Skills demonstrated

A scan-friendly index of what's in this repo — each item maps to
specific code, results, and design decisions you can dig into:

| Area | What's here |
| --- | --- |
| **Audio ML** | Log-mel spectrogram frontend matched bit-for-bit between training and inference ([`nano_kws/data/features.py`](nano_kws/data/features.py)); 12-class Speech Commands setup. |
| **Model design** | DS-CNN (depthwise-separable convs) with a width multiplier so the same architecture sweeps from 18 K to 224 K parameters ([`nano_kws/model.py`](nano_kws/model.py)). |
| **Hardware-aware tradeoffs** | Multi-size sweep reporting accuracy vs parameter count vs MACs vs latency ([sweep table](#model-size-sweep)). |
| **Quantization (PTQ)** | Static post-training quantization to INT8 via ONNX Runtime — 2.6× smaller, faster, with measured accuracy delta ([benchmark table](#tldr-technical)). |
| **Quantization (QAT)** | Custom straight-through-estimator fake-quant + per-channel weight quantization, 5-epoch fine-tune; recovers the PTQ accuracy gap ([`nano_kws/qat.py`](nano_kws/qat.py)). |
| **Edge inference** | ONNX Runtime in Python *and* C++ (`cpp/infer.cpp`); same model, same pre/post-processing, byte-identical outputs. |
| **Streaming inference** | Sliding-window classifier with EMA-smoothed posteriors and per-keyword peak-picking ([`nano_kws/streaming.py`](nano_kws/streaming.py)) — the building block of every wake-word detector. |
| **Hand-written kernels** | Depthwise + pointwise conv in plain C with AVX2 + FMA intrinsics, benchmarked against PyTorch's MKL-DNN ([`cpp/microbench/`](cpp/microbench/)). |
| **Software engineering** | `pyproject.toml`, `ruff`, `pre-commit`, `pytest` (131 tests), GitHub Actions CI on Python 3.11 + 3.12, `Makefile` with one-command targets. |
| **Deployment** | Streamlit web demo deployed on Streamlit Community Cloud with the bundled INT8 model — zero training required for a fresh clone. |

> **Why this project exists:** I built `nano-kws` while preparing for a
> Machine Learning Engineering Intern interview at **Syntiant**, who
> design ultra-low-power Neural Decision Processors for on-device voice
> AI. The repo is structured to mirror their actual work: train in
> PyTorch, quantize to INT8, deploy to constrained hardware, measure
> everything. It's a 1-2 week portfolio sprint, not a research project
> — scope was kept ruthlessly narrow to make the engineering bar
> visible.

---

## TL;DR (technical)

> The bundled checkpoint is a **30-epoch CPU run** of DS-CNN at width
> multiplier 0.5 (62 K params, 44 M MACs). Headline numbers below are
> on Speech Commands v0.02 test split (4,888 clips, 12 classes).
> Static post-training quantization shrinks the model **2.6x on disk**
> at the cost of ~7 pp top-1 — that gap is what motivated the QAT
> stretch deliverable. The *INT8 (QAT)* row is a **5-epoch fine-tune
> from the same fp32 checkpoint with augmentation disabled**,
> fake-quant active in the forward pass, then standard PTQ on the
> resulting weights. Two effects compound there: training against the
> INT8 grid (the textbook QAT effect) plus adapting to the cleaner
> test distribution (Speech Commands test is mostly studio-clean and
> the original 30-epoch run trained with heavy SpecAugment +
> background-noise mixing). See [`MODEL_CARD.md`](MODEL_CARD.md) for
> the per-effect attribution caveat.

<!-- BEGIN_BENCHMARK_TABLE -->

| Variant | Runtime | Params | MACs | Top-1 acc | Size on disk | Latency mean | Latency p50 | Latency p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-CNN small fp32 | PyTorch (CPU) | 62.1 K | 44.13 M | 79.32% | n/a | 3.866 ms | 3.758 ms | 5.652 ms |
| DS-CNN small fp32 | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 79.32% | 242.6 KB | 0.586 ms | 0.496 ms | 0.945 ms |
| DS-CNN small INT8 (PTQ) | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 72.20% | 93.4 KB | 0.413 ms | 0.405 ms | 0.526 ms |
| DS-CNN small INT8 (QAT) | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 90.67% | 93.4 KB | 0.436 ms | 0.424 ms | 0.514 ms |

**INT8 (PTQ) vs fp32 (ONNX Runtime):**
- Size: 38.5% of fp32 (242.6 KB -> 93.4 KB)
- Latency: 1.42x (0.586 ms -> 0.413 ms mean)
- Top-1 accuracy: -7.12 pp

**INT8 (QAT) vs fp32 (ONNX Runtime):**
- Top-1 accuracy: +11.35 pp
- vs PTQ-only INT8: +18.47 pp top-1 recovered by QAT.

<!-- END_BENCHMARK_TABLE -->

A pre-trained INT8 model is committed to [`assets/`](assets/) so a fresh
clone can run the benchmark and live demo **without training**.

---

## Model size sweep

Hardware-aware ML in miniature: sweep the DS-CNN width multiplier and
read off accuracy vs parameter count vs MACs / inference. Numbers
populated by `make sweep`; raw artefacts land in `runs/sweep/` (gitignored)
and the rendered table mirrors [`assets/sweep_table.md`](assets/sweep_table.md).
The accuracy-vs-MACs plot is at [`assets/sweep_plot.png`](assets/sweep_plot.png).

<!-- BEGIN_SWEEP_TABLE -->

| Width | Params | MACs | fp32 top-1 | INT8 top-1 | Δ acc | fp32 size | INT8 size | Size ratio | fp32 latency | INT8 latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 18.5 K | 12.63 M | 73.57% | 61.62% | -11.95 pp | 74.4 KB | 43.5 KB | 58.4% | 0.330 ms | 0.306 ms |
| 0.5 | 62.1 K | 44.13 M | 79.32% | 72.20% | -7.12 pp | 242.6 KB | 93.4 KB | 38.5% | 0.976 ms | 0.459 ms |
| 1 | 224.5 K | 163.73 M | 87.19% | 79.56% | -7.63 pp | 873.1 KB | 266.9 KB | 30.6% | 2.606 ms | 1.006 ms |

<!-- END_SWEEP_TABLE -->

---

## Hand-written conv kernels vs ATen

The DS-CNN's compute is dominated by two ops: 1×1 pointwise conv
(channel mixing) and 3×3 depthwise conv (per-channel spatial filter).
[`cpp/microbench/`](cpp/microbench/) is a from-scratch C
implementation of both — naive triple-nested-loop and AVX2 + FMA
intrinsics — pitted against PyTorch's MKL-DNN-backed
`torch.nn.functional.conv2d`. The point isn't to beat ATen; it's to
*quantify* the gap. See [`cpp/microbench/README.md`](cpp/microbench/README.md)
for methodology and "what would close the gap" notes.

Build the C kernels with `make microbench-build` (needs CMake +
MSVC/GCC/Clang with AVX2), then run `make microbench`. Without the
build, only the NumPy + ATen rows populate.

<!-- BEGIN_MICROBENCH_TABLE -->

Inputs: `(C, H, W) = (56, 16, 47)`, fp32, single-thread (`torch.set_num_threads(1)`). All times wall-clock from `perf_counter_ns`.

### Pointwise (1x1)

| Implementation | Mean (ms) | p50 (ms) | p95 (ms) | Speedup vs C naive | Correct? |
| --- | ---: | ---: | ---: | ---: | :---: |
| ATen (reference) | 0.1593 | 0.1631 | 0.1955 | — | ref |
| NumPy einsum | 0.5209 | 0.5314 | 0.6870 | — | yes (err 7.6e-06) |

### Depthwise 3x3

| Implementation | Mean (ms) | p50 (ms) | p95 (ms) | Speedup vs C naive | Correct? |
| --- | ---: | ---: | ---: | ---: | :---: |
| ATen (reference) | 0.4449 | 0.4119 | 0.6304 | — | ref |

<!-- END_MICROBENCH_TABLE -->

---

## The edge-AI pipeline

The deployment pipeline this repo walks through, step by step:

1. **Train** a small architecture in PyTorch (DS-CNN here; TC-ResNet,
   BC-ResNet, MobileNet are all peers in this regime).
2. **Match the audio frontend bit-for-bit** between training and
   inference. The most common edge-deploy bug is "training mel ≠
   inference mel" — one off-by-one in window size and accuracy
   collapses on the device.
3. **Quantize to INT8** and verify the accuracy delta is acceptable.
   Two paths shipped here: PTQ (post-training, fast, easy) and QAT
   (quantization-aware fine-tune, slower, recovers accuracy).
4. **Export to a portable inference format** (ONNX) and benchmark
   against the fp32 baseline on the same hardware. Latency, peak
   memory, model file size — all measured, all in the README.
5. **Deploy through a vendor toolchain** to the target NPU. Here the
   `cpp/infer.cpp` harness stands in for that step (ONNX Runtime C++
   API instead of a vendor SDK).

The bundled artefacts in [`assets/`](assets/) (~400 KB total) let a
fresh clone reproduce every number in the README without training
anything.

---

## Architecture

```
            ┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
 16 kHz PCM │  log-mel front  │ →   │   DS-CNN     │ →   │  12-class    │
   1-s clip │  40 bins        │     │  (PyTorch)   │     │  softmax     │
            │  30 ms / 10 ms  │     └──────┬───────┘     └──────────────┘
            └─────────────────┘            │
                                           │ static PTQ
                                           ▼
                                   ┌──────────────┐     ┌──────────────┐
                                   │  INT8 ONNX   │ →   │ ONNX Runtime │
                                   │  (assets/)   │     │  / C++ API   │
                                   └──────────────┘     └──────────────┘
```

A higher-resolution diagram lives at [`assets/arch_diagram.png`](assets/arch_diagram.png).

The Streamlit demo wraps the same exported INT8 ONNX in a sliding-window
[`StreamingClassifier`](nano_kws/streaming.py) for its **Continuous** tab
— 1-second window, configurable hop, EMA-smoothed posteriors,
per-keyword peak-picking with a refractory window. The model itself
remains a single-window classifier; streaming is built on top of that
primitive in pure Python, exactly the way wake-word detection is built
on top of single-window NN inference in production deployments.

---

## Quickstart

```bash
# 1. Install (Python 3.11+)
make install

# 2. Run the benchmark on the bundled INT8 model — no training required
make benchmark

# 3. Launch the live mic demo
make app
```

To reproduce training from scratch:

```bash
make download-data    # ~2.4 GB Speech Commands v2 → ~/.cache/nano_kws/
make train            # ~30 min on a single modern CPU; faster on GPU
make quantize         # PTQ → INT8 ONNX in assets/
make benchmark        # regenerates the TL;DR table
```

---

## Repo layout

```
nano_kws/        importable package: data, model, train, quantize, infer, benchmark
scripts/         dataset fetcher, multi-size sweep
app/             Streamlit live mic demo
cpp/             ONNX Runtime C++ inference harness + hand-written AVX2 conv kernels (microbench)
assets/          committed INT8 model + benchmark snapshot for zero-setup demo
tests/           pytest suite (uses synthetic audio, no dataset required)
docs/            extended results, design notes, MAC budget derivations
notebooks/       one-off EDA
```

---

## Results

### Accuracy / size / latency sweep

| Width multiplier | Params | MACs | Top-1 (fp32) | Top-1 (INT8) | INT8 size |
| ---------------- | ------ | ---- | ------------ | ------------ | --------- |
| 0.25             | _TBD_  | _TBD_| _TBD_        | _TBD_        | _TBD_     |
| 0.50 (default)   | _TBD_  | _TBD_| _TBD_        | _TBD_        | _TBD_     |
| 1.00             | _TBD_  | _TBD_| _TBD_        | _TBD_        | _TBD_     |

Plot: `docs/accuracy_vs_macs.png` *(generated by `scripts/sweep_sizes.py`)*.

### Per-class confusion (small INT8)

See [`docs/benchmark.md`](docs/benchmark.md).

---

## Design decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Quantization (PTQ) | Static PTQ via `onnxruntime.quantization`, QDQ format, per-channel weights | The format every modern edge runtime ingests cleanly; per-channel weights matter at this model scale. |
| Quantization (QAT) | Custom STE fake-quant + per-channel weight quantization, fine-tune from fp32 ckpt for 5 epochs, freeze observers after epoch 2, then run standard PTQ on the QAT-trained weights | Vanilla PTQ on this DS-CNN drops ~7 pp top-1 — QAT closes most of that gap. Custom (vs `torch.ao.quantization.quantize_fx`) keeps the trained model structurally identical to a vanilla DS-CNN, so the existing export + quantize pipeline consumes it unchanged. |
| Primary export | ONNX + ONNX Runtime | Native from PyTorch, broadly accepted by edge toolchains. TFLite path is fragile and out of scope. |
| Audio frontend | Log-mel (40 bins, 30 ms / 10 ms, 16 kHz, 1-s clips) | Matches "Hello Edge" / DS-CNN baseline so results are comparable to published work. |
| Class set | 12-class (10 keywords + `_silence_` + `_unknown_`) | Standard Speech Commands setup; sanity-checks accuracy against the literature. |
| Featurizer location | `nano_kws.data.features` — single source of truth | The most common edge-deploy bug is "training mel ≠ inference mel". One function, both paths. |
| Inference mode | Single 1-s window for the exported ONNX; Python-side sliding window + EMA + peak-pick for the demo's Continuous tab | The model contract stays a fixed-shape 1-s classifier so the export is portable to any edge runtime. Streaming is the per-deployment glue (ring buffer + posterior smoother) — done in pure Python here, would be a hand-tuned C/DSP loop on real hardware. |
| Hand-written kernels | Pointwise + depthwise 3×3 in plain C with AVX2 + FMA intrinsics, loaded into Python via `ctypes` | The two ops are 95%+ of DS-CNN's compute. Writing them from scratch (and honestly benchmarking the gap to MKL-DNN) is the closest analogue in this repo to the hand-tuned-kernel work that vendor edge stacks ship. |

---

## Roadmap

**MVP**

- [x] Phase 0 — repo scaffold, CI, packaging
- [x] Phase 1 — data pipeline + log-mel featurizer
- [x] Phase 2 — DS-CNN model + training loop _(30-epoch w=0.5 checkpoint bundled in `assets/`)_
- [x] Phase 3 — PTQ → INT8 ONNX export _(bundled INT8 ONNX in `assets/`)_
- [x] Phase 4 — fp32 vs INT8 benchmark + README TL;DR populated
- [x] Phase 5 — multi-size sweep table + plot, real numbers from the overnight run

**Stretch**

- [x] Streamlit live mic demo (`make app`)
- [x] C++ inference harness (ONNX Runtime C++ API) — `cpp/`
- [x] Quantization-aware training (`make qat`) — custom STE fake-quant + per-channel weight quantization, 5-epoch fine-tune from the fp32 checkpoint, then standard PTQ; see `nano_kws/qat.py` and the *INT8 (QAT)* row in the TL;DR table.
- [x] Hand-written depthwise-separable conv microbenchmark (NumPy → C scalar → C AVX2 vs ATen) — `cpp/microbench/` + `scripts/conv_microbench.py`. Run `make microbench-build && make microbench` to populate the [microbench table](#hand-written-conv-kernels-vs-aten) above.

---

## Citing prior work

- Zhang, Y. et al. *Hello Edge: Keyword Spotting on Microcontrollers.* (2017)
- Warden, P. *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* (2018)

---

## License & model card

- Code: [MIT](LICENSE)
- Model card: [`MODEL_CARD.md`](MODEL_CARD.md)
