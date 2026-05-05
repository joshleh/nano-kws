# nano-kws

**Edge-deployable keyword spotter.** A small DS-CNN trained on Google Speech
Commands, quantized to INT8, exported to ONNX, and benchmarked against its
fp32 baseline — plus a live mic demo and a C++ inference reference.

[![CI](https://github.com/joshualee/nano-kws/actions/workflows/ci.yml/badge.svg)](https://github.com/joshualee/nano-kws/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## TL;DR

> The bundled checkpoint is a **30-epoch CPU run** of DS-CNN at width
> multiplier 0.5 (62 K params, 44 M MACs). Headline numbers below are
> on Speech Commands v0.02 test split (4,888 clips, 12 classes).
> Static post-training quantization shrinks the model **2.6x on disk**
> and gives a **2.1x latency speedup** on host-CPU ONNX Runtime, at the
> cost of **~7 pp top-1**. That accuracy gap is the headline argument
> for the next stretch deliverable on the roadmap (quantization-aware
> training) — PTQ alone is leaving real accuracy on the table here.

<!-- BEGIN_BENCHMARK_TABLE -->

| Variant | Runtime | Params | MACs | Top-1 acc | Size on disk | Latency mean | Latency p50 | Latency p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-CNN small fp32 | PyTorch (CPU) | 62.1 K | 44.13 M | 79.32% | n/a | 3.238 ms | 3.308 ms | 4.517 ms |
| DS-CNN small fp32 | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 79.32% | 242.6 KB | 0.442 ms | 0.424 ms | 0.583 ms |
| DS-CNN small INT8 | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 72.20% | 93.4 KB | 0.399 ms | 0.378 ms | 0.510 ms |

**INT8 vs fp32 (ONNX Runtime):**
- Size: 38.5% of fp32 (242.6 KB -> 93.4 KB)
- Latency: 1.11x (0.442 ms -> 0.399 ms mean)
- Top-1 accuracy: -7.12 pp

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

## Why this project

Modern always-on voice interfaces — wake-word detection in earbuds, hearables,
and smart speakers — run on **ultra-low-power neural decision processors**
that have on the order of **~100 KB of weight memory** and a power budget
under **1 mW**. Getting a Python-trained model onto that hardware is a
specific engineering pipeline:

1. Train a small architecture in PyTorch (DS-CNN, TC-ResNet, BC-ResNet, …).
2. Match the on-device audio frontend (log-mel) **bit-for-bit** during training.
3. Post-training quantize to INT8 and verify the accuracy delta is acceptable.
4. Export to a portable inference format (ONNX / TFLite) and benchmark
   against the fp32 baseline on the same hardware.
5. Deploy through a vendor toolchain to the target NPU.

`nano-kws` is a focused walk through steps 1–4 of that pipeline, plus a C++
inference harness as a stand-in for step 5. It is intentionally narrow —
**one model family, one dataset, one quantization path, real numbers** —
and it exists to make the edge-AI competency concrete and reviewable.

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
cpp/             ONNX Runtime C API inference harness (stretch)
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
| Quantization | Static PTQ (FX graph mode) | DS-CNN typically loses <1 pp top-1 with PTQ; QAT only justified if PTQ drop is meaningful. |
| Primary export | ONNX + ONNX Runtime | Native from PyTorch, broadly accepted by edge toolchains. TFLite path is fragile and out of scope. |
| Audio frontend | Log-mel (40 bins, 30 ms / 10 ms, 16 kHz, 1-s clips) | Matches "Hello Edge" / DS-CNN baseline so results are comparable to published work. |
| Class set | 12-class (10 keywords + `_silence_` + `_unknown_`) | Standard Speech Commands setup; sanity-checks accuracy against the literature. |
| Featurizer location | `nano_kws.data.features` — single source of truth | The most common edge-deploy bug is "training mel ≠ inference mel". One function, both paths. |
| Inference mode | Windowed (1 s clips) | Streaming is its own engineering problem (ring buffer + posterior smoothing) and out of scope. |

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
- [ ] Quantization-aware training (only if PTQ accuracy drop is large)
- [ ] Hand-written depthwise-separable conv microbenchmark (NumPy → C/SIMD vs ATen)

---

## Citing prior work

- Zhang, Y. et al. *Hello Edge: Keyword Spotting on Microcontrollers.* (2017)
- Warden, P. *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* (2018)

---

## License & model card

- Code: [MIT](LICENSE)
- Model card: [`MODEL_CARD.md`](MODEL_CARD.md)
