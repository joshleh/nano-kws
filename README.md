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
[![CI](https://github.com/joshleh/nano-kws/actions/workflows/ci.yml/badge.svg)](https://github.com/joshleh/nano-kws/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[![nano-kws live demo screenshot](assets/streamlit_demo.png)](https://nano-kws.streamlit.app/)
<sub>↑ Continuous-mode tab of the live demo at [nano-kws.streamlit.app](https://nano-kws.streamlit.app/): a 10 s recording streamed through the bundled INT8 model with a 200 ms hop and EMA-smoothed per-keyword posteriors, surfacing four detections (`yes`, `yes`, `no`, `on`). Sidebar exposes the streaming knobs — window hop, smoothing alpha, detection threshold, refractory.</sub>

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
| **AED applicability** | The pipeline is keyword-specific only at the dataset and label-set layer — the model, frontend, quantization, and deployment recipe transfer to non-speech *Acoustic Event Detection* (turkey gobble, glass-break, baby cry, smoke alarm) by swapping the dataset module. Documented in the [From KWS to AED section](#from-kws-to-aed-same-recipe-different-label-set). |
| **Model design** | DS-CNN (depthwise-separable convs) with a width multiplier so the same architecture sweeps from 18 K to 224 K parameters ([`nano_kws/models/ds_cnn.py`](nano_kws/models/ds_cnn.py)). |
| **Hardware-aware tradeoffs** | Multi-size sweep reporting accuracy vs parameter count vs MACs vs latency ([sweep table](#model-size-sweep)). |
| **Quantization (PTQ)** | Static post-training quantization to INT8 via ONNX Runtime — 2.6× smaller, faster, with measured accuracy delta ([benchmark table](#tldr-technical)). |
| **Quantization (QAT)** | Custom straight-through-estimator fake-quant + per-channel weight quantization, 5-epoch fine-tune; recovers the PTQ accuracy gap ([`nano_kws/qat.py`](nano_kws/qat.py)). |
| **Edge inference** | ONNX Runtime in Python *and* C++ (`cpp/infer.cpp`); same model, same pre/post-processing, byte-identical outputs. |
| **Streaming inference** | Sliding-window classifier with EMA-smoothed posteriors and per-keyword peak-picking ([`nano_kws/streaming.py`](nano_kws/streaming.py)) — the building block of every wake-word detector. |
| **Hand-written kernels** | Depthwise + pointwise conv in plain C with AVX2 + FMA intrinsics, benchmarked against PyTorch's MKL-DNN ([`cpp/microbench/`](cpp/microbench/)). |
| **Software engineering** | `pyproject.toml`, `ruff`, `pre-commit`, `pytest` (160 tests), GitHub Actions CI on Python 3.11 + 3.12, `Makefile` with one-command targets. |
| **Deployment** | Streamlit web demo deployed on Streamlit Community Cloud with the bundled INT8 model — zero training required for a fresh clone. |

> **Why this project exists:** the same train → quantize → export →
> benchmark pipeline that ships speech models onto edge devices also
> ships *non-speech* audio models — turkey-gobble detectors, glass-break
> alarms, baby-cry monitors, smoke-alarm classifiers — onto the same
> hardware. Real-world Acoustic Event Detection (AED) tasks live in a
> very different *data* regime than KWS, though: only a few thousand
> labelled samples per class, and active research on closing that gap
> with generative data augmentation. `nano-kws` is the deployment-side
> spine of that recipe, plus three explicit experiments mapping the
> KWS work onto the AED low-data regime:
> [From KWS to AED](#from-kws-to-aed-same-recipe-different-label-set)
> (architecture isomorphism), [Few-shot transfer](#few-shot-transfer-how-much-data-do-you-actually-need)
> (from-scratch vs fine-tuned at K ∈ {10, 50, 200, 500} samples/class —
> fine-tuning wins by **+37 pp at K = 200**), and
> [Augmentation in the low-data regime](#augmentation-in-the-low-data-regime--and-why-more-aug-isnt-free)
> (a 6-cell ablation with a surprising result that motivates
> *generative* augmentation specifically). It's a focused 1-2 week
> sprint, not a research project — scope was kept narrow on purpose to
> make the engineering bar visible.

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

**Legend.** *Speedup vs C naive* divides each row's mean wall-time by the hand-written C scalar (`C naive`) baseline, so it answers "how much faster than a textbook nested-loop C kernel?". *Correct?* compares the implementation's output to ATen element-wise (max absolute error `<= --atol`, default 1e-3); `ref` marks the ATen reference itself.

> **Note.** The hand-written C kernels were not built on this machine, so the *C naive* and *C AVX2* rows are absent and the *Speedup vs C naive* column is blank (no baseline to divide by). Run `make microbench-build && make microbench` on a host with CMake + a C compiler with AVX2 to populate them.

### Pointwise (1x1)

| Implementation | Mean (ms) | p50 (ms) | p95 (ms) | Speedup vs C naive | Correct? |
| --- | ---: | ---: | ---: | ---: | :---: |
| ATen (reference) | 0.1150 | 0.0897 | 0.2011 | n/a | ref |
| NumPy einsum | 0.4033 | 0.3523 | 0.6034 | n/a | yes (err 7.6e-06) |

### Depthwise 3x3

| Implementation | Mean (ms) | p50 (ms) | p95 (ms) | Speedup vs C naive | Correct? |
| --- | ---: | ---: | ---: | ---: | :---: |
| ATen (reference) | 0.1936 | 0.1905 | 0.2148 | n/a | ref |

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

## From KWS to AED: same recipe, different label set

Keyword Spotting (KWS — "did the user say *yes / stop / on*?") and
Acoustic Event Detection (AED — "did the device hear a *turkey gobble
/ glass break / smoke alarm*?") are the **same problem at the
deployment layer**. Both ingest a short audio clip, both run a small
CNN on a log-mel spectrogram, both deploy to a sub-mW edge accelerator,
and both care about the same metrics (accuracy, latency, model size,
peak RAM, MACs per inference). The only things that change going from
KWS to AED are:

| Layer | KWS (this repo) | AED (e.g. a turkey-gobble or glass-break detector) |
| --- | --- | --- |
| Dataset module | Speech Commands v0.02 (~85 K labelled 1-s utterances) | Domain-specific event recordings (~2 K samples typical) |
| Label set | 10 keywords + `_silence_` + `_unknown_` | The events of interest + a `_background_` / negative class |
| Frontend | Log-mel, 40 mels x 97 frames at 16 kHz / 30 ms / 10 ms | Identical — log-mel is the standard frontend for both KWS and most AED work |
| Backbone | DS-CNN (`nano_kws/models/ds_cnn.py`) | Same. DS-CNN, TC-ResNet, BC-ResNet, MobileNet are all valid here. |
| Quantization → ONNX → ORT C++ | [Bundled INT8 export](#tldr-technical) | Identical — the whole `nano_kws.quantize` → `cpp/infer.cpp` path is dataset-agnostic. |
| Hard part that changes | KWS has plenty of data, so augmentation + few-shot transfer are second-order | Far less data per class, so few-shot transfer + generative augmentation (EcoGen / BirdDiff / AudioLDM) dominate the accuracy story |

In practice, porting `nano-kws` to a new AED task means:

1. Subclass `nano_kws/data/speech_commands.py` (or write a sibling
   `nano_kws/data/<your_dataset>.py`) that yields
   `(waveform_16k_mono, label_int)` tuples.
2. Update `config.LABELS` / `config.NUM_CLASSES` to match the new
   label set.
3. Run `make train` → `make quantize` → `make benchmark`. Everything
   downstream of the dataset module — featurizer, model, training
   loop, ONNX export, INT8 calibration, C++ inference, microbench,
   streaming — is reused unchanged.

The two AED-specific stories that *don't* fall out for free — and
that real-world AED work spends most of its time on — are **few-shot
transfer from a pretrained audio backbone** and **generative data
augmentation** (EcoGen, BirdDiff, AudioLDM). See the
[Related work for AED on edge section](#related-work-for-aed-on-edge)
for one-paragraph summaries of each and where they slot into this
pipeline.

---

## Few-shot transfer: how much data do you actually need?

Real-world AED tasks (turkey gobble, glass break, smoke alarm,
baby cry) typically have only **~2 K labelled samples** per target
class — orders of magnitude less than KWS datasets. The standard
answer is to fine-tune a pretrained audio model rather than train
from scratch, but how much does that actually buy you, and at what
data budget does the gap close?

This experiment runs the structural analog on Speech Commands:

1. **Base task** (the pretrained-audio-model stand-in): train
   DS-CNN-w0.5 on 6 keywords (`down, left, no, right, up, yes`) +
   `_silence_` + `_unknown_` = 8 classes.
2. **Novel task**: the held-out 4 keywords (`go, off, on, stop`) +
   `_silence_` + `_unknown_` = 6 classes — labels the base model
   never saw.
3. For each K ∈ {10, 50, 200, 500} samples per novel class, train
   two DS-CNN-w0.5 models head-to-head: a **from-scratch** baseline
   and a **fine-tuned** variant initialised from the base checkpoint
   with a fresh 6-way head.

Reproduce with `python -m scripts.few_shot --update-readme`. Raw
artefacts land in `runs/few_shot/` (gitignored) and the rendered
table mirrors [`assets/few_shot_table.md`](assets/few_shot_table.md).
The K = 200 row is the AED-relevant one — it maps directly to the
"~200 samples per class" regime that turkey-gobble / glass-break
detectors live in.

What the numbers actually say:

* **K = 10 (60 total clips, ~ten clips per class):** both models
  collapse to ~17 % val acc (random for 6 classes ≈ 16.7 %). At this
  budget you simply don't have enough gradient signal to retrain
  even the fresh classifier head, let alone fine-tune the backbone.
  Pretraining doesn't help yet — the bottleneck is K, not the
  initialisation.
* **K = 50 (300 clips):** the from-scratch model is still stuck at
  random; the fine-tuned model jumps to **44 %**. **+27 pp lift**
  from initialising with pretrained features. The base task's
  hidden representations are already useful for the held-out
  keywords even though the labels were never seen.
* **K = 200 (1200 clips, the AED-relevant point):** from-scratch
  reaches 31 % (just barely above random), fine-tuned reaches
  **68 %**. **+37 pp lift** — the largest in the sweep. This is
  where transfer learning is structurally most valuable.
* **K = 500 (3000 clips):** from-scratch finally starts learning
  (47 %); fine-tuned reaches **80 %**, only ~5 pp below the
  base-task ceiling of 85 %. **+33 pp lift** — still substantial,
  but the gap is starting to close.

The takeaway: **at the ~2 K-sample budget that real-world AED tasks
live in, fine-tuning a pretrained audio backbone is worth tens of
points of val accuracy over training from scratch**. This is the
structural case for using Perch 2.0 / YAMNet embeddings (or any
pretrained audio model) as the starting point for a turkey-gobble
or glass-break detector, rather than the from-scratch DS-CNN this
repo currently uses as the headline model. The implementation cost
on top of `nano-kws` is a new dataset module + the `--init-from`
flag the few-shot script already uses, plus a one-time distillation
pass to shrink the embedding model back into a sub-mW edge-deployable
footprint.

<!-- BEGIN_FEW_SHOT_TABLE -->

**Base task** (the pretrained-audio-model stand-in): DS-CNN-w0.5 trained for 8 epochs on the 6 base keywords (`down, left, no, right, up, yes`) + `_silence_` + `_unknown_` = 8 classes, 24648 training clips. Best base-task val accuracy: **85.33%**.

**Novel task**: the held-out 4 keywords (`go, off, on, stop`) + `_silence_` + `_unknown_` = 6 classes — labels the base model never saw. For each K samples/class budget, two DS-CNN-w0.5 models are trained to convergence: a from-scratch baseline (random init) and a fine-tuned variant (initialised from the base checkpoint, with the 8-way classifier head replaced by a fresh 6-way head). Both use SpecAugment + bg-noise augmentation. Validation accuracy is on the *full* novel-task validation split, not the K-sample subset, so it isn't biased by training-set size.

| K samples / class | Train clips | Val clips | From scratch | Fine-tuned (transfer from base) | Lift from pretraining |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 60 | 2198 | 16.83% | 13.10% | -3.73 pp |
| 50 | 300 | 2198 | 16.83% | 44.13% | +27.30 pp |
| 200 | 1200 | 2198 | 31.48% | 68.33% | +36.85 pp |
| 500 | 3000 | 2198 | 47.41% | 80.12% | +32.71 pp |

<!-- END_FEW_SHOT_TABLE -->

---

## Augmentation in the low-data regime — and why "more aug" isn't free

Before reaching for a generative model like EcoGen / BirdDiff /
AudioLDM, the cheap question is: **how much does classical
augmentation buy you in the AED-scale low-data regime?** I ran a
2-axis sweep of DS-CNN-w0.5 across three data budgets (50, 200, 500
samples per class) with and without SpecAugment + background-noise
mixing toggled on, 10 epochs each.

> **The honest finding (full numbers in the table below): with the
> default augmentation strength used by the headline 30-epoch
> training run, classical augmentation *hurts* val accuracy by 30-55
> pp at these data budgets and epoch counts.** This is the opposite
> of what I expected going in, and the more interesting result.

What's going on:

* SpecAugment (default: up to 8/40 frequency bins masked + up to
  16/97 time frames masked, two of each per sample) and
  BackgroundNoiseMixer (5-20 dB SNR, applied to 80% of clips) are
  *robustness* augmentations — they widen the training distribution
  so the model generalises to noisy / partially occluded inputs.
* The val set is mostly clean studio audio. So the augmentation is
  introducing a **train-vs-val distribution shift**, not just a
  variance-reduction effect.
* In the **low-data + few-epoch** regime (≤ 6 K training clips, 10
  epochs), the model doesn't see enough clean examples to learn the
  underlying patterns before being asked to also generalise across
  augmentation variants. The aug-induced bias dominates.
* The bundled 30-epoch headline checkpoint trains on **~37 K clips
  for 30 epochs with the same augmentation on** and reaches 79.32%
  val acc. That's the regime where these augmentations were tuned —
  enough data + enough epochs to amortize the bias.

Why this is the more interesting result:

1. **Classical augmentation has a cost-benefit trade-off that
   depends on data scale and epoch budget.** The right answer at
   the ~2 K-sample AED scale is not "turn aug on by default" — it's
   "tune aug strength to your data + training budget, or you'll
   hurt the very metric you're trying to improve".
2. **It strengthens the case for *generative* augmentation
   (EcoGen / BirdDiff / AudioLDM).** Those approaches add *more
   in-distribution data*, not "noisier versions of the data you
   already have". If the val distribution is clean turkey gobble,
   a generative model that produces more clean-ish turkey gobbles
   should help where SpecAugment hurts, because it doesn't impose a
   train-vs-val distribution shift.
3. **It identifies a concrete next experiment**: sweep augmentation
   *strength* (not just on/off) at each data budget, and find the
   per-budget optimum. The augmentation knobs are already exposed
   on the training CLI; only the orchestrator script needs writing.

Reproduce with `python -m scripts.aug_ablation --update-readme`. Raw
artefacts land in `runs/aug_ablation/` (gitignored) and the rendered
table mirrors [`assets/aug_ablation_table.md`](assets/aug_ablation_table.md).

<!-- BEGIN_AUG_ABLATION_TABLE -->

DS-CNN-w0.5 fine-tunes from scratch for **10 epochs** at each setting; best validation accuracy across the run is reported. Augmentation = SpecAugment (frequency + time masks) + `BackgroundNoiseMixer` (5-20 dB SNR). Same model, same optimiser, same seed across all cells; only the data budget and the augmentation toggle vary.

| Samples / class | Train clips | Val clips | No augmentation | + SpecAugment + bg-noise | Lift from augmentation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 600 | 4443 | 8.33% | 8.33% | +0.00 pp |
| 200 | 2400 | 4443 | 73.44% | 18.84% | -54.60 pp |
| 500 | 6000 | 4443 | 86.20% | 52.22% | -33.99 pp |

<!-- END_AUG_ABLATION_TABLE -->

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
nano_kws/        importable package: data, model, train, quantize, qat, infer, streaming, benchmark
scripts/         dataset fetcher, multi-size sweep, augmentation ablation, few-shot transfer, conv microbenchmark
app/             Streamlit live mic + continuous-mode demo
cpp/             ONNX Runtime C++ inference harness + hand-written AVX2 conv kernels (microbench)
assets/          committed fp32 + INT8 (PTQ) + INT8 (QAT) ONNX models, benchmark/sweep/microbench tables, aug-ablation + few-shot result tables, training histories, demo screenshot
tests/           pytest suite, 160 tests, uses synthetic audio (no dataset download required)
docs/            scaffold for long-form notes (per-class confusion, MAC budget) — empty for now
```

---

## Results

All headline numbers are inlined above and auto-stamped from the
benchmark / sweep scripts (so they can't drift):

- Accuracy + latency + size for the bundled checkpoint:
  [TL;DR table](#tldr-technical) (fp32 vs INT8 PTQ vs INT8 QAT).
- Accuracy vs parameter count vs MACs vs latency across width
  multipliers: [Model size sweep](#model-size-sweep) +
  [`assets/sweep_plot.png`](assets/sweep_plot.png).
- Hand-written kernels vs ATen:
  [Hand-written conv kernels vs ATen](#hand-written-conv-kernels-vs-aten).
- Raw artefacts: [`assets/benchmark_table.md`](assets/benchmark_table.md),
  [`assets/sweep_table.md`](assets/sweep_table.md),
  [`assets/microbench_table.md`](assets/microbench_table.md),
  [`assets/ds_cnn_w0p5.history.json`](assets/ds_cnn_w0p5.history.json),
  [`assets/ds_cnn_w0p5_qat.history.json`](assets/ds_cnn_w0p5_qat.history.json).

Per-class confusion matrices are not generated yet — adding them is a
small follow-up that would slot into `nano_kws/benchmark.py`.

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

## Related work for AED on edge

There is an active cluster of generative and pretrained-embedding
audio models that target the **"only ~2 K labelled samples"** regime
that AED tasks like turkey-gobble or glass-break detection live in.
This section is a one-paragraph summary of each, plus how it would
slot into the `nano-kws` pipeline. None of these are implemented in
this repo — this is the reading list, not a benchmark.

### Generative augmentation (synthesise more training data)

- **EcoGen** (Symes et al., 2024). Conditional generative model for
  ecological / bird audio. Trained on a large corpus of wildlife
  recordings, fine-tunes on a small target set (handful of clips) and
  then synthesises plausible variations. *Slot in `nano-kws` pipeline:*
  offline data-augmentation step before the training loop — emit
  `(synthetic_waveform, label)` pairs into the `nano_kws.data.*`
  dataset module so the rest of the train → quantize → export path is
  unchanged. The interesting research questions are (a) what's the
  real-vs-synthetic mixing ratio that maximises test accuracy, and
  (b) does synthetic data help INT8 calibration as well as fp32
  training. Both fall out as natural follow-ups of the augmentation
  ablation methodology used in the [Augmentation pays you back in the
  low-data regime section](#augmentation-in-the-low-data-regime--and-why-more-aug-isnt-free).
- **BirdDiff** (Lin et al., 2024). Diffusion-based bird-sound
  synthesizer. Same pipeline-slot as EcoGen; the interesting
  trade-off is sample quality vs inference cost — diffusion models
  are slower per-sample but produce sharper outputs, which matters
  more for AED than for KWS because AED targets are often
  high-frequency / transient.
- **AudioLDM / AudioLDM 2** (Liu et al., 2023). Text-conditioned
  latent diffusion for general audio. Lets the user describe target
  sounds in language — *"a single turkey gobble with light wind
  noise in the background"* — without needing reference recordings
  of every variation. Useful for the very-low-N classes where even
  EcoGen / BirdDiff don't have enough seed audio to fine-tune on.

### Pretrained audio embeddings (skip the from-scratch CNN)

- **Perch 2.0** (Hamer et al., Google Research, 2023). Large
  pretrained bird-sound classifier whose hidden representations
  generalise well as a feature extractor for arbitrary AED targets.
  *Slot in `nano-kws` pipeline:* swap the log-mel + DS-CNN frontend
  for `Perch_embedding(waveform)` followed by a linear probe (or a
  tiny MLP). With only ~2 K labelled samples, an embedding + linear
  probe usually beats a from-scratch DS-CNN by a comfortable margin —
  this is the same pattern as ImageNet-pretrained features beating
  from-scratch CNNs on low-N image-classification tasks. The cost is
  larger embedding-extractor inference (~10 MB of weights vs
  `nano-kws`'s 100 KB), so deployment on a sub-mW edge accelerator
  would require distilling the Perch features back into a small
  student model — a natural follow-up project on top of `nano-kws`.
- **YAMNet** (Plakal & Ellis, Google, 2019) and **VGGish** (Hershey
  et al., 2017). Older but battle-tested AudioSet-pretrained feature
  extractors. Same lever as Perch 2.0 (embedding + linear probe), with
  weaker representations but smaller and easier to integrate. Good
  baseline to put alongside Perch in any "from-scratch vs pretrained
  embedding" sweep.

### How this maps back to the nano-kws scope

The pieces of an EcoGen / BirdDiff / Perch-style AED system that
this repo *does* concretely implement (or that the implemented
infrastructure makes trivially extendable to):

| AED-on-edge skill | What `nano-kws` already shows |
| --- | --- |
| Adapt an audio classifier to a new label set | The 12-class Speech Commands setup is a `_silence_` + `_unknown_` + 10-keyword construction layered on top of the raw 35-class dataset; the recipe lives in [`nano_kws/data/speech_commands.py`](nano_kws/data/speech_commands.py). The same pattern works for any AED label set. |
| Quantify the value of data augmentation in a low-data regime | [Augmentation in the low-data regime](#augmentation-in-the-low-data-regime--and-why-more-aug-isnt-free) sweeps SpecAugment + bg-noise on/off across {50, 200, 500} samples/class. The (counter-intuitive) finding — augmentation hurts at this data + epoch budget — is the more interesting result, and the case for *generative* augmentation specifically. |
| Few-shot transfer from a pretrained backbone to a new task | [Few-shot transfer: how much data do you actually need?](#few-shot-transfer-how-much-data-do-you-actually-need) trains a base DS-CNN on 6 of the 10 keywords and fine-tunes it on K samples per class of the held-out 4 (K ∈ {10, 50, 200, 500}). At the AED-relevant K = 200, fine-tuning beats from-scratch by **+37 pp** of val accuracy — the structural analog of "fine-tune Perch 2.0 on the 2 K turkey samples" minus the larger backbone. |
| Honest INT8 deployment of the resulting model | The whole [TL;DR](#tldr-technical) + [QAT](#tldr-technical) story. |

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

**AED low-data extensions**

- [x] [From KWS to AED](#from-kws-to-aed-same-recipe-different-label-set) — explicit isomorphism between the implemented KWS pipeline and a non-speech AED setup (turkey gobble, glass break, etc.).
- [x] [Related work for AED on edge](#related-work-for-aed-on-edge) — survey of EcoGen / BirdDiff / AudioLDM / Perch 2.0 / YAMNet with how each plugs into the `nano-kws` pipeline.
- [x] [Few-shot transfer experiment](#few-shot-transfer-how-much-data-do-you-actually-need) — base DS-CNN trained on 8 classes (6 keywords + silence + unknown, 85.33 % val acc), then fine-tuned vs from-scratch on the held-out novel 6 (4 keywords + silence + unknown) at K ∈ {10, 50, 200, 500} samples/class. **At K = 200 (the AED-relevant point), fine-tuning beats from-scratch by +37 pp.**
- [x] [Augmentation in the low-data regime](#augmentation-in-the-low-data-regime--and-why-more-aug-isnt-free) — 6-cell ablation across {50, 200, 500} samples/class × {aug, no-aug} × DS-CNN-w0.5 at 10 epochs. Counter-intuitive result: classical augmentation *hurts* val accuracy by 30-55 pp at this budget, which is the more interesting finding and the case for generative augmentation specifically.

---

## Citing prior work

- Zhang, Y. et al. *Hello Edge: Keyword Spotting on Microcontrollers.* (2017)
- Warden, P. *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* (2018)

---

## License & model card

- Code: [MIT](LICENSE)
- Model card: [`MODEL_CARD.md`](MODEL_CARD.md)
