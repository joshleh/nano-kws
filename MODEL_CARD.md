# Model card — `ds_cnn_small` (nano-kws)

## Overview

| Field | Value |
| --- | --- |
| Model family | DS-CNN (depthwise-separable convolutional network) |
| Reference | Zhang, Y. et al. *Hello Edge: Keyword Spotting on Microcontrollers.* (2017) |
| Task | 12-class keyword spotting on 1-second 16 kHz audio clips |
| Classes | `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `_silence_`, `_unknown_` |
| Width multiplier (default) | 0.5 |
| Parameters | 62,060 |
| MACs / inference | 44,134,720 |
| Training data | Google Speech Commands v0.02 (Warden, 2018) |
| Featurizer | Log-mel: 16 kHz, 40 mel bins, 30 ms window, 10 ms hop |
| Framework | PyTorch 2.x → ONNX → ONNX Runtime |
| Quantization (PTQ) | Static post-training (QDQ), INT8 weights + activations, per-channel weights, MinMax calibration |
| Quantization (QAT) | Custom STE fake-quant + per-output-channel symmetric INT8 weight quantization + per-tensor symmetric INT8 activation observers; 5-epoch fine-tune from fp32 checkpoint at LR 1e-4, AdamW, cosine schedule, observers frozen after epoch 2 |
| Calibration | 50 batches × 16 samples drawn from the training split (PTQ and QAT-then-PTQ paths both use this) |
| License | MIT |

## Intended use

- Demonstrate the train → quantize → export → benchmark pipeline for an
  edge-deployable audio classifier.
- Reference implementation of the train → export → quantize → benchmark
  loop for edge-deployed audio classifiers.
- Reference recipe for adapting the same pipeline to non-speech
  **Acoustic Event Detection** (AED) — turkey gobble, glass break,
  baby cry, smoke alarm, etc. The model, frontend, quantization, and
  C++ inference path are dataset-agnostic; the work to retarget is
  contained in the dataset module + label set. See the
  [README "From KWS to AED" section](README.md#from-kws-to-aed-same-recipe-different-label-set).

## Out-of-scope use

- **Not** a production wake-word system. No streaming front-end, no posterior
  smoothing, no false-accept rate tuning, no on-device deployment to a real
  NPU in this repo.
- **Not** robust to far-field audio, heavy reverberation, codec artefacts,
  non-English keywords, or domains beyond the Speech Commands distribution.
- **Not** validated for accessibility-critical or safety-critical use.

## Metrics

Top-1 accuracy is reported on the official Speech Commands v0.02 test split,
12-class setup. Latency is single-inference wall-clock on a host CPU,
averaged over 1000 runs after 100 warmup runs.

| Variant | Top-1 acc | Mean latency | Model size |
| --- | ---: | ---: | ---: |
| fp32 (ONNX Runtime, CPU) | 79.32 % | 0.586 ms | 242.6 KB |
| INT8 PTQ (ONNX Runtime, CPU) | 72.20 % | 0.413 ms | 93.4 KB |
| INT8 QAT (ONNX Runtime, CPU) | 90.67 % | 0.436 ms | 93.4 KB |

The accuracy-vs-MACs sweep over widths {0.25, 0.5, 1.0} is in
[`assets/sweep_table.md`](assets/sweep_table.md) and rendered as
[`assets/sweep_plot.png`](assets/sweep_plot.png). The PTQ row's `-7.12
pp` accuracy drop at width 0.5 is what motivated the QAT stretch
deliverable; the QAT row reflects a **5-epoch fine-tune from the same
30-epoch fp32 checkpoint with augmentation disabled**, fake-quant
active in the forward pass, then standard PTQ on the resulting
weights. Two effects compound to produce the +18.47 pp gain over
PTQ-only:

1. The quantization-aware fine-tune trains the weights to be robust to
   INT8 rounding (the textbook QAT effect).
2. Disabling background-noise mixing and SpecAugment during the
   fine-tune lets the model adapt to the cleaner test distribution
   (Speech Commands test is mostly studio-clean).

A fairer "QAT in isolation" comparison would re-train the fp32 baseline
for the same 5 extra epochs without augmentation; that ablation is
left as next-session work. For the engineering question this repo is
meant to answer — "can you ship an INT8 KWS model from PyTorch to
something an edge runtime ingests?" — what matters is that the bundled
INT8 checkpoint reaches 90.67 % top-1 at 93.4 KB.

## Training

- Optimizer: AdamW, weight decay 1e-4
- Schedule: cosine, ~30 epochs
- Batch size: 256
- Augmentation: SpecAugment (time + frequency masking) + background-noise mixin
- Hardware: single CPU or single consumer GPU is sufficient
- Reproducibility: fixed seed, deterministic dataloader where possible

## Limitations and ethical considerations

- **Dataset bias.** Speech Commands v2 was crowdsourced and skews toward
  North American English speakers in quiet rooms. The model will perform
  worse on accented speech, low-SNR conditions, and child speakers, and
  this is not characterised here.
- **Closed vocabulary.** The 12-class setup collapses everything outside the
  10 target keywords into `_unknown_`. The model is not designed to handle
  open-vocabulary speech.
- **Quantization.** INT8 PTQ can have non-uniform per-class accuracy impact;
  per-class confusion matrices are not currently generated and are
  flagged as a follow-up in the [README Results section](README.md#results).

## Citation

If you reference this work:

```
@misc{lee2026nanokws,
  author = {Joshua Lee},
  title  = {nano-kws: Edge-deployable keyword spotter},
  year   = {2026},
  url    = {https://github.com/joshleh/nano-kws}
}
```
