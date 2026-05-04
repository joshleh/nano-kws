# Model card — `ds_cnn_small` (nano-kws)

> _This card is a template. Concrete numbers are populated after Phase 4
> (benchmark) lands and the bundled INT8 asset is committed._

## Overview

| Field | Value |
| --- | --- |
| Model family | DS-CNN (depthwise-separable convolutional network) |
| Reference | Zhang, Y. et al. *Hello Edge: Keyword Spotting on Microcontrollers.* (2017) |
| Task | 12-class keyword spotting on 1-second 16 kHz audio clips |
| Classes | `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `_silence_`, `_unknown_` |
| Width multiplier (default) | 0.5 |
| Parameters | _TBD_ |
| MACs / inference | _TBD_ |
| Training data | Google Speech Commands v0.02 (Warden, 2018) |
| Featurizer | Log-mel: 16 kHz, 40 mel bins, 30 ms window, 10 ms hop |
| Framework | PyTorch 2.x → ONNX → ONNX Runtime |
| Quantization | Static post-training, INT8 weights and activations (FX graph mode) |
| Calibration | _TBD_ batches from the training set |
| License | MIT |

## Intended use

- Demonstrate the train → quantize → export → benchmark pipeline for an
  edge-deployable audio classifier.
- Reference implementation for portfolio / interview discussion of edge ML
  systems work.

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

| Variant | Top-1 acc | Mean latency | p95 latency | Model size |
| --- | --- | --- | --- | --- |
| fp32 (PyTorch)         | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| fp32 (ONNX Runtime)    | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| INT8 (ONNX Runtime)    | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Per-class confusion and accuracy-vs-MACs sweep are in
[`docs/benchmark.md`](docs/benchmark.md).

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
  see the per-class confusion matrix in `docs/benchmark.md` after Phase 4.

## Citation

If you reference this work:

```
@misc{lee2026nanokws,
  author = {Joshua Lee},
  title  = {nano-kws: Edge-deployable keyword spotter},
  year   = {2026},
  url    = {https://github.com/joshualee/nano-kws}
}
```
