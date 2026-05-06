**Base task** (the pretrained-audio-model stand-in): DS-CNN-w0.5 trained for 8 epochs on the 6 base keywords (`down, left, no, right, up, yes`) + `_silence_` + `_unknown_` = 8 classes, 24648 training clips. Best base-task val accuracy: **85.33%**.

**Novel task**: the held-out 4 keywords (`go, off, on, stop`) + `_silence_` + `_unknown_` = 6 classes — labels the base model never saw. For each K samples/class budget, two DS-CNN-w0.5 models are trained to convergence: a from-scratch baseline (random init) and a fine-tuned variant (initialised from the base checkpoint, with the 8-way classifier head replaced by a fresh 6-way head). Both use SpecAugment + bg-noise augmentation. Validation accuracy is on the *full* novel-task validation split, not the K-sample subset, so it isn't biased by training-set size.

| K samples / class | Train clips | Val clips | From scratch | Fine-tuned (transfer from base) | Lift from pretraining |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 60 | 2198 | 16.83% | 13.10% | -3.73 pp |
| 50 | 300 | 2198 | 16.83% | 44.13% | +27.30 pp |
| 200 | 1200 | 2198 | 31.48% | 68.33% | +36.85 pp |
| 500 | 3000 | 2198 | 47.41% | 80.12% | +32.71 pp |
