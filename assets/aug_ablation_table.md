DS-CNN-w0.5 fine-tunes from scratch for **10 epochs** at each setting; best validation accuracy across the run is reported. Augmentation = SpecAugment (frequency + time masks) + `BackgroundNoiseMixer` (5-20 dB SNR). Same model, same optimiser, same seed across all cells; only the data budget and the augmentation toggle vary.

| Samples / class | Train clips | Val clips | No augmentation | + SpecAugment + bg-noise | Lift from augmentation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 600 | 4443 | 8.33% | 8.33% | +0.00 pp |
| 200 | 2400 | 4443 | 73.44% | 18.84% | -54.60 pp |
| 500 | 6000 | 4443 | 86.20% | 52.22% | -33.99 pp |
