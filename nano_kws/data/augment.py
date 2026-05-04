"""Training-time audio augmentation.

Phase 1 implements:

* SpecAugment-style time and frequency masking on the log-mel spectrogram.
* Background-noise mixing at random SNR using the ``_background_noise_``
  clips bundled with Speech Commands.
* Random time shift (+/- 100 ms) of the raw waveform before featurization.

These are train-only; evaluation and inference use the raw featurizer.
"""

from __future__ import annotations


class SpecAugment:
    """SpecAugment time + frequency masking. Implemented in Phase 1."""

    def __init__(
        self,
        freq_mask_param: int = 8,
        time_mask_param: int = 16,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ) -> None:
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        raise NotImplementedError("Phase 1: implement SpecAugment.")
