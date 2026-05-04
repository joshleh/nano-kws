"""Training-time audio augmentation.

Two transforms, applied at different points of the pipeline:

* :class:`BackgroundNoiseMixer` — operates on raw waveforms, mixes in a
  random window of background noise at a random SNR. Use as a transform
  before featurization.
* :class:`SpecAugment` — operates on log-mel spectrograms, masks random
  contiguous bands of frequency and time. Use after featurization,
  before the model.

Both are :class:`torch.nn.Module` so they compose cleanly inside a
training loop and can be conditionally disabled by switching the module
to eval mode.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torchaudio.transforms as T

from nano_kws import config


class BackgroundNoiseMixer(nn.Module):
    """Mix a random window of background noise into the input waveform.

    Parameters
    ----------
    background_clips
        List of 1-D ``float32`` tensors at ``config.SAMPLE_RATE``. Pass
        the same clips loaded by :class:`SpeechCommandsKWS`.
    p
        Probability of applying the mix on any given call.
    snr_db_range
        Inclusive ``(low, high)`` range for the random SNR in decibels.
        Higher = quieter noise relative to signal.
    """

    def __init__(
        self,
        background_clips: list[torch.Tensor],
        *,
        p: float = 0.8,
        snr_db_range: tuple[float, float] = (5.0, 20.0),
    ) -> None:
        super().__init__()
        if not background_clips:
            raise ValueError("BackgroundNoiseMixer needs at least one background clip.")
        self._bg = background_clips
        self.p = p
        self.snr_db_range = snr_db_range

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return waveform

        bg = random.choice(self._bg)
        n = waveform.shape[-1]
        if len(bg) < n:
            return waveform
        start = random.randrange(0, len(bg) - n + 1)
        noise = bg[start : start + n].to(device=waveform.device, dtype=waveform.dtype)

        # Power-based SNR mixing: scale noise so that
        # 10 * log10(P_signal / P_noise) == snr_db.
        snr_db = random.uniform(*self.snr_db_range)
        sig_power = waveform.pow(2).mean().clamp(min=1e-10)
        noise_power = noise.pow(2).mean().clamp(min=1e-10)
        scale = (sig_power / (noise_power * (10 ** (snr_db / 10)))).sqrt()
        return waveform + scale * noise


class SpecAugment(nn.Module):
    """SpecAugment-style frequency + time masking on log-mel spectrograms.

    Operates on tensors of shape ``(..., N_MELS, N_FRAMES)``.
    """

    def __init__(
        self,
        freq_mask_param: int = 8,
        time_mask_param: int = 16,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ) -> None:
        super().__init__()
        if not 0 <= freq_mask_param <= config.N_MELS:
            raise ValueError(f"freq_mask_param must be in [0, {config.N_MELS}].")
        if not 0 <= time_mask_param <= config.N_FRAMES:
            raise ValueError(f"time_mask_param must be in [0, {config.N_FRAMES}].")
        self.freq_masks = nn.ModuleList(
            T.FrequencyMasking(freq_mask_param=freq_mask_param) for _ in range(n_freq_masks)
        )
        self.time_masks = nn.ModuleList(
            T.TimeMasking(time_mask_param=time_mask_param) for _ in range(n_time_masks)
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spectrogram
        for m in self.freq_masks:
            spectrogram = m(spectrogram)
        for m in self.time_masks:
            spectrogram = m(spectrogram)
        return spectrogram


__all__ = ["BackgroundNoiseMixer", "SpecAugment"]
