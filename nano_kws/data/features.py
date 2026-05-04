"""Log-mel spectrogram featurizer.

This is the **single source of truth** for the audio frontend. Both the
training dataloader and the inference helper (:mod:`nano_kws.infer`) call
into this module so that the features the model sees at inference time
match the features it was trained on, bit-for-bit.

The implementation is a thin wrapper over
:class:`torchaudio.transforms.MelSpectrogram` configured to produce the
exact ``(1, N_MELS, N_FRAMES)`` tensor that the DS-CNN consumes.

Frame count derivation
----------------------
With ``center=False``, the number of frames produced for a length-``T``
input is ``floor((T - n_fft) / hop_length) + 1`` (note: torch's STFT
frames against ``n_fft``, not ``win_length``, even when the two differ).
Substituting the constants from :mod:`nano_kws.config`::

    floor((16000 - 512) / 160) + 1 = floor(15488 / 160) + 1 = 96 + 1 = 97

so the output is exactly ``N_FRAMES`` frames with no trim/pad needed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

from nano_kws import config

LOG_EPS: float = 1e-6
"""Floor added before ``log`` to keep silence frames finite."""


class LogMelSpectrogram(nn.Module):
    """Compute log-mel spectrograms with the project's audio-frontend config.

    The output of :meth:`forward` is the canonical model input:

    * ``(1, N_MELS, N_FRAMES)`` for a single waveform of ``CLIP_SAMPLES``
      samples (input shape ``(CLIP_SAMPLES,)`` or ``(1, CLIP_SAMPLES)``).
    * ``(B, 1, N_MELS, N_FRAMES)`` for a batched input of shape
      ``(B, CLIP_SAMPLES)`` or ``(B, 1, CLIP_SAMPLES)``.

    Subclassing :class:`torch.nn.Module` (rather than exposing a free
    function) lets callers move the featurizer to a GPU and amortize the
    cost of building the mel filterbank across many batches.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            win_length=config.WIN_LENGTH,
            hop_length=config.HOP_LENGTH,
            f_min=config.F_MIN,
            f_max=config.F_MAX,
            n_mels=config.N_MELS,
            power=2.0,
            center=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (samples,) -> (1, samples)
            squeeze_batch = True
        elif waveform.dim() == 2 and waveform.shape[0] == 1:
            squeeze_batch = True
        else:
            squeeze_batch = False

        if waveform.dim() == 3:
            # (B, 1, samples) -> (B, samples) for MelSpectrogram.
            waveform = waveform.squeeze(1)

        mel = self.mel(waveform)
        logmel = torch.log(mel + LOG_EPS)

        if logmel.shape[-1] != config.N_FRAMES:
            raise RuntimeError(
                f"LogMelSpectrogram produced {logmel.shape[-1]} frames, "
                f"expected {config.N_FRAMES}. "
                f"Input length was {waveform.shape[-1]} samples; expected "
                f"{config.CLIP_SAMPLES}. Run `pad_or_crop` first."
            )

        # Add the channel dim that the DS-CNN expects.
        logmel = logmel.unsqueeze(-3)  # (..., N_MELS, N_FRAMES) -> (..., 1, N_MELS, N_FRAMES)

        if squeeze_batch and logmel.dim() == 4:
            logmel = logmel.squeeze(0)
        return logmel


# Module-level singleton so the simple `waveform_to_logmel(...)` call path
# doesn't pay for filterbank construction on every invocation.
_DEFAULT_FEATURIZER: LogMelSpectrogram | None = None


def _featurizer() -> LogMelSpectrogram:
    global _DEFAULT_FEATURIZER
    if _DEFAULT_FEATURIZER is None:
        _DEFAULT_FEATURIZER = LogMelSpectrogram()
        _DEFAULT_FEATURIZER.eval()
    return _DEFAULT_FEATURIZER


def waveform_to_logmel(waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert a waveform into a log-mel spectrogram.

    Parameters
    ----------
    waveform
        Shape ``(samples,)``, ``(1, samples)``, ``(B, samples)``, or
        ``(B, 1, samples)``. Length along the last axis must equal
        ``config.CLIP_SAMPLES``; pad/crop with :func:`pad_or_crop` first
        if not.

    Returns
    -------
    torch.Tensor
        ``config.INPUT_SHAPE`` for single inputs, or ``(B, 1, N_MELS,
        N_FRAMES)`` for batched inputs. Always ``torch.float32``.
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    waveform = waveform.to(dtype=torch.float32)

    with torch.no_grad():
        return _featurizer()(waveform)


def pad_or_crop(waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Pad with zeros or center-crop a waveform to exactly ``CLIP_SAMPLES``.

    Operates along the last dimension. Accepts numpy or torch input;
    always returns a ``torch.float32`` tensor with the same leading dims.
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    waveform = waveform.to(dtype=torch.float32)

    n = waveform.shape[-1]
    target = config.CLIP_SAMPLES

    if n == target:
        return waveform
    if n < target:
        pad = target - n
        # symmetric-ish zero pad: half before, the remainder after.
        left = pad // 2
        right = pad - left
        return torch.nn.functional.pad(waveform, (left, right), mode="constant", value=0.0)
    # n > target: center crop.
    start = (n - target) // 2
    return waveform[..., start : start + target]


__all__ = [
    "LOG_EPS",
    "LogMelSpectrogram",
    "pad_or_crop",
    "waveform_to_logmel",
]
