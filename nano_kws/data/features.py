"""Log-mel spectrogram featurizer.

This is the **single source of truth** for the audio frontend. Both the
training dataloader and the inference helper (`nano_kws.infer`) call into
this module so that the features the model sees at inference time match
the features it was trained on, bit-for-bit.

Implemented in Phase 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch


def waveform_to_logmel(waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert a 1-D waveform at SAMPLE_RATE into a log-mel spectrogram.

    Parameters
    ----------
    waveform
        Shape ``(samples,)`` or ``(1, samples)``, dtype float32 in ``[-1, 1]``.
        Length must equal ``CLIP_SAMPLES`` after pad/crop performed by the
        caller; this function does not pad.

    Returns
    -------
    torch.Tensor
        Shape ``INPUT_SHAPE`` = ``(1, N_MELS, N_FRAMES)``, dtype float32.
        Values are natural-log mel energies with a numerical floor.

    Notes
    -----
    Implemented in Phase 1 against ``torchaudio.transforms.MelSpectrogram``
    with the parameters from :mod:`nano_kws.config`.
    """
    raise NotImplementedError("Phase 1: implement log-mel featurizer.")


def pad_or_crop(waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Pad with zeros or center-crop a waveform to exactly ``CLIP_SAMPLES``.

    Implemented in Phase 1.
    """
    raise NotImplementedError("Phase 1: implement pad_or_crop.")
