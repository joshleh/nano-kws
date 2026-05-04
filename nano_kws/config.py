"""Project-wide constants.

This module is the single source of truth for audio frontend parameters,
class labels, and on-disk paths. Anything that touches the audio pipeline
(training dataloader, inference helper, Streamlit demo, C++ harness via
the exported ONNX metadata) reads from here.

Changing a value here is a real model-affecting change — bump the model
version in `MODEL_CARD.md` if you do.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Audio frontend (must match between training and inference, bit-for-bit).
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000
"""Input sample rate, Hz. Speech Commands is natively 16 kHz."""

CLIP_DURATION_S: float = 1.0
"""All clips are pad/cropped to exactly this many seconds before feature
extraction."""

CLIP_SAMPLES: int = int(SAMPLE_RATE * CLIP_DURATION_S)
"""Samples per clip after pad/crop. = 16000."""

WIN_LENGTH_MS: float = 30.0
HOP_LENGTH_MS: float = 10.0

WIN_LENGTH: int = int(SAMPLE_RATE * WIN_LENGTH_MS / 1000)  # 480 samples
HOP_LENGTH: int = int(SAMPLE_RATE * HOP_LENGTH_MS / 1000)  # 160 samples

N_FFT: int = 512
"""Power-of-two FFT size >= WIN_LENGTH. 512 fits a 480-sample window."""

N_MELS: int = 40
"""Mel filterbank size. Matches the Hello Edge / DS-CNN baseline."""

F_MIN: float = 20.0
F_MAX: float = SAMPLE_RATE / 2  # Nyquist

# Number of mel frames per 1-second clip. With 16 kHz audio and a 10 ms hop,
# torchaudio's MelSpectrogram with center=True yields ceil(16000 / 160) + 1 = 101
# frames. We trim to a fixed N_FRAMES for a deterministic input shape.
N_FRAMES: int = 98

INPUT_SHAPE: tuple[int, int, int] = (1, N_MELS, N_FRAMES)
"""(channels, mel_bins, frames) tensor layout the model consumes."""

# ---------------------------------------------------------------------------
# Class labels (12-class Speech Commands setup).
# ---------------------------------------------------------------------------

KEYWORDS: tuple[str, ...] = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)

SILENCE_LABEL: str = "_silence_"
UNKNOWN_LABEL: str = "_unknown_"

LABELS: tuple[str, ...] = (*KEYWORDS, SILENCE_LABEL, UNKNOWN_LABEL)
NUM_CLASSES: int = len(LABELS)  # 12

LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
INDEX_TO_LABEL: dict[int, str] = dict(enumerate(LABELS))

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ASSETS_DIR: Path = REPO_ROOT / "assets"
DOCS_DIR: Path = REPO_ROOT / "docs"

# Dataset cache lives outside the repo so it survives `git clean -fdx` and
# isn't accidentally committed. Override with NANO_KWS_DATA_DIR.
DATA_DIR: Path = Path(os.environ.get("NANO_KWS_DATA_DIR", Path.home() / ".cache" / "nano_kws"))

# Default committed asset paths used by Makefile targets and the demo app.
DEFAULT_INT8_ONNX: Path = ASSETS_DIR / "ds_cnn_small_int8.onnx"
DEFAULT_FP32_ONNX: Path = ASSETS_DIR / "ds_cnn_small_fp32.onnx"
DEFAULT_LABEL_MAP: Path = ASSETS_DIR / "label_map.json"

__all__ = [
    "ASSETS_DIR",
    "CLIP_DURATION_S",
    "CLIP_SAMPLES",
    "DATA_DIR",
    "DEFAULT_FP32_ONNX",
    "DEFAULT_INT8_ONNX",
    "DEFAULT_LABEL_MAP",
    "DOCS_DIR",
    "F_MAX",
    "F_MIN",
    "HOP_LENGTH",
    "HOP_LENGTH_MS",
    "INDEX_TO_LABEL",
    "INPUT_SHAPE",
    "KEYWORDS",
    "LABELS",
    "LABEL_TO_INDEX",
    "NUM_CLASSES",
    "N_FFT",
    "N_FRAMES",
    "N_MELS",
    "REPO_ROOT",
    "SAMPLE_RATE",
    "SILENCE_LABEL",
    "UNKNOWN_LABEL",
    "WIN_LENGTH",
    "WIN_LENGTH_MS",
]
