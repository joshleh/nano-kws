"""Google Speech Commands v0.02 dataset wrapper for the 12-class KWS setup.

Wraps :class:`torchaudio.datasets.SPEECHCOMMANDS` and remaps its native
35-class label space into the 12 classes from :mod:`nano_kws.config`:

* 10 target keywords (``yes``, ``no``, ..., ``go``).
* ``_unknown_`` — every other word in the dataset, randomly subsampled so
  it doesn't dominate the loss.
* ``_silence_`` — synthesised by sampling random 1-second windows from
  the bundled ``_background_noise_`` clips, scaled to a low gain.

Each :meth:`__getitem__` call returns ``(waveform, label_index)`` where
``waveform`` is a ``(CLIP_SAMPLES,)`` ``float32`` tensor in roughly
``[-1, 1]`` (already pad/cropped) and ``label_index`` is in
``range(NUM_CLASSES)``. Featurization happens downstream so the
training loop can amortize the mel filterbank cost across the batch on
GPU.

The CLI ``python -m nano_kws.data.speech_commands --sanity`` builds one
batch and prints its featurized shape — this is the Phase 1 checkpoint.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS

from nano_kws import config
from nano_kws.data.features import LogMelSpectrogram, pad_or_crop

logger = logging.getLogger(__name__)

_BACKGROUND_FOLDER: str = "_background_noise_"
_KEYWORD_SET: frozenset[str] = frozenset(config.KEYWORDS)

# ─── Interview note: the 12-class Speech Commands setup ─────────────────────
# Speech Commands v0.02 ships with 35 word labels. The standard KWS benchmark
# (Hello Edge, every paper since) collapses that to 12 classes:
#   * 10 keywords (the Speech Commands "core" set: yes/no/up/down/left/right/
#     on/off/stop/go).
#   * `_unknown_` — every other word, randomly subsampled. Without
#     subsampling, "unknown" would be ~25x larger than any single keyword and
#     dominate the cross-entropy loss → model collapses to predicting unknown.
#   * `_silence_` — synthesized by sampling 1-second windows from the
#     dataset's bundled `_background_noise_` clips at low gain. The dataset
#     ships these specifically so models learn "no keyword present" as a
#     real class, not just "low energy".
# This 12-class framing is what the model card, every benchmark, and every
# published comparison number assumes. Deviating from it (e.g. 20 classes,
# different unknown ratio) makes the headline accuracy incomparable to
# literature.
# ────────────────────────────────────────────────────────────────────────────


def _classify_label(label: str) -> int:
    """Map a raw Speech Commands label to a 12-class label index."""
    if label in _KEYWORD_SET:
        return config.LABEL_TO_INDEX[label]
    return config.LABEL_TO_INDEX[config.UNKNOWN_LABEL]


def _load_wav_mono(path: str | Path) -> torch.Tensor:
    """Load a WAV file as a 1-D float32 mono torch tensor at SAMPLE_RATE.

    Uses ``soundfile`` directly rather than ``torchaudio.load``: torchaudio
    2.9+ delegates loading to ``torchcodec`` (which pulls ffmpeg as a heavy
    dependency), and Speech Commands clips are simple 16 kHz 16-bit PCM
    WAVs that ``soundfile`` reads natively. Keeping the runtime dependency
    surface small matters more for an edge-AI portfolio piece than the
    convenience of torchaudio's loader.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != config.SAMPLE_RATE:
        raise RuntimeError(
            f"{path} has sample rate {sr}; expected {config.SAMPLE_RATE}. "
            "Speech Commands v0.02 is uniformly 16 kHz; resampling is not implemented "
            "(it would also break the bit-exact match between training and inference)."
        )
    if data.ndim == 2:
        data = data.mean(axis=1)  # mono
    return torch.from_numpy(np.ascontiguousarray(data))


def _load_background_noise(root: Path) -> list[torch.Tensor]:
    """Load all WAVs in ``<root>/_background_noise_/`` as 1-D float32 tensors."""
    bg_dir = root / _BACKGROUND_FOLDER
    if not bg_dir.is_dir():
        raise FileNotFoundError(
            f"Expected background noise directory at {bg_dir}. "
            "Has the dataset been downloaded? Try `make download-data`."
        )
    clips = [_load_wav_mono(p) for p in sorted(bg_dir.glob("*.wav"))]
    if not clips:
        raise FileNotFoundError(f"No background noise WAVs found in {bg_dir}.")
    return clips


class SpeechCommandsKWS(Dataset):
    """12-class Speech Commands dataset for keyword spotting.

    Parameters
    ----------
    root
        Directory that contains (or will contain) the
        ``SpeechCommands/`` folder. Defaults to :data:`config.DATA_DIR`.
    subset
        One of ``"training"``, ``"validation"``, ``"testing"``.
    download
        If ``True``, fetch the dataset (~2.4 GB) when missing.
    unknown_per_class_ratio
        Number of ``_unknown_`` samples expressed as a multiple of the
        per-keyword average. ``1.0`` keeps the unknown class roughly the
        same size as a single keyword class.
    silence_per_class_ratio
        Same convention as ``unknown_per_class_ratio`` but for synthesised
        silence.
    silence_max_gain
        Maximum amplitude scale applied to background noise when building
        a silence sample. ``0.1`` matches the standard recipe.
    seed
        RNG seed for the deterministic subsampling and silence-window
        selection. Two ``SpeechCommandsKWS`` instances with the same args
        and seed produce identical indices.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        subset: str = "training",
        *,
        download: bool = False,
        unknown_per_class_ratio: float = 1.0,
        silence_per_class_ratio: float = 1.0,
        silence_max_gain: float = 0.1,
        seed: int = 0,
    ) -> None:
        if subset not in {"training", "validation", "testing"}:
            raise ValueError(f"subset must be training/validation/testing, got {subset!r}.")

        self.root = Path(root) if root is not None else config.DATA_DIR
        self.root.mkdir(parents=True, exist_ok=True)

        self._base = SPEECHCOMMANDS(
            root=str(self.root),
            subset=subset,
            download=download,
        )
        archive_root = Path(self._base._path)

        # ------------------------------------------------------------------
        # Walk the underlying dataset and bucket each clip by class.
        # We index via get_metadata() to avoid loading every waveform here.
        # ------------------------------------------------------------------
        keyword_indices: list[int] = []
        unknown_indices: list[int] = []
        for i in range(len(self._base)):
            label = self._base.get_metadata(i)[2]
            if label in _KEYWORD_SET:
                keyword_indices.append(i)
            else:
                unknown_indices.append(i)

        per_class_avg = max(1, len(keyword_indices) // len(config.KEYWORDS))
        n_unknown = min(len(unknown_indices), int(per_class_avg * unknown_per_class_ratio))
        n_silence = int(per_class_avg * silence_per_class_ratio)

        rng = random.Random(seed)
        unknown_sampled = rng.sample(unknown_indices, n_unknown) if n_unknown else []

        # ------------------------------------------------------------------
        # Pre-build the silence-sample plan: deterministic per-index
        # (bg_file_idx, start_offset, gain) tuples drawn from the seeded RNG.
        # Train-time augmentation can layer additional jitter on top.
        # ------------------------------------------------------------------
        self._bg_clips = _load_background_noise(archive_root)
        bg_lengths = [len(c) for c in self._bg_clips]
        silence_plan: list[tuple[int, int, float]] = []
        for _ in range(n_silence):
            bg_idx = rng.randrange(len(self._bg_clips))
            max_start = max(0, bg_lengths[bg_idx] - config.CLIP_SAMPLES)
            start = rng.randrange(max_start + 1) if max_start > 0 else 0
            gain = rng.uniform(0.0, silence_max_gain)
            silence_plan.append((bg_idx, start, gain))

        # ------------------------------------------------------------------
        # Final flat index. Each entry is ("clip", base_idx) or
        # ("silence", silence_plan_idx).
        # ------------------------------------------------------------------
        self._index: list[tuple[str, int]] = []
        for i in keyword_indices:
            self._index.append(("clip", i))
        for i in unknown_sampled:
            self._index.append(("clip", i))
        for i in range(len(silence_plan)):
            self._index.append(("silence", i))
        self._silence_plan = silence_plan

        # Stable shuffle of the final index, using a fresh RNG so that
        # downstream DataLoader shuffling is the only source of order
        # variability across epochs.
        rng.shuffle(self._index)

        self._counts = Counter(self._label_for(kind, payload) for kind, payload in self._index)
        logger.info(
            "SpeechCommandsKWS(%s): %d clips total | per-class counts: %s",
            subset,
            len(self._index),
            {config.INDEX_TO_LABEL[k]: v for k, v in sorted(self._counts.items())},
        )

    # ----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        kind, payload = self._index[i]
        if kind == "clip":
            # Bypass self._base[payload] (which would call torchaudio.load ->
            # torchcodec) and read the WAV directly with soundfile.
            file_path, _sr, label, *_ = self._base.get_metadata(payload)
            full_path = Path(self._base._archive) / file_path
            wav = pad_or_crop(_load_wav_mono(full_path))
            return wav, _classify_label(label)

        # silence
        bg_idx, start, gain = self._silence_plan[payload]
        bg = self._bg_clips[bg_idx]
        end = start + config.CLIP_SAMPLES
        wav = pad_or_crop(bg[start:]) if end > len(bg) else bg[start:end].clone()
        wav = wav * gain
        return wav, config.LABEL_TO_INDEX[config.SILENCE_LABEL]

    # ----------------------------------------------------------------------

    def _label_for(self, kind: str, payload: int) -> int:
        if kind == "silence":
            return config.LABEL_TO_INDEX[config.SILENCE_LABEL]
        return _classify_label(self._base.get_metadata(payload)[2])

    @property
    def class_counts(self) -> dict[int, int]:
        """``{label_index: n_samples}`` for the materialised dataset."""
        return dict(self._counts)


# ---------------------------------------------------------------------------
# CLI sanity check.
# ---------------------------------------------------------------------------


def _sanity_check(args: argparse.Namespace) -> None:
    """Build one batch and print its featurized shape.

    This is the Phase 1 checkpoint: it confirms the data pipeline is
    wired end-to-end and produces tensors of the shape the model expects.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    dataset = SpeechCommandsKWS(
        root=args.root,
        subset=args.subset,
        download=args.download,
    )
    print(f"Dataset size: {len(dataset)} clips ({args.subset} split)")
    print("Per-class counts:")
    for idx, count in sorted(dataset.class_counts.items()):
        print(f"  {idx:2d} {config.INDEX_TO_LABEL[idx]:>10s} {count:>6d}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    waveforms, labels = next(iter(loader))
    print(f"\nWaveform batch:  shape={tuple(waveforms.shape)}, dtype={waveforms.dtype}")
    print(f"Label batch:     shape={tuple(labels.shape)}, dtype={labels.dtype}")

    featurizer = LogMelSpectrogram().eval()
    with torch.no_grad():
        features = featurizer(waveforms)
    print(f"Log-mel batch:   shape={tuple(features.shape)}, dtype={features.dtype}")
    print(f"Expected shape:  {(args.batch_size, *config.INPUT_SHAPE)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Build one batch from the dataset and print its featurized shape.",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("NANO_KWS_DATA_DIR"),
        help="Dataset root (defaults to NANO_KWS_DATA_DIR / config.DATA_DIR).",
    )
    parser.add_argument(
        "--subset",
        choices=("training", "validation", "testing"),
        default="validation",
        help="Which split to load (validation is fastest to walk).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset (~2.4 GB) if it isn't already cached.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    if not args.sanity:
        parser.error("Pass --sanity to run the Phase 1 checkpoint.")
    _sanity_check(args)


if __name__ == "__main__":
    main()
