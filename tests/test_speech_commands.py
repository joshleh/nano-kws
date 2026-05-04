"""Phase 1 — Speech Commands wrapper.

Most assertions here require the ~2.4 GB dataset to be cached locally.
CI does not download it; these tests auto-skip when the cache is absent.

Run locally after::

    make download-data
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from nano_kws import config
from nano_kws.data.features import LogMelSpectrogram

_ARCHIVE_ROOT: Path = config.DATA_DIR / "SpeechCommands" / "speech_commands_v0.02"

requires_dataset = pytest.mark.skipif(
    not _ARCHIVE_ROOT.is_dir(),
    reason=f"Speech Commands cache not found at {_ARCHIVE_ROOT}; run `make download-data`.",
)


@requires_dataset
def test_validation_split_loads_and_yields_correct_shapes() -> None:
    from nano_kws.data.speech_commands import SpeechCommandsKWS

    ds = SpeechCommandsKWS(subset="validation", silence_per_class_ratio=0.5)
    assert len(ds) > 0

    waveform, label = ds[0]
    assert waveform.shape == (config.CLIP_SAMPLES,)
    assert waveform.dtype == torch.float32
    assert 0 <= int(label) < config.NUM_CLASSES


@requires_dataset
def test_validation_split_includes_all_12_classes() -> None:
    from nano_kws.data.speech_commands import SpeechCommandsKWS

    ds = SpeechCommandsKWS(subset="validation", silence_per_class_ratio=0.5)
    assert set(ds.class_counts.keys()) == set(range(config.NUM_CLASSES))


@requires_dataset
def test_dataloader_batch_featurizes_to_expected_shape() -> None:
    from nano_kws.data.speech_commands import SpeechCommandsKWS

    ds = SpeechCommandsKWS(subset="validation", silence_per_class_ratio=0.5)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    waveforms, labels = next(iter(loader))
    assert waveforms.shape == (4, config.CLIP_SAMPLES)
    assert labels.shape == (4,)

    featurizer = LogMelSpectrogram().eval()
    with torch.no_grad():
        features = featurizer(waveforms)
    assert features.shape == (4, *config.INPUT_SHAPE)


@requires_dataset
def test_seed_reproduces_index() -> None:
    from nano_kws.data.speech_commands import SpeechCommandsKWS

    a = SpeechCommandsKWS(subset="validation", silence_per_class_ratio=0.5, seed=123)
    b = SpeechCommandsKWS(subset="validation", silence_per_class_ratio=0.5, seed=123)
    assert a._index == b._index
    assert a._silence_plan == b._silence_plan
