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


@requires_dataset
def test_filtered_dataset_restricts_to_label_subset() -> None:
    from nano_kws.data.speech_commands import FilteredKwsDataset, SpeechCommandsKWS

    base = SpeechCommandsKWS(subset="validation", silence_per_class_ratio=0.5, seed=0)
    kept = ["yes", "no", config.SILENCE_LABEL, config.UNKNOWN_LABEL]
    filtered = FilteredKwsDataset(base, kept_label_names=kept, seed=0)

    assert filtered.num_classes == 4
    assert filtered.kept_label_names == kept
    assert set(filtered.class_counts.keys()) == set(range(4))
    assert len(filtered) == sum(filtered.class_counts.values())

    waveform, label = filtered[0]
    assert waveform.shape == (config.CLIP_SAMPLES,)
    assert waveform.dtype == torch.float32
    assert 0 <= int(label) < 4


@requires_dataset
def test_filtered_dataset_caps_each_class_to_max_samples_per_class() -> None:
    from nano_kws.data.speech_commands import FilteredKwsDataset, SpeechCommandsKWS

    base = SpeechCommandsKWS(subset="validation", silence_per_class_ratio=1.0, seed=0)
    kept = ["yes", "no", "_silence_", "_unknown_"]
    n = 17
    filtered = FilteredKwsDataset(base, kept_label_names=kept, max_samples_per_class=n, seed=0)

    counts = filtered.class_counts
    assert all(v <= n for v in counts.values()), counts
    # All four classes should have at least n in the validation split, so the
    # cap should bind exactly.
    assert all(v == n for v in counts.values()), counts
    assert len(filtered) == n * len(kept)


@requires_dataset
def test_filtered_dataset_is_seed_reproducible() -> None:
    from nano_kws.data.speech_commands import FilteredKwsDataset, SpeechCommandsKWS

    base = SpeechCommandsKWS(subset="validation", silence_per_class_ratio=0.5, seed=0)
    kept = ["yes", "no", "_silence_", "_unknown_"]
    a = FilteredKwsDataset(base, kept_label_names=kept, max_samples_per_class=20, seed=42)
    b = FilteredKwsDataset(base, kept_label_names=kept, max_samples_per_class=20, seed=42)
    assert a._chosen_base_indices == b._chosen_base_indices


@requires_dataset
def test_filtered_dataset_relabels_to_contiguous_index_space() -> None:
    from nano_kws.data.speech_commands import FilteredKwsDataset, SpeechCommandsKWS

    base = SpeechCommandsKWS(subset="validation", silence_per_class_ratio=0.5, seed=0)
    # Pick keywords whose 12-class indices are non-contiguous, so the
    # remap is meaningful.
    kept = ["yes", "stop", "_unknown_"]
    filtered = FilteredKwsDataset(base, kept_label_names=kept, max_samples_per_class=8, seed=0)

    seen = set()
    for i in range(len(filtered)):
        _, label = filtered[i]
        seen.add(int(label))
    assert seen == {0, 1, 2}


def test_filtered_dataset_rejects_empty_keyword_subset() -> None:
    from nano_kws.data.speech_commands import FilteredKwsDataset

    with pytest.raises(ValueError, match="cannot be empty"):
        FilteredKwsDataset(base=None, kept_label_names=[])  # type: ignore[arg-type]


def test_filtered_dataset_rejects_unknown_label_names() -> None:
    from nano_kws.data.speech_commands import FilteredKwsDataset

    with pytest.raises(ValueError, match="Unknown label name"):
        FilteredKwsDataset(  # type: ignore[arg-type]
            base=None,
            kept_label_names=["yes", "not_a_real_label"],
        )
