"""Sanity checks on the project-wide audio constants.

These are cheap invariants worth pinning early so that nobody silently
changes one number (e.g. SAMPLE_RATE) and breaks the entire pipeline.
"""

from __future__ import annotations

import nano_kws
from nano_kws import config


def test_package_has_version() -> None:
    assert isinstance(nano_kws.__version__, str)
    assert nano_kws.__version__ != ""


def test_clip_samples_matches_sample_rate_and_duration() -> None:
    assert config.CLIP_SAMPLES == int(config.SAMPLE_RATE * config.CLIP_DURATION_S)


def test_window_and_hop_in_samples() -> None:
    assert config.WIN_LENGTH == int(config.SAMPLE_RATE * config.WIN_LENGTH_MS / 1000)
    assert config.HOP_LENGTH == int(config.SAMPLE_RATE * config.HOP_LENGTH_MS / 1000)
    assert config.HOP_LENGTH < config.WIN_LENGTH


def test_n_fft_covers_window() -> None:
    assert config.N_FFT >= config.WIN_LENGTH
    # power of two
    assert config.N_FFT & (config.N_FFT - 1) == 0


def test_label_set_is_12_class() -> None:
    assert config.NUM_CLASSES == 12
    assert len(config.LABELS) == 12
    assert config.SILENCE_LABEL in config.LABELS
    assert config.UNKNOWN_LABEL in config.LABELS
    for kw in config.KEYWORDS:
        assert kw in config.LABELS


def test_label_index_round_trip() -> None:
    for i, label in enumerate(config.LABELS):
        assert config.LABEL_TO_INDEX[label] == i
        assert config.INDEX_TO_LABEL[i] == label


def test_input_shape_is_three_dim() -> None:
    assert config.INPUT_SHAPE == (1, config.N_MELS, config.N_FRAMES)
