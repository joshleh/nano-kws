"""Tests for the sliding-window streaming classifier.

Most tests use a duck-typed mock inferencer so we exercise the
sliding/smoothing/peak-picking math without spinning up ONNX Runtime.
A final smoke test runs against the bundled INT8 ONNX to confirm the
end-to-end glue works on real model output.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from nano_kws import config
from nano_kws.streaming import (
    DEFAULT_HOP_MS,
    Detection,
    StreamingClassifier,
    StreamingResult,
)

# ---------------------------------------------------------------------------
# Mock inferencer (duck-typed: only needs .labels and .predict).
# ---------------------------------------------------------------------------


class _FixedSequenceInferencer:
    """Returns a pre-baked probability vector per call, in order."""

    def __init__(self, probs_per_call: np.ndarray, labels: tuple[str, ...]):
        if probs_per_call.ndim != 2 or probs_per_call.shape[1] != len(labels):
            raise ValueError(
                f"probs_per_call shape {probs_per_call.shape} doesn't match "
                f"len(labels)={len(labels)}"
            )
        self._probs = probs_per_call.astype(np.float32, copy=False)
        self._call = 0
        self.labels = labels

    def predict(self, waveform: np.ndarray) -> np.ndarray:
        i = min(self._call, self._probs.shape[0] - 1)
        self._call += 1
        return self._probs[i].copy()


def _uniform_probs(n_calls: int, labels: tuple[str, ...]) -> np.ndarray:
    """All classes equiprobable on every call."""
    return np.full((n_calls, len(labels)), 1.0 / len(labels), dtype=np.float32)


# ---------------------------------------------------------------------------
# Construction / parameter validation.
# ---------------------------------------------------------------------------


def test_constructor_rejects_bad_alpha() -> None:
    inf = _FixedSequenceInferencer(_uniform_probs(1, config.LABELS), config.LABELS)
    with pytest.raises(ValueError):
        StreamingClassifier(inf, ema_alpha=0.0)
    with pytest.raises(ValueError):
        StreamingClassifier(inf, ema_alpha=1.5)


def test_constructor_rejects_bad_hop() -> None:
    inf = _FixedSequenceInferencer(_uniform_probs(1, config.LABELS), config.LABELS)
    with pytest.raises(ValueError):
        StreamingClassifier(inf, hop_ms=0.0)
    with pytest.raises(ValueError):
        StreamingClassifier(inf, hop_ms=-50.0)


def test_constructor_rejects_negative_refractory() -> None:
    inf = _FixedSequenceInferencer(_uniform_probs(1, config.LABELS), config.LABELS)
    with pytest.raises(ValueError):
        StreamingClassifier(inf, detection_refractory_s=-0.1)


def test_hop_samples_matches_default_hop_ms() -> None:
    inf = _FixedSequenceInferencer(_uniform_probs(1, config.LABELS), config.LABELS)
    sc = StreamingClassifier(inf)
    expected = round(DEFAULT_HOP_MS * config.SAMPLE_RATE / 1000)
    assert sc.hop_samples == expected
    assert sc.window_samples == config.CLIP_SAMPLES


# ---------------------------------------------------------------------------
# Window count math.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration_s", "hop_ms", "expected_windows"),
    [
        # Exactly one window.
        (1.0, 200.0, 1),
        # 2 seconds @ 200 ms hop = floor((32000 - 16000) / 3200) + 1 = 6
        (2.0, 200.0, 6),
        # 3 seconds @ 200 ms hop = floor((48000 - 16000) / 3200) + 1 = 11
        (3.0, 200.0, 11),
        # 5 seconds @ 100 ms hop = floor((80000 - 16000) / 1600) + 1 = 41
        (5.0, 100.0, 41),
    ],
)
def test_window_count_matches_formula(duration_s, hop_ms, expected_windows):
    n_samples = int(duration_s * config.SAMPLE_RATE)
    waveform = np.zeros(n_samples, dtype=np.float32)
    inf = _FixedSequenceInferencer(_uniform_probs(expected_windows, config.LABELS), config.LABELS)
    sc = StreamingClassifier(inf, hop_ms=hop_ms)
    result = sc.classify(waveform)
    assert result.probs.shape == (expected_windows, config.NUM_CLASSES)
    assert result.smoothed.shape == (expected_windows, config.NUM_CLASSES)
    assert result.times_s.shape == (expected_windows,)


def test_short_waveform_padded_to_one_window() -> None:
    inf = _FixedSequenceInferencer(_uniform_probs(1, config.LABELS), config.LABELS)
    sc = StreamingClassifier(inf)
    short = np.zeros(int(0.3 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(short)
    assert result.probs.shape == (1, config.NUM_CLASSES)


def test_classify_rejects_2d_input() -> None:
    inf = _FixedSequenceInferencer(_uniform_probs(1, config.LABELS), config.LABELS)
    sc = StreamingClassifier(inf)
    with pytest.raises(ValueError):
        sc.classify(np.zeros((2, config.CLIP_SAMPLES), dtype=np.float32))


def test_window_center_times_are_centered() -> None:
    """First window's t should be 0.5 s (= half of a 1 s window)."""
    inf = _FixedSequenceInferencer(_uniform_probs(3, config.LABELS), config.LABELS)
    sc = StreamingClassifier(inf, hop_ms=200.0)
    waveform = np.zeros(int(1.5 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    assert math.isclose(result.times_s[0], 0.5, rel_tol=1e-6)
    if result.times_s.shape[0] > 1:
        assert math.isclose(result.times_s[1] - result.times_s[0], 0.2, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# EMA smoothing math.
# ---------------------------------------------------------------------------


def test_ema_smoothed_first_step_equals_raw() -> None:
    labels = config.LABELS
    probs = np.zeros((4, len(labels)), dtype=np.float32)
    probs[0, 0] = 0.9
    probs[1, 0] = 0.1
    inf = _FixedSequenceInferencer(probs, labels)
    sc = StreamingClassifier(inf, hop_ms=200.0, ema_alpha=0.5)
    waveform = np.zeros(int(1.6 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    assert result.smoothed[0, 0] == pytest.approx(0.9)


def test_ema_recurrence_holds() -> None:
    """smoothed[t] = a*probs[t] + (1-a)*smoothed[t-1] for all t > 0."""
    labels = config.LABELS
    rng = np.random.default_rng(0)
    # 2.0 s @ 200 ms hop => floor((32000 - 16000)/3200) + 1 = 6 windows.
    n = 6
    probs = rng.random((n, len(labels)), dtype=np.float32)
    inf = _FixedSequenceInferencer(probs, labels)
    alpha = 0.3
    sc = StreamingClassifier(inf, hop_ms=200.0, ema_alpha=alpha)
    waveform = np.zeros(int(2.0 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    assert result.smoothed.shape[0] == n  # guard against drift in the formula above
    for t in range(1, result.smoothed.shape[0]):
        expected = alpha * probs[t] + (1.0 - alpha) * result.smoothed[t - 1]
        np.testing.assert_allclose(result.smoothed[t], expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Peak-picking and detections.
# ---------------------------------------------------------------------------


def _probs_with_keyword_peak(
    n_steps: int,
    peak_step: int,
    peak_class: str,
    peak_value: float = 0.95,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build a (T, C) probability matrix that's flat low except for one peak."""
    labels = config.LABELS
    probs = np.full((n_steps, len(labels)), 0.05 / (len(labels) - 1), dtype=np.float32)
    # Push silence up so each row sums close to 1 with the peak.
    silence_idx = labels.index(config.SILENCE_LABEL)
    probs[:, silence_idx] = 0.95
    # Inject a single peak.
    peak_idx = labels.index(peak_class)
    probs[peak_step, :] = 0.05 / (len(labels) - 1)
    probs[peak_step, peak_idx] = peak_value
    probs[peak_step, silence_idx] = 1.0 - peak_value - (len(labels) - 2) * 0.05 / (len(labels) - 1)
    return probs, labels


def test_peak_pick_finds_single_keyword_peak() -> None:
    probs, labels = _probs_with_keyword_peak(n_steps=5, peak_step=2, peak_class="yes")
    inf = _FixedSequenceInferencer(probs, labels)
    sc = StreamingClassifier(
        inf,
        hop_ms=200.0,
        ema_alpha=1.0,  # disable smoothing so the peak survives unchanged
        detection_threshold=0.5,
    )
    waveform = np.zeros(int(2.0 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.label == "yes"
    assert det.probability >= 0.9


def test_peak_pick_respects_threshold() -> None:
    probs, labels = _probs_with_keyword_peak(
        n_steps=4, peak_step=1, peak_class="yes", peak_value=0.45
    )
    inf = _FixedSequenceInferencer(probs, labels)
    sc = StreamingClassifier(inf, hop_ms=200.0, ema_alpha=1.0, detection_threshold=0.5)
    waveform = np.zeros(int(1.6 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    assert result.detections == []


def test_peak_pick_skips_silence_and_unknown_even_at_high_prob() -> None:
    """Silence/unknown should never produce detections, even if dominant."""
    labels = config.LABELS
    probs = np.full((5, len(labels)), 0.001, dtype=np.float32)
    silence_idx = labels.index(config.SILENCE_LABEL)
    probs[:, silence_idx] = 0.99
    probs[2, silence_idx] = 0.999  # an obvious "peak" — but on silence
    inf = _FixedSequenceInferencer(probs, labels)
    sc = StreamingClassifier(inf, hop_ms=200.0, ema_alpha=1.0, detection_threshold=0.5)
    waveform = np.zeros(int(1.8 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    assert result.detections == []


def test_peak_pick_respects_refractory() -> None:
    """Two adjacent windows above threshold for the same class should
    only produce one detection if they're inside the refractory window."""
    labels = config.LABELS
    probs = np.full((6, len(labels)), 0.05 / (len(labels) - 1), dtype=np.float32)
    silence_idx = labels.index(config.SILENCE_LABEL)
    yes_idx = labels.index("yes")
    probs[:, silence_idx] = 0.95
    # Two equal peaks 200 ms apart (one hop).
    for t in (2, 3):
        probs[t, :] = 0.05 / (len(labels) - 1)
        probs[t, yes_idx] = 0.9
        probs[t, silence_idx] = 0.05
    inf = _FixedSequenceInferencer(probs, labels)
    sc = StreamingClassifier(
        inf,
        hop_ms=200.0,
        ema_alpha=1.0,
        detection_threshold=0.5,
        detection_refractory_s=0.5,  # 500 ms > 200 ms hop, so the second is suppressed
    )
    waveform = np.zeros(int(2.2 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    assert len(result.detections) == 1
    assert result.detections[0].label == "yes"


def test_peak_pick_allows_separate_keywords_without_refractory_interference() -> None:
    """Refractory is per-class, so 'yes' then 'stop' inside the refractory
    window should still produce two detections."""
    labels = config.LABELS
    probs = np.full((5, len(labels)), 0.05 / (len(labels) - 1), dtype=np.float32)
    silence_idx = labels.index(config.SILENCE_LABEL)
    probs[:, silence_idx] = 0.95
    # 'yes' peak at step 1, 'stop' peak at step 2 — different classes.
    probs[1, :] = 0.05 / (len(labels) - 1)
    probs[1, labels.index("yes")] = 0.9
    probs[1, silence_idx] = 0.05
    probs[2, :] = 0.05 / (len(labels) - 1)
    probs[2, labels.index("stop")] = 0.9
    probs[2, silence_idx] = 0.05
    inf = _FixedSequenceInferencer(probs, labels)
    sc = StreamingClassifier(
        inf,
        hop_ms=200.0,
        ema_alpha=1.0,
        detection_threshold=0.5,
        detection_refractory_s=1.0,
    )
    waveform = np.zeros(int(1.8 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    detected_labels = [d.label for d in result.detections]
    assert "yes" in detected_labels
    assert "stop" in detected_labels


def test_detections_are_sorted_by_time() -> None:
    labels = config.LABELS
    n = 6
    probs = np.full((n, len(labels)), 0.05 / (len(labels) - 1), dtype=np.float32)
    silence_idx = labels.index(config.SILENCE_LABEL)
    probs[:, silence_idx] = 0.95
    for t, kw in [(1, "yes"), (3, "no"), (5, "stop")]:
        probs[t, :] = 0.05 / (len(labels) - 1)
        probs[t, labels.index(kw)] = 0.9
        probs[t, silence_idx] = 0.05
    inf = _FixedSequenceInferencer(probs, labels)
    sc = StreamingClassifier(inf, hop_ms=200.0, ema_alpha=1.0, detection_threshold=0.5)
    waveform = np.zeros(int(2.2 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    times = [d.time_s for d in result.detections]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# StreamingResult metadata.
# ---------------------------------------------------------------------------


def test_streaming_result_carries_window_and_hop_metadata() -> None:
    inf = _FixedSequenceInferencer(_uniform_probs(2, config.LABELS), config.LABELS)
    sc = StreamingClassifier(inf, hop_ms=300.0)
    waveform = np.zeros(int(1.4 * config.SAMPLE_RATE), dtype=np.float32)
    result = sc.classify(waveform)
    assert isinstance(result, StreamingResult)
    assert result.window_size_s == config.CLIP_DURATION_S
    assert result.hop_s == pytest.approx(0.3)
    assert result.labels == tuple(config.LABELS)


def test_detection_dataclass_is_frozen() -> None:
    d = Detection(time_s=1.0, label="yes", probability=0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.label = "no"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# End-to-end smoke against the bundled INT8 ONNX (uses real ORT).
# ---------------------------------------------------------------------------


def test_streaming_against_real_inferencer_returns_well_formed_result() -> None:
    """Drive the full path against the bundled ONNX with synthetic silence.

    Asserts shape contracts and dtypes — does NOT assert specific
    detection content, since silence is ambiguous and the model's
    behaviour on pure zeros is implementation-defined.
    """
    from nano_kws.infer import KwsInferencer

    model_path = config.DEFAULT_INT8_ONNX
    if not model_path.is_file():
        pytest.skip(f"bundled INT8 ONNX not present at {model_path}")
    inf = KwsInferencer(model_path=model_path)
    sc = StreamingClassifier(inf, hop_ms=200.0)
    rng = np.random.default_rng(0)
    waveform = (0.001 * rng.standard_normal(int(2.5 * config.SAMPLE_RATE))).astype(np.float32)
    result = sc.classify(waveform)
    assert result.probs.dtype == np.float32
    assert result.smoothed.dtype == np.float32
    assert result.probs.shape == result.smoothed.shape
    assert result.probs.shape[1] == config.NUM_CLASSES
    np.testing.assert_allclose(result.probs.sum(axis=1), 1.0, atol=1e-3)
    assert result.times_s.shape[0] == result.probs.shape[0]
