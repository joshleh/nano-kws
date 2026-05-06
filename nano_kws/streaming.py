"""Sliding-window streaming inference for the bundled KWS model.

The exported ONNX takes exactly one 1-second log-mel and returns one
softmax-ready logits vector. Real-time keyword spotting is built on
top of that primitive by sliding a 1-second window across the input
audio at a small hop (typically 100-300 ms), exponentially smoothing
the per-window posteriors over time, and peak-picking on the smoothed
track. This module implements that glue in pure Python on top of
:class:`nano_kws.infer.KwsInferencer`.

This is exactly what wake-word detection in production looks like —
structurally identical to what runs on the dedicated audio
accelerators in earbuds and smart speakers, where the same
window-based primitive gets wrapped in a hardware-friendly ring
buffer + posterior smoother. We do the software version here purely
so the Streamlit demo can run against arbitrarily long recordings
without lying about what the model is fundamentally classifying (a
1-second clip).

Out of scope on purpose:

* Online streaming featurization (we recompute the full mel
  spectrogram per window). The dominant cost in the demo is the
  per-window ONNX run, not the featurizer; an incremental STFT would
  be the right next step on actually constrained hardware.
* Hard real-time guarantees. The Streamlit demo runs on whatever
  Python thread the script is on; latency-critical deployments push
  this loop into a C++ or DSP context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from nano_kws import config
from nano_kws.infer import KwsInferencer

logger = logging.getLogger("nano_kws.streaming")


# ─── Design note: this whole module is the "streaming wrapper" recipe ──────
# Given a fixed-window classifier, the canonical way to build a wake-word
# detector on top is exactly the three-stage pipeline implemented here:
#   1. Slide the fixed-window classifier across the audio at a small hop
#      (100-300 ms is typical; smaller = better localization but more compute).
#   2. Smooth the per-window posteriors over time. Without smoothing you get
#      jittery single-window detections; with too much smoothing you delay
#      the trigger. EMA is the simplest tunable smoother — production teams
#      use anything from a moving median to a learned RNN smoother.
#   3. Peak-pick: trigger when the smoothed posterior crosses a threshold
#      AND is a local maximum AND is outside the refractory window of the
#      last trigger. The refractory window prevents a single keyword from
#      firing N adjacent windows.
# The hardware version on a dedicated edge accelerator looks the same
# logically but runs on a circular ring buffer of audio + an incrementally
# updated mel spectrogram (no per-window recomputation), with the smoothing
# + peak logic in a tiny C state machine. This module is the software
# reference.
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_HOP_MS: float = 200.0
"""Stride between adjacent windows. 200 ms is a common default in the KWS
literature — small enough to localize keyword onsets to ~5 frames, large
enough that we run the model 5 Hz instead of 50 Hz."""

DEFAULT_EMA_ALPHA: float = 0.4
"""Weight on the new posterior in the EMA: lower = more smoothing."""

DEFAULT_DETECTION_THRESHOLD: float = 0.5
"""Smoothed probability above which a keyword class can be declared."""

DEFAULT_DETECTION_REFRACTORY_S: float = 0.5
"""Minimum spacing between two reported detections of the same class.
Without this we'd report adjacent windows that all sit above the
threshold around the same true keyword."""


@dataclass(frozen=True)
class Detection:
    """A single keyword spot in a streaming pass."""

    time_s: float
    """Center time of the window the detection was peaked at."""

    label: str
    """Class label (always a keyword — silence/unknown are filtered out)."""

    probability: float
    """Smoothed posterior probability at the peak."""


@dataclass(frozen=True)
class StreamingResult:
    """Output of a sliding-window classification pass.

    Attributes
    ----------
    times_s
        ``(T,)`` center time of each window in seconds.
    probs
        ``(T, NUM_CLASSES)`` raw per-window softmax (unsmoothed).
    smoothed
        ``(T, NUM_CLASSES)`` EMA-smoothed posteriors (what peak-picking
        runs against).
    detections
        Sorted list of :class:`Detection` events.
    window_size_s
        The classifier's input clip length (always
        :data:`nano_kws.config.CLIP_DURATION_S`).
    hop_s
        Stride between windows, in seconds. Equal to ``hop_ms / 1000``
        rounded to the nearest sample.
    labels
        Class labels in the order they appear along axis 1 of ``probs``
        and ``smoothed`` (cached from the underlying inferencer).
    """

    times_s: np.ndarray
    probs: np.ndarray
    smoothed: np.ndarray
    detections: list[Detection]
    window_size_s: float
    hop_s: float
    labels: tuple[str, ...]


class StreamingClassifier:
    """Slides :class:`KwsInferencer`'s 1-second classifier across longer audio.

    Parameters
    ----------
    inferencer
        Underlying single-window classifier. Reused as-is — this class
        is pure orchestration around it.
    hop_ms
        Stride between adjacent windows, in milliseconds. Defaults to
        :data:`DEFAULT_HOP_MS`.
    ema_alpha
        Exponential moving-average weight on the new posterior:

        .. code-block:: text

            smoothed[t] = alpha * probs[t] + (1 - alpha) * smoothed[t-1]

        Lower alpha = more smoothing, longer reaction time, fewer
        spurious peaks. Higher alpha = jumpier track that follows the
        raw classifier. Defaults to :data:`DEFAULT_EMA_ALPHA`.
    detection_threshold
        Smoothed probability above which a *keyword* class can be
        declared as a detection. Silence and unknown never produce
        detections.
    detection_refractory_s
        Minimum spacing between two reported detections of the same
        class. Defaults to :data:`DEFAULT_DETECTION_REFRACTORY_S`.
    """

    def __init__(
        self,
        inferencer: KwsInferencer,
        *,
        hop_ms: float = DEFAULT_HOP_MS,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
        detection_refractory_s: float = DEFAULT_DETECTION_REFRACTORY_S,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if hop_ms <= 0:
            raise ValueError(f"hop_ms must be positive, got {hop_ms}")
        if detection_refractory_s < 0:
            raise ValueError(
                f"detection_refractory_s must be non-negative, got {detection_refractory_s}"
            )
        self.inferencer = inferencer
        self.hop_ms = float(hop_ms)
        self.ema_alpha = float(ema_alpha)
        self.detection_threshold = float(detection_threshold)
        self.detection_refractory_s = float(detection_refractory_s)

    # ------------------------------------------------------------------
    # Derived parameters.
    # ------------------------------------------------------------------

    @property
    def hop_samples(self) -> int:
        """Stride between adjacent windows, in samples."""
        return round(self.hop_ms * config.SAMPLE_RATE / 1000)

    @property
    def window_samples(self) -> int:
        """Window length in samples (= one full classifier input)."""
        return config.CLIP_SAMPLES

    # ------------------------------------------------------------------
    # Main entry point.
    # ------------------------------------------------------------------

    def classify(self, waveform: np.ndarray) -> StreamingResult:
        """Slide the classifier across ``waveform`` and return per-window posteriors.

        Audio shorter than one window is zero-padded on the right; audio
        that doesn't divide evenly into hops drops the trailing partial
        window (i.e. only complete windows are classified).
        """
        if waveform.ndim != 1:
            raise ValueError(f"expected 1D waveform, got shape {waveform.shape}")

        wav = np.asarray(waveform, dtype=np.float32)
        win = self.window_samples
        hop = self.hop_samples

        if wav.shape[0] < win:
            wav = np.concatenate([wav, np.zeros(win - wav.shape[0], dtype=np.float32)])

        n_windows = (wav.shape[0] - win) // hop + 1
        n_classes = len(self.inferencer.labels)

        probs = np.empty((n_windows, n_classes), dtype=np.float32)
        for i in range(n_windows):
            start = i * hop
            window = wav[start : start + win]
            probs[i] = self.inferencer.predict(window)

        # Center-of-window timestamps. Reporting the *middle* of the
        # window (rather than its leading edge) puts the marker closer
        # to where the keyword was actually spoken, which matches what
        # human listeners expect when scrubbing the audio.
        times_s = (np.arange(n_windows) * hop + win / 2) / config.SAMPLE_RATE

        smoothed = self._ema_smooth(probs)
        detections = self._peak_pick(times_s, smoothed)

        logger.info(
            "Streaming classify: %d windows over %.2fs (%d Hz frame rate), %d detections",
            n_windows,
            wav.shape[0] / config.SAMPLE_RATE,
            round(config.SAMPLE_RATE / hop) if hop else 0,
            len(detections),
        )

        return StreamingResult(
            times_s=times_s,
            probs=probs,
            smoothed=smoothed,
            detections=detections,
            window_size_s=config.CLIP_DURATION_S,
            hop_s=hop / config.SAMPLE_RATE,
            labels=tuple(self.inferencer.labels),
        )

    # ------------------------------------------------------------------
    # Building blocks.
    # ------------------------------------------------------------------

    def _ema_smooth(self, probs: np.ndarray) -> np.ndarray:
        """Apply the per-class EMA recurrence along the time axis."""
        smoothed = np.empty_like(probs)
        smoothed[0] = probs[0]
        a = self.ema_alpha
        # Plain Python loop over T — T is small (10s of windows for a
        # demo recording). Vectorising would be possible with
        # `lfilter` but adds a scipy dep we don't currently carry.
        for t in range(1, probs.shape[0]):
            smoothed[t] = a * probs[t] + (1.0 - a) * smoothed[t - 1]
        return smoothed

    def _peak_pick(
        self,
        times_s: np.ndarray,
        smoothed: np.ndarray,
    ) -> list[Detection]:
        """Find local maxima per *keyword* class above the threshold.

        A detection at time ``t`` for class ``c`` requires:

        1. ``smoothed[t, c] >= detection_threshold``
        2. ``smoothed[t, c] >= smoothed[t-1, c]`` and
           ``smoothed[t, c] >= smoothed[t+1, c]`` (local-max with
           plateau-friendly ``>=`` so adjacent equal samples don't both
           qualify; the refractory check below cleans those up too)
        3. No previously reported detection of the same class within
           ``detection_refractory_s`` seconds.

        Silence and unknown classes are excluded — they're useful for
        suppressing false fires inside the model but aren't meaningful
        as positive detections in a demo.
        """
        labels = self.inferencer.labels
        keyword_indices = [
            i
            for i, lbl in enumerate(labels)
            if lbl not in (config.SILENCE_LABEL, config.UNKNOWN_LABEL)
        ]

        n_t = smoothed.shape[0]
        detections: list[Detection] = []
        last_time_per_class: dict[int, float] = {}

        for t in range(n_t):
            for ci in keyword_indices:
                p = float(smoothed[t, ci])
                if p < self.detection_threshold:
                    continue
                left = float(smoothed[t - 1, ci]) if t > 0 else -1.0
                right = float(smoothed[t + 1, ci]) if t < n_t - 1 else -1.0
                if not (p >= left and p >= right):
                    continue
                last_t = last_time_per_class.get(ci)
                if last_t is not None and times_s[t] - last_t < self.detection_refractory_s:
                    continue
                detections.append(
                    Detection(
                        time_s=float(times_s[t]),
                        label=labels[ci],
                        probability=p,
                    )
                )
                last_time_per_class[ci] = float(times_s[t])

        detections.sort(key=lambda d: d.time_s)
        return detections


__all__ = [
    "DEFAULT_DETECTION_REFRACTORY_S",
    "DEFAULT_DETECTION_THRESHOLD",
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_HOP_MS",
    "Detection",
    "StreamingClassifier",
    "StreamingResult",
]
