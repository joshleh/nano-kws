"""Streamlit live demo for the bundled INT8 keyword spotter.

The demo runs against ``assets/ds_cnn_small_int8.onnx`` so a fresh clone
can launch it with ``make app`` without any training. Two input paths
are supported:

* **Mic capture** (default): click *Record 1 second* to capture audio
  from the default input device via ``sounddevice``.
* **WAV upload**: drop a 16 kHz mono WAV (or anything ``soundfile`` can
  decode); we resample / pad / crop to 1 s for you.

Both paths flow through the *same* preprocessing as training
(:class:`nano_kws.data.features.LogMelSpectrogram` +
:func:`nano_kws.data.features.pad_or_crop`) so the spectrogram you see
is bit-identical to what the model received.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from nano_kws import config
from nano_kws.data.features import LogMelSpectrogram, pad_or_crop, waveform_to_logmel
from nano_kws.infer import KwsInferencer
from nano_kws.streaming import (
    DEFAULT_DETECTION_REFRACTORY_S,
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_EMA_ALPHA,
    DEFAULT_HOP_MS,
    StreamingClassifier,
    StreamingResult,
)

DEFAULT_MODEL = config.ASSETS_DIR / "ds_cnn_small_int8.onnx"
RECORD_SECONDS = config.CLIP_DURATION_S
CONTINUOUS_MAX_SECONDS: float = 10.0


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading INT8 model ...")
def _load_inferencer(model_path: str) -> KwsInferencer:
    return KwsInferencer(model_path=model_path)


@st.cache_resource
def _featurizer() -> LogMelSpectrogram:
    return LogMelSpectrogram().eval()


def _streaming_classifier(
    inferencer: KwsInferencer,
    *,
    hop_ms: float,
    ema_alpha: float,
    detection_threshold: float,
    detection_refractory_s: float,
) -> StreamingClassifier:
    """Build a fresh StreamingClassifier (cheap) wrapping the cached inferencer."""
    return StreamingClassifier(
        inferencer,
        hop_ms=hop_ms,
        ema_alpha=ema_alpha,
        detection_threshold=detection_threshold,
        detection_refractory_s=detection_refractory_s,
    )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _record_from_mic(seconds: float, sample_rate: int) -> np.ndarray | None:
    """Capture ``seconds`` of mono audio from the default input device.

    Returns ``None`` (and posts an error to the page) if ``sounddevice``
    isn't available or the device can't be opened — common on CI / SSH
    sessions where there's no microphone.
    """
    try:
        import sounddevice as sd
    except ImportError:
        st.error("`sounddevice` is not installed. Run `pip install sounddevice`.")
        return None

    n_frames = round(seconds * sample_rate)
    try:
        audio = sd.rec(n_frames, samplerate=sample_rate, channels=1, dtype="float32", blocking=True)
    except Exception as exc:
        st.error(f"Couldn't open the input device: {exc}")
        return None
    return audio.reshape(-1).astype(np.float32, copy=False)


def _decode_uploaded_wav(uploaded: io.BytesIO, target_sr: int) -> np.ndarray:
    """Decode an uploaded file and downmix / resample to ``target_sr`` mono."""
    data, sr = sf.read(uploaded, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        ratio = target_sr / sr
        n_out = round(data.shape[0] * ratio)
        x_old = np.linspace(0.0, 1.0, num=data.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32, copy=False)
    return data


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_prediction(inferencer: KwsInferencer, waveform: np.ndarray) -> None:
    # pad_or_crop always returns a torch.float32 tensor (see its docstring).
    # Inferencer + featurizer accept either, but st.audio wants ndarray.
    waveform = pad_or_crop(waveform).detach().cpu().numpy()

    logmel = waveform_to_logmel(waveform).numpy()  # (1, n_mels, n_frames)
    probs = inferencer.predict(waveform)
    label, confidence = inferencer.predict_label(waveform)

    # Top prediction banner.
    st.subheader(f"Prediction: `{label}`  ({confidence * 100:.1f}%)")
    st.audio(waveform, sample_rate=config.SAMPLE_RATE)

    col_spec, col_probs = st.columns([1, 1], gap="large")

    with col_spec:
        st.caption(
            f"Log-mel spectrogram — {config.N_MELS} mels x {config.N_FRAMES} frames "
            f"({config.WIN_LENGTH_MS} ms window / {config.HOP_LENGTH_MS} ms hop)"
        )
        spec = logmel[0]
        spec_norm = (spec - spec.min()) / max(spec.max() - spec.min(), 1e-6)
        st.image(
            np.flipud(spec_norm),
            caption="low mel <- ... -> high mel",
            use_container_width=True,
            clamp=True,
        )

    with col_probs:
        st.caption("Per-class probability")
        chart_data = {
            "label": list(config.LABELS),
            "probability": [float(p) for p in probs],
        }
        st.bar_chart(chart_data, x="label", y="probability", height=320)


def _render_streaming_result(
    waveform: np.ndarray,
    result: StreamingResult,
) -> None:
    """Render the timeline + detection table for a streaming pass."""
    st.audio(waveform, sample_rate=config.SAMPLE_RATE)

    keyword_cols = [
        i
        for i, lbl in enumerate(result.labels)
        if lbl not in (config.SILENCE_LABEL, config.UNKNOWN_LABEL)
    ]
    keyword_labels = [result.labels[i] for i in keyword_cols]

    chart_df = pd.DataFrame(
        result.smoothed[:, keyword_cols],
        columns=keyword_labels,
        index=pd.Index(result.times_s, name="time (s)"),
    )

    st.caption(
        f"Smoothed per-keyword posterior over time — sliding "
        f"{result.window_size_s:g}s window with a {result.hop_s * 1000:.0f}ms hop "
        f"({len(result.times_s)} windows). Silence and unknown classes hidden."
    )
    st.line_chart(chart_df, height=320)

    if result.detections:
        det_df = pd.DataFrame(
            [
                {
                    "time (s)": round(d.time_s, 3),
                    "label": d.label,
                    "probability": round(d.probability, 3),
                }
                for d in result.detections
            ]
        )
        st.subheader(f"Detections ({len(result.detections)})")
        st.dataframe(det_df, hide_index=True, use_container_width=True)
    else:
        st.info(
            "No keyword detections. Try lowering the threshold in the sidebar, "
            "raising the smoothing alpha (less smoothing), or speaking closer to the mic."
        )


def _render_continuous_controls() -> tuple[float, float, float, float]:
    """Sidebar controls for the streaming/continuous mode."""
    st.sidebar.header("Continuous-mode controls")
    hop_ms = st.sidebar.slider(
        "Window hop (ms)",
        min_value=50.0,
        max_value=500.0,
        value=float(DEFAULT_HOP_MS),
        step=50.0,
        help="Stride between adjacent 1-second classifier windows. Smaller = "
        "tighter time resolution but more inference calls.",
    )
    ema_alpha = st.sidebar.slider(
        "Smoothing alpha",
        min_value=0.05,
        max_value=1.0,
        value=float(DEFAULT_EMA_ALPHA),
        step=0.05,
        help="EMA weight on the new posterior. Lower = more smoothing, longer "
        "reaction time. 1.0 disables smoothing entirely.",
    )
    detection_threshold = st.sidebar.slider(
        "Detection threshold",
        min_value=0.1,
        max_value=0.95,
        value=float(DEFAULT_DETECTION_THRESHOLD),
        step=0.05,
        help="Smoothed probability above which a keyword can fire as a detection.",
    )
    detection_refractory_s = st.sidebar.slider(
        "Refractory (s)",
        min_value=0.0,
        max_value=2.0,
        value=float(DEFAULT_DETECTION_REFRACTORY_S),
        step=0.1,
        help="Minimum spacing between two reported detections of the same class.",
    )
    return hop_ms, ema_alpha, detection_threshold, detection_refractory_s


def _render_sidebar() -> tuple[str, float]:
    st.sidebar.header("Model")
    model_path = st.sidebar.text_input("ONNX model path", value=str(DEFAULT_MODEL))
    if not Path(model_path).is_file():
        st.sidebar.error("File not found. Run `make quantize` to produce one.")
        st.stop()
    seconds = st.sidebar.slider(
        "Recording length (s)",
        min_value=0.5,
        max_value=2.0,
        value=float(RECORD_SECONDS),
        step=0.25,
    )
    st.sidebar.markdown(
        f"**Sample rate:** {config.SAMPLE_RATE} Hz  \n"
        f"**Clip length:** {config.CLIP_DURATION_S} s (audio is padded/cropped)  \n"
        f"**Classes ({config.NUM_CLASSES}):** " + ", ".join(f"`{lbl}`" for lbl in config.LABELS)
    )
    return model_path, seconds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _render_about_section() -> None:
    """Recruiter / non-technical-friendly project explainer.

    Shown expanded by default for first-time visitors so the demo
    has context. Visitors who already know what they're looking at
    can collapse it.
    """
    with st.expander("**About this project** (click to expand / collapse)", expanded=True):
        st.markdown(
            """
**What is this?** A live demo of a tiny keyword recognition model
that listens for one of 10 words: *yes, no, up, down, left, right,
on, off, stop, go*. It's the same kind of model that runs inside
smart speakers, earbuds, and doorbells when they listen for a
wake word like "Hey Alexa".

**Why is it interesting?** The whole model fits in **under 100 KB**
of memory — about the size of a small image — and decides what you
said in **under half a millisecond** on a regular laptop CPU. That
tiny size and speed is what makes it deployable on the kind of
ultra-low-power chip you'd find in a battery-powered device.

**How to try it:**

1. Use the **Upload WAV** tab on the right and drop in a short audio
   clip of yourself (or anyone) saying one of the 10 keywords.
2. Or use the **Continuous** tab to upload a longer recording and
   watch the model fire detections over time.
3. The **Microphone** tab works only when you run the demo locally on
   your own computer — the cloud server here has no microphone
   attached.

**Want the technical story?** Source code, design decisions, accuracy
benchmarks, and the model card all live at the
[GitHub repo](https://github.com/joshleh/nano-kws).
            """.strip()
        )


def _render_glossary() -> None:
    """Quick technical-term explainer for non-ML viewers."""
    with st.expander("Glossary — what the technical words mean", expanded=False):
        st.markdown(
            """
- **Keyword spotter (KWS)** — A neural network that listens to a short
  audio clip and decides whether it contains one of a fixed list of
  keywords. Wake-word detectors (e.g. "Hey Siri") are KWS models.
- **DS-CNN** — Depthwise-Separable Convolutional Neural Network. A
  family of small, efficient image/audio models popular for
  on-device deployment. Roughly 8-9x cheaper than a regular CNN at
  the same kernel size.
- **INT8** — 8-bit integer arithmetic. Standard floating-point models
  use 32-bit numbers (FP32). Squeezing them down to 8-bit integers
  cuts model size by ~4x and speeds inference up significantly, at
  the cost of a small accuracy drop. The art is in keeping that
  drop small.
- **Log-mel spectrogram** — How we turn raw audio into something a
  neural network can chew on. The waveform is sliced into 30 ms
  windows, each window is converted to a frequency representation,
  and that's compressed onto 40 perceptually-spaced "mel" bins. The
  result is a 40 x 97 image — and the model is essentially an image
  classifier on those.
- **ONNX Runtime** — The open-source inference engine we use to
  actually run the trained model. Once the model is exported to ONNX,
  the same file runs in Python, C++, mobile, browser, and most
  embedded toolchains.
- **Quantization** (PTQ / QAT) — The process of converting a model
  from FP32 to INT8. PTQ (post-training) does it after training, in
  a few seconds. QAT (quantization-aware) bakes quantization noise
  into the training loop so the model learns to be robust to it —
  more accurate, more setup work.
- **MAC** — Multiply-Accumulate. The basic arithmetic operation in a
  neural network: `c += a * b`. People count MACs the way you'd
  count miles per gallon — fewer MACs per inference = less power
  drawn on the chip.
            """.strip()
        )


def main() -> None:
    st.set_page_config(page_title="nano-kws — live demo", page_icon=None, layout="wide")
    st.title("nano-kws — INT8 keyword spotter")
    st.caption(
        "A < 100 KB neural network that listens for 10 keywords. "
        "[Source on GitHub](https://github.com/joshleh/nano-kws)."
    )

    _render_about_section()
    _render_glossary()

    model_path, seconds = _render_sidebar()
    hop_ms, ema_alpha, detection_threshold, detection_refractory_s = _render_continuous_controls()
    inferencer = _load_inferencer(model_path)

    tab_mic, tab_upload, tab_continuous = st.tabs(["Microphone", "Upload WAV", "Continuous"])

    with tab_mic:
        st.info(
            "Microphone capture only works locally — the hosted Streamlit "
            "Community Cloud container has no audio input device. If you're "
            "viewing this on the deployed demo, use the **Upload WAV** tab "
            "instead. Run `make app` locally to use the mic path."
        )
        if st.button(f"Record {seconds:g} second(s)", type="primary"):
            with st.spinner("Recording ..."):
                waveform = _record_from_mic(seconds, config.SAMPLE_RATE)
            if waveform is not None:
                _render_prediction(inferencer, waveform)

    with tab_upload:
        uploaded = st.file_uploader(
            "WAV file (any sample rate; mono is recommended)", type=["wav", "flac", "ogg"]
        )
        if uploaded is not None:
            waveform = _decode_uploaded_wav(uploaded, config.SAMPLE_RATE)
            _render_prediction(inferencer, waveform)

    with tab_continuous:
        st.markdown(
            "Slide the 1-second classifier across a longer recording, smooth "
            "the per-window posteriors, and peak-pick keyword detections — "
            "this is structurally what wake-word detection does in production. "
            "Tune the hop, smoothing, and threshold from the sidebar."
        )

        sub_mic, sub_upload = st.tabs(["Record (local)", "Upload long WAV"])

        with sub_mic:
            st.info(
                "Microphone capture only works locally. On the cloud demo, use "
                "**Upload long WAV** below — it works the same way."
            )
            cont_seconds = st.slider(
                "Recording length (s)",
                min_value=2.0,
                max_value=CONTINUOUS_MAX_SECONDS,
                value=5.0,
                step=0.5,
                key="continuous_seconds",
            )
            if st.button(
                f"Record {cont_seconds:g} second(s) and stream-classify",
                type="primary",
                key="continuous_record_button",
            ):
                with st.spinner(f"Recording {cont_seconds:g}s ..."):
                    waveform = _record_from_mic(cont_seconds, config.SAMPLE_RATE)
                if waveform is not None:
                    classifier = _streaming_classifier(
                        inferencer,
                        hop_ms=hop_ms,
                        ema_alpha=ema_alpha,
                        detection_threshold=detection_threshold,
                        detection_refractory_s=detection_refractory_s,
                    )
                    with st.spinner("Running sliding-window classifier ..."):
                        result = classifier.classify(waveform)
                    _render_streaming_result(waveform, result)

        with sub_upload:
            cont_uploaded = st.file_uploader(
                "Long WAV file — any length, any sample rate (resampled to 16 kHz mono)",
                type=["wav", "flac", "ogg"],
                key="continuous_uploader",
            )
            if cont_uploaded is not None:
                waveform = _decode_uploaded_wav(cont_uploaded, config.SAMPLE_RATE)
                classifier = _streaming_classifier(
                    inferencer,
                    hop_ms=hop_ms,
                    ema_alpha=ema_alpha,
                    detection_threshold=detection_threshold,
                    detection_refractory_s=detection_refractory_s,
                )
                with st.spinner(
                    f"Sliding the classifier across {waveform.shape[0] / config.SAMPLE_RATE:.1f}s "
                    f"of audio ..."
                ):
                    result = classifier.classify(waveform)
                _render_streaming_result(waveform, result)

    with st.expander("Model details"):
        st.json(
            {
                "model_path": str(inferencer.model_path),
                "size_bytes": Path(inferencer.model_path).stat().st_size,
                "labels": list(inferencer.labels),
                "input_shape": (1, *config.INPUT_SHAPE),
                "sample_rate": config.SAMPLE_RATE,
                "frontend": {
                    "n_mels": config.N_MELS,
                    "n_frames": config.N_FRAMES,
                    "win_length_ms": config.WIN_LENGTH_MS,
                    "hop_length_ms": config.HOP_LENGTH_MS,
                },
            }
        )


if __name__ == "__main__":
    main()
