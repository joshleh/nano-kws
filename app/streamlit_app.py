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
import soundfile as sf
import streamlit as st

from nano_kws import config
from nano_kws.data.features import LogMelSpectrogram, pad_or_crop, waveform_to_logmel
from nano_kws.infer import KwsInferencer

DEFAULT_MODEL = config.ASSETS_DIR / "ds_cnn_small_int8.onnx"
RECORD_SECONDS = config.CLIP_DURATION_S


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading INT8 model ...")
def _load_inferencer(model_path: str) -> KwsInferencer:
    return KwsInferencer(model_path=model_path)


@st.cache_resource
def _featurizer() -> LogMelSpectrogram:
    return LogMelSpectrogram().eval()


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
    waveform = pad_or_crop(waveform).astype(np.float32, copy=False)

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


def main() -> None:
    st.set_page_config(page_title="nano-kws — live demo", page_icon=None, layout="wide")
    st.title("nano-kws — INT8 keyword spotter")
    st.write(
        "Live demo of the bundled DS-CNN INT8 model. "
        "Speak one of the 10 keywords (`yes / no / up / down / left / right / on / off / stop / go`) "
        "or upload a 16 kHz mono WAV. The model is < 100 KB on disk and inference runs on a "
        "single CPU thread via ONNX Runtime."
    )

    model_path, seconds = _render_sidebar()
    inferencer = _load_inferencer(model_path)

    tab_mic, tab_upload = st.tabs(["Microphone", "Upload WAV"])

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
