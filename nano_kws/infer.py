"""Single-clip inference helper.

This is the **only** code path that reads a 1-second waveform and
produces a class-probability vector at inference time. The Streamlit
demo, the test suite, and the benchmark all flow through it; the C++
harness in ``cpp/`` mirrors the same pipeline against the same exported
ONNX so the two implementations stay in lockstep.

Pipeline::

    np.ndarray waveform (any length, any dtype)
        --> pad_or_crop  (-> CLIP_SAMPLES, float32)
        --> LogMelSpectrogram  (-> (1, N_MELS, N_FRAMES) float32)
        --> ONNX Runtime InferenceSession
        --> softmax
        --> length-NUM_CLASSES probability vector

The pre/post-processing has zero ONNX-side state, so swapping the
underlying model (fp32 -> INT8) is just a constructor argument.

Implemented in Phase 3 alongside ONNX export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from nano_kws import config
from nano_kws.data.features import LogMelSpectrogram, pad_or_crop

logger = logging.getLogger("nano_kws.infer")

DEFAULT_INPUT_NAME: str = "input"
DEFAULT_OUTPUT_NAME: str = "logits"

# ─── Design note: ONNX Runtime as the inference backend ─────────────────
# Three reasons, in order of importance:
#   1. Portability. The same ONNX file that the Python demo loads is what
#      the C++ harness in cpp/infer.cpp consumes (also via ORT) and what a
#      vendor toolchain would ingest as the source format for a hardware
#      compile. Standardizing on ONNX means we don't have to reason about
#      "which serialization does which deployment target like" — there's
#      one model file, one set of input/output names, one tensor layout.
#   2. INT8 support. ORT's CPU EP has solid QDQ kernels and is what
#      onnxruntime.quantization quantizes *for* — so the calibration we
#      ran in nano_kws.quantize is exactly tuned for this runtime.
#   3. CPUExecutionProvider only, deliberately. We force CPU EP so the
#      benchmark numbers and the demo behavior are identical across
#      machines (no "fast on my GPU box, slow in CI"). A real edge
#      deployment would swap the EP for a vendor-specific one (CoreML,
#      QNN, etc.) — this code path doesn't change.
# ───────────────────────────────────────────────────────────────────────────

# ─── Design note: why softmax in Python rather than baked into the graph? ──
# The exported model emits raw logits, not probabilities. Reasons:
#   * Numerical stability — the subtract-max trick below is easier to
#     reason about and audit in 4 lines of NumPy than buried inside the
#     graph. Quantization of the softmax op is finicky (exponentials over
#     wide ranges) and we'd rather keep it out of the INT8 path.
#   * A consumer that wants top-k or just argmax doesn't need to compute
#     the exponentials at all; raw logits give them that flexibility.
#   * The C++ harness and the Streamlit demo can share this exact softmax
#     implementation by porting these 4 lines, which is what infer.cpp does.
# Trade-off: the model isn't "self-contained" — every consumer must apply
# softmax. For a 12-class classifier with one consumer pattern (top-1) that's
# trivial; for larger output distributions you might bake it in.
# ─────────────────────────────────────────────────────────────────────────────


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


class KwsInferencer:
    """Load an ONNX model once, run KWS inference on 1-second waveforms.

    Parameters
    ----------
    model_path
        Path to an ONNX file (fp32 or INT8). Both work; the calling code
        does not need to know which it loaded.
    label_map_path
        Optional path to the JSON label map written next to the model.
        If unset, falls back to :data:`nano_kws.config.LABELS`.
    providers
        ONNX Runtime execution providers, in priority order. Defaults to
        ``["CPUExecutionProvider"]`` so behaviour is identical across
        machines and tests.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        label_map_path: str | Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self._featurizer = LogMelSpectrogram().eval()

        # Cache the input/output names from the session so we work with
        # any opset / export pipeline that names tensors differently.
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Label map: prefer the JSON sidecar, fall back to config.LABELS.
        if label_map_path is None:
            sidecar = self.model_path.with_suffix(".label_map.json")
            label_map_path = sidecar if sidecar.is_file() else None
        if label_map_path is not None:
            payload = json.loads(Path(label_map_path).read_text())
            self.labels: tuple[str, ...] = tuple(payload["labels"])
            if len(self.labels) != config.NUM_CLASSES:
                logger.warning(
                    "Label map has %d entries but config.NUM_CLASSES is %d; using sidecar.",
                    len(self.labels),
                    config.NUM_CLASSES,
                )
        else:
            self.labels = config.LABELS

        logger.info(
            "Loaded %s (input=%s, output=%s, %d labels)",
            self.model_path.name,
            self._input_name,
            self._output_name,
            len(self.labels),
        )

    # ------------------------------------------------------------------
    # Inference entry points
    # ------------------------------------------------------------------

    def featurize(self, waveform: np.ndarray | torch.Tensor) -> np.ndarray:
        """Apply pad/crop + log-mel; returns a ``(1, N_MELS, N_FRAMES)`` array."""
        wav = pad_or_crop(waveform)
        with torch.no_grad():
            features = self._featurizer(wav)
        return features.unsqueeze(0).numpy() if features.dim() == 3 else features.numpy()

    def predict(self, waveform: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return a length-``NUM_CLASSES`` softmax probability vector."""
        features = self.featurize(waveform)
        logits = self._session.run([self._output_name], {self._input_name: features})[0]
        return _softmax(logits)[0]

    def predict_label(self, waveform: np.ndarray | torch.Tensor) -> tuple[str, float]:
        """Return ``(top_1_label, top_1_probability)`` for the waveform."""
        probs = self.predict(waveform)
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    def predict_batch(self, waveforms: np.ndarray | torch.Tensor) -> np.ndarray:
        """Batched variant: ``(B, samples)`` waveforms -> ``(B, NUM_CLASSES)`` probs."""
        wav = pad_or_crop(waveforms)
        with torch.no_grad():
            features = self._featurizer(wav).numpy()
        if features.ndim == 3:
            features = features[None]
        logits = self._session.run([self._output_name], {self._input_name: features})[0]
        return _softmax(logits)


__all__ = ["KwsInferencer"]
