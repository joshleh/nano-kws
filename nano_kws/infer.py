"""Single-clip inference helper.

Designed to be the **only** code path that reads a waveform and produces a
class probability vector at inference time. The Streamlit demo, the test
suite, the benchmark, and the C++ harness (via the exported ONNX) all flow
through equivalent logic.

Implemented in Phase 3 (alongside ONNX export).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class KwsInferencer:
    """Load an ONNX model once, run inference on 1-second waveforms.

    Parameters
    ----------
    model_path
        Path to an ONNX file (fp32 or INT8).
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        raise NotImplementedError("Phase 3: implement KwsInferencer.")

    def predict(self, waveform: np.ndarray) -> np.ndarray:
        """Return a length-NUM_CLASSES softmax probability vector."""
        raise NotImplementedError("Phase 3: implement predict().")
