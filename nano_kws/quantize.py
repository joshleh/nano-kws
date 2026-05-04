"""Static post-training quantization of a DS-CNN to INT8 ONNX.

Pipeline
--------
1. Load the float PyTorch checkpoint and export it to fp32 ONNX
   (delegated to :mod:`nano_kws.export_onnx`).
2. Build a calibration dataset by drawing ``--calibration-batches`` mini
   batches from the training split, featurising them with the canonical
   :class:`LogMelSpectrogram`, and feeding the resulting tensors to
   ONNX Runtime's :func:`quantize_static`.
3. Emit an INT8 ONNX file in the QDQ format (insert
   QuantizeLinear/DequantizeLinear around fp32 ops). QDQ is the format
   most modern edge runtimes (ONNX Runtime, TensorRT, vendor toolchains)
   ingest cleanly, and it's the format the C++ harness in ``cpp/`` will
   target.
4. Verify the INT8 model loads in ONNX Runtime and produces predictions
   that agree with the source PyTorch model in argmax on the calibration
   batches (a soft sanity bound — element-wise agreement is not expected
   from INT8).

Notes on toolchain choice
-------------------------
We use ``onnxruntime.quantization`` rather than PyTorch FX-graph-mode
quantization because the FX → ONNX export path has historically been
fragile for non-trivial models. The ONNX-side toolchain is what
production edge deployments actually use, and it lets the same INT8
artefact serve both Python (Streamlit demo, benchmark) and C/C++
(``cpp/`` harness) consumers without re-quantising.

Quantization-aware training (QAT) is a stretch goal; not in MVP.

Usage::

    python -m nano_kws.quantize \\
        --checkpoint assets/ds_cnn_w0p5.pt \\
        --output     assets/ds_cnn_small_int8.onnx
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch.utils.data import DataLoader

from nano_kws import config
from nano_kws.data.features import LogMelSpectrogram
from nano_kws.data.speech_commands import SpeechCommandsKWS
from nano_kws.evaluate import load_checkpoint
from nano_kws.export_onnx import INPUT_NAME, OUTPUT_NAME, export_to_onnx, write_label_map

logger = logging.getLogger("nano_kws.quantize")


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------


class FeatureCalibrationReader(CalibrationDataReader):
    """Adapt an iterable of pre-featurised numpy batches into the OR API.

    The reader is single-use: ``quantize_static`` walks it once, collecting
    activation statistics. ``get_next`` returns ``None`` to signal the end
    of the stream.
    """

    def __init__(self, batches: Iterable[np.ndarray], input_name: str = INPUT_NAME) -> None:
        self._iter: Iterator[np.ndarray] = iter(batches)
        self._input_name = input_name

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            arr = next(self._iter)
        except StopIteration:
            return None
        return {self._input_name: arr.astype(np.float32, copy=False)}


def synthetic_calibration_batches(
    n_batches: int, batch_size: int = 8, seed: int = 0
) -> list[np.ndarray]:
    """Generate distribution-plausible random log-mel features.

    Used by tests (and as a fallback when the dataset isn't downloaded)
    so the quantization round-trip can be exercised without 2.4 GB of
    audio. The numbers are statistically reasonable (zero-centred,
    moderate variance) so calibration doesn't degenerate.
    """
    rng = np.random.default_rng(seed)
    return [
        rng.normal(loc=-5.0, scale=3.0, size=(batch_size, *config.INPUT_SHAPE)).astype(np.float32)
        for _ in range(n_batches)
    ]


def real_calibration_batches(
    *,
    data_root: Path | None,
    batch_size: int,
    n_batches: int,
    seed: int,
) -> list[np.ndarray]:
    """Materialise ``n_batches`` log-mel batches from the training split.

    The full set is held in memory so :class:`FeatureCalibrationReader` is
    free of dataloader / multi-worker complexity at quantisation time.
    """
    dataset = SpeechCommandsKWS(
        root=data_root,
        subset="training",
        unknown_per_class_ratio=1.0,
        silence_per_class_ratio=1.0,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    featurizer = LogMelSpectrogram().eval()

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for waveforms, _labels in loader:
            features = featurizer(waveforms).cpu().numpy()
            batches.append(features.astype(np.float32, copy=False))
            if len(batches) >= n_batches:
                break
    if len(batches) < n_batches:
        logger.warning(
            "Requested %d calibration batches but the loader only yielded %d.",
            n_batches,
            len(batches),
        )
    return batches


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def quantize_onnx(
    *,
    fp32_path: str | Path,
    int8_path: str | Path,
    calibration_batches: list[np.ndarray],
    per_channel: bool = True,
    calibration_method: CalibrationMethod = CalibrationMethod.MinMax,
) -> Path:
    """Run static PTQ on an existing fp32 ONNX file.

    Returns the resolved INT8 output path.
    """
    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    if not fp32_path.is_file():
        raise FileNotFoundError(f"fp32 ONNX not found: {fp32_path}")
    if not calibration_batches:
        raise ValueError("calibration_batches must contain at least one batch.")

    # Run shape inference + graph optimisations before quantisation. This is
    # the onnxruntime-recommended pre-processing step: it produces a model
    # with all tensor shapes resolved, which lets the quantizer place
    # QuantizeLinear / DequantizeLinear nodes more accurately and avoids
    # the "Please consider to run pre-processing" warnings.
    preprocessed = int8_path.with_suffix(".preprocessed.onnx")
    quant_pre_process(input_model_path=str(fp32_path), output_model_path=str(preprocessed))

    reader = FeatureCalibrationReader(calibration_batches)

    try:
        quantize_static(
            model_input=str(preprocessed),
            model_output=str(int8_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            reduce_range=False,
            calibrate_method=calibration_method,
        )
    finally:
        preprocessed.unlink(missing_ok=True)

    logger.info(
        "Wrote INT8 ONNX to %s (%.1f KB; fp32 was %.1f KB)",
        int8_path,
        int8_path.stat().st_size / 1024,
        fp32_path.stat().st_size / 1024,
    )
    return int8_path


def verify_int8_argmax_agreement(
    *,
    int8_path: str | Path,
    fp32_model: torch.nn.Module,
    batches: list[np.ndarray],
) -> float:
    """Return the fraction of samples where INT8 argmax matches fp32 argmax.

    INT8 will not match fp32 element-wise, but for a well-calibrated
    static PTQ on a small classifier the top-1 prediction should agree
    on the vast majority of inputs.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    fp32_model = fp32_model.cpu().eval()

    n_total = 0
    n_agree = 0
    with torch.no_grad():
        for batch in batches:
            int8_logits = session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0]
            fp32_logits = fp32_model(torch.from_numpy(batch)).numpy()
            int8_argmax = int8_logits.argmax(axis=1)
            fp32_argmax = fp32_logits.argmax(axis=1)
            n_total += int8_argmax.shape[0]
            n_agree += int(np.sum(int8_argmax == fp32_argmax))
    return n_agree / max(1, n_total)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pt produced by nano_kws.train."
    )
    parser.add_argument("--output", required=True, help="Destination INT8 .onnx path.")
    parser.add_argument(
        "--fp32-output",
        default=None,
        help="Where to persist the intermediate fp32 ONNX (default: alongside output).",
    )
    parser.add_argument("--calibration-batches", type=int, default=100)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--data-root", default=None, help="Override config.DATA_DIR.")
    parser.add_argument(
        "--synthetic-calibration",
        action="store_true",
        help="Use synthetic random features instead of Speech Commands "
        "(only for smoke-testing the toolchain; produces a model with poor accuracy).",
    )
    parser.add_argument(
        "--no-per-channel",
        action="store_true",
        help="Disable per-channel weight quantization (use per-tensor).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    int8_path = Path(args.output)
    fp32_path = (
        Path(args.fp32_output)
        if args.fp32_output
        else int8_path.with_name(int8_path.stem.replace("int8", "fp32") + ".onnx")
    )
    if fp32_path == int8_path:  # if int8/fp32 substitution didn't change the name
        fp32_path = int8_path.with_name(f"{int8_path.stem}_fp32.onnx")

    # 1. Load checkpoint and export fp32 ONNX.
    model, ckpt = load_checkpoint(args.checkpoint, device="cpu")
    logger.info(
        "Loaded checkpoint %s (width=%s, val_acc=%.4f)",
        args.checkpoint,
        ckpt["model_config"]["width_multiplier"],
        ckpt.get("val_acc", float("nan")),
    )
    export_to_onnx(model=model, output_path=fp32_path)

    # 2. Build calibration data.
    if args.synthetic_calibration:
        logger.warning("Using SYNTHETIC calibration data — INT8 model will have degraded accuracy.")
        batches = synthetic_calibration_batches(
            n_batches=args.calibration_batches,
            batch_size=args.calibration_batch_size,
            seed=args.seed,
        )
    else:
        logger.info(
            "Materialising %d calibration batches of size %d from Speech Commands ...",
            args.calibration_batches,
            args.calibration_batch_size,
        )
        batches = real_calibration_batches(
            data_root=Path(args.data_root) if args.data_root else None,
            batch_size=args.calibration_batch_size,
            n_batches=args.calibration_batches,
            seed=args.seed,
        )

    # 3. Quantize.
    quantize_onnx(
        fp32_path=fp32_path,
        int8_path=int8_path,
        calibration_batches=batches,
        per_channel=not args.no_per_channel,
    )

    # 4. Sanity-check argmax agreement on the calibration batches.
    agreement = verify_int8_argmax_agreement(
        int8_path=int8_path,
        fp32_model=model,
        batches=batches,
    )
    logger.info("INT8 vs fp32 argmax agreement on calibration batches: %.4f", agreement)

    # 5. Persist the label map alongside the INT8 model.
    label_map_path = int8_path.with_suffix(".label_map.json")
    write_label_map(label_map_path)
    logger.info("Wrote label map to %s", label_map_path)


if __name__ == "__main__":
    main()
