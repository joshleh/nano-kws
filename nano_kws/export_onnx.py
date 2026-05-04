"""Export a trained PyTorch checkpoint to fp32 ONNX.

The exported ONNX file is the input to two downstream paths:

* :mod:`nano_kws.quantize` runs static PTQ on it to produce the INT8
  ONNX that the live demo and benchmark consume.
* :mod:`nano_kws.benchmark` runs ONNX Runtime against it directly to
  measure the fp32 baseline latency and accuracy.

We export with a dynamic batch axis so a single .onnx file serves both
single-clip inference (the demo) and batched inference (the benchmark).
The exported graph is verified end-to-end by re-loading it in ONNX
Runtime and asserting that predictions match the source PyTorch model
within a tight numerical tolerance.

Usage::

    python -m nano_kws.export_onnx \\
        --checkpoint assets/ds_cnn_w0p5.pt \\
        --output     assets/ds_cnn_small_fp32.onnx
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from nano_kws import config
from nano_kws.evaluate import load_checkpoint

logger = logging.getLogger("nano_kws.export_onnx")

INPUT_NAME: str = "input"
OUTPUT_NAME: str = "logits"
DEFAULT_OPSET: int = 17


def export_to_onnx(
    *,
    model: torch.nn.Module,
    output_path: str | Path,
    opset: int = DEFAULT_OPSET,
    verify_tolerance: float = 1e-4,
) -> Path:
    """Export ``model`` to ``output_path`` and verify the round-trip.

    The model is exported in eval mode with a dynamic batch dimension.
    A single forward pass is then run through ONNX Runtime and compared
    against PyTorch on the same dummy input; if max absolute element-wise
    diff exceeds ``verify_tolerance``, raises ``RuntimeError``.

    Returns the resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu().eval()
    dummy = torch.randn(1, *config.INPUT_SHAPE, dtype=torch.float32)

    # Use the legacy TorchScript-based exporter (`dynamo=False`) rather than
    # the newer dynamo-based one. The TorchScript path has no extra runtime
    # deps (the dynamo path requires `onnxscript`), produces stable graphs
    # for the simple Conv2d / BatchNorm2d / ReLU / Linear ops in DS-CNN, and
    # is what production edge-AI toolchains still consume today.
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
        dynamo=False,
    )
    logger.info("Wrote fp32 ONNX to %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)

    # Verify: ONNX Runtime output must match PyTorch within tolerance.
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run([OUTPUT_NAME], {INPUT_NAME: dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    max_diff = float(np.max(np.abs(onnx_out - torch_out)))
    if max_diff > verify_tolerance:
        raise RuntimeError(
            f"ONNX vs PyTorch max abs diff = {max_diff:.2e} exceeds "
            f"tolerance {verify_tolerance:.2e}. The export is not numerically faithful."
        )
    logger.info(
        "Verified: ONNX vs PyTorch max abs diff = %.2e (tol %.2e)", max_diff, verify_tolerance
    )
    return output_path


def write_label_map(path: str | Path) -> Path:
    """Persist the index -> label mapping alongside the model.

    Downstream inference code (and the C++ harness in cpp/) loads this
    file rather than reaching into nano_kws.config, so the .onnx + .json
    pair is fully self-describing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": list(config.LABELS),
        "num_classes": config.NUM_CLASSES,
        "sample_rate": config.SAMPLE_RATE,
        "input_shape": list(config.INPUT_SHAPE),
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a .pt produced by nano_kws.train."
    )
    parser.add_argument("--output", required=True, help="Destination .onnx path.")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument(
        "--label-map",
        default=None,
        help="Where to write the JSON label map (default: <output>.label_map.json next to the .onnx).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    model, ckpt = load_checkpoint(args.checkpoint, device="cpu")
    logger.info(
        "Loaded checkpoint %s (width=%s, val_acc=%.4f)",
        args.checkpoint,
        ckpt["model_config"]["width_multiplier"],
        ckpt.get("val_acc", float("nan")),
    )

    out = export_to_onnx(model=model, output_path=args.output, opset=args.opset)

    label_map_path = Path(args.label_map) if args.label_map else out.with_suffix(".label_map.json")
    write_label_map(label_map_path)
    logger.info("Wrote label map to %s", label_map_path)


if __name__ == "__main__":
    main()
