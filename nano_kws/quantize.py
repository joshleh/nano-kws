"""Static post-training quantization of a trained DS-CNN to INT8 ONNX.

Approach (Phase 3):

1. Load the float checkpoint and run FX-graph-mode static PTQ
   (``torch.ao.quantization.quantize_fx``).
2. Calibrate on a fixed number of batches drawn from the training set.
3. Export the quantized model to ONNX (INT8 weights and activations).
4. Sanity-check that ONNX Runtime predictions agree with the in-PyTorch
   quantized model to within a small tolerance.

Quantization-aware training (QAT) is a stretch goal; not in MVP.
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Float checkpoint to quantize.")
    parser.add_argument("--output", required=True, help="Destination INT8 .onnx path.")
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=100,
        help="Number of training batches used for activation calibration.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    raise NotImplementedError("Phase 3: implement static PTQ + INT8 ONNX export.")


if __name__ == "__main__":
    main()
