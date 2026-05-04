"""Export a trained PyTorch checkpoint to fp32 ONNX.

Used both as the fp32 baseline for the benchmark and as the input to the
INT8 quantization step in :mod:`nano_kws.quantize`.

Implemented in Phase 3.
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--output", required=True, help="Destination .onnx path.")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    raise NotImplementedError("Phase 3: implement fp32 ONNX export.")


if __name__ == "__main__":
    main()
