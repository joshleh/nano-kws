"""Train a single DS-CNN configuration.

Usage:
    python -m nano_kws.train --width 0.5 --epochs 30

Implemented in Phase 2.
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=float, default=0.5, help="DS-CNN width multiplier.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="assets/ds_cnn_small.pt",
        help="Where to write the trained checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    raise NotImplementedError("Phase 2: implement training loop.")


if __name__ == "__main__":
    main()
