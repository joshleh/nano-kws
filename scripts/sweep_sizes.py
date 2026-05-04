"""Train DS-CNN at multiple width multipliers and emit a comparison table.

For widths in {0.25, 0.5, 1.0}: train, evaluate, quantize, measure MACs and
disk size. Renders results to ``docs/sweep.md`` and a matplotlib plot at
``docs/accuracy_vs_macs.png``.

Implemented in Phase 5.
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--widths",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0],
        help="Width multipliers to train.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    raise NotImplementedError("Phase 5: implement size sweep.")


if __name__ == "__main__":
    main()
