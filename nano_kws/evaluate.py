"""Evaluate a trained checkpoint on the test split.

Implemented in Phase 2 (basic accuracy) and extended in Phase 4 (per-class
confusion + per-variant comparison).
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    raise NotImplementedError("Phase 2: implement evaluation.")


if __name__ == "__main__":
    main()
