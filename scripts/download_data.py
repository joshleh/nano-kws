"""Explicitly download Speech Commands v0.02 to the local cache.

Mirrors the secom-fault-detection data-fetcher pattern: download to a
deterministic path, verify checksum, fail loudly if anything is off.

Implemented in Phase 1.
"""

from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset is already cached.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    raise NotImplementedError("Phase 1: implement Speech Commands downloader.")


if __name__ == "__main__":
    main()
