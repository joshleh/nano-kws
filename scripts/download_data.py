"""Explicitly download Speech Commands v0.02 to the local cache.

Delegates to ``torchaudio.datasets.SPEECHCOMMANDS(download=True)`` and
then verifies the cache directory layout. The actual checksum is
enforced upstream by torchaudio's ``download_url_to_file`` call.

Usage::

    python -m scripts.download_data            # download to config.DATA_DIR
    python -m scripts.download_data --root /tmp/sc

This is a one-time ~2.4 GB fetch. Subsequent calls are no-ops unless
``--force`` is passed.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from torchaudio.datasets import SPEECHCOMMANDS

from nano_kws import config

logger = logging.getLogger("download_data")

_BACKGROUND_FOLDER = "_background_noise_"
_VALIDATION_LIST = "validation_list.txt"
_TESTING_LIST = "testing_list.txt"


def _verify_layout(archive_root: Path) -> None:
    """Sanity-check that all expected pieces of the archive are present."""
    missing: list[str] = []
    if not (archive_root / _BACKGROUND_FOLDER).is_dir():
        missing.append(f"{_BACKGROUND_FOLDER}/ (silence synthesis source)")
    for name in (_VALIDATION_LIST, _TESTING_LIST):
        if not (archive_root / name).is_file():
            missing.append(name)
    keyword_dirs = [d for d in archive_root.iterdir() if d.is_dir() and d.name in config.KEYWORDS]
    if len(keyword_dirs) != len(config.KEYWORDS):
        present = {d.name for d in keyword_dirs}
        missing.extend(f"{kw}/ (keyword dir)" for kw in config.KEYWORDS if kw not in present)
    if missing:
        raise RuntimeError(
            f"Cache at {archive_root} is incomplete:\n  - "
            + "\n  - ".join(missing)
            + "\nRe-run with --force to re-download."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(config.DATA_DIR),
        help="Where to download the dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing cache and re-download.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    archive_root = root / "SpeechCommands" / "speech_commands_v0.02"

    if args.force and archive_root.exists():
        logger.info("--force: removing existing cache at %s", archive_root)
        shutil.rmtree(archive_root)

    if archive_root.is_dir():
        logger.info("Cache already exists at %s; skipping download.", archive_root)
    else:
        logger.info("Downloading Speech Commands v0.02 to %s (this is ~2.4 GB) ...", root)
        SPEECHCOMMANDS(root=str(root), download=True)
        logger.info("Download complete.")

    _verify_layout(archive_root)
    n_keywords = sum(1 for _ in archive_root.glob("*/*.wav"))
    n_bg = sum(1 for _ in (archive_root / _BACKGROUND_FOLDER).glob("*.wav"))
    logger.info(
        "Cache OK: %s | %d keyword/utterance WAVs | %d background-noise WAVs",
        archive_root,
        n_keywords,
        n_bg,
    )


if __name__ == "__main__":
    sys.exit(main())
