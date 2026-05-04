"""Google Speech Commands v0.02 dataset wrapper.

Wraps :class:`torchaudio.datasets.SPEECHCOMMANDS` and remaps the underlying
35-class label space into our 12-class setup:

  * 10 target keywords (see :data:`nano_kws.config.KEYWORDS`)
  * ``_unknown_`` — every other word in the dataset, subsampled to keep the
    class roughly balanced with the keyword classes.
  * ``_silence_`` — synthesised by sampling random 1-second windows from the
    bundled background-noise clips.

Implemented in Phase 1.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """CLI: ``python -m nano_kws.data.speech_commands --sanity``.

    Loads one batch and prints its tensor shape so a fresh clone can verify
    the data pipeline before training.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity", action="store_true", help="Print one-batch sanity check.")
    parser.parse_args()
    raise NotImplementedError("Phase 1: implement Speech Commands loader.")


if __name__ == "__main__":
    main()
