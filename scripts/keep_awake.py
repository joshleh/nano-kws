"""Prevent the host (Windows) from sleeping while a long job runs.

Usage:

    python -m scripts.keep_awake                # block forever, prints heartbeats
    python -m scripts.keep_awake --duration 8h  # auto-exit after 8 hours
    python -m scripts.keep_awake --quiet        # no heartbeat prints

On Windows this calls ``SetThreadExecutionState`` with
``ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED | ES_CONTINUOUS`` so the OS
treats the calling thread as user-active. The flag clears as soon as
the process exits, so you can safely ``Ctrl-C`` to release the hold.

On macOS / Linux this is a no-op (prints a warning) — run the long job
inside ``caffeinate`` or under ``systemd-inhibit`` instead.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time

logger = logging.getLogger("nano_kws.keep_awake")

# Win32 SetThreadExecutionState constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040


def _engage_windows() -> bool:
    """Set the execution state. Returns True on success, False otherwise."""
    if not sys.platform.startswith("win"):
        return False
    import ctypes

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    prev = ctypes.windll.kernel32.SetThreadExecutionState(flags)
    return prev != 0


def _release_windows() -> None:
    if not sys.platform.startswith("win"):
        return
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def _parse_duration(s: str) -> float:
    """Parse '8h' / '30m' / '90s' / '3600' (seconds) into seconds (float)."""
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"can't parse duration: {s!r}")
    value, unit = float(m.group(1)), m.group(2)
    return value * {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--duration",
        type=_parse_duration,
        default=None,
        help="Auto-release after this long. Accepts '8h' / '30m' / '90s' / raw seconds. "
        "Default: run until killed.",
    )
    parser.add_argument(
        "--heartbeat",
        type=_parse_duration,
        default=300.0,
        help="Seconds between status prints. Default: 300 (5 min).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress heartbeat prints.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    if not sys.platform.startswith("win"):
        logger.warning(
            "keep_awake is a no-op on %s. Use `caffeinate` (macOS) or "
            "`systemd-inhibit` (Linux) to wrap your long job.",
            sys.platform,
        )
        return 0

    if not _engage_windows():
        logger.error("SetThreadExecutionState returned 0 (failure). Aborting.")
        return 1

    deadline = None if args.duration is None else time.monotonic() + args.duration
    logger.info(
        "Sleep prevention active (system + display). Heartbeat every %.0fs. %s",
        args.heartbeat,
        f"Auto-release in {args.duration / 3600:.2f}h." if args.duration else "Run until killed.",
    )

    try:
        while True:
            time.sleep(args.heartbeat)
            if not args.quiet:
                elapsed = time.strftime("%H:%M:%S", time.localtime())
                logger.info("still awake @ %s", elapsed)
            if deadline is not None and time.monotonic() >= deadline:
                logger.info("Duration elapsed; releasing.")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted; releasing.")
    finally:
        _release_windows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
