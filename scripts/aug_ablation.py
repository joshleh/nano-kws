"""Augmentation ablation in the low-data regime.

Why this exists
---------------
Real-world Acoustic Event Detection (AED) tasks like turkey-gobble,
glass-break, or smoke-alarm detection typically have only a few
thousand labelled samples per class — the scale where state-of-the-art
work reaches for *generative data augmentation* (EcoGen, BirdDiff,
AudioLDM) to synthesise more training data.

Before reaching for a generative model, the cheap question is:
**how much accuracy does classical augmentation buy you in that
low-data regime?** That's what this script measures.

We run a 2-axis sweep:

* ``samples-per-class`` ∈ {50, 200, 500} — three points along the
  data-scarcity axis. 200/class ≈ 2.4 K total samples, the same
  scale as a representative AED training set.
* ``augmentation`` ∈ {SpecAugment + bg-noise, none} — the existing
  classical-augmentation knob already used by training, toggled via
  ``--no-spec-aug --no-bg-mixer``.

For each combination we run a short fine-tune of DS-CNN-w0.5 from
scratch and record the best val accuracy. The result table is stamped
into the README between BEGIN/END markers.

Reading the table:

* Higher accuracy at the same ``samples-per-class`` with augmentation
  on means classical augmentation is doing useful work.
* The *gap* between the two columns tells you the upper bound of what
  *classical* augmentation can recover. Generative augmentation
  (EcoGen, BirdDiff, AudioLDM) only beats this if it produces data
  with structure that SpecAugment + bg-noise can't synthesise.

Usage::

    # Default sweep (~30-60 min CPU on a modern laptop)
    python -m scripts.aug_ablation --update-readme

    # Smaller / faster smoke
    python -m scripts.aug_ablation --samples-per-class 50 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from nano_kws import config

logger = logging.getLogger("nano_kws.aug_ablation")

README_BEGIN: str = "<!-- BEGIN_AUG_ABLATION_TABLE -->"
README_END: str = "<!-- END_AUG_ABLATION_TABLE -->"


@dataclass
class AblationCell:
    samples_per_class: int
    augmentation: bool
    best_val_acc: float
    n_train: int
    n_val: int
    epochs: int
    seconds: float


def _run_training(
    *,
    samples_per_class: int,
    augmentation: bool,
    epochs: int,
    out_dir: Path,
    base_args: list[str],
) -> AblationCell:
    """Spawn ``python -m nano_kws.train`` for one (N, aug) combination."""
    aug_tag = "aug" if augmentation else "noaug"
    tag = f"n{samples_per_class}_{aug_tag}"
    ckpt = out_dir / f"ds_cnn_w0p5_{tag}.pt"
    history = out_dir / f"ds_cnn_w0p5_{tag}.history.json"

    cmd = [
        sys.executable,
        "-m",
        "nano_kws.train",
        "--width",
        "0.5",
        "--epochs",
        str(epochs),
        "--max-samples-per-class",
        str(samples_per_class),
        "--output",
        str(ckpt),
        "--history",
        str(history),
        *base_args,
    ]
    if not augmentation:
        cmd += ["--no-spec-aug", "--no-bg-mixer"]

    logger.info("Running %s ...  (logs to %s)", " ".join(cmd), out_dir / f"{tag}.log")
    log_path = out_dir / f"{tag}.log"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        # Print last few lines of the log inline so the failure is visible.
        # `errors="replace"` because Python's logger may emit em-dashes etc.
        # encoded as CP1252 on Windows even though we wrote in text mode.
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        for line in tail:
            logger.error("  %s", line)
        raise RuntimeError(
            f"Training run for samples_per_class={samples_per_class} "
            f"augmentation={augmentation} failed; see {log_path}"
        )

    # Pull the result + dataset sizes out of the history JSON.
    h = json.loads(history.read_text(encoding="utf-8"))
    best_val = float(h["best_val_acc"])
    # Dataset sizes are emitted by SpeechCommandsKWS / FilteredKwsDataset
    # at INFO level — not in the history JSON. Parse from the log instead.
    n_train, n_val = _parse_split_sizes(log_path)

    return AblationCell(
        samples_per_class=samples_per_class,
        augmentation=augmentation,
        best_val_acc=best_val,
        n_train=n_train,
        n_val=n_val,
        epochs=epochs,
        seconds=elapsed,
    )


_SPLIT_RE = re.compile(r"Train:\s*(\d+)\s*clips\s*\|\s*Val:\s*(\d+)\s*clips")


def _parse_split_sizes(log_path: Path) -> tuple[int, int]:
    # `errors="replace"` handles non-UTF-8 bytes that Python's logger
    # emits on Windows (e.g. em-dashes encoded as CP1252).
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = _SPLIT_RE.search(text)
    if not m:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def render_markdown(cells: list[AblationCell]) -> str:
    """Render results as a Markdown ablation table."""
    cells_by_n: dict[int, dict[bool, AblationCell]] = {}
    for c in cells:
        cells_by_n.setdefault(c.samples_per_class, {})[c.augmentation] = c

    lines: list[str] = []
    sample_counts = sorted(cells_by_n.keys())
    epochs = cells[0].epochs if cells else 0

    lines.append(
        f"DS-CNN-w0.5 fine-tunes from scratch for **{epochs} epochs** at each "
        f"setting; best validation accuracy across the run is reported. "
        f"Augmentation = SpecAugment (frequency + time masks) + "
        f"`BackgroundNoiseMixer` (5-20 dB SNR). Same model, same optimiser, "
        f"same seed across all cells; only the data budget and the augmentation "
        f"toggle vary."
    )
    lines.append("")
    lines.append(
        "| Samples / class | Train clips | Val clips | No augmentation | "
        "+ SpecAugment + bg-noise | Lift from augmentation |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")

    for n in sample_counts:
        row = cells_by_n[n]
        no_aug = row.get(False)
        with_aug = row.get(True)
        if no_aug is None or with_aug is None:
            continue
        lift_pp = (with_aug.best_val_acc - no_aug.best_val_acc) * 100.0
        n_train = with_aug.n_train if with_aug.n_train > 0 else no_aug.n_train
        n_val = with_aug.n_val if with_aug.n_val > 0 else no_aug.n_val
        lines.append(
            f"| {n} | {n_train} | {n_val} | "
            f"{no_aug.best_val_acc * 100:.2f}% | "
            f"{with_aug.best_val_acc * 100:.2f}% | "
            f"{lift_pp:+.2f} pp |"
        )

    return "\n".join(lines).rstrip() + "\n"


def update_readme_section(readme_path: Path, table_md: str) -> bool:
    """Replace the auto-stamped block between BEGIN/END markers."""
    if not readme_path.is_file():
        return False
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"({re.escape(README_BEGIN)})(.*?)({re.escape(README_END)})",
        flags=re.DOTALL,
    )
    if not pattern.search(text):
        return False
    new_section = f"{README_BEGIN}\n\n{table_md}\n{README_END}"
    new_text = pattern.sub(new_section, text)
    if new_text != text:
        readme_path.write_text(new_text, encoding="utf-8")
        return True
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        nargs="+",
        default=[50, 200, 500],
        help="N values to sweep along the data-scarcity axis.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help=(
            "Epochs per training run. 10 is enough for the low-data regime "
            "to reach a stable best-val; 5 is fine as a smoke."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="runs/aug_ablation",
        help="Where to write per-cell checkpoints, history JSONs, and logs.",
    )
    parser.add_argument(
        "--table-out",
        default="assets/aug_ablation_table.md",
        help="Where to write the rendered Markdown table.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Also stamp the table between BEGIN/END markers in README.md.",
    )
    parser.add_argument("--readme", default="README.md")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Forwarded to nano_kws.train --seed for reproducibility.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Forwarded to nano_kws.train --num-workers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Forwarded to nano_kws.train --batch-size. Smaller default than "
            "the headline 256 because the low-data sweeps would otherwise "
            "have only a handful of batches per epoch."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Forwarded to nano_kws.train --device.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    out_dir = config.REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]

    cells: list[AblationCell] = []
    for n in args.samples_per_class:
        for aug in (False, True):
            logger.info("=" * 70)
            logger.info("Cell: samples_per_class=%d  augmentation=%s", n, aug)
            cell = _run_training(
                samples_per_class=n,
                augmentation=aug,
                epochs=args.epochs,
                out_dir=out_dir,
                base_args=base_args,
            )
            cells.append(cell)
            logger.info("  → best val acc %.4f  (%.1fs)", cell.best_val_acc, cell.seconds)

    table_md = render_markdown(cells)
    table_path = config.REPO_ROOT / args.table_out
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table_md, encoding="utf-8")
    logger.info("Wrote table to %s", table_path)
    print("\n" + table_md)

    if args.update_readme:
        readme_path = config.REPO_ROOT / args.readme
        if update_readme_section(readme_path, table_md):
            logger.info("Stamped table into %s between BEGIN/END markers.", readme_path)
        else:
            logger.warning(
                "Could not find %s / %s markers in %s; README not updated.",
                README_BEGIN,
                README_END,
                readme_path,
            )


if __name__ == "__main__":
    main()


__all__ = [
    "AblationCell",
    "main",
    "render_markdown",
    "update_readme_section",
]
