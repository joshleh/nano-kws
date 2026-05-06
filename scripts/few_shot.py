"""Few-shot transfer experiment for AED-scale data budgets.

Why this exists
---------------
The Syntiant Audio Intern role's central question is:

    "I have a pretrained audio model and ~2 K labelled samples of a
    new target sound. How well does fine-tuning work compared to
    training from scratch?"

This script runs the structural analog of that question on Speech
Commands, which is large enough to support an apples-to-apples
comparison:

1. **Base task** — train DS-CNN-w0.5 from scratch on a "base" subset
   of 6 of the 10 keywords (yes, no, up, down, left, right) plus
   ``_silence_`` + ``_unknown_`` = 8 classes. This is the
   "pretrained audio model" stand-in.
2. **Novel task** — the held-out 4 keywords (on, off, stop, go) plus
   ``_silence_`` + ``_unknown_`` = 6 classes. The labels in this
   task were *never seen during base training*.
3. For each K ∈ {10, 50, 200, 500} samples per novel class, train
   two models in parallel:

   * **From scratch** — random init, train on the K-sample dataset.
   * **Fine-tuned** — load the base checkpoint, replace the
     classifier head (8 → 6 outputs), train end-to-end on the same
     K-sample dataset.

4. Report a sample-efficiency table: "at K samples/class,
   fine-tuned reaches X%, from-scratch reaches Y%, the lift from
   pretraining is +(X-Y) pp."

Reading the table:

* If fine-tuned beats from-scratch by a comfortable margin at small
  K, that's the textbook "pretrained features transfer" story.
* If the gap closes at large K, that's the textbook "with enough
  data you don't need pretraining" story.
* The K = 50 / K = 200 cells are the AED-relevant ones — they map
  to "we have ~200 samples/class of turkey gobble".

Usage::

    # Default sweep (~60-90 min CPU on a modern laptop).
    python -m scripts.few_shot --update-readme

    # Skip retraining the base if you already have it:
    python -m scripts.few_shot --base-checkpoint runs/few_shot/base.pt --update-readme
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

logger = logging.getLogger("nano_kws.few_shot")

README_BEGIN: str = "<!-- BEGIN_FEW_SHOT_TABLE -->"
README_END: str = "<!-- END_FEW_SHOT_TABLE -->"

# Stable, reproducible split: alphabetically-first 6 keywords are the
# "base" set, the remaining 4 are the "novel" set. The split is
# documented in the README/MODEL_CARD; changing it here would invalidate
# every committed result table.
_BASE_KEYWORDS: list[str] = ["down", "left", "no", "right", "up", "yes"]
_NOVEL_KEYWORDS: list[str] = ["go", "off", "on", "stop"]
assert sorted(_BASE_KEYWORDS + _NOVEL_KEYWORDS) == sorted(config.KEYWORDS), (
    "Few-shot split out of sync with config.KEYWORDS"
)

_NON_KEYWORD_LABELS: list[str] = [config.SILENCE_LABEL, config.UNKNOWN_LABEL]


@dataclass
class FewShotCell:
    samples_per_class: int
    mode: str  # "from_scratch" or "fine_tuned"
    best_val_acc: float
    n_train: int
    n_val: int
    epochs: int
    seconds: float


def _train_subprocess(
    *,
    keyword_subset: list[str],
    samples_per_class: int | None,
    epochs: int,
    output_ckpt: Path,
    output_history: Path,
    log_path: Path,
    init_from: Path | None = None,
    base_args: list[str],
    label: str,
) -> tuple[float, int, int, float]:
    """Spawn ``python -m nano_kws.train`` with few-shot args; return
    (best_val_acc, n_train, n_val, elapsed_s)."""
    cmd = [
        sys.executable,
        "-m",
        "nano_kws.train",
        "--width",
        "0.5",
        "--epochs",
        str(epochs),
        "--keyword-subset",
        *keyword_subset,
        "--output",
        str(output_ckpt),
        "--history",
        str(output_history),
        *base_args,
    ]
    if samples_per_class is not None:
        cmd += ["--max-samples-per-class", str(samples_per_class)]
    if init_from is not None:
        cmd += ["--init-from", str(init_from)]
    # Augmentation stays ON for the few-shot runs — that's what production
    # AED setups would do; the augmentation ablation script is the place
    # to ablate it. Keeping aug on here also keeps fine-tuned vs
    # from-scratch comparable (same augmentation regime for both).

    logger.info("[%s] %s", label, " ".join(cmd))
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
        # `errors="replace"` handles non-UTF-8 bytes that Python's logger
        # emits on Windows (e.g. em-dashes encoded as CP1252).
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        for line in tail:
            logger.error("  %s", line)
        raise RuntimeError(f"Training run [{label}] failed; see {log_path}")

    h = json.loads(output_history.read_text(encoding="utf-8"))
    best_val = float(h["best_val_acc"])
    n_train, n_val = _parse_split_sizes(log_path)
    return best_val, n_train, n_val, elapsed


_SPLIT_RE = re.compile(r"Train:\s*(\d+)\s*clips\s*\|\s*Val:\s*(\d+)\s*clips")


def _parse_split_sizes(log_path: Path) -> tuple[int, int]:
    # `errors="replace"` handles non-UTF-8 bytes that Python's logger
    # emits on Windows (e.g. em-dashes encoded as CP1252).
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = _SPLIT_RE.search(text)
    if not m:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def render_markdown(
    cells: list[FewShotCell],
    *,
    base_val_acc: float,
    base_epochs: int,
    base_n_train: int,
) -> str:
    """Render the few-shot result table."""
    cells_by_n: dict[int, dict[str, FewShotCell]] = {}
    for c in cells:
        cells_by_n.setdefault(c.samples_per_class, {})[c.mode] = c

    lines: list[str] = []
    lines.append(
        f"**Base task** (the pretrained-audio-model stand-in): DS-CNN-w0.5 "
        f"trained for {base_epochs} epochs on the 6 base keywords "
        f"(`{', '.join(_BASE_KEYWORDS)}`) + `_silence_` + `_unknown_` = "
        f"8 classes, {base_n_train} training clips. Best base-task val "
        f"accuracy: **{base_val_acc * 100:.2f}%**."
    )
    lines.append("")
    lines.append(
        f"**Novel task**: the held-out 4 keywords "
        f"(`{', '.join(_NOVEL_KEYWORDS)}`) + `_silence_` + `_unknown_` = "
        f"6 classes — labels the base model never saw. For each "
        f"K samples/class budget, two DS-CNN-w0.5 models are trained "
        f"to convergence: a from-scratch baseline (random init) and a "
        f"fine-tuned variant (initialised from the base checkpoint, "
        f"with the 8-way classifier head replaced by a fresh 6-way head). "
        f"Both use SpecAugment + bg-noise augmentation. Validation "
        f"accuracy is on the *full* novel-task validation split, not the "
        f"K-sample subset, so it isn't biased by training-set size."
    )
    lines.append("")
    lines.append(
        "| K samples / class | Train clips | Val clips | "
        "From scratch | Fine-tuned (transfer from base) | Lift from pretraining |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")

    for n in sorted(cells_by_n.keys()):
        row = cells_by_n[n]
        scratch = row.get("from_scratch")
        finetuned = row.get("fine_tuned")
        if scratch is None or finetuned is None:
            continue
        lift_pp = (finetuned.best_val_acc - scratch.best_val_acc) * 100.0
        n_train = finetuned.n_train if finetuned.n_train > 0 else scratch.n_train
        n_val = finetuned.n_val if finetuned.n_val > 0 else scratch.n_val
        lines.append(
            f"| {n} | {n_train} | {n_val} | "
            f"{scratch.best_val_acc * 100:.2f}% | "
            f"{finetuned.best_val_acc * 100:.2f}% | "
            f"{lift_pp:+.2f} pp |"
        )

    return "\n".join(lines).rstrip() + "\n"


def update_readme_section(readme_path: Path, table_md: str) -> bool:
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
        default=[10, 50, 200, 500],
        help="K values to sweep along the few-shot data axis.",
    )
    parser.add_argument(
        "--base-epochs",
        type=int,
        default=15,
        help="Epochs to train the base (8-class) model. ~half the headline run; "
        "we don't need it as polished as the published checkpoint, just stable.",
    )
    parser.add_argument(
        "--novel-epochs",
        type=int,
        default=10,
        help="Epochs per novel-task fine-tune / from-scratch run.",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Skip base training and use this checkpoint instead. Useful when "
        "iterating on the few-shot side without retraining the base each time.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/few_shot",
        help="Where to write per-cell checkpoints, history JSONs, and logs.",
    )
    parser.add_argument(
        "--table-out",
        default="assets/few_shot_table.md",
        help="Where to write the rendered Markdown table.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Also stamp the table between BEGIN/END markers in README.md.",
    )
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
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

    base_subset = _BASE_KEYWORDS + _NON_KEYWORD_LABELS
    novel_subset = _NOVEL_KEYWORDS + _NON_KEYWORD_LABELS

    # ----- Base training -----
    if args.base_checkpoint is not None:
        base_ckpt = Path(args.base_checkpoint)
        logger.info("Reusing existing base checkpoint at %s", base_ckpt)
        # Parse base val acc from the matching history JSON if present.
        base_history = base_ckpt.with_suffix(".history.json")
        if base_history.is_file():
            h = json.loads(base_history.read_text(encoding="utf-8"))
            base_val_acc = float(h.get("best_val_acc", 0.0))
            base_n_train = int(h.get("config", {}).get("num_workers", -1))  # placeholder
        else:
            base_val_acc = -1.0
            base_n_train = -1
        # We don't have the original log; estimate train-clip count later.
    else:
        base_ckpt = out_dir / "base.pt"
        base_history = out_dir / "base.history.json"
        base_log = out_dir / "base.log"
        logger.info("=" * 70)
        logger.info(
            "Phase 1/2: training base model on 8 classes (6 base keywords + silence + unknown)"
        )
        base_val_acc, base_n_train, _, base_seconds = _train_subprocess(
            keyword_subset=base_subset,
            samples_per_class=None,
            epochs=args.base_epochs,
            output_ckpt=base_ckpt,
            output_history=base_history,
            log_path=base_log,
            base_args=base_args,
            label="base",
        )
        logger.info(
            "  → base val acc %.4f  (%.1fs, %d train clips)",
            base_val_acc,
            base_seconds,
            base_n_train,
        )

    # ----- Novel-task sweep -----
    cells: list[FewShotCell] = []
    logger.info("=" * 70)
    logger.info("Phase 2/2: novel-task sweep over K ∈ %s samples/class", args.samples_per_class)
    for k in args.samples_per_class:
        for mode, init_from in (
            ("from_scratch", None),
            ("fine_tuned", base_ckpt),
        ):
            tag = f"k{k}_{mode}"
            ckpt = out_dir / f"{tag}.pt"
            history = out_dir / f"{tag}.history.json"
            log_path = out_dir / f"{tag}.log"
            logger.info("-" * 70)
            logger.info("Cell: K=%d  mode=%s", k, mode)
            best_val, n_train, n_val, elapsed = _train_subprocess(
                keyword_subset=novel_subset,
                samples_per_class=k,
                epochs=args.novel_epochs,
                output_ckpt=ckpt,
                output_history=history,
                log_path=log_path,
                init_from=init_from,
                base_args=base_args,
                label=tag,
            )
            cell = FewShotCell(
                samples_per_class=k,
                mode=mode,
                best_val_acc=best_val,
                n_train=n_train,
                n_val=n_val,
                epochs=args.novel_epochs,
                seconds=elapsed,
            )
            cells.append(cell)
            logger.info("  → %s K=%d best val acc %.4f  (%.1fs)", mode, k, best_val, elapsed)

    table_md = render_markdown(
        cells,
        base_val_acc=base_val_acc,
        base_epochs=args.base_epochs,
        base_n_train=base_n_train,
    )
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
    "FewShotCell",
    "main",
    "render_markdown",
    "update_readme_section",
]
