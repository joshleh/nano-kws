"""Evaluate a trained checkpoint on the test split.

Loads a checkpoint produced by :mod:`nano_kws.train`, reconstructs the
model from the embedded ``model_config``, runs inference on the test
split, and reports overall top-1 accuracy plus per-class accuracy and a
confusion matrix.

Usage::

    python -m nano_kws.evaluate --checkpoint assets/ds_cnn_w0p5.pt
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nano_kws import config
from nano_kws.data.features import LogMelSpectrogram
from nano_kws.data.speech_commands import SpeechCommandsKWS
from nano_kws.models.ds_cnn import build_ds_cnn

logger = logging.getLogger("nano_kws.evaluate")


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Reconstruct a DS-CNN from a checkpoint produced by :mod:`nano_kws.train`.

    Returns ``(model, checkpoint_dict)``. The model is on ``device`` and in
    eval mode.
    """
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model_cfg = ckpt["model_config"]
    model = build_ds_cnn(
        num_classes=model_cfg.get("num_classes", config.NUM_CLASSES),
        width_multiplier=model_cfg["width_multiplier"],
        n_blocks=model_cfg.get("n_blocks", 4),
        base_channels=model_cfg.get("base_channels", 224),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_dataset(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    featurizer: nn.Module | None = None,
) -> dict[str, Any]:
    """Run inference and return overall + per-class accuracy and a confusion matrix."""
    if featurizer is None:
        featurizer = LogMelSpectrogram().to(device)
    featurizer.eval()
    model.eval()

    n_total = 0
    n_correct = 0
    per_class_total: Counter[int] = Counter()
    per_class_correct: Counter[int] = Counter()
    confusion = torch.zeros(config.NUM_CLASSES, config.NUM_CLASSES, dtype=torch.long)

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = featurizer(waveforms)
        preds = model(features).argmax(dim=1)

        n_total += labels.size(0)
        n_correct += (preds == labels).sum().item()
        for label, pred in zip(labels.cpu().tolist(), preds.cpu().tolist(), strict=True):
            per_class_total[label] += 1
            per_class_correct[label] += int(label == pred)
            confusion[label, pred] += 1

    return {
        "n_total": n_total,
        "top1_accuracy": n_correct / max(1, n_total),
        "per_class_accuracy": {
            i: per_class_correct[i] / max(1, per_class_total[i]) for i in range(config.NUM_CLASSES)
        },
        "per_class_total": dict(per_class_total),
        "confusion_matrix": confusion.tolist(),
    }


def _format_per_class_table(result: dict[str, Any]) -> str:
    lines = [f"  {'idx':>3s} {'label':>10s} {'n':>6s} {'acc':>8s}"]
    for i in range(config.NUM_CLASSES):
        n = result["per_class_total"].get(i, 0)
        acc = result["per_class_accuracy"][i]
        lines.append(f"  {i:>3d} {config.INDEX_TO_LABEL[i]:>10s} {n:>6d} {acc:>8.4f}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a .pt produced by nano_kws.train."
    )
    parser.add_argument("--subset", choices=("validation", "testing"), default="testing")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--data-root", default=None, help="Override config.DATA_DIR.")
    parser.add_argument(
        "--silence-per-class-ratio",
        type=float,
        default=1.0,
        help="Match the value used at training time so per-class counts are comparable.",
    )
    parser.add_argument("--unknown-per-class-ratio", type=float, default=1.0)
    return parser.parse_args(argv)


def _pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    device = _pick_device(args.device)
    logger.info("Device: %s", device)

    model, ckpt = load_checkpoint(args.checkpoint, device=device)
    logger.info(
        "Loaded %s | width=%s | best train val_acc=%.4f at epoch %s",
        args.checkpoint,
        ckpt["model_config"]["width_multiplier"],
        ckpt.get("val_acc", float("nan")),
        ckpt.get("epoch", "?"),
    )

    data_root = Path(args.data_root) if args.data_root else None
    dataset = SpeechCommandsKWS(
        root=data_root,
        subset=args.subset,
        unknown_per_class_ratio=args.unknown_per_class_ratio,
        silence_per_class_ratio=args.silence_per_class_ratio,
        seed=ckpt["train_config"].get("seed", 0) if "train_config" in ckpt else 0,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    result = evaluate_dataset(model=model, loader=loader, device=device)
    print(
        f"\n{args.subset} accuracy (top-1): {result['top1_accuracy']:.4f}  ({result['n_total']} clips)"
    )
    print("\nPer-class accuracy:")
    print(_format_per_class_table(result))


if __name__ == "__main__":
    main()
