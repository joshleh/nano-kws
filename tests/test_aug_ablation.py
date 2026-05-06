"""Smoke tests for ``scripts/aug_ablation.py`` rendering + stamping logic.

The training subprocess itself is not exercised here (it requires the
~2.4 GB Speech Commands cache and several minutes of CPU per cell).
This file only covers the rendering and README-stamping helpers, which
are pure functions over the per-cell results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.aug_ablation import (
    README_BEGIN,
    README_END,
    AblationCell,
    render_markdown,
    update_readme_section,
)


def _make_cells(*entries: tuple[int, bool, float]) -> list[AblationCell]:
    return [
        AblationCell(
            samples_per_class=n,
            augmentation=aug,
            best_val_acc=acc,
            n_train=n * 12,
            n_val=4443,
            epochs=10,
            seconds=12.3,
        )
        for (n, aug, acc) in entries
    ]


def test_render_markdown_includes_lift_column_for_each_n() -> None:
    cells = _make_cells(
        (50, False, 0.10),
        (50, True, 0.30),
        (200, False, 0.45),
        (200, True, 0.65),
    )
    md = render_markdown(cells)

    # Headline row
    assert "| Samples / class |" in md
    assert "Lift from augmentation" in md

    # Each N appears with both columns and the lift in pp.
    assert "| 50 |" in md
    assert "| 200 |" in md
    assert "10.00%" in md
    assert "30.00%" in md
    assert "45.00%" in md
    assert "65.00%" in md
    assert "+20.00 pp" in md  # 50: 30 - 10
    assert "+20.00 pp" in md  # 200: 65 - 45 (also 20 pp; appears twice)


def test_render_markdown_skips_rows_with_only_one_cell() -> None:
    cells = _make_cells(
        (50, False, 0.10),
        # missing (50, True, _)
        (200, False, 0.45),
        (200, True, 0.65),
    )
    md = render_markdown(cells)
    assert "| 200 |" in md
    # The N=50 row must NOT appear because we don't have both halves.
    lines = [line for line in md.splitlines() if line.startswith("| 50 |")]
    assert lines == []


def test_render_markdown_records_negative_lift_when_aug_hurts() -> None:
    cells = _make_cells(
        (500, False, 0.80),
        (500, True, 0.70),
    )
    md = render_markdown(cells)
    assert "-10.00 pp" in md


def test_update_readme_section_replaces_block_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        f"Before content.\n\n{README_BEGIN}\n_old placeholder_\n{README_END}\n\nAfter content.\n",
        encoding="utf-8",
    )

    table_md = "| col |\n| --- |\n| 1 |\n"
    assert update_readme_section(readme, table_md)

    text = readme.read_text(encoding="utf-8")
    assert "Before content." in text
    assert "After content." in text
    assert "old placeholder" not in text
    assert "| col |" in text
    assert text.count(README_BEGIN) == 1
    assert text.count(README_END) == 1


def test_update_readme_section_returns_false_when_markers_missing(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("Just some content with no markers.\n", encoding="utf-8")
    assert not update_readme_section(readme, "| x |\n| - |\n| 1 |\n")


def test_update_readme_section_returns_false_when_file_missing(tmp_path: Path) -> None:
    assert not update_readme_section(tmp_path / "does_not_exist.md", "table")


def test_render_markdown_caption_mentions_epochs_and_augmentation_kind() -> None:
    cells = _make_cells((50, False, 0.10), (50, True, 0.30))
    md = render_markdown(cells)
    assert "10 epochs" in md
    assert "SpecAugment" in md
    assert "BackgroundNoiseMixer" in md or "bg-noise" in md or "background" in md.lower()


def test_render_markdown_handles_multiple_n_values_in_sorted_order() -> None:
    cells = _make_cells(
        (500, False, 0.80),
        (500, True, 0.85),
        (50, False, 0.10),
        (50, True, 0.30),
        (200, False, 0.45),
        (200, True, 0.65),
    )
    md = render_markdown(cells)
    # The rendered table should list rows in increasing N order.
    pos50 = md.find("| 50 |")
    pos200 = md.find("| 200 |")
    pos500 = md.find("| 500 |")
    assert 0 < pos50 < pos200 < pos500


@pytest.mark.parametrize("acc", [0.0, 1.0])
def test_render_markdown_handles_extreme_accuracies(acc: float) -> None:
    md = render_markdown(_make_cells((50, False, acc), (50, True, acc)))
    assert "0.00 pp" in md  # 0 lift either way
