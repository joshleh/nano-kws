"""Smoke tests for ``scripts/few_shot.py`` rendering + stamping logic.

The training subprocesses themselves are not exercised here (each one
needs the ~2.4 GB Speech Commands cache + several minutes of CPU).
This file only covers the rendering and README-stamping helpers, which
are pure functions over the per-cell results plus the static keyword
split.
"""

from __future__ import annotations

from pathlib import Path

from scripts.few_shot import (
    _BASE_KEYWORDS,
    _NOVEL_KEYWORDS,
    FewShotCell,
    README_BEGIN,
    README_END,
    render_markdown,
    update_readme_section,
)


def _make_cells(*entries: tuple[int, str, float]) -> list[FewShotCell]:
    return [
        FewShotCell(
            samples_per_class=k,
            mode=mode,
            best_val_acc=acc,
            n_train=k * 6,
            n_val=2200,
            epochs=10,
            seconds=20.0,
        )
        for (k, mode, acc) in entries
    ]


def test_base_and_novel_keyword_sets_are_disjoint_and_cover_kwords() -> None:
    """Catch accidental drift between the few-shot split and the canonical
    keyword set."""
    from nano_kws import config

    assert set(_BASE_KEYWORDS).isdisjoint(set(_NOVEL_KEYWORDS))
    assert sorted(_BASE_KEYWORDS + _NOVEL_KEYWORDS) == sorted(config.KEYWORDS)


def test_render_markdown_renders_lift_per_k() -> None:
    cells = _make_cells(
        (10, "from_scratch", 0.20),
        (10, "fine_tuned", 0.45),
        (50, "from_scratch", 0.55),
        (50, "fine_tuned", 0.75),
    )
    md = render_markdown(cells, base_val_acc=0.85, base_epochs=8, base_n_train=18000)

    # Headline row
    assert "K samples / class" in md
    assert "From scratch" in md
    assert "Fine-tuned" in md
    assert "Lift from pretraining" in md

    # K rows present with both columns
    assert "| 10 |" in md
    assert "| 50 |" in md
    assert "20.00%" in md and "45.00%" in md
    assert "55.00%" in md and "75.00%" in md
    # Lifts: 25 pp at K=10, 20 pp at K=50
    assert "+25.00 pp" in md
    assert "+20.00 pp" in md

    # Base-task summary
    assert "85.00%" in md
    assert "8 epochs" in md


def test_render_markdown_skips_rows_that_dont_have_both_modes() -> None:
    cells = _make_cells(
        (10, "fine_tuned", 0.45),
        # missing (10, "from_scratch", _)
        (50, "from_scratch", 0.55),
        (50, "fine_tuned", 0.75),
    )
    md = render_markdown(cells, base_val_acc=0.85, base_epochs=8, base_n_train=18000)
    assert "| 50 |" in md
    assert not any(line.startswith("| 10 |") for line in md.splitlines())


def test_render_markdown_records_negative_lift_when_pretraining_hurts() -> None:
    cells = _make_cells(
        (500, "from_scratch", 0.90),
        (500, "fine_tuned", 0.80),
    )
    md = render_markdown(cells, base_val_acc=0.85, base_epochs=8, base_n_train=18000)
    assert "-10.00 pp" in md


def test_render_markdown_lists_keyword_subsets_in_caption() -> None:
    cells = _make_cells((10, "from_scratch", 0.2), (10, "fine_tuned", 0.4))
    md = render_markdown(cells, base_val_acc=0.85, base_epochs=8, base_n_train=18000)
    for kw in _BASE_KEYWORDS:
        assert kw in md
    for kw in _NOVEL_KEYWORDS:
        assert kw in md


def test_update_readme_section_replaces_block_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "Header.\n\n"
        f"{README_BEGIN}\n_placeholder_\n{README_END}\n\n"
        "Footer.\n",
        encoding="utf-8",
    )
    assert update_readme_section(readme, "| col |\n| --- |\n| ok |\n")
    text = readme.read_text(encoding="utf-8")
    assert "_placeholder_" not in text
    assert "| col |" in text
    assert text.count(README_BEGIN) == 1
    assert text.count(README_END) == 1


def test_update_readme_section_returns_false_when_markers_missing(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("No markers here.\n", encoding="utf-8")
    assert not update_readme_section(readme, "table")


def test_update_readme_section_returns_false_when_file_missing(tmp_path: Path) -> None:
    assert not update_readme_section(tmp_path / "missing.md", "table")
