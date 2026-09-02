"""What the parity gate compares, and what it must refuse to call a pass.

The gate itself re-measures cells and takes hours; these tests cover the
comparison it performs, which is where the migration slipped through. The
previous version of the gate read one recording-level row and one window-level
row out of each artifact and compared those. Every assertion it made was true,
and the grid it approved had been cut from 15 seed pairs to 1.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from parity_gate import compare_cell  # noqa: E402

SPLIT_SEEDS = (42, 7, 1234, 2026, 99)
TRAIN_SEEDS = (0, 1, 2)


def _grid() -> list[dict[str, Any]]:
    """A full published cell: 15 seed pairs, both arms, 30 runs."""
    rows = []
    for strategy, (split, train) in itertools.product(
        ("recording-level", "window-level"), itertools.product(SPLIT_SEEDS, TRAIN_SEEDS)
    ):
        rows.append(
            {
                "strategy": strategy,
                "split_seed": split,
                "train_seed": train,
                "test_accuracy": 0.5 + 0.001 * (split % 7) + (0.03 if strategy[0] == "w" else 0.0),
                "train_accuracy": 0.9,
                "train_windows": 1120,
                "test_windows": 400,
            }
        )
    return rows


def test_an_unchanged_cell_passes() -> None:
    grid = _grid()
    assert compare_cell("cell", grid, _grid()) == []
    assert len(grid) == 30


def test_a_shrunken_grid_is_caught_even_when_every_value_matches() -> None:
    """The migration bug, as a test.

    The first seed pair is reproduced bit-for-bit. Nothing a value comparison
    looks at disagrees. It is still a different measurement, and reporting it
    as the published one would put a standard error of zero on the page.
    """
    recorded = _grid()
    first_pair = [r for r in recorded if r["split_seed"] == 42 and r["train_seed"] == 0]
    assert len(first_pair) == 2

    failures = compare_cell("cell", first_pair, recorded)

    assert failures, "a 2-run grid passed against a 30-run artifact"
    assert any("measured 2 runs" in f and "holds 30" in f for f in failures), failures


def test_the_right_number_of_the_wrong_seeds_is_caught() -> None:
    """Sample size alone is not enough: the seeds have to be the same seeds."""
    recorded = _grid()
    measured = [{**row, "split_seed": row["split_seed"] + 1} for row in recorded]

    failures = compare_cell("cell", measured, recorded)

    assert len(measured) == len(recorded)
    assert any("seed pairs do not match" in f for f in failures), failures


def test_a_single_changed_accuracy_is_caught() -> None:
    """One row out of thirty, matched by seed rather than by position."""
    recorded = _grid()
    measured = [dict(row) for row in recorded]
    measured[17]["test_accuracy"] += 1e-9

    failures = compare_cell("cell", measured, recorded)

    assert len(failures) == 1, failures
    assert "test_accuracy" in failures[0]


def test_rows_are_matched_by_seed_and_not_by_position() -> None:
    """Reordering is not a difference; the gate must not report one."""
    recorded = _grid()
    assert compare_cell("cell", list(reversed(recorded)), recorded) == []


def test_a_selector_that_matches_nothing_is_a_failure() -> None:
    """An empty expectation must not be vacuously satisfied.

    A mistyped cell selector returns no recorded rows, and every per-row
    comparison then trivially passes. That is the shape of a gate that reports
    success while checking nothing.
    """
    failures = compare_cell("cell", _grid(), [])

    assert failures, "an empty artifact selection was reported as a pass"
    assert "selector" in failures[0]
