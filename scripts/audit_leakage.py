"""Is a high test accuracy real, or is it leakage?

When accuracy climbs above 98% the default reaction should be to audit, not to
celebrate. This script runs four independent checks over a built dataset:

  1. Recording disjointness - no recording may be in more than one split
     (SPEC §5.6).
  2. Window twinning - the highest cosine similarity between a test window and a
     training window. Neighbouring windows from the same recording overlap by
     50%, so a high similarity WITHIN a recording is normal; a high similarity
     ACROSS splits is leakage.

     KNOWN BLIND SPOT, do not read a low number here as "no overlap". Cosine
     similarity is computed on flattened windows, which compares sample k of one
     against sample k of the other. Two windows that share half their samples
     hold them at DIFFERENT positions, so this scores them like two unrelated
     windows. It detects exact duplicates and nothing else, while every real
     case is a partial overlap. Check 1 is the one that actually holds. See
     ROADMAP.md for the replacement: intersect sample-index ranges, not content.
  3. Offset independence - the class x carrier offset contingency per split.
  4. Burst position independence - the same check for the burst start.
     `--balance-by` balances only the field it is given; another field can still
     leak.

Usage:
    python scripts/audit_leakage.py <dataset_dir> [--source examples]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from iqforge.io import load

#: A similarity above this threshold means "the same window" in practice.
DUPLICATE_THRESHOLD = 0.999


def load_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, list[int]]:
    """Load every window of a split into memory."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["splits"][split]
    if not entry["shards"]:
        return np.zeros((0, 0)), []
    arrays = [np.load(dataset_dir / s) for s in entry["shards"]]
    stacked = np.concatenate(arrays, axis=0)
    return stacked.reshape(stacked.shape[0], -1), entry["labels"]


def max_cross_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Highest and mean similarity between two window sets, plus a twin count.

    Returns:
        `(max, mean, twins)`. The twin count is the number of pairs above
        `DUPLICATE_THRESHOLD`; that is the signature of real leakage. A high but
        sub-unity similarity comes from the reference tone every recording
        shares, which carries no class information.
    """
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), 0
    a_unit = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_unit = b / np.linalg.norm(b, axis=1, keepdims=True)
    similarity = np.abs(a_unit @ b_unit.T)
    duplicates = int((similarity > DUPLICATE_THRESHOLD).sum())
    return float(similarity.max()), float(similarity.mean()), duplicates


def within_record_similarity(dataset_dir: Path, split: str) -> float:
    """Highest similarity between windows inside one split, as a reference value.

    The stride is smaller than the window, so neighbouring windows overlap. This
    number sets the scale for judging how high a cross-split similarity is.
    """
    windows, _ = load_split(dataset_dir, split)
    if windows.shape[0] < 2:
        return float("nan")
    unit = windows / np.linalg.norm(windows, axis=1, keepdims=True)
    similarity = np.abs(unit @ unit.T)
    np.fill_diagonal(similarity, 0.0)
    return float(similarity.max())


def contingency_deviation(cells: dict[tuple[str, object], int]) -> float:
    """Largest absolute deviation from independence (0 means fully independent)."""
    labels = sorted({label for label, _ in cells})
    values = sorted({value for _, value in cells}, key=str)
    n = sum(cells.values())
    if n == 0:
        return 0.0
    worst = 0.0
    for label in labels:
        row = sum(cells.get((label, v), 0) for v in values)
        for value in values:
            column = sum(cells.get((lab, value), 0) for lab in labels)
            worst = max(worst, abs(cells.get((label, value), 0) - row * column / n))
    return worst


def _sort_key(value: object) -> tuple[int, float, str]:
    """Sort column headings numerically when possible, alphabetically otherwise.

    Offset headings can carry a suffix such as '+180k', so calling float()
    directly would raise.
    """
    text = str(value)
    try:
        return (0, float(text.rstrip("k")), "")
    except ValueError:
        return (1, 0.0, text)


def print_contingency(title: str, cells: dict[tuple[str, object], int]) -> float:
    """Print a contingency table and return its deviation."""
    labels = sorted({label for label, _ in cells})
    values = sorted({value for _, value in cells}, key=_sort_key)
    print(f"    {title}")
    print("      " + "".join(f"{v:>10}" for v in values) + f"{'total':>10}")
    for label in labels:
        row = "".join(f"{cells.get((label, v), 0):>10d}" for v in values)
        total = sum(cells.get((label, v), 0) for v in values)
        print(f"      {label:<6}{row}{total:>10d}")
    deviation = contingency_deviation(cells)
    print(f"      -> {'INDEPENDENT' if deviation < 1e-9 else f'DEPENDENT (dev {deviation:.2f})'}")
    return deviation


def main() -> None:
    """Run the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--source", type=Path, default=Path("examples"))
    args = parser.parse_args()

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = ("train", "val", "test")

    print("1) RECORDING DISJOINTNESS")
    where: dict[str, list[str]] = defaultdict(list)
    for split in splits:
        for record in manifest["splits"][split]["records"]:
            where[record["id"]].append(split)
    overlapping = {k: v for k, v in where.items() if len(v) > 1}
    print(f"   recordings: {len(where)}")
    print(f"   in more than one split: {overlapping or 'NONE'}")

    print("\n2) WINDOW TWINNING (absolute cosine similarity)")
    train_windows, _ = load_split(args.dataset_dir, "train")
    for split in ("val", "test"):
        windows, _ = load_split(args.dataset_dir, split)
        peak, mean, duplicates = max_cross_similarity(windows, train_windows)
        print(
            f"   {split} vs train : max {peak:.4f}  mean {mean:.4f}  "
            f"twins (>{DUPLICATE_THRESHOLD}): {duplicates}"
        )
    within = within_record_similarity(args.dataset_dir, "train")
    print(f"   within train (overlapping neighbours): max {within:.4f}")
    print(
        "   note: the baseline is high because every window carries the same +100 kHz\n"
        "         reference tone. A twin count > 0 is leakage; a twin count of 0 is\n"
        "         NOT evidence of none -- this check is blind to offset overlap,\n"
        "         which is the form leakage actually takes. See the docstring."
    )

    print("\n3) CLASS x CARRIER OFFSET")
    worst_offset = 0.0
    for split in splits:
        records = manifest["splits"][split]["records"]
        if not records:
            continue
        cells: dict[tuple[str, object], int] = defaultdict(int)
        for record in records:
            cells[(record["label"], f"{record['carrier_offset_hz'] / 1e3:+.0f}k")] += 1
        worst_offset = max(
            worst_offset, print_contingency(f"{split} ({len(records)} recordings)", cells)
        )

    print("\n4) CLASS x BURST START")
    starts: dict[str, int] = {}
    for path in sorted(args.source.glob("*.sigmf-meta")):
        rec = load(path)
        burst = next(a for a in rec.annotations if a.label != "ref_tone")
        starts[path.name] = burst.sample_start

    worst_start = 0.0
    for split in splits:
        records = manifest["splits"][split]["records"]
        if not records:
            continue
        cells = defaultdict(int)
        for record in records:
            cells[(record["label"], str(starts[Path(record["id"]).name]))] += 1
        worst_start = max(
            worst_start, print_contingency(f"{split} ({len(records)} recordings)", cells)
        )

    offset_verdict = "full" if worst_offset < 1e-9 else f"deviation {worst_offset:.2f}"
    start_verdict = "full" if worst_start < 1e-9 else f"deviation {worst_start:.2f}"
    print("\nSUMMARY")
    print(f"   recording leakage        : {'NONE' if not overlapping else 'PRESENT'}")
    print(f"   offset independence      : {offset_verdict}")
    print(f"   burst position independence: {start_verdict}")
    print(
        "\n   Caveat: in splits holding a single recording per class (val/test) the\n"
        "   independence of NO recording-level attribute can be demonstrated - any\n"
        "   field that differs between the two recordings also appears to 'separate'\n"
        "   the classes. At n=2 the contingency table is degenerate, and a deviation\n"
        "   there is not evidence of leakage."
    )


if __name__ == "__main__":
    main()
