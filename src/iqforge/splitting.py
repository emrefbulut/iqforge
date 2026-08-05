"""Stratified, RECORDING-LEVEL train/val/test splitting (SPEC §5.6).

This module has one rule: windows from the same recording go to the same split.
Splitting at the window level puts neighbouring windows in both train and test,
which inflates test accuracy.

If the rule cannot be honoured, iqforge does NOT quietly fall back to
window-level splitting — it raises.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from iqforge.io import IQForgeError

#: Split names, in manifest order.
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitPlan:
    """Which recording goes to which split.

    Attributes:
        assignment: Recording id -> split name.
        ratios: The ratios used.
        seed: The seed used.
        groups: Recording id -> nuisance-variable group when balancing was used;
            empty otherwise.
    """

    assignment: dict[str, str]
    ratios: tuple[float, float, float]
    seed: int
    groups: dict[str, str] = field(default_factory=dict)

    def records_in(self, split: str) -> list[str]:
        """Return the recording ids in a split, sorted."""
        return sorted(rid for rid, name in self.assignment.items() if name == split)


def parse_ratios(text: str) -> tuple[float, float, float]:
    """Parse a ratio string of the form `0.7,0.15,0.15`.

    Raises:
        IQForgeError: If the format is wrong, a value is negative, or the values
            do not sum to 1.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise IQForgeError(
            f"--split expects three values (train,val,test), got {len(parts)}: '{text}'. "
            "Example: --split 0.7,0.15,0.15"
        )
    try:
        values = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise IQForgeError(
            f"--split values must be numeric: '{text}'. Example: --split 0.7,0.15,0.15"
        ) from exc

    if any(v < 0 for v in values):
        raise IQForgeError(f"--split values cannot be negative: '{text}'.")
    if abs(sum(values) - 1.0) > 1e-6:
        raise IQForgeError(f"--split values must sum to 1, got {sum(values):.6g}.")
    return values  # type: ignore[return-value]


def _allocate(n: int, ratios: tuple[float, float, float], active: list[int]) -> list[int]:
    """Share `n` recordings out by ratio, giving every active split at least one.

    Standard largest-remainder allocation first, then a transfer from the
    fullest split to any active split left empty.

    Applying the minimum afterwards rather than up front matters: reserving one
    recording per split first systematically inflates the small splits (with 10
    recordings and 0.7/0.15/0.15 that gives 6/2/2 instead of the correct 7/2/1).
    """
    exact = [n * ratios[i] if i in active else 0.0 for i in range(3)]
    counts = [int(value) for value in exact]

    order = sorted(active, key=lambda i: (-(exact[i] - int(exact[i])), i))
    for k in range(n - sum(counts)):
        counts[order[k % len(order)]] += 1

    for i in active:
        if counts[i] == 0:
            donor = max(active, key=lambda j: (counts[j], -j))
            counts[donor] -= 1
            counts[i] += 1
    return counts


def _shuffled(values: list[str], rng: np.random.Generator) -> list[str]:
    """Shuffle deterministically; the input is sorted first so the result does
    not depend on the caller's ordering."""
    ordered = sorted(values)
    return [ordered[i] for i in rng.permutation(len(ordered))]


def _assign_balanced(
    by_label: dict[str, list[str]],
    groups: dict[str, str],
    ratios: tuple[float, float, float],
    active: list[int],
    rng: np.random.Generator,
) -> dict[str, str]:
    """Do the stratified split while spreading the nuisance variable across splits.

    The per-class recording count of each split is fixed by `_allocate`, so the
    stratification is untouched. What changes is WHICH recording goes where.

    Goal: **inside a split, group and label must be independent.** Assignment
    therefore works in "rounds": a round is one recording of every class from a
    single group. A round goes to one split as a unit, so the group cannot give
    the label away within that split.

    This rule is essential, because the opposite is a disaster: if groups are
    handed out complementarily between classes (bpsk taking the positive offsets
    in train, qpsk the negative ones), the group predicts the label exactly
    WITHIN a split. The model learns that shortcut, scores 100% on training, and
    collapses to 0% on a test split where the relationship is reversed — below
    chance.

    Rounds are processed group by group in rotation, and each round goes to the
    split with the largest PROPORTIONAL deficit (not the largest absolute
    capacity). With absolute capacity, train fills up completely before any
    group reaches val or test, so no group is ever shared between train and the
    evaluation splits and evaluation always happens on unseen groups.

    If a group holds recordings of only one class, no round can be formed; the
    leftovers go to the emptiest split for their label and `leakage_warnings`
    reports the situation.
    """
    targets: dict[tuple[str, int], int] = {}
    for label, records in by_label.items():
        counts = _allocate(len(records), ratios, active)
        for i in active:
            targets[(label, i)] = counts[i]
    remaining = dict(targets)

    # cell[group][label] = recordings in that group carrying that label
    cell: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for label, records in by_label.items():
        for record_id in records:
            cell[groups[record_id]].setdefault(label, []).append(record_id)
    for buckets in cell.values():
        for label in buckets:
            buckets[label] = _shuffled(buckets[label], rng)

    def _deficit(split: int, labels: list[str]) -> float:
        """Fraction of the split's target that is still unfilled."""
        total = sum(targets[(label, split)] for label in labels)
        left = sum(remaining[(label, split)] for label in labels)
        return left / total if total else 0.0

    assignment: dict[str, str] = {}
    rotation = _shuffled(list(cell), rng)
    while any(any(cell[group].values()) for group in rotation):
        for group in rotation:
            buckets = cell[group]
            if not any(buckets.values()):
                continue
            present = [label for label, records in buckets.items() if records]
            rounds = {i: min(remaining[(label, i)] for label in present) for i in active}
            candidates = [i for i in active if rounds[i] > 0]

            if candidates:
                best = max(candidates, key=lambda i: (_deficit(i, present), rounds[i], -i))
                for label in present:
                    record_id = buckets[label].pop()
                    assignment[record_id] = SPLIT_NAMES[best]
                    remaining[(label, best)] -= 1
                continue

            # A whole round does not fit: place the rest per label, emptiest split first.
            for label in present:
                record_id = buckets[label].pop()
                target = max(
                    (i for i in active if remaining[(label, i)] > 0),
                    key=lambda i: (remaining[(label, i)], -i),
                )
                assignment[record_id] = SPLIT_NAMES[target]
                remaining[(label, target)] -= 1

    return assignment


def stratified_record_split(
    record_labels: dict[str, str],
    ratios: tuple[float, float, float],
    seed: int,
    record_groups: dict[str, str] | None = None,
) -> SplitPlan:
    """Distribute recordings across splits, stratified by label.

    The split is at the RECORDING level: every window of a recording lands in
    the same split. The same `seed` reproduces the result exactly.

    If `record_groups` is given, the nuisance variable is also spread across the
    splits while the stratification is preserved (see `--balance-by`).

    Args:
        record_labels: Recording id -> that recording's dominant label.
        ratios: `(train, val, test)` ratios.
        seed: Seed for deterministic shuffling.
        record_groups: Recording id -> the group value to balance.

    Returns:
        The recording assignment.

    Raises:
        IQForgeError: If a class has too few recordings to fill the non-empty
            splits the ratios ask for.
    """
    if not record_labels:
        raise IQForgeError("Nothing to split: no labelled windows were found in the input.")

    active = [i for i, r in enumerate(ratios) if r > 0]
    by_label: dict[str, list[str]] = defaultdict(list)
    for record_id, label in record_labels.items():
        by_label[label].append(record_id)

    for label in sorted(by_label):
        available = len(by_label[label])
        if available < len(active):
            needed = ", ".join(f"{SPLIT_NAMES[i]}={ratios[i]:g}" for i in active)
            raise IQForgeError(
                f"Cannot stratify by recording: class '{label}' has only {available} "
                f"recording{'' if available == 1 else 's'}, but a "
                f"{'/'.join(f'{r:g}' for r in ratios)} split needs at least "
                f"{len(active)} ({needed}).\n\n"
                "Windows from one recording must not appear in more than one split "
                "(SPEC §5.6); falling back to window-level splitting inflates test "
                "accuracy.\n\n"
                "Options:\n"
                "  - provide more recordings per class (pass a directory)\n"
                "  - reduce the split ratios, e.g. --split 0.5,0.25,0.25\n"
                "  - build a training set only: --split 1.0,0,0"
            )

    rng = np.random.default_rng(seed)

    if record_groups is not None:
        assignment = _assign_balanced(by_label, record_groups, ratios, active, rng)
        return SplitPlan(
            assignment=assignment, ratios=ratios, seed=seed, groups=dict(record_groups)
        )

    assignment = {}
    for label in sorted(by_label):
        shuffled = _shuffled(by_label[label], rng)
        counts = _allocate(len(shuffled), ratios, active)
        cursor = 0
        for split_index in range(3):
            for record_id in shuffled[cursor : cursor + counts[split_index]]:
                assignment[record_id] = SPLIT_NAMES[split_index]
            cursor += counts[split_index]

    return SplitPlan(assignment=assignment, ratios=ratios, seed=seed)


def balance_warnings(plan: SplitPlan, field_name: str) -> list[str]:
    """Check how well balancing worked and return warning texts.

    Balancing can be structurally impossible — for instance when there are more
    groups than the smallest split can hold. That is not an ERROR, the split is
    still valid, but the user should know because the residual skew can affect
    the results.

    Returns:
        Warning texts; an empty list when there is nothing to report.
    """
    if not plan.groups:
        return []

    warnings: list[str] = []
    distinct = sorted(set(plan.groups.values()))
    if len(distinct) < 2:
        only = distinct[0] if distinct else "-"
        return [
            f"--balance-by '{field_name}' produced a single group value ({only}); "
            "balancing had no effect."
        ]
    if len(distinct) == len(plan.groups):
        warnings.append(
            f"--balance-by '{field_name}' gave every recording its own group "
            f"({len(distinct)} groups / {len(plan.groups)} recordings); balancing is "
            "meaningless. Pick a coarser field."
        )

    return warnings


def leakage_warnings(plan: SplitPlan, record_labels: dict[str, str], field_name: str) -> list[str]:
    """Check whether the group gives the label away WITHIN a split.

    This is the dangerous case: if each class falls into different groups inside
    one split, the group predicts the label exactly. The model learns that
    shortcut, and when the relationship changes in another split accuracy drops
    below chance.

    Also reports whether evaluation happens on groups never seen in training.
    That is not an error, but it should be known when reading the numbers.

    Returns:
        Warning texts; an empty list when there is nothing to report.
    """
    if not plan.groups:
        return []

    warnings: list[str] = []
    for name in SPLIT_NAMES:
        records = plan.records_in(name)
        if len(records) < 2:
            continue
        by_label: dict[str, set[str]] = defaultdict(set)
        for record_id in records:
            by_label[record_labels[record_id]].add(plan.groups[record_id])
        if len(by_label) < 2:
            continue
        if not set.intersection(*by_label.values()):
            detail = ", ".join(f"{label}={sorted(gs)}" for label, gs in sorted(by_label.items()))
            warnings.append(
                f"LEAKAGE RISK - in the '{name}' split, '{field_name}' separates the classes "
                f"exactly ({detail}). The model can read the label off this field; the "
                "accuracy numbers are not trustworthy. Provide more recordings or change "
                "the --balance-by field."
            )

    train_groups = {plan.groups[r] for r in plan.records_in("train")}
    for name in ("val", "test"):
        records = plan.records_in(name)
        if not records or not train_groups:
            continue
        unseen = {plan.groups[r] for r in records} - train_groups
        if unseen == {plan.groups[r] for r in records}:
            warnings.append(
                f"every recording in the '{name}' split has a '{field_name}' value never "
                f"seen during training ({sorted(unseen)}). This is an extrapolation test; "
                "accuracy may come out lower than expected, and that is not a bug."
            )
    return warnings
