"""Tests for iqforge.splitting - the recording-level split rule of SPEC §5.6."""

from __future__ import annotations

import pytest

from iqforge.io import IQForgeError
from iqforge.splitting import (
    SPLIT_NAMES,
    balance_warnings,
    leakage_warnings,
    parse_ratios,
    stratified_record_split,
)

DEFAULT = (0.7, 0.15, 0.15)

#: Same shape as the example dataset: two classes, four carrier-offset groups,
#: each class using each group once.
OFFSET_GROUPS = ("-280", "-180", "+180", "+280")


def _records(counts: dict[str, int]) -> dict[str, str]:
    """`{'bpsk': 4}` -> `{'bpsk_00': 'bpsk', ...}`"""
    return {f"{label}_{i:02d}": label for label, n in counts.items() for i in range(n)}


def _offset_groups(records: dict[str, str]) -> dict[str, str]:
    """Assign each recording a carrier-offset group, round robin."""
    groups: dict[str, str] = {}
    per_label: dict[str, int] = {}
    for record_id, label in sorted(records.items()):
        index = per_label.get(label, 0)
        groups[record_id] = OFFSET_GROUPS[index % len(OFFSET_GROUPS)]
        per_label[label] = index + 1
    return groups


def test_parse_ratios_accepts_valid_input() -> None:
    """A valid ratio string parses into three floats."""
    assert parse_ratios("0.7,0.15,0.15") == (0.7, 0.15, 0.15)
    assert parse_ratios(" 1.0 , 0 , 0 ") == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("0.7,0.3", "three values"),
        ("a,b,c", "numeric"),
        ("0.5,0.5,0.5", "sum to 1"),
        ("1.5,-0.5,0", "negative"),
    ],
)
def test_parse_ratios_rejects_invalid_input(text: str, fragment: str) -> None:
    """A malformed ratio string raises an error that says what to do."""
    with pytest.raises(IQForgeError, match=fragment):
        parse_ratios(text)


def test_every_record_lands_in_exactly_one_split() -> None:
    """A recording goes to exactly one split - the fundamental rule."""
    records = _records({"bpsk": 4, "qpsk": 4})
    plan = stratified_record_split(records, DEFAULT, seed=42)

    assert set(plan.assignment) == set(records)
    placed = [r for name in SPLIT_NAMES for r in plan.records_in(name)]
    assert sorted(placed) == sorted(records)
    assert len(placed) == len(set(placed)), "a recording appears in more than one split"


def test_each_class_is_present_in_every_split() -> None:
    """Stratified splitting represents every class in every split."""
    records = _records({"bpsk": 4, "qpsk": 4})
    plan = stratified_record_split(records, DEFAULT, seed=42)

    for name in SPLIT_NAMES:
        labels = {records[r] for r in plan.records_in(name)}
        assert labels == {"bpsk", "qpsk"}, f"class missing from the {name} split: {labels}"


def test_split_sizes_follow_ratios_as_closely_as_possible() -> None:
    """Four recordings per class and 0.7/0.15/0.15 gives 2/1/1."""
    plan = stratified_record_split(_records({"bpsk": 4, "qpsk": 4}), DEFAULT, seed=42)

    assert [len(plan.records_in(n)) for n in SPLIT_NAMES] == [4, 2, 2]


def test_same_seed_gives_identical_split() -> None:
    """The same seed reproduces the split exactly (SPEC §5.6 determinism)."""
    records = _records({"bpsk": 4, "qpsk": 4})

    first = stratified_record_split(records, DEFAULT, seed=42)
    second = stratified_record_split(records, DEFAULT, seed=42)

    assert first.assignment == second.assignment


def test_different_seed_changes_the_split() -> None:
    """A different seed gives a different split, otherwise the seed does nothing."""
    records = _records({"bpsk": 4, "qpsk": 4})

    assignments = {
        tuple(sorted(stratified_record_split(records, DEFAULT, seed=s).assignment.items()))
        for s in range(8)
    }

    assert len(assignments) > 1


def test_record_order_does_not_affect_the_split() -> None:
    """The ordering of the input dictionary must not change the result."""
    records = _records({"bpsk": 4, "qpsk": 4})
    reversed_records = dict(reversed(list(records.items())))

    a = stratified_record_split(records, DEFAULT, seed=42)
    b = stratified_record_split(reversed_records, DEFAULT, seed=42)

    assert a.assignment == b.assignment


def test_too_few_records_raises_instead_of_falling_back() -> None:
    """Too few recordings must RAISE, never fall back to window-level splitting."""
    with pytest.raises(IQForgeError) as exc:
        stratified_record_split(_records({"bpsk": 1, "qpsk": 4}), DEFAULT, seed=42)

    message = str(exc.value)
    assert "class 'bpsk' has only 1 recording" in message
    assert "needs at least 3" in message
    assert "--split 1.0,0,0" in message, "the error must offer a way out"
    assert "inflates" in message, "the error must say why"


def test_two_records_per_class_fails_for_three_way_split() -> None:
    """Two recordings cannot fill three splits."""
    with pytest.raises(IQForgeError, match="needs at least 3"):
        stratified_record_split(_records({"bpsk": 2, "qpsk": 2}), DEFAULT, seed=42)


def test_two_way_split_needs_only_two_records() -> None:
    """Splits with a zero ratio do not count towards the minimum."""
    plan = stratified_record_split(_records({"bpsk": 2, "qpsk": 2}), (0.5, 0.5, 0.0), seed=42)

    assert [len(plan.records_in(n)) for n in SPLIT_NAMES] == [2, 2, 0]


def test_single_record_allowed_with_train_only_split() -> None:
    """`--split 1.0,0,0` is the explicit escape hatch for a single recording."""
    plan = stratified_record_split({"solo": "bpsk"}, (1.0, 0.0, 0.0), seed=42)

    assert plan.records_in("train") == ["solo"]
    assert plan.records_in("val") == []
    assert plan.records_in("test") == []


def test_empty_input_is_rejected() -> None:
    """With no labellable recording the error must be explicit."""
    with pytest.raises(IQForgeError, match="Nothing to split"):
        stratified_record_split({}, DEFAULT, seed=42)


def test_uneven_class_sizes_are_stratified_independently() -> None:
    """Classes of different sizes are apportioned within themselves.

    10 recordings and 0.7/0.15/0.15 gives 7.0/1.5/1.5, so 7/2/1 by largest
    remainder. 4 recordings gives 2.8/0.6/0.6 -> 3/1/0, then 2/1/1 once the
    minimum is enforced.
    """
    records = _records({"bpsk": 10, "qpsk": 4})
    plan = stratified_record_split(records, DEFAULT, seed=42)

    bpsk_per_split = {
        name: [records[r] for r in plan.records_in(name)].count("bpsk") for name in SPLIT_NAMES
    }
    qpsk_per_split = {
        name: [records[r] for r in plan.records_in(name)].count("qpsk") for name in SPLIT_NAMES
    }

    assert bpsk_per_split == {"train": 7, "val": 2, "test": 1}
    assert qpsk_per_split == {"train": 2, "val": 1, "test": 1}


@pytest.mark.parametrize("seed", range(8))
def test_group_never_predicts_the_label_within_a_split(seed: int) -> None:
    """Inside a split, the group must not be able to predict the label.

    This is the whole point of balancing. Violating it is a disaster: if groups
    are handed out complementarily between classes (bpsk taking the positive
    offsets in train, qpsk the negative ones), the model learns the shortcut,
    scores 100% on training, and drops to 0% on a test split where the
    relationship is reversed — below chance. That regression actually happened;
    this test locks it out.
    """
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=seed, record_groups=groups)

    for name in SPLIT_NAMES:
        in_split = plan.records_in(name)
        if len(in_split) < 2:
            continue
        by_label: dict[str, set[str]] = {}
        for record_id in in_split:
            by_label.setdefault(records[record_id], set()).add(groups[record_id])
        assert set.intersection(*by_label.values()), (
            f"seed {seed}, '{name}': the group gives the label away -> {by_label}"
        )


def test_each_group_stays_in_a_single_split_when_it_has_one_record_per_class() -> None:
    """With one recording per class per group, a group cannot be divided.

    That is the price of the independence guarantee above: train and test cannot
    share a group. `leakage_warnings` reports it as an extrapolation warning.
    """
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)

    where: dict[str, set[str]] = {}
    for record_id, split in plan.assignment.items():
        where.setdefault(groups[record_id], set()).add(split)
    assert all(len(splits) == 1 for splits in where.values()), where


def test_balancing_preserves_class_stratification() -> None:
    """Balancing must not change the per-class recording counts of a split."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plain = stratified_record_split(records, DEFAULT, seed=42)
    balanced = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)

    for name in SPLIT_NAMES:
        for label in ("bpsk", "qpsk"):
            assert [records[r] for r in plain.records_in(name)].count(label) == [
                records[r] for r in balanced.records_in(name)
            ].count(label)


def test_balancing_is_deterministic() -> None:
    """The same seed reproduces a balanced split exactly too."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    first = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)
    second = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)

    assert first.assignment == second.assignment
    assert first.groups == groups


def test_balancing_keeps_the_record_level_guarantee() -> None:
    """Balancing must not break the recording-level rule."""
    records = _records({"bpsk": 6, "qpsk": 6})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=3, record_groups=groups)

    placed = [r for name in SPLIT_NAMES for r in plan.records_in(name)]
    assert sorted(placed) == sorted(records)
    assert len(placed) == len(set(placed))


def test_balance_warning_when_every_record_is_its_own_group() -> None:
    """One group per recording makes balancing meaningless; warn about it."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = {r: f"group_{r}" for r in records}

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)
    warnings = balance_warnings(plan, "core:description")

    assert any("own group" in w for w in warnings)


def test_balance_warning_when_only_one_group_exists() -> None:
    """A single group means balancing did nothing; warn about it."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = dict.fromkeys(records, "single")

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)
    warnings = balance_warnings(plan, "core:hw")

    assert any("no effect" in w for w in warnings)


def test_leakage_warning_when_group_separates_classes_within_a_split() -> None:
    """A group that separates the classes inside a split must warn loudly.

    This can happen without balancing, or when the chosen field correlates with
    the class. The user must know before trusting the accuracy numbers.
    """
    records = _records({"bpsk": 2, "qpsk": 2})
    # The group coincides exactly with the label: the worst case.
    groups = {r: ("g_bpsk" if records[r] == "bpsk" else "g_qpsk") for r in records}

    plan = stratified_record_split(records, (0.5, 0.5, 0.0), seed=42, record_groups=groups)
    warnings = leakage_warnings(plan, records, "core:hw")

    assert any("LEAKAGE RISK" in w for w in warnings)


def test_leakage_warning_when_evaluation_groups_are_unseen_in_training() -> None:
    """Evaluating on groups never seen in training is extrapolation; report it."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)
    warnings = leakage_warnings(plan, records, "core:freq_lower_edge")

    assert any("never seen during training" in w and "test" in w for w in warnings)


def test_no_leakage_warning_without_balancing() -> None:
    """Without `--balance-by` the leakage check stays quiet."""
    records = _records({"bpsk": 4, "qpsk": 4})
    plan = stratified_record_split(records, DEFAULT, seed=42)

    assert leakage_warnings(plan, records, "core:hw") == []


def test_balance_warnings_are_empty_when_balancing_works() -> None:
    """No warnings when balancing does its job."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)

    assert balance_warnings(plan, "core:freq_lower_edge") == []


def test_balance_warnings_empty_without_balancing() -> None:
    """No warnings when `--balance-by` was not used."""
    plan = stratified_record_split(_records({"bpsk": 4, "qpsk": 4}), DEFAULT, seed=42)

    assert balance_warnings(plan, "core:hw") == []


def test_minimum_guarantee_does_not_inflate_small_splits() -> None:
    """The minimum must steal from the largest split, not skew the ratios up front.

    Reserving one recording per split first (the wrong approach) gives 6/2/2 for
    10 recordings; the correct answer is 7/2/1.
    """
    plan = stratified_record_split(_records({"bpsk": 10}), DEFAULT, seed=42)

    assert [len(plan.records_in(n)) for n in SPLIT_NAMES] == [7, 2, 1]
