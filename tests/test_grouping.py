"""Tests for `--group-by`: keeping non-independent recordings in one split."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest

from helpers import write_record
from iqforge.grouping import grouping_warnings, resolve_group_keys
from iqforge.io import IQForgeError
from iqforge.splitting import stratified_record_split

RATIOS = (0.6, 0.2, 0.2)

#: The message the splitter raises, byte for byte. Written as a literal so that
#: any change to the wording fails here first.
UNGROUPED_SPLIT_ERROR = (
    "Cannot stratify by recording: class 'bpsk' has only 1 recording, but a "
    "0.7/0.15/0.15 split needs at least 3 (train=0.7, val=0.15, test=0.15).\n\n"
    "Windows from one recording must not appear in more than one split "
    "(SPEC §5.6); falling back to window-level splitting inflates test "
    "accuracy.\n\n"
    "Options:\n"
    "  - provide more recordings per class (pass a directory)\n"
    "  - reduce the split ratios, e.g. --split 0.5,0.25,0.25\n"
    "  - build a training set only: --split 1.0,0,0"
)


def _unwrapped(text: str) -> str:
    """Collapse single line breaks so a rendered block can be compared to source.

    The README quotes the message as it appeared in a terminal, so its first
    paragraph carries a line break the message itself does not have. Comparing
    raw strings would fail on that alone. Blank lines are kept, since those are
    in the message.
    """
    return "\n\n".join(" ".join(p.split()) for p in text.split("\n\n"))


def _labels(pairs: dict[str, str]) -> dict[str, str]:
    return dict(pairs)


# --------------------------------------------------------------------------
# The README block must not move
# --------------------------------------------------------------------------


def test_split_error_is_byte_identical_without_grouping() -> None:
    """Ungrouped, the message is unchanged, character for character.

    Adding `--group-by` rewrote this message's construction -- a noun that
    varies, an inserted paragraph, an extra option line. None of that may reach
    the ungrouped path, because the README prints this block as the tool's
    answer when it refuses to split.
    """
    with pytest.raises(IQForgeError) as exc:
        stratified_record_split({"a.sigmf-meta": "bpsk"}, (0.7, 0.15, 0.15), seed=42)

    assert str(exc.value) == UNGROUPED_SPLIT_ERROR


def test_the_readme_still_shows_that_message() -> None:
    """And the README block still says the same thing, so the two cannot drift.

    Compared with line wrapping collapsed: the README quotes terminal output,
    which is the message wrapped to a width. The wrapping is a rendering
    detail; the wording is not.
    """
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    block = re.search(r"\n```\n(Error: Cannot stratify by recording.*?)\n```\n", readme, re.S)

    assert block is not None, "README no longer contains the split-error block"
    assert _unwrapped(block.group(1)) == _unwrapped(f"Error: {UNGROUPED_SPLIT_ERROR}")


def test_split_error_explains_itself_when_grouping_is_on() -> None:
    """Grouped, the message says groups -- and says how many recordings collapsed."""
    labels = {f"r{i}.sigmf-meta": "bpsk" for i in range(6)}
    units = dict.fromkeys(labels, "one-unit")

    with pytest.raises(IQForgeError) as exc:
        stratified_record_split(
            labels, (0.7, 0.15, 0.15), seed=42, record_units=units, group_by="path:(x)"
        )

    message = str(exc.value)
    assert "has only 1 group," in message
    assert "collapsed 6 recording(s) of class 'bpsk' into 1 group(s)" in message
    assert "relax the grouping" in message


# --------------------------------------------------------------------------
# Grouping holds
# --------------------------------------------------------------------------


def test_a_group_never_spans_two_splits() -> None:
    """The one guarantee the feature exists to provide."""
    labels = {f"{cls}_{i:02d}": cls for cls in ("bpsk", "qpsk") for i in range(1, 9)}
    units = {r: f"{r[:4]}_pair{(int(r[-2:]) + 1) // 2}" for r in labels}

    plan = stratified_record_split(labels, RATIOS, seed=7, record_units=units)

    where: dict[str, set[str]] = {}
    for record_id, split in plan.assignment.items():
        where.setdefault(units[record_id], set()).add(split)
    assert all(len(splits) == 1 for splits in where.values()), where


def test_grouping_actually_changes_the_split() -> None:
    """Without grouping the same pairs do get separated.

    A test that only checks "groups are together" would pass on a split that
    happened to keep them together by luck. This pins that the constraint is
    doing work: ungrouped, at least one pair is broken.
    """
    labels = {f"{cls}_{i:02d}": cls for cls in ("bpsk", "qpsk") for i in range(1, 9)}
    units = {r: f"{r[:4]}_pair{(int(r[-2:]) + 1) // 2}" for r in labels}

    ungrouped = stratified_record_split(labels, RATIOS, seed=7)
    where: dict[str, set[str]] = {}
    for record_id, split in ungrouped.assignment.items():
        where.setdefault(units[record_id], set()).add(split)

    assert any(len(splits) > 1 for splits in where.values())


def test_stratification_survives_grouping() -> None:
    """Every class still reaches every split."""
    labels = {f"{cls}_{i:02d}": cls for cls in ("bpsk", "qpsk") for i in range(1, 13)}
    units = {r: f"{r[:4]}_pair{(int(r[-2:]) + 1) // 2}" for r in labels}

    plan = stratified_record_split(labels, RATIOS, seed=3, record_units=units)

    for split in ("train", "val", "test"):
        classes = {labels[r] for r in plan.records_in(split)}
        assert classes == {"bpsk", "qpsk"}, (split, classes)


def test_units_are_recorded_on_the_plan() -> None:
    """The plan carries the mapping so the manifest can be audited."""
    labels = {"a": "x", "b": "x", "c": "y", "d": "y"}
    units = {"a": "u1", "b": "u1", "c": "u2", "d": "u2"}

    plan = stratified_record_split(labels, (1.0, 0.0, 0.0), seed=1, record_units=units)

    assert plan.units == units
    assert stratified_record_split(labels, (1.0, 0.0, 0.0), seed=1).units == {}


def test_a_group_spanning_two_classes_is_an_error() -> None:
    """A unit goes to one split, so it cannot carry two strata."""
    labels = {"a": "bpsk", "b": "qpsk"}
    units = {"a": "same", "b": "same"}

    with pytest.raises(IQForgeError, match="more than one class"):
        stratified_record_split(labels, (1.0, 0.0, 0.0), seed=1, record_units=units)


# --------------------------------------------------------------------------
# Key resolution
# --------------------------------------------------------------------------


def test_path_scheme_joins_capture_groups() -> None:
    ids = ["CH0/rec1/a.sigmf-meta", "CH0/rec2/b.sigmf-meta", "CH93/rec1/c.sigmf-meta"]

    keys = resolve_group_keys(ids, r"path:(CH\d+)")

    assert keys[ids[0]] == keys[ids[1]] == "CH0"
    assert keys[ids[2]] == "CH93"


def test_path_scheme_without_capture_groups_uses_the_match() -> None:
    keys = resolve_group_keys(["loc7/x.sigmf-meta"], r"path:loc\d+")

    assert keys["loc7/x.sigmf-meta"] == "loc7"


def test_unmatched_recordings_stay_independent() -> None:
    """An unmatched recording gets its own key, never a shared bucket.

    Pooling them under one "unmatched" key would glue unrelated recordings
    together, which is the failure grouping exists to prevent.
    """
    ids = ["CH0/a.sigmf-meta", "other/b.sigmf-meta", "other/c.sigmf-meta"]

    keys = resolve_group_keys(ids, r"path:(CH\d+)")

    assert keys[ids[1]] != keys[ids[2]]


def test_csv_scheme_maps_by_file_name(tmp_path: Path) -> None:
    table = tmp_path / "g.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recording", "group"])
        writer.writerows([["a.sigmf-meta", "u1"], ["b.sigmf-meta", "u1"]])

    keys = resolve_group_keys(["deep/a.sigmf-meta", "b.sigmf-meta"], f"csv:{table}")

    assert keys["deep/a.sigmf-meta"] == keys["b.sigmf-meta"] == "u1"


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("core:hw", "must be '<scheme>:<argument>'"),
        ("nope:x", "must be '<scheme>:<argument>'"),
        ("path:", "must be '<scheme>:<argument>'"),
        (r"path:(unclosed", "not a valid regex"),
        ("csv:missing.csv", "not found"),
    ],
)
def test_bad_specs_explain_themselves(spec: str, expected: str) -> None:
    with pytest.raises(IQForgeError, match=re.escape(expected)):
        resolve_group_keys(["a.sigmf-meta"], spec)


def test_csv_without_the_expected_columns_is_an_error(tmp_path: Path) -> None:
    table = tmp_path / "g.csv"
    table.write_text("file,unit\na,b\n", encoding="utf-8")

    with pytest.raises(IQForgeError, match="'recording' and 'group' columns"):
        resolve_group_keys(["a.sigmf-meta"], f"csv:{table}")


# --------------------------------------------------------------------------
# Warnings for a grouping that did nothing
# --------------------------------------------------------------------------


def test_warns_when_the_pattern_matches_nothing() -> None:
    keys = resolve_group_keys(["a.sigmf-meta", "b.sigmf-meta"], r"path:(ZZZ\d+)")

    (warning,) = grouping_warnings(keys, r"path:(ZZZ\d+)")
    assert "matched no recording" in warning


def test_warns_when_every_unit_holds_one_recording() -> None:
    """A grouping that separates everything is not grouping."""
    ids = ["CH0/a.sigmf-meta", "CH1/b.sigmf-meta"]
    keys = resolve_group_keys(ids, r"path:(CH\d+)")

    (warning,) = grouping_warnings(keys, r"path:(CH\d+)")
    assert "one group per recording" in warning


def test_warns_when_only_some_recordings_match() -> None:
    ids = ["CH0/a.sigmf-meta", "CH0/b.sigmf-meta", "loose/c.sigmf-meta"]
    keys = resolve_group_keys(ids, r"path:(CH\d+)")

    (warning,) = grouping_warnings(keys, r"path:(CH\d+)")
    assert "did not match 1 of 3" in warning


def test_no_warning_when_the_grouping_did_something() -> None:
    ids = ["CH0/a.sigmf-meta", "CH0/b.sigmf-meta", "CH1/c.sigmf-meta", "CH1/d.sigmf-meta"]
    keys = resolve_group_keys(ids, r"path:(CH\d+)")

    assert grouping_warnings(keys, r"path:(CH\d+)") == []


# --------------------------------------------------------------------------
# Interaction with --balance-by
# --------------------------------------------------------------------------


def test_units_with_disagreeing_balance_values_are_marked_mixed() -> None:
    """A unit that spans two nuisance values carries none into its split."""
    from iqforge.splitting import MIXED_BALANCE, _unit_balance_values

    members = {"u1": ["a", "b"], "u2": ["c", "d"]}
    groups = {"a": "+180", "b": "-180", "c": "+280", "d": "+280"}

    values = _unit_balance_values(members, groups)

    assert values["u1"] == MIXED_BALANCE
    assert values["u2"] == "+280"


def test_grouping_and_balancing_together_still_hold_units(tmp_path: Path) -> None:
    """Grouping wins: units stay whole even while balancing runs."""
    labels = {f"{cls}_{i:02d}": cls for cls in ("bpsk", "qpsk") for i in range(1, 9)}
    units = {r: f"{r[:4]}_pair{(int(r[-2:]) + 1) // 2}" for r in labels}
    # Deliberately crossed: each pair spans two offsets, so every unit is mixed.
    offsets = {r: ("+180" if int(r[-2:]) % 2 else "-180") for r in labels}

    plan = stratified_record_split(
        labels, RATIOS, seed=11, record_groups=offsets, record_units=units
    )

    where: dict[str, set[str]] = {}
    for record_id, split in plan.assignment.items():
        where.setdefault(units[record_id], set()).add(split)
    assert all(len(splits) == 1 for splits in where.values())
    assert plan.groups == offsets


def test_written_dataset_records_the_group(tmp_path: Path) -> None:
    """The manifest carries the unit, so a split can be audited afterwards."""
    import json

    from typer.testing import CliRunner

    from iqforge.cli import app

    samples = (np.arange(8192) + 1j * np.arange(8192)).astype(np.complex64)
    for cls in ("bpsk", "qpsk"):
        for i in range(1, 7):
            write_record(
                tmp_path / "src" / cls,
                samples,
                name=f"{cls}_{i:02d}",
                annotations=[
                    {"core:sample_start": 0, "core:sample_count": 8192, "core:label": cls}
                ],
            )

    out = tmp_path / "ds"
    result = CliRunner().invoke(
        app,
        [
            "build", str(tmp_path / "src"), "-o", str(out),
            "--labels", "dirname",
            "--group-by", r"path:(bpsk|qpsk)_0(\d)",
            "--window", "1024", "--stride", "4096",
            "--split", "0.6,0.2,0.2",
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["group_by"] == r"path:(bpsk|qpsk)_0(\d)"
    entries = [e for s in ("train", "val", "test") for e in manifest["splits"][s]["records"]]
    assert all(e["group"] for e in entries)
