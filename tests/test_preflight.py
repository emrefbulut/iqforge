"""Refuse path for `iqforge measure-leakage`.

The command does not train. These tests lock the six categories that
eliminated four public datasets and let LoRaIQ through (methodology §6), the
quotable block, and the `--force` header.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from helpers import loraiq_paths, loraiq_skip_reason, write_record
from iqforge.audit import WIDTH, AuditReport, Finding, RecordFeatures, Status
from iqforge.cli import _audit_folder, app
from iqforge.grouping import resolve_group_keys
from iqforge.preflight import (
    CATEGORY_META,
    Category,
    DecisionStatus,
    decide,
    render_json,
    render_text,
)

runner = CliRunner()


def _torch_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("torch") is not None


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"

_VEGA_STAMPS = (
    dt.datetime(2022, 7, 24, 18, 47, 38, tzinfo=dt.UTC),
    dt.datetime(2022, 7, 24, 19, 25, 49, tzinfo=dt.UTC),
    dt.datetime(2022, 7, 24, 19, 29, 2, tzinfo=dt.UTC),
)
_VEGA_SATS = ("ASTROBIO", "CELESTA", "MTCube-2", "NESS", "ROBUSTA-3A")


def _samples(n: int = 4096, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def _iso(stamp: dt.datetime) -> str:
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _invoke(path: Path, *args: str):
    return runner.invoke(app, ["measure-leakage", str(path), *args])


def _write_airid(root: Path) -> Path:
    """GENESYS-style `cf16_le`: the reader rejects every file."""
    folder = root / "airid"
    for index, name in enumerate(("uav1", "uav2", "uav3")):
        write_record(folder / name, _samples(seed=index), name="burst", datatype="cf16_le")
    return folder


def _write_vega_c(root: Path) -> Path:
    """Five satellites, three shared stamps, passes that do not overlap."""
    folder = root / "vega"
    seed = 0
    for sat in _VEGA_SATS:
        for stamp in _VEGA_STAMPS:
            write_record(
                folder / sat,
                _samples(seed=seed),
                name=f"pass_{stamp:%H_%M_%S}",
                sample_rate=40_000.0,
                capture_extra={"core:datetime": _iso(stamp)},
            )
            seed += 1
    return folder


def _write_dash7_indoor(root: Path) -> Path:
    """Two locations, pairs of 2 s captures 43 s apart: same room, two runs."""
    folder = root / "ds_indoor"
    start = dt.datetime(2024, 4, 11, 11, 35, 36, tzinfo=dt.UTC)
    # 2048 samples at 1024 S/s = 2 s, above the 1 s floor that keeps LoRaIQ
    # frame snippets from looking like this pattern. Locations do not share
    # timestamps: that would be Vega-C, not this case.
    seed = 0
    for loc_index, loc in enumerate(("loc1", "loc2")):
        loc_start = start + dt.timedelta(days=loc_index)
        for index in range(3):
            pair_start = loc_start + dt.timedelta(hours=index)
            for offset_s, name in ((0, "a"), (43, "b")):
                stamp = pair_start + dt.timedelta(seconds=offset_s)
                write_record(
                    folder / loc,
                    _samples(2048, seed=seed),
                    name=f"ch0_{index}_{name}",
                    sample_rate=1024.0,
                    capture_extra={"core:datetime": _iso(stamp)},
                )
                seed += 1
    return folder


def _write_ceiling(root: Path) -> Path:
    """Class *is* a carrier offset: the DASH7 cabled situation, in miniature."""
    folder = root / "cabled"
    centre = 100_000_000.0
    seed = 0
    for label, offset in (("ch0", 0.0), ("ch1", 500_000.0)):
        lo, hi = centre + offset - 10_000.0, centre + offset + 10_000.0
        for index in range(3):
            write_record(
                folder / label,
                _samples(seed=seed),
                name=f"rec{index}",
                center_freq=centre,
                annotations=[
                    {
                        "core:sample_start": 0,
                        "core:sample_count": 4096,
                        "core:freq_lower_edge": lo,
                        "core:freq_upper_edge": hi,
                        "core:label": label,
                    }
                ],
            )
            seed += 1
    return folder


def _write_loraiq_pattern(root: Path) -> Path:
    """Simultaneous receptions, short segments, enough independent units.

    Two classes, three transmissions each, two receivers per transmission.
    The files that share air time share a group key that does not cross class.
    """
    folder = root / "loraiq"
    start = dt.datetime(2023, 6, 1, 12, 0, 0, tzinfo=dt.UTC)
    seed = 0
    for label_index, label in enumerate(("drone_los", "pedestrian_nlos")):
        for tx in range(3):
            # Stamps are per class so simultaneous receptions do not cross
            # a class boundary (a unit that did would be unsplittable).
            stamp = start + dt.timedelta(hours=label_index, minutes=tx * 10)
            for rx in range(2):
                write_record(
                    folder / label / f"tx{tx}",
                    _samples(seed=seed),
                    name=f"rx{rx}",
                    sample_rate=1_000_000.0,
                    capture_extra={"core:datetime": _iso(stamp)},
                )
                seed += 1
    return folder


# --------------------------------------------------------------------------
# Categories 1-4: the datasets methodology §6 eliminated
# --------------------------------------------------------------------------


def test_airid_pattern_is_refused_as_category_1(tmp_path: Path) -> None:
    result = _invoke(_write_airid(tmp_path))

    assert result.exit_code == 1
    assert "REFUSED" in result.output
    assert "1  unreadable format" in result.output
    assert "6.1" in result.output
    assert "cf16_le" in result.output


def test_vega_c_pattern_is_refused_as_category_2(tmp_path: Path) -> None:
    result = _invoke(_write_vega_c(tmp_path))

    assert result.exit_code == 1
    assert "REFUSED" in result.output
    assert "2  shared timestamp" in result.output
    assert "6.2" in result.output
    assert "Vega-C" in result.output


def test_dash7_indoor_pattern_is_refused_as_category_3(tmp_path: Path) -> None:
    result = _invoke(_write_dash7_indoor(tmp_path))

    assert result.exit_code == 1
    assert "REFUSED" in result.output
    assert "3  physical independence" in result.output
    assert "6.3" in result.output


def test_ceiling_is_refused_as_category_4(tmp_path: Path) -> None:
    result = _invoke(_write_ceiling(tmp_path))

    assert result.exit_code == 1
    assert "REFUSED" in result.output
    assert "4  ceiling" in result.output
    assert "6.4" in result.output


# --------------------------------------------------------------------------
# Category 5 / LoRaIQ: overlapping air time is a leak unless grouped
# --------------------------------------------------------------------------


def test_ungrouped_simultaneous_receptions_are_a_structural_leak(tmp_path: Path) -> None:
    result = _invoke(_write_loraiq_pattern(tmp_path), "--dirname-level", "2")

    assert result.exit_code == 1
    assert "REFUSED" in result.output
    assert "5  structural leak" in result.output
    assert "overlapping air time" in result.output


def test_loraiq_pattern_is_not_refused_when_grouped(tmp_path: Path) -> None:
    """§6.5: simultaneous receptions held together, task not at the ceiling."""
    folder = _write_loraiq_pattern(tmp_path)
    result = _invoke(folder, "--dirname-level", "2", "--group-by", r"path:([^/]+/tx\d+)")

    assert result.exit_code == 0, result.output
    assert "WOULD MEASURE" in result.output
    assert "REFUSED" not in result.output
    if _torch_installed():
        assert "MEASUREMENT" in result.output
    else:
        assert "stops before training" in result.output


@pytest.mark.skipif(loraiq_skip_reason() is not None, reason=loraiq_skip_reason() or "")
def test_loraiq_recordings_are_not_refused() -> None:
    """The dataset that carried §3 must pass preflight.

    This stays a preflight test on purpose: full measurement is exercised in
    `tests/test_measurement.py`, and rerunning the whole training cell here
    would make category coverage tests prohibitively slow.
    """
    paths = loraiq_paths()
    assert paths is not None
    source, _index, labels, groups = paths
    report = _audit_folder(
        source,
        1024,
        512,
        "csv",
        1,
        labels,
    )
    group_keys = resolve_group_keys(
        [f.record_id for f in report.features],
        f"csv:{groups}",
        collections={f.record_id: f.collection for f in report.features},
    )
    decision = decide(report, group_keys=group_keys, seconds_per_window_epoch=None)
    assert decision.status is DecisionStatus.WOULD_MEASURE
    assert decision.category is None


# --------------------------------------------------------------------------
# --force, the quotable block, category 6
# --------------------------------------------------------------------------


def test_force_puts_the_overridden_verdict_in_the_header(tmp_path: Path) -> None:
    result = _invoke(_write_ceiling(tmp_path), "--force")

    assert result.exit_code == 0, result.output
    first_content = next(
        line for line in result.output.splitlines() if line.startswith("iqforge leakage")
    )
    assert "FORCED PAST audit VERDICT 'ceiling'" in first_content
    assert "WOULD MEASURE" in result.output
    assert "4  ceiling" in result.output
    assert "this run is not a clean" in result.output
    assert "measurement" in result.output
    assert "—" not in result.output


def test_too_few_recordings_is_category_6(tmp_path: Path) -> None:
    folder = tmp_path / "tiny"
    write_record(folder / "a", _samples(seed=1), name="one")
    write_record(folder / "b", _samples(seed=2), name="two")
    result = _invoke(folder)

    assert result.exit_code == 1
    assert "6  cannot split" in result.output
    assert "5.6" in result.output


def test_the_report_is_78_columns_of_ascii(tmp_path: Path) -> None:
    result = _invoke(_write_ceiling(tmp_path), "--force")
    text = result.output
    assert WIDTH == 78
    block_lines: list[str] = []
    for line in text.splitlines():
        if line.strip().endswith("MEASUREMENT"):
            break
        block_lines.append(line)
    for line in block_lines:
        assert len(line) <= WIDTH, line
    "\n".join(block_lines).encode("ascii")


def test_json_carries_the_same_category(tmp_path: Path) -> None:
    result = _invoke(_write_ceiling(tmp_path), "--format", "json")
    payload = json.loads(result.output)

    assert result.exit_code == 1
    assert payload["status"] == "REFUSED"
    assert payload["category"] == 4
    assert payload["name"] == "ceiling"
    assert payload["cites"] == "methodology 6.4"
    assert payload["forced"] is False
    assert "snr" not in payload


def _measure_leakage_option_flags() -> set[str]:
    from typer.main import get_group

    group = get_group(app)
    cmd = group.commands["measure-leakage"]
    flags: set[str] = set()
    for param in cmd.params:
        flags.update(getattr(param, "opts", ()) or ())
    return flags


def test_measure_leakage_device_defaults_to_cpu() -> None:
    """CUDA is opt-in. A GPU being present must not change the default."""
    from typer.main import get_group

    flags = _measure_leakage_option_flags()
    assert "--device" in flags
    group = get_group(app)
    param = next(
        p for p in group.commands["measure-leakage"].params if "--device" in (p.opts or ())
    )
    assert param.default == "cpu"

    train = next(p for p in group.commands["train"].params if "--device" in (p.opts or ()))
    assert train.default == "cpu"


def test_help_does_not_offer_sweep_snr() -> None:
    result = runner.invoke(
        app,
        ["measure-leakage", "--help"],
        env={"COLUMNS": "200", "NO_COLOR": "1"},
    )
    assert result.exit_code == 0
    flags = _measure_leakage_option_flags()
    assert "--sweep" in result.output or "--sweep" in flags
    assert "stride" in result.output.lower() or "--stride" in flags
    assert "snr" not in result.output.lower()


def test_examples_would_measure() -> None:
    """The project's own recordings are the trivial 'not refused' case."""
    if not any(EXAMPLES.glob("*.sigmf-meta")):
        pytest.skip("examples/ recordings have not been generated")
    result = _invoke(EXAMPLES)
    assert result.exit_code == 0, result.output
    assert "WOULD MEASURE" in result.output


def test_measure_leakage_without_torch_still_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refuse path must not grow a torch dependency."""
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    for name in ("iqforge.dataset", "iqforge.models", "iqforge.training"):
        monkeypatch.delitem(__import__("sys").modules, name, raising=False)
    result = _invoke(_write_airid(tmp_path))
    assert result.exit_code == 1
    assert "1  unreadable format" in result.output


# --------------------------------------------------------------------------
# Library: grouping holds overlap, force, inconclusive
# --------------------------------------------------------------------------


def test_decide_does_not_refuse_grouped_overlap() -> None:
    start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
    features = [
        RecordFeatures(
            record_id="rrh1/1",
            label="los",
            capture_time=start,
            duration_samples=15_244,
            sample_rate=1_000_000.0,
        ),
        RecordFeatures(
            record_id="rrh2/1",
            label="los",
            capture_time=start + dt.timedelta(microseconds=300),
            duration_samples=15_244,
            sample_rate=1_000_000.0,
        ),
        RecordFeatures(record_id="rrh1/2", label="los", duration_samples=15_244, sample_rate=1e6),
        RecordFeatures(record_id="rrh1/3", label="nlos", duration_samples=15_244, sample_rate=1e6),
        RecordFeatures(record_id="rrh1/4", label="nlos", duration_samples=15_244, sample_rate=1e6),
        RecordFeatures(record_id="rrh1/5", label="nlos", duration_samples=15_244, sample_rate=1e6),
    ]
    report = AuditReport(
        tool_version="0.0.0",
        generated="2026-01-01T00:00:00Z",
        mode="recording folder (no manifest)",
        input_path="loraiq/",
        input_lines=["6 recordings"],
        fingerprint="sha256:deadbeef",
        findings=[
            Finding(
                Status.LEAK,
                "recording time overlap",
                "1 pair(s) claim overlapping air time: rrh1/1 / rrh2/1",
            )
        ],
        verdict="unknown - test",
        features=features,
    )
    grouped = decide(
        report,
        group_keys={"rrh1/1": "tx1", "rrh2/1": "tx1"},
        seconds_per_window_epoch=None,
    )
    assert grouped.category is not Category.STRUCTURAL_LEAK

    ungrouped = decide(report, seconds_per_window_epoch=None)
    assert ungrouped.status is DecisionStatus.REFUSED
    assert ungrouped.category is Category.STRUCTURAL_LEAK


def test_force_on_inconclusive_names_the_audit() -> None:
    report = AuditReport(
        tool_version="0.0.0",
        generated="2026-01-01T00:00:00Z",
        mode="built dataset (manifest_schema 1)",
        input_path="ds/",
        input_lines=["sources missing"],
        fingerprint="sha256:00000000",
        findings=[Finding(Status.NOT_CHECKED, "class axes", "source recordings not found")],
        verdict="unknown - no axis was measurable, so nothing was ruled out",
    )
    decision = decide(report, force=True, seconds_per_window_epoch=None)
    assert decision.status is DecisionStatus.WOULD_MEASURE
    assert decision.forced
    assert "INCONCLUSIVE" in (decision.forced_past or "")
    text = render_text(decision)
    assert "FORCED PAST an INCONCLUSIVE audit" in text
    for line in text.splitlines():
        assert len(line) <= WIDTH
    render_json(decision).encode("ascii")


def test_short_same_class_frames_are_not_the_indoor_pattern() -> None:
    """LoRaIQ consecutive transmissions are milliseconds long, not 8 s rooms."""
    start = dt.datetime(2023, 1, 1, tzinfo=dt.UTC)
    features = []
    for index in range(6):
        label = "los" if index < 3 else "nlos"
        features.append(
            RecordFeatures(
                record_id=f"{label}/{index}",
                label=label,
                capture_time=start + dt.timedelta(seconds=index * 2),
                duration_samples=15_244,
                sample_rate=1_000_000.0,
            )
        )
    report = AuditReport(
        tool_version="0.0.0",
        generated="2026-01-01T00:00:00Z",
        mode="recording folder (no manifest)",
        input_path="x/",
        input_lines=["6 recordings"],
        fingerprint="sha256:00",
        verdict="unknown - test",
        features=features,
    )
    decision = decide(report, seconds_per_window_epoch=None)
    assert decision.category is not Category.INDEPENDENCE


# --------------------------------------------------------------------------
# --force: what it can and cannot override
# --------------------------------------------------------------------------


def test_force_cannot_override_an_unreadable_set(tmp_path: Path) -> None:
    """Category 1 is not a judgement, so there is nothing to overrule.

    Every forcible category is this tool inferring what a pattern means, and
    someone who knows their own recordings can be right where the inference is
    wrong. This one says the reader could not open the files: there is nothing
    to measure, and accepting `--force` would only move the failure further in.
    """
    result = _invoke(_write_airid(tmp_path), "--force")

    assert result.exit_code == 1, result.output
    assert "1  unreadable format" in result.output
    assert "FORCED PAST" not in result.output
    assert "cannot be overridden" in result.output


def test_force_cannot_override_a_split_that_cannot_be_made(tmp_path: Path) -> None:
    """Category 6: `build` would refuse, so no measurement can be constructed."""
    folder = tmp_path / "tiny"
    write_record(folder / "a", _samples(seed=1), name="one")
    write_record(folder / "b", _samples(seed=2), name="two")
    result = _invoke(folder, "--force")

    assert result.exit_code == 1, result.output
    assert "6  cannot split" in result.output
    assert "FORCED PAST" not in result.output
    assert "cannot be overridden" in result.output


def test_the_json_reports_a_refused_force(tmp_path: Path) -> None:
    result = _invoke(_write_airid(tmp_path), "--force", "--format", "json")
    payload = json.loads(result.output)

    assert payload["status"] == "REFUSED"
    assert payload["category"] == 1
    assert payload["forced"] is False
    assert payload["forced_past"] is None
    assert "cannot be overridden" in payload["force_refused"]


def test_force_still_overrides_a_heuristic_refusal(tmp_path: Path) -> None:
    """The forcible four must keep working, or the distinction became a ban.

    Library-level rather than through the CLI: `decide` is where the split
    lives, and going through the command would train the forced cell.
    """
    report = _audit_folder(_write_vega_c(tmp_path), 1024, 512, "dirname", 1)
    decision = decide(report, force=True, seconds_per_window_epoch=None)

    assert decision.category is Category.SHARED_TIMESTAMP
    assert decision.status is DecisionStatus.WOULD_MEASURE
    assert decision.forced is True
    assert decision.forced_past
    assert decision.force_refused is None


def test_every_category_is_either_forcible_or_structural() -> None:
    """No category may fall outside the split as new ones are added."""
    from iqforge.preflight import STRUCTURAL_CATEGORIES

    assert STRUCTURAL_CATEGORIES == {Category.UNREADABLE, Category.CANNOT_SPLIT}
    forcible = set(Category) - set(STRUCTURAL_CATEGORIES)
    assert forcible == {
        Category.SHARED_TIMESTAMP,
        Category.INDEPENDENCE,
        Category.CEILING,
        Category.STRUCTURAL_LEAK,
    }


# --------------------------------------------------------------------------
# The report must describe the run it is actually making
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _torch_installed(), reason="training must be possible to contradict it")
def test_a_run_that_trains_does_not_say_it_stops_before_training(tmp_path: Path) -> None:
    """The block used to announce the opposite of what happened next.

    `started  no. This version of the command stops before training` was
    printed unconditionally on the WOULD MEASURE path, and the measurement it
    denied followed four lines later. A refuse path whose report cannot be
    trusted about its own behaviour has no standing to be trusted about the
    recordings.
    """
    result = _invoke(
        _write_ceiling(tmp_path), "--force", "--split-seeds", "42", "--train-seeds", "0"
    )

    assert result.exit_code == 0, result.output
    assert "stops before training" not in result.output
    started = next(line for line in result.output.splitlines() if line.startswith("started"))
    assert "yes" in started, started
    assert "MEASUREMENT" in result.output


def test_the_started_line_reports_each_outcome(tmp_path: Path) -> None:
    """All three branches, at library level so no environment is required."""
    report = _audit_folder(_write_loraiq_pattern(tmp_path), 1024, 512, "dirname", 2)
    keys = resolve_group_keys(
        [f.record_id for f in report.features],
        r"path:([^/]+/tx\d+)",
        collections={f.record_id: f.collection for f in report.features},
    )

    trains = decide(report, group_keys=keys, seconds_per_window_epoch=None)
    assert trains.status is DecisionStatus.WOULD_MEASURE
    assert "started       yes." in render_text(trains)

    held = decide(
        report,
        group_keys=keys,
        seconds_per_window_epoch=None,
        no_train_reason="torch is not installed, so this run ends at the classification above",
    )
    text = render_text(held)
    assert "started       no. torch is not installed" in text
    assert "stops before training" not in text

    refused = decide(_report_for_category_4(tmp_path), seconds_per_window_epoch=None)
    assert refused.status is DecisionStatus.REFUSED
    assert "nothing was built and nothing was trained" in render_text(refused)


def _report_for_category_4(tmp_path: Path):
    return _audit_folder(_write_ceiling(tmp_path / "ceil"), 1024, 512, "dirname", 1)


def test_a_run_that_cannot_train_says_why(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The torch-free branch, forced open on a machine that has torch.

    Without this the branch is unreachable here and a CLI that stopped passing
    the reason would still look correct: every local run trains, so "yes" is
    right by accident. The assertion is that the report names the obstacle,
    not merely that it avoids the old wrong sentence.
    """
    monkeypatch.setattr("iqforge.cli._torch_available", lambda: False)
    result = _invoke(
        _write_loraiq_pattern(tmp_path),
        "--dirname-level",
        "2",
        "--group-by",
        r"path:([^/]+/tx\d+)",
    )

    assert result.exit_code == 0, result.output
    assert "WOULD MEASURE" in result.output
    started = next(line for line in result.output.splitlines() if line.startswith("started"))
    assert "no." in started, started
    assert "torch is not installed" in result.output
    assert "stops before training" not in result.output
    assert "MEASUREMENT" not in result.output


def test_the_three_documents_agree_on_the_category_numbers() -> None:
    """Code, SPEC and methodology must not drift apart on what `category N` means.

    The command prints a number and a citation; a reader follows it into
    methodology §6. Nothing kept those in step, and §6 did not mention
    categories at all -- so `category 5` and `§6.5` looked like the same thing
    while meaning opposite ones: a refusal, and the dataset that passed.
    """
    import re

    root = Path(__file__).resolve().parent.parent
    spec = (root / "SPEC.md").read_text(encoding="utf-8")
    methodology = (root / "docs" / "methodology.md").read_text(encoding="utf-8")

    for category in Category:
        name = CATEGORY_META[category][0]
        row = rf"\|\s*{int(category)}\s*\|\s*{re.escape(name)}\s*\|"
        assert re.search(row, spec), f"SPEC has no row for category {int(category)} '{name}'"
        assert re.search(row, methodology), (
            f"methodology has no row for category {int(category)} '{name}'"
        )

    # The collision the table exists to head off. Compared with whitespace
    # removed: these documents are hard-wrapped, so a phrase that fits on one
    # line today can straddle two after an unrelated edit.
    squeezed = "".join(methodology.split())
    assert "".join("`category 5` and `§6.5` are not the same thing".split()) in squeezed
    assert "".join("There is deliberately no category for it.".split()) in squeezed
