"""Tests for the paired leakage-measurement core.

The expensive LoRaIQ bit-exact gate needs the recordings on disk and is skipped
when they are not. What always runs is the part that does not train: argv
construction, the window-level deal, paired arithmetic, and reproducing the
published tables from the recorded run files.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from helpers import loraiq_paths, loraiq_skip_reason
from iqforge.measurement import (
    BuildSpec,
    Run,
    build_command,
    build_recording_level,
    build_window_level,
    cell_stats,
    check_environment,
    current_environment,
    guard_artifact_rows,
    paired_differences,
    split_counts,
    summarise_snr_table,
    summarise_stride_table,
)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
ARTIFACTS = ROOT / "artifacts"

#: The published LoRaIQ cell. `artifacts/leakage_loraiq_runs.json`, first pair.
LORAIQ_CELL = {
    "split_seed": 42,
    "train_seed": 0,
    "stride": 1024,
    "recording-level": {
        "test_accuracy": 0.6498829039812647,
        "train_windows": 2926,
        "test_windows": 854,
    },
    "window-level": {
        "test_accuracy": 0.6787383177570093,
        "train_windows": 2925,
        "test_windows": 856,
    },
}


def _runs_from_json(path: Path) -> list[Run]:
    known = {f.name for f in fields(Run)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Run(**{k: v for k, v in row.items() if k in known}) for row in payload]


def test_loraiq_build_flags_hold_transmissions_together() -> None:
    """Without --group-by the LoRaIQ split leaks air time; the argv must name it."""
    spec = BuildSpec(
        window=1024,
        stride=1024,
        split="0.6,0.2,0.2",
        labels="csv",
        label_file=Path("labels.csv"),
        group_by="csv:groups.csv",
    )
    argv = build_command(Path("rec"), Path("out"), 42, spec)
    assert argv[argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--stride") + 1] == "1024"
    assert argv[argv.index("--labels") + 1] == "csv"
    assert argv[argv.index("--group-by") + 1] == "csv:groups.csv"


def test_synthetic_build_omits_stride_when_the_cell_uses_the_default() -> None:
    """The SNR sweep leaves the tool's default stride alone."""
    spec = BuildSpec(balance_by="core:freq_lower_edge", assert_offsets_shared=True)
    argv = build_command(Path("rec"), Path("out"), 42, spec)
    assert "--stride" not in argv
    assert "--window" not in argv
    assert argv[argv.index("--balance-by") + 1] == "core:freq_lower_edge"


def test_dash7_build_reads_the_channel_from_the_directory() -> None:
    spec = BuildSpec(
        window=1024, stride=512, labels="dirname", dirname_level=2, split="0.6,0.2,0.2"
    )
    argv = build_command(Path("rec"), Path("out"), 7, spec)
    assert argv[argv.index("--labels") + 1] == "dirname"
    assert argv[argv.index("--dirname-level") + 1] == "2"
    assert argv[argv.index("--stride") + 1] == "512"


def test_paired_differences_cancel_seed_and_isolate_the_assignment() -> None:
    runs = [
        Run(0, 0, "recording-level", 42, 0, 0.50, 0.9, 10, 4),
        Run(0, 0, "window-level", 42, 0, 0.60, 0.9, 10, 4),
        Run(0, 0, "recording-level", 7, 1, 0.80, 0.9, 10, 4),
        Run(0, 0, "window-level", 7, 1, 0.85, 0.9, 10, 4),
    ]
    assert paired_differences(runs) == pytest.approx([0.10, 0.05])


def test_cell_stats_wait_for_both_arms() -> None:
    """A checkpoint mid-cell must not become a one-sided inflation figure."""
    half = [Run(0, 0, "recording-level", 42, 0, 0.50, 0.9, 10, 4, stride=1024)]
    assert cell_stats(half) is None
    both = half + [Run(0, 0, "window-level", 42, 0, 0.60, 0.9, 10, 4, stride=1024)]
    stats = cell_stats(both)
    assert stats is not None
    assert stats.n == 1
    assert stats.mean_diff == pytest.approx(0.10)


def test_a_single_pair_prints_no_spread() -> None:
    """One seed pair has nothing to estimate, and must not look like it does.

    A standard deviation over one sample is 0, and "± 0.0" in a results table
    is the strongest claim the format can make -- printed exactly where the
    evidence is weakest.
    """
    runs = [
        Run(0, 0, "recording-level", 42, 0, 0.50, 0.9, 10, 4, stride=1024),
        Run(0, 0, "window-level", 42, 0, 0.60, 0.9, 10, 4, stride=1024),
    ]
    stats = cell_stats(runs)
    assert stats is not None
    assert stats.n == 1
    assert not stats.has_spread

    cells = stats.accuracy_cells()
    assert "±" not in cells, cells
    assert "(not estimated)" in cells, cells
    assert cells.endswith("| 1 |"), cells


def test_a_repeated_cell_still_prints_its_spread() -> None:
    """The n = 1 wording must not leak into rows that do have a sample."""
    runs = [
        Run(0, 0, "recording-level", 42, 0, 0.50, 0.9, 10, 4, stride=1024),
        Run(0, 0, "window-level", 42, 0, 0.60, 0.9, 10, 4, stride=1024),
        Run(0, 0, "recording-level", 7, 1, 0.55, 0.9, 10, 4, stride=1024),
        Run(0, 0, "window-level", 7, 1, 0.70, 0.9, 10, 4, stride=1024),
    ]
    stats = cell_stats(runs)
    assert stats is not None
    assert stats.n == 2
    assert stats.has_spread

    cells = stats.accuracy_cells()
    assert cells.count("±") == 3, cells
    assert "(not estimated)" not in cells, cells


def test_a_half_measured_row_does_not_invent_a_spread() -> None:
    """Checkpointing writes the table mid-cell, before the second arm exists.

    That row cannot report an inflation, which `cell_stats` already refuses.
    What it must also not do is print the one accuracy it does have as though
    it had been repeated.
    """
    table = summarise_snr_table(
        [Run(0.08, 5.0, "recording-level", 42, 0, 0.50, 0.9, 10, 4)],
        caption="one arm measured so far",
    )
    assert "± 0.0%" not in table, table
    assert "nan" not in table, table
    assert "50.0%" in table, table


def test_window_level_split_is_deterministic(tmp_path: Path) -> None:
    """Same seed, same window pool, same deal -- twice."""
    if not any(EXAMPLES.glob("*.sigmf-meta")):
        pytest.skip("examples/ recordings have not been generated")
    rec = tmp_path / "rec"
    win_a = tmp_path / "win_a"
    win_b = tmp_path / "win_b"
    spec = BuildSpec(window=1024, stride=512)
    build_recording_level(EXAMPLES, rec, 42, spec)
    build_window_level(rec, win_a, 42)
    build_window_level(rec, win_b, 42)
    rec_train, rec_test = split_counts(rec)
    assert rec_train > 0 and rec_test > 0
    assert split_counts(win_a) == split_counts(win_b)
    labels_a = json.loads((win_a / "manifest.json").read_text(encoding="utf-8"))["splits"]
    labels_b = json.loads((win_b / "manifest.json").read_text(encoding="utf-8"))["splits"]
    assert labels_a["train"]["labels"] == labels_b["train"]["labels"]
    assert labels_a["test"]["labels"] == labels_b["test"]["labels"]


def test_summarise_reproduces_the_published_loraiq_table() -> None:
    """Collapsing five summarise functions must not change a published table."""
    runs = _runs_from_json(ARTIFACTS / "leakage_loraiq_runs.json")
    table = summarise_stride_table(
        runs,
        window=1024,
        segment=15_244,
        caption=(
            "LoRaIQ, class = propagation environment (drone_los, drone_nlos, "
            "pedestrian_partial_los, pedestrian_nlos, indoor), 312 recordings over 13 "
            "capture sessions, grouped by transmission id so simultaneous receptions stay "
            "in one split. Window fixed at 1024 samples over a 15244-sample segment "
            "centred on each frame; no noise added. Overlap is the only thing that moves "
            "between rows. Inflation is the mean paired difference ± its standard error."
        ),
    )
    published = (ARTIFACTS / "leakage_loraiq_table.md").read_text(encoding="utf-8").strip()
    assert table.strip() == published


def test_summarise_snr_rows_match_the_published_synthetic_table() -> None:
    runs = _runs_from_json(ARTIFACTS / "leakage_runs.json")
    table = summarise_snr_table(runs, caption="ignored")
    published_rows = [
        line
        for line in (ARTIFACTS / "leakage_table.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    ]
    got_rows = [line for line in table.splitlines() if line.startswith("|")]
    assert got_rows == published_rows


def test_summarise_stride_rows_match_the_published_synthetic_stride_table() -> None:
    runs = _runs_from_json(ARTIFACTS / "leakage_stride_runs.json")
    table = summarise_stride_table(runs, window=1024, caption="ignored")
    published_rows = [
        line
        for line in (ARTIFACTS / "leakage_stride_table.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    ]
    got_rows = [line for line in table.splitlines() if line.startswith("|")]
    assert got_rows == published_rows


@pytest.mark.skipif(loraiq_skip_reason() is not None, reason=loraiq_skip_reason() or "")
def test_loraiq_cell_reproduces_the_recorded_number(tmp_path: Path) -> None:
    """The Faz 2 gate: stride 1024 / split 42 / train 0, both arms.

    Layered: window counts first (no training), then the recorded accuracies.
    """
    pytest.importorskip("torch")
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from leakage_loraiq import WINDOW, frame_offsets, prepare  # noqa: E402

    from iqforge.measurement import measure_pair

    paths = loraiq_paths()
    assert paths is not None
    source, index, labels, groups = paths
    prepared = tmp_path / "prepared"
    prepare(source, prepared, frame_offsets(index))
    spec = BuildSpec(
        window=WINDOW,
        stride=1024,
        split="0.6,0.2,0.2",
        labels="csv",
        label_file=labels,
        group_by=f"csv:{groups}",
    )
    rec, win = measure_pair(prepared, spec, split_seed=42, train_seed=0)
    expect_rec = LORAIQ_CELL["recording-level"]
    expect_win = LORAIQ_CELL["window-level"]
    assert rec.train_windows == expect_rec["train_windows"]
    assert rec.test_windows == expect_rec["test_windows"]
    assert win.train_windows == expect_win["train_windows"]
    assert win.test_windows == expect_win["test_windows"]
    assert rec.test_accuracy == expect_rec["test_accuracy"]
    assert win.test_accuracy == expect_win["test_accuracy"]


# --------------------------------------------------------------------------
# Guards that protect the published grids
# --------------------------------------------------------------------------


def test_check_environment_stops_on_a_checkpoint_that_records_nothing(tmp_path):
    """The published grids carry `environment: null` and were waved through.

    They were written before environment stamping existed, so the guard's
    early return applied to exactly the files it most needed to protect.
    "I cannot tell whether these are comparable" is a reason to stop.
    """
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps([{"environment": None} for _ in range(180)]), encoding="utf-8")
    with pytest.raises(SystemExit, match="record no environment"):
        check_environment(path)


def test_check_environment_continues_when_the_environment_matches(tmp_path):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps([{"environment": current_environment()}]), encoding="utf-8")
    check_environment(path)


def test_guard_refuses_to_shrink_a_published_grid(tmp_path):
    """15 seed pairs to 1 reproduces the first pair and loses the measurement."""
    path = tmp_path / "grid.json"
    path.write_text(json.dumps([{} for _ in range(180)]), encoding="utf-8")
    with pytest.raises(SystemExit, match="must not lose sample size"):
        guard_artifact_rows(path, 12)


def test_guard_allows_the_same_size_or_larger(tmp_path):
    path = tmp_path / "grid.json"
    path.write_text(json.dumps([{} for _ in range(180)]), encoding="utf-8")
    guard_artifact_rows(path, 180)
    guard_artifact_rows(path, 300)
    guard_artifact_rows(tmp_path / "absent.json", 1)


def test_the_scripts_plan_the_published_grid_sizes():
    """A regression on the exact failure: seeds collapsing to one pair.

    Checked as arithmetic rather than by running the grids, which take hours.
    If a future change hardcodes seeds again, these products stop matching the
    row counts the tables in docs/methodology.md report.
    """
    import sys

    sys.path.insert(0, "scripts")
    import leakage_experiment as synthetic
    import leakage_loraiq as loraiq
    import leakage_real as real

    pairs = len(synthetic.SPLIT_SEEDS.split(",")) * len(synthetic.TRAIN_SEEDS.split(","))
    assert pairs == 15
    assert len(synthetic.NOISE_LEVELS) * pairs * 2 == 180
    assert len(synthetic.STRIDES) * pairs * 2 == 150
    assert len(real.STRIDES) * pairs * 2 == 150
    assert len(loraiq.STRIDES) * pairs * 2 == 150


def test_current_environment_degrades_without_torch(monkeypatch):
    """The comparability guards must work in a torch-free install.

    `audit` and the refuse path both run there, and asking whether two sets of
    runs are comparable does not need a training backend. numpy and scipy are
    still reported, because windowing and normalisation depend on them either
    way.
    """
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for module in [m for m in list(sys.modules) if m.startswith(("torch", "iqforge.training"))]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    env = current_environment()
    assert "numpy" in env
    assert "scipy" in env
    assert "torch" not in env


def test_train_once_forwards_the_requested_device(tmp_path, monkeypatch):
    """measure-leakage's training path must honour --device, not ignore it."""
    pytest.importorskip("torch")
    from iqforge.measurement import train_once

    (tmp_path / "manifest.json").write_text(
        json.dumps({"splits": {"train": {"count": 1}, "test": {"count": 1}}}),
        encoding="utf-8",
    )
    seen: dict[str, str] = {}

    class _Result:
        test_accuracy = 0.5
        final_train_accuracy = 0.75
        environment = {"device": "cuda"}

    def fake_train_baseline(dataset, *, epochs, seed, device_choice="cpu"):
        seen["device_choice"] = device_choice
        return _Result()

    monkeypatch.setattr("iqforge.training.train_baseline", fake_train_baseline)
    train_once(tmp_path, train_seed=0, epochs=1, device_choice="cuda")
    assert seen["device_choice"] == "cuda"

    train_once(tmp_path, train_seed=0, epochs=1)
    assert seen["device_choice"] == "cpu"


def test_current_environment_stamps_an_opt_in_cuda_device(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args, **kwargs: "mocked-gpu")
    assert current_environment("cpu")["device"] == "cpu"
    assert current_environment("cuda")["device"] == "cuda"
    assert current_environment()["device"] == "cpu"
