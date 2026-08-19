"""Paired leakage measurement: recording-level split versus window-level split.

This is a library, not a command. The deliberately wrong split lives here and
is never reachable from the CLI. Dataset-specific preparation -- generating
synthetic recordings, locating a DASH7 packet, cutting a LoRaIQ frame segment
-- stays in `scripts/`. What this module owns is the part that is the same
every time: build the honest split, re-deal the same windows the wrong way,
train both arms, and summarise the paired difference.

Torch is imported only when a run is actually trained, so `build` and `audit`
do not grow a torch dependency by existing next to this file.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from iqforge import __version__
from iqforge.storage import ShardWriter, write_manifest

#: Held fixed across every published table. The stride sweep varies overlap at
#: this length; it does not vary the length itself.
DEFAULT_WINDOW = 1024

#: Split ratios used by every published grid. Chosen so a balanced nuisance
#: variable can appear on both sides; see `scripts/leakage_experiment.py`.
DEFAULT_SPLIT = "0.6,0.2,0.2"

#: Epochs the baseline was measured at. Changing this changes the numbers.
EPOCHS = 20


@dataclass
class Run:
    """One training run's outcome."""

    noise_sigma: float
    snr_db: float
    strategy: str
    split_seed: int
    train_seed: int
    test_accuracy: float
    train_accuracy: float
    train_windows: int
    test_windows: int
    #: None when the cell left the tool's default stride alone.
    stride: int | None = None
    #: Device, torch, CUDA, and numeric-stack versions this run was measured on.
    #: Rows measured on different environments are not comparable, and a table
    #: that does not carry this has already lost the ability to say so.
    environment: dict[str, str] = field(default_factory=dict)


MEASUREMENT_SCHEMA = 1

#: Split seeds the published grids used. Exposed as a default rather than
#: hardcoded so a run's sample size is a stated choice, not an accident.
DEFAULT_SPLIT_SEEDS = "42,7,1234,2026,99"

#: Training seeds the published grids used. Five split seeds times these three
#: is the 15 pairs every table in docs/methodology.md reports.
DEFAULT_TRAIN_SEEDS = "0,1,2"


@dataclass(frozen=True)
class BuildSpec:
    """How to build the honest, recording-level arm.

    The three experiment scripts differed only in these flags. One spec is the
    single code path; the scripts fill it in.
    """

    window: int = DEFAULT_WINDOW
    #: `None` omits `--window`/`--stride` and keeps the tool default.
    stride: int | None = None
    split: str = DEFAULT_SPLIT
    labels: str | None = None
    label_file: Path | None = None
    dirname_level: int | None = None
    group_by: str | None = None
    balance_by: str | None = None
    #: Directly from the manifest, not from the absence of a warning. A
    #: partially confounded split does not warn.
    assert_offsets_shared: bool = False


@dataclass(frozen=True)
class GridCell:
    """One folder of recordings, ready to measure, with the fields a `Run` needs.

    Preparation has already happened. This cell does not write recordings.
    """

    records: Path
    spec: BuildSpec
    noise_sigma: float = 0.0
    snr_db: float = float("nan")
    stride: int | None = None
    #: Prefix on the progress line (`stride=1024`, `snr= +3`).
    label: str = ""


def build_command(records: Path, out: Path, seed: int, spec: BuildSpec) -> list[str]:
    """Argv for the honest `iqforge build`. Kept as a list so tests can pin it."""
    argv = [
        sys.executable,
        "-m",
        "iqforge",
        "build",
        str(records),
        "-o",
        str(out),
        "--split",
        spec.split,
        "--seed",
        str(seed),
    ]
    if spec.labels is not None:
        argv += ["--labels", spec.labels]
    if spec.label_file is not None:
        argv += ["--label-file", str(spec.label_file)]
    if spec.dirname_level is not None:
        argv += ["--dirname-level", str(spec.dirname_level)]
    if spec.group_by is not None:
        argv += ["--group-by", spec.group_by]
    if spec.balance_by is not None:
        argv += ["--balance-by", spec.balance_by]
    if spec.stride is not None:
        argv += ["--window", str(spec.window), "--stride", str(spec.stride)]
    return argv


def build_recording_level(records: Path, out: Path, seed: int, spec: BuildSpec) -> None:
    """Build a dataset the normal way, through the CLI.

    Raises:
        RuntimeError: If iqforge warns. Swallowing that warning is how the first
            version of the synthetic experiment measured a confounded split
            instead of leakage.
    """
    if out.exists():
        shutil.rmtree(out)
    completed = subprocess.run(
        build_command(records, out, seed, spec),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if "warning" in output.lower():
        raise RuntimeError(
            f"iqforge warned while building the baseline split (seed {seed}); the "
            f"comparison would not be valid:\n{output.strip()}"
        )
    if spec.assert_offsets_shared:
        _assert_offsets_shared(out, seed)


def _assert_offsets_shared(dataset: Path, seed: int) -> None:
    """Check every carrier offset in test also appears in train."""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))

    def groups(split: str) -> set[str]:
        return {
            str(record.get("balance_group"))
            for record in manifest["splits"][split]["records"]
            if record.get("balance_group") is not None
        }

    unseen = groups("test") - groups("train")
    if unseen:
        raise RuntimeError(
            f"split seed {seed}: carrier offset(s) {sorted(unseen)} appear in test but "
            f"never in train. The recording-level arm would be measuring distribution "
            f"shift, not leakage. Adjust the split ratios or the recording count."
        )


def _load_all_windows(dataset: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read every window of every split back, with its label."""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    arrays: list[np.ndarray] = []
    labels: list[int] = []
    for split in ("train", "val", "test"):
        entry = manifest["splits"][split]
        for shard in entry["shards"]:
            arrays.append(np.load(dataset / shard))
        labels.extend(entry["labels"])
    return np.concatenate(arrays, axis=0), np.asarray(labels, dtype=np.int64), manifest


def build_window_level(source: Path, out: Path, seed: int) -> None:
    """Re-split the SAME windows at the window level.

    Takes the window pool of a recording-level dataset and deals the windows out
    individually, keeping each split's window count identical. That keeps
    training-set size out of the comparison: the only difference is which
    windows ended up where.

    The deal is stratified by class, so the label distribution also matches --
    otherwise a class imbalance would be confounded with the leakage.
    """
    windows, labels, manifest = _load_all_windows(source)
    counts = {s: manifest["splits"][s]["count"] for s in ("train", "val", "test")}

    rng = np.random.default_rng(seed)
    per_class: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        per_class.setdefault(int(label), []).append(index)
    for indices in per_class.values():
        rng.shuffle(indices)

    total = len(labels)
    assigned: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for _label, indices in sorted(per_class.items()):
        cursor = 0
        for split in ("train", "val", "test"):
            take = round(len(indices) * counts[split] / total)
            assigned[split].extend(indices[cursor : cursor + take])
            cursor += take
        assigned["test"].extend(indices[cursor:])  # rounding remainder

    if out.exists():
        shutil.rmtree(out)
    splits: dict[str, dict[str, Any]] = {}
    for split, indices in assigned.items():
        rng.shuffle(indices)
        writer = ShardWriter(out, split)
        chosen = np.asarray(indices, dtype=np.int64)
        writer.add(windows[chosen], [int(labels[i]) for i in chosen])
        writer.flush()
        splits[split] = {
            "shards": writer.shards,
            "labels": writer.labels,
            "count": writer.count,
            # Deliberately empty: under window-level splitting no split owns a
            # recording, which is exactly the property being measured.
            "records": [],
        }

    config = dict(manifest["config"])
    config["split_strategy"] = "window-level (DELIBERATELY WRONG, experiment only)"
    write_manifest(
        out,
        version=__version__,
        config=config,
        label_map=manifest["label_map"],
        source_files=manifest["source_files"],
        splits=splits,
    )


def split_counts(dataset: Path) -> tuple[int, int]:
    """`(train windows, test windows)` from a built dataset's manifest."""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    return int(manifest["splits"]["train"]["count"]), int(manifest["splits"]["test"]["count"])


def train_once(
    dataset: Path, train_seed: int, epochs: int = EPOCHS
) -> tuple[float, float, int, int]:
    """Train the baseline and return accuracies and split sizes."""
    from iqforge.training import train_baseline

    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    result = train_baseline(dataset, epochs=epochs, seed=train_seed)
    return (
        result.test_accuracy or 0.0,
        result.final_train_accuracy,
        manifest["splits"]["train"]["count"],
        manifest["splits"]["test"]["count"],
    )


def current_environment() -> dict[str, str]:
    """Device, torch, CUDA, and numeric-stack versions of this process.

    Degrades rather than raising when torch is absent. The guards that use this
    are about whether two sets of runs are comparable, and that question is
    still worth answering in a torch-free install -- `audit` and the refuse
    path both run there. Windowing and normalisation are numpy and scipy work
    regardless, so those versions are reported either way.
    """
    import numpy
    import scipy
    import sigmf

    base = {
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sigmf": sigmf.__version__,
    }
    try:
        from iqforge.training import DEFAULT_DEVICE, describe_environment, resolve_device
    except ModuleNotFoundError:
        return {"device": "none (torch not installed)", **base}
    return describe_environment(resolve_device(DEFAULT_DEVICE))


def check_environment(runs_path: Path) -> None:
    """Refuse to extend a checkpoint that was measured on another environment.

    An existing checkpoint that records no environment at all is refused too.
    Returning quietly in that case is what left the published grids unguarded:
    they were written before environment stamping existed, so every one of them
    carries `environment: null`, and the guard waved them through. "I cannot
    tell whether these are comparable" is a reason to stop, not to continue.
    """
    if not runs_path.exists():
        return
    try:
        existing = json.loads(runs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not existing:
        return
    recorded = next((r.get("environment") for r in existing if r.get("environment")), None)
    if not recorded:
        raise SystemExit(
            f"{runs_path.name} holds {len(existing)} run(s) that record no environment, "
            f"so there is no way to tell whether they are comparable with this process "
            f"({current_environment()}). Move the checkpoint aside to start a fresh grid."
        )
    now = current_environment()
    if recorded != now:
        raise SystemExit(
            f"{runs_path.name} holds runs measured on {recorded}, but this process "
            f"would measure on {now}. Rows from different environments are not "
            f"comparable and the table would not show it. Move the checkpoint aside "
            f"to start a fresh grid, or run in the original environment."
        )


def measure_cell_via_cli(
    records: Path,
    spec: BuildSpec,
    *,
    split_seeds: str = DEFAULT_SPLIT_SEEDS,
    train_seeds: str = DEFAULT_TRAIN_SEEDS,
    force: bool = False,
    force_reason: str = "",
) -> list[Run]:
    """Measure one cell by invoking the shipped command, and return every run.

    The experiment scripts go through the CLI rather than calling `run_grid`
    directly, so the numbers in `artifacts/` come from the same path a user
    gets. This adapter is the single copy of that call: three near-identical
    versions of it existed, and one of them silently dropped the seed lists.

    Returns every `Run` the command produced, not the first pair. Returning a
    pair is what reduced the published grids from 15 seed pairs to 1 while
    still reproducing the first pair exactly.

    Args:
        force: Pass `--force`. Requires `force_reason`, because an unexplained
            override of the refuse path is the one flag that must never be
            copied without understanding it.
    """
    if force and not force_reason:
        raise ValueError("measure_cell_via_cli(force=True) requires force_reason")
    command = [
        sys.executable, "-m", "iqforge", "measure-leakage", str(records),
        "--format", "json",
        "--split", spec.split,
        "--split-seeds", split_seeds,
        "--train-seeds", train_seeds,
    ]  # fmt: skip
    if force:
        command.append("--force")
    if spec.stride is not None:
        command += ["--window", str(spec.window), "--stride", str(spec.stride)]
    if spec.labels is not None:
        command += ["--labels", spec.labels]
    if spec.label_file is not None:
        command += ["--label-file", str(spec.label_file)]
    if spec.dirname_level is not None:
        command += ["--dirname-level", str(spec.dirname_level)]
    if spec.group_by is not None:
        command += ["--group-by", spec.group_by]
    if spec.balance_by is not None:
        command += ["--balance-by", spec.balance_by]

    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "measure-leakage refused or failed for "
            f"{records}:\n{completed.stdout}\n{completed.stderr}"
        )
    return runs_from_payload(json.loads(completed.stdout), fallback_stride=spec.stride)


def runs_from_payload(payload: dict[str, Any], *, fallback_stride: int | None = None) -> list[Run]:
    """Rebuild `Run` objects from a `measure-leakage --format json` payload.

    One parser instead of three hand-written dictionary walks. It checks
    `measurement_schema` for the same reason `read_manifest` checks
    `manifest_schema`: a reader that guesses at a shape it does not recognise
    produces a plausible wrong answer.
    """
    measured = payload["measurement"]
    schema = measured.get("measurement_schema")
    if schema is not None and schema > MEASUREMENT_SCHEMA:
        raise RuntimeError(
            f"measurement payload declares schema {schema}, this build understands "
            f"{MEASUREMENT_SCHEMA}. Upgrade iqforge or regenerate with a matching build."
        )
    rows = measured.get("rows")
    if rows is None:
        raise RuntimeError(
            "measurement payload carries no 'rows'. A payload without every run cannot "
            "be aggregated, and taking the first pair is how sample size gets lost."
        )
    return [
        Run(
            noise_sigma=0.0,
            snr_db=float("nan"),
            strategy=str(row["strategy"]),
            split_seed=int(row["split_seed"]),
            train_seed=int(row["train_seed"]),
            test_accuracy=float(row["test_accuracy"]),
            train_accuracy=float(row["train_accuracy"]),
            train_windows=int(row["train_windows"]),
            test_windows=int(row["test_windows"]),
            stride=row.get("stride", fallback_stride),
            environment=dict(row.get("environment") or {}),
        )
        for row in rows
    ]


def guard_artifact_rows(runs_path: Path, new_rows: int) -> None:
    """Refuse to replace a published grid with a smaller one.

    `artifacts/` is quoted by `docs/methodology.md` and `README.md`, and the
    scripts write their checkpoints straight over those files. A migration that
    reduced the seed grid from 15 pairs to 1 would have rewritten a 180-run
    table as a 12-run one, reproducing the first pair exactly and reporting a
    standard error of zero. Value equality is not enough; the shape has to hold.

    Raises:
        SystemExit: If the file exists and holds more runs than are about to
            replace it.
    """
    if not runs_path.exists():
        return
    try:
        existing = json.loads(runs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(existing, list) and len(existing) > new_rows:
        raise SystemExit(
            f"{runs_path.name} holds {len(existing)} runs and this grid would write "
            f"{new_rows}. A published artifact must not lose sample size: the tables in "
            f"docs/methodology.md and README.md quote these files. Check the seed lists, "
            f"or move the file aside deliberately if the smaller grid is what you want."
        )


def run_grid(
    cells: Sequence[GridCell],
    split_seeds: Sequence[int],
    train_seeds: Sequence[int],
    *,
    checkpoint: Callable[[list[Run]], None] | None = None,
    epochs: int = EPOCHS,
    verbose: bool = True,
) -> list[Run]:
    """Build both arms and train them, for every cell and seed pair.

    Args:
        cells: Prepared recording folders plus the build flags and the numbers
            to stamp on each `Run`. One loop serves every sweep so the scripts
            cannot drift apart in how they build or train.
        split_seeds: Seeds for the recording-level split.
        train_seeds: Seeds for weight init and batch order.
        checkpoint: Called with the runs so far after every run.
        epochs: Training length. Default is the published-table value.
    """
    runs: list[Run] = []
    environment = current_environment()
    work = Path(tempfile.mkdtemp(prefix="iqforge-measure-"))
    total = len(cells) * len(split_seeds) * len(train_seeds) * 2
    done = 0
    started = time.time()
    try:
        for cell in cells:
            spec = cell.spec
            if cell.stride is not None and spec.stride is None:
                spec = replace(spec, stride=cell.stride)
            for split_seed in split_seeds:
                rec_ds = work / f"r_{id(cell)}_{split_seed}"
                win_ds = work / f"w_{id(cell)}_{split_seed}"
                build_recording_level(cell.records, rec_ds, split_seed, spec)
                build_window_level(rec_ds, win_ds, split_seed)
                for strategy, dataset in (("recording-level", rec_ds), ("window-level", win_ds)):
                    for train_seed in train_seeds:
                        t0 = time.time()
                        acc, train_acc, n_train, n_test = train_once(
                            dataset, train_seed, epochs=epochs
                        )
                        done += 1
                        run = Run(
                            noise_sigma=cell.noise_sigma,
                            snr_db=cell.snr_db,
                            strategy=strategy,
                            split_seed=split_seed,
                            train_seed=train_seed,
                            test_accuracy=acc,
                            train_accuracy=train_acc,
                            train_windows=n_train,
                            test_windows=n_test,
                            stride=cell.stride if cell.stride is not None else spec.stride,
                            environment=environment,
                        )
                        runs.append(run)
                        if checkpoint is not None:
                            checkpoint(runs)
                        rate = (time.time() - started) / done
                        label = cell.label or (
                            f"stride={run.stride:4d}" if run.stride is not None else "cell"
                        )
                        if verbose:
                            print(
                                f"  [{done:3d}/{total}] {label} {strategy:15s} "
                                f"split={split_seed} train={train_seed}  "
                                f"test={acc:6.2%}  train={train_acc:6.2%}  "
                                f"({time.time() - t0:.0f}s/run, {n_train}/{n_test} windows, "
                                f"~{rate * (total - done) / 60:.0f} min left)",
                                flush=True,
                            )
                shutil.rmtree(rec_ds, ignore_errors=True)
                shutil.rmtree(win_ds, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if verbose:
        print(f"\ntotal wall time {(time.time() - started) / 60:.1f} min for {done} runs")
    return runs


def measure_pair(
    records: Path,
    spec: BuildSpec,
    *,
    split_seed: int,
    train_seed: int,
    epochs: int = EPOCHS,
    verbose: bool = True,
) -> tuple[Run, Run]:
    """One seed pair, both arms. The unit the LoRaIQ acceptance cell is."""
    cell = GridCell(
        records=records,
        spec=spec,
        stride=spec.stride,
        label=f"stride={spec.stride}" if spec.stride is not None else "",
    )
    runs = run_grid([cell], [split_seed], [train_seed], epochs=epochs, verbose=verbose)
    rec = next(r for r in runs if r.strategy == "recording-level")
    win = next(r for r in runs if r.strategy == "window-level")
    return rec, win


# --------------------------------------------------------------------------
# Paired statistics
# --------------------------------------------------------------------------


def paired_differences(runs: Sequence[Run]) -> list[float]:
    """Window-level minus recording-level accuracy, per (split seed, train seed).

    The design is paired by construction: for one split seed and one training
    seed, both arms use the same recordings, the same window pool, the same
    split sizes and the same weight initialisation. Only the assignment differs,
    so the difference within a pair isolates the effect while the seed-to-seed
    scatter -- which dominates everything else in this project -- cancels.
    """
    paired: dict[tuple[int, int], dict[str, float]] = {}
    for run in runs:
        paired.setdefault((run.split_seed, run.train_seed), {})[run.strategy] = run.test_accuracy
    return [
        arms["window-level"] - arms["recording-level"] for arms in paired.values() if len(arms) == 2
    ]


def _sd(values: list[float]) -> float:
    """Standard deviation, 0 for a single sample (a smoke grid)."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


@dataclass(frozen=True)
class CellStats:
    """Accuracy means and the paired inflation for one table row."""

    rec_mean: float
    rec_sd: float
    win_mean: float
    win_sd: float
    mean_diff: float
    stderr: float
    n: int

    def accuracy_cells(self) -> str:
        """The four right-hand columns shared by every published table."""
        return (
            f"{self.rec_mean:.1%} ± {self.rec_sd:.1%} | "
            f"{self.win_mean:.1%} ± {self.win_sd:.1%} | "
            f"**{self.mean_diff * 100:+.1f} pp** ± {self.stderr * 100:.1f} | {self.n} |"
        )


def cell_stats(runs: Sequence[Run]) -> CellStats | None:
    """Paired inflation for a set of runs that share a table row.

    Returns `None` when either arm is missing -- checkpointing writes the table
    mid-cell, before the second arm has run, and a row with one side is not a
    number.
    """
    rec = [r.test_accuracy for r in runs if r.strategy == "recording-level"]
    win = [r.test_accuracy for r in runs if r.strategy == "window-level"]
    diffs = paired_differences(runs)
    if not rec or not win or not diffs:
        return None
    stderr = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    return CellStats(
        rec_mean=statistics.mean(rec),
        rec_sd=_sd(rec),
        win_mean=statistics.mean(win),
        win_sd=_sd(win),
        mean_diff=statistics.mean(diffs),
        stderr=stderr,
        n=len(diffs),
    )


def summarise_stride_table(
    runs: Sequence[Run],
    *,
    window: int = DEFAULT_WINDOW,
    segment: int | None = None,
    caption: str,
) -> str:
    """Stride-sweep table. Overlap is the mechanism; stride 1024 carries the claim."""
    has_windows = segment is not None
    header = (
        "| stride | overlap | windows/rec | recording-level | window-level | inflation (paired) | n |"  # noqa: E501
        if has_windows
        else "| stride | overlap | recording-level | window-level | inflation (paired) | n |"
    )
    rule = "|---|---|---|---|---|---|---|" if has_windows else "|---|---|---|---|---|---|"
    lines = [header, rule]
    strides = sorted({r.stride for r in runs if r.stride is not None}, reverse=True)
    for stride in strides:
        stats = cell_stats([r for r in runs if r.stride == stride])
        if stats is None:
            continue
        overlap = 1.0 - stride / window
        left = f"| {stride} | {overlap:.0%} |"
        if has_windows:
            assert segment is not None
            left += f" {(segment - window) // stride + 1} |"
        lines.append(f"{left} {stats.accuracy_cells()}")
    lines += ["", caption]
    return "\n".join(lines)


def summarise_snr_table(
    runs: Sequence[Run],
    *,
    caption: str,
    in_band_gain_db: float | None = None,
) -> str:
    """SNR-sweep table, grouped by `noise_sigma` or by `snr_db` when in-band is shown."""
    if in_band_gain_db is None:
        lines = [
            "| burst SNR | recording-level | window-level | inflation (paired) | n |",
            "|---|---|---|---|---|",
        ]
        keys = sorted({r.noise_sigma for r in runs}, reverse=True)
        for noise in keys:
            at = [r for r in runs if r.noise_sigma == noise]
            stats = cell_stats(at)
            inflation = stats.accuracy_cells() if stats else "- | 0 |"
            if stats is None:
                rec = [r.test_accuracy for r in at if r.strategy == "recording-level"]
                win = [r.test_accuracy for r in at if r.strategy == "window-level"]
                rec_s = f"{statistics.mean(rec):.1%} ± {_sd(rec):.1%}" if rec else "nan ± 0.0%"
                win_s = f"{statistics.mean(win):.1%} ± {_sd(win):.1%}" if win else "nan ± 0.0%"
                inflation = f"{rec_s} | {win_s} | - | 0 |"
            lines.append(f"| {at[0].snr_db:+.1f} dB | {inflation}")
    else:
        lines = [
            "| wideband SNR | in-band SNR | recording-level | window-level | inflation (paired) | n |",  # noqa: E501
            "|---|---|---|---|---|---|",
        ]
        keys = sorted({r.snr_db for r in runs}, reverse=True)
        for snr in keys:
            at = [r for r in runs if r.snr_db == snr]
            stats = cell_stats(at)
            wide = "raw" if math.isinf(snr) else f"{snr:+.0f} dB"
            inband = "-" if math.isinf(snr) else f"{snr + in_band_gain_db:+.1f} dB"
            if stats is None:
                rec = [r.test_accuracy for r in at if r.strategy == "recording-level"]
                win = [r.test_accuracy for r in at if r.strategy == "window-level"]
                rec_s = f"{statistics.mean(rec):.1%} ± {_sd(rec):.1%}" if rec else "nan ± 0.0%"
                win_s = f"{statistics.mean(win):.1%} ± {_sd(win):.1%}" if win else "nan ± 0.0%"
                inflation = f"{rec_s} | {win_s} | - | {0} |"
            else:
                inflation = stats.accuracy_cells()
            lines.append(f"| {wide} | {inband} | {inflation}")
    lines += ["", caption]
    return "\n".join(lines)
