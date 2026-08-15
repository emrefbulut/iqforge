"""Measure how much window-level splitting inflates reported test accuracy.

This is the experiment behind the claim the README makes. It compares two ways
of splitting the SAME windows:

  recording-level  every window of a recording goes to one split (what iqforge
                   does, SPEC §5.6)
  window-level     windows are dealt out individually, so neighbouring windows
                   -- which overlap by half a window -- land on both sides

Both are trained with the same model, the same epochs, and the same number of
training windows. The only thing that differs is which split each window went
to, so the gap between the two reported accuracies is the inflation.

Two design points worth stating, because they decide whether the number means
anything:

1. The deliberately wrong split lives HERE, not in the CLI. `iqforge build`
   refuses to split at the window level; proving why that refusal is right
   requires doing the wrong thing on purpose, and that capability must not ship
   to users.

2. It sweeps SNR. At the SNR `examples/` ships with, the honest split already
   scores ~100%, so a leaky split cannot score higher and the measured gap is
   zero -- which would look like evidence AGAINST the concern. Leakage only
   shows up when the task is hard enough that the model needs the shortcut. The
   result is therefore a curve, not a number: it says WHEN leakage matters.

`examples/` is never touched. Recordings are generated into a temporary
directory at each noise level.

Usage:
    uv run python scripts/leakage_experiment.py --quick     # smoke, ~2 min
    uv run python scripts/leakage_experiment.py             # full grid
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_example as gen  # noqa: E402

from iqforge import __version__  # noqa: E402
from iqforge.storage import ShardWriter, write_manifest  # noqa: E402
from iqforge.training import train_baseline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

#: Noise levels to sweep, as the standard deviation per quadrature.
#:
#: Every level here is in the band where the answer can vary. A first run swept
#: 0.02 to 0.25 and measured both arms at 99-100% for 0.02, 0.05 and 0.08 alike:
#: with the honest split already at the ceiling, a leaky one cannot beat it and
#: half the compute bought a row of zeros. 0.08 is kept as the ceiling anchor --
#: the curve needs one end where the gap is provably absent -- and the rest of
#: the resolution goes below it, where the model starts needing the shortcut.
#: `examples/` ships at 0.02, well inside the flat region.
NOISE_LEVELS = (0.08, 0.11, 0.14, 0.17, 0.20, 0.25)

#: Ratios and dataset size are chosen together so that every carrier offset is
#: present in both train and test, for every split seed used below -- measured,
#: not assumed. With `examples/`-sized data the balancer leaves whole offsets to
#: a single split, the recording-level arm is then tested on a carrier it never
#: trained on, and it collapses to chance. That is a distribution shift, not
#: leakage, and it would swamp the effect being measured.
#: 48 recordings at 0.6/0.2/0.2 gives 28/10/10 with all four offsets shared.
SPLIT_RATIOS = "0.6,0.2,0.2"
RECORDS_PER_CELL = 6
EPOCHS = 20

#: Window length held fixed for the stride sweep, so overlap is the only thing
#: that moves.
WINDOW = 1024

#: Strides to sweep, at a fixed SNR. Overlap is the actual MECHANISM of the
#: leak: at stride 1024 windows are disjoint and a window-level split can only
#: put *different* samples on both sides, while at stride 128 each window shares
#: 7/8 of its samples with its neighbour and the test set is nearly a copy of
#: the training set. The SNR sweep shows when leakage matters; this shows why.
STRIDES = (1024, 768, 512, 256, 128)

#: Fixed noise for the stride sweep: -0.8 dB, in the band where the SNR sweep
#: found the largest inflation, so there is room for overlap to move the number.
STRIDE_NOISE = 0.17


def burst_snr_db(noise_sigma: float) -> float:
    """SNR of the burst against the additive noise, in dB.

    The burst is normalised to `BURST_RMS`, so its power is that squared.
    Complex Gaussian noise with `sigma` per quadrature has power `2 * sigma^2`.
    """
    return 10.0 * math.log10(gen.BURST_RMS**2 / (2.0 * noise_sigma**2))


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
    #: None for the SNR sweep, which leaves the tool's default stride alone.
    stride: int | None = None
    #: Device / torch / CUDA versions this run was measured on. Rows measured on
    #: different devices are not comparable, and a table that does not carry
    #: this has already lost the ability to say so.
    environment: dict[str, str] = field(default_factory=dict)


def generate_recordings(noise_sigma: float, out_dir: Path) -> None:
    """Write the experiment's recordings at a given noise level."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for plan in gen._build_plans(RECORDS_PER_CELL):
        gen.write_record(plan, gen.build_signal(plan, noise_sigma=noise_sigma), out_dir)


def build_recording_level(records: Path, out: Path, seed: int, stride: int | None = None) -> None:
    """Build a dataset the normal way, through the CLI.

    Args:
        records: Directory of SigMF recordings.
        out: Dataset output directory.
        seed: Split seed.
        stride: Step between windows. `None` keeps the tool's default.

    Raises:
        RuntimeError: If iqforge warns that the nuisance variable could not be
            balanced. Swallowing that warning is how the first version of this
            experiment measured a confounded split instead of leakage -- the
            recording-level arm was being tested on carrier offsets it had never
            trained on, which looks like a catastrophic result and is really a
            distribution shift. If this fires, the grid is wrong, not the tool.
    """
    if out.exists():
        shutil.rmtree(out)
    argv = [
        sys.executable,
        "-m",
        "iqforge",
        "build",
        str(records),
        "-o",
        str(out),
        "--split",
        SPLIT_RATIOS,
        "--seed",
        str(seed),
        "--balance-by",
        "core:freq_lower_edge",
    ]
    if stride is not None:
        argv += ["--window", str(WINDOW), "--stride", str(stride)]
    completed = subprocess.run(
        argv,
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
    _assert_offsets_shared(out, seed)


def _assert_offsets_shared(dataset: Path, seed: int) -> None:
    """Check every carrier offset in test also appears in train.

    The absence of a warning is not enough: iqforge warns only when a split
    collapses to a single group, and a partially confounded split passes that
    check while still leaving the model tested on unseen carriers. The baseline
    arm has to be a fair one or the whole comparison is meaningless, so this
    verifies the property directly instead of trusting the warning.
    """
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
            f"shift, not leakage. Adjust SPLIT_RATIOS / RECORDS_PER_CELL."
        )


def _load_all_windows(dataset: Path) -> tuple[np.ndarray, np.ndarray, dict]:
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

    # Deal each class's windows across the splits in the target proportions, so
    # both the split sizes and the class balance match the recording-level run.
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
    splits: dict[str, dict] = {}
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


def train_once(dataset: Path, train_seed: int) -> tuple[float, float, int, int]:
    """Train the baseline and return accuracies and split sizes."""
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    result = train_baseline(dataset, epochs=EPOCHS, seed=train_seed)
    return (
        result.test_accuracy or 0.0,
        result.final_train_accuracy,
        manifest["splits"]["train"]["count"],
        manifest["splits"]["test"]["count"],
    )


def current_environment() -> dict[str, str]:
    """Device / torch / CUDA of the process about to run a grid."""
    from iqforge.training import DEFAULT_DEVICE, describe_environment, resolve_device

    return describe_environment(resolve_device(DEFAULT_DEVICE))


def check_environment(runs_path: Path) -> None:
    """Refuse to extend a checkpoint that was measured on another device.

    A grid resumed on a different device produces a table whose rows are not
    comparable, and nothing in the output would show it. Better to stop: this
    is the same class of mistake as the acquisition method in §7 -- each run
    individually correct, the aggregate invalid, invisible in the result.
    """
    if not runs_path.exists():
        return
    try:
        existing = json.loads(runs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    recorded = next((r.get("environment") for r in existing if r.get("environment")), None)
    if not recorded:
        return
    now = current_environment()
    if recorded != now:
        raise SystemExit(
            f"{runs_path.name} holds runs measured on {recorded}, but this process "
            f"would measure on {now}. Rows from different devices are not comparable "
            f"and the table would not show it. Move the checkpoint aside to start a "
            f"fresh grid, or run on the original device."
        )


def run_grid(
    cells: list[tuple[float, int | None]],
    split_seeds: list[int],
    train_seeds: list[int],
    checkpoint: Callable[[list[Run]], None] | None = None,
) -> list[Run]:
    """Run the experiment over a list of cells.

    Args:
        cells: `(noise_sigma, stride)` pairs. `stride=None` keeps the tool's
            default, which is what the SNR sweep uses; the stride sweep holds
            the noise fixed and varies the stride instead. One loop serves both
            so the two sweeps cannot drift apart in how they build or train.
        split_seeds: Seeds for the recording-level split.
        train_seeds: Seeds for weight init and batch order.
        checkpoint: Called with the runs so far after every run. The full grid
            takes the better part of an hour; without this, a run that is
            interrupted at 31 of 72 leaves nothing on disk, which is exactly
            what happened the first time.
    """
    runs: list[Run] = []
    environment = current_environment()
    work = Path(tempfile.mkdtemp(prefix="iqforge-leakage-"))
    total = len(cells) * len(split_seeds) * len(train_seeds) * 2
    done = 0
    started = time.time()
    try:
        for noise, stride in cells:
            records = work / f"records_{noise:.2f}"
            if not records.exists():
                generate_recordings(noise, records)
            snr = burst_snr_db(noise)
            tag = f"{noise:.2f}_{stride}"
            for split_seed in split_seeds:
                rec_ds = work / f"rec_{tag}_{split_seed}"
                win_ds = work / f"win_{tag}_{split_seed}"
                build_recording_level(records, rec_ds, split_seed, stride=stride)
                build_window_level(rec_ds, win_ds, split_seed)
                for strategy, dataset in (("recording-level", rec_ds), ("window-level", win_ds)):
                    for train_seed in train_seeds:
                        acc, train_acc, n_train, n_test = train_once(dataset, train_seed)
                        runs.append(
                            Run(
                                noise_sigma=noise,
                                snr_db=snr,
                                strategy=strategy,
                                split_seed=split_seed,
                                train_seed=train_seed,
                                test_accuracy=acc,
                                train_accuracy=train_acc,
                                train_windows=n_train,
                                test_windows=n_test,
                                stride=stride,
                                environment=environment,
                            )
                        )
                        done += 1
                        if checkpoint is not None:
                            checkpoint(runs)
                        rate = (time.time() - started) / done
                        label = f"snr={snr:+5.1f}dB" if stride is None else f"stride={stride:4d}"
                        print(
                            f"  [{done:3d}/{total}] {label} {strategy:15s} "
                            f"split={split_seed} train={train_seed}  "
                            f"test={acc:6.2%}  (~{rate * (total - done) / 60:.0f} min left)",
                            flush=True,
                        )
                shutil.rmtree(rec_ds, ignore_errors=True)
                shutil.rmtree(win_ds, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return runs


def _sd(values: list[float]) -> float:
    """Standard deviation, 0 for a single sample (the --quick smoke grid)."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarise_strides(runs: list[Run]) -> str:
    """Build the stride-sweep table.

    Overlap is the mechanism. At stride == window the windows are disjoint, so a
    window-level split still separates distinct samples and there is little to
    gain; as the stride shrinks each window shares more of itself with its
    neighbours, until the test set is close to a copy of the training set.
    """
    lines = [
        "| stride | overlap | recording-level | window-level | inflation (paired) | n |",
        "|---|---|---|---|---|---|",
    ]
    for stride in sorted({r.stride for r in runs if r.stride is not None}, reverse=True):
        at = [r for r in runs if r.stride == stride]
        rec = [r.test_accuracy for r in at if r.strategy == "recording-level"]
        win = [r.test_accuracy for r in at if r.strategy == "window-level"]
        diffs = paired_differences(at)
        if not rec or not win:
            # Checkpointing writes this table mid-cell, before the second arm
            # has run. Leave the row out until both sides exist.
            continue
        mean_diff = statistics.mean(diffs) if diffs else float("nan")
        stderr = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        overlap = 1.0 - stride / WINDOW
        lines.append(
            f"| {stride} | {overlap:.0%} | "
            f"{statistics.mean(rec):.1%} ± {_sd(rec):.1%} | "
            f"{statistics.mean(win):.1%} ± {_sd(win):.1%} | "
            f"**{mean_diff * 100:+.1f} pp** ± {stderr * 100:.1f} | {len(diffs)} |"
        )
    lines.append("")
    lines.append(
        f"Window fixed at {WINDOW} samples; noise fixed at sigma={STRIDE_NOISE} "
        f"({burst_snr_db(STRIDE_NOISE):+.1f} dB burst SNR), the region where the SNR "
        "sweep found the largest inflation. Inflation is the mean paired difference "
        "± its standard error."
    )
    return "\n".join(lines)


def summarise(runs: list[Run]) -> str:
    """Build the results table."""

    def stats(subset: list[Run]) -> tuple[float, float]:
        values = [r.test_accuracy for r in subset]
        if not values:
            return float("nan"), 0.0
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        return statistics.mean(values), spread

    lines = [
        "| burst SNR | recording-level | window-level | inflation (paired) | n |",
        "|---|---|---|---|---|",
    ]
    for noise in sorted({r.noise_sigma for r in runs}, reverse=True):
        at = [r for r in runs if r.noise_sigma == noise]
        rec_mean, rec_sd = stats([r for r in at if r.strategy == "recording-level"])
        win_mean, win_sd = stats([r for r in at if r.strategy == "window-level"])
        diffs = paired_differences(at)
        if diffs:
            mean_diff = statistics.mean(diffs)
            stderr = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
            inflation = f"**{mean_diff * 100:+.1f} pp** ± {stderr * 100:.1f}"
        else:
            inflation = "-"
        lines.append(
            f"| {at[0].snr_db:+.1f} dB | {rec_mean:.1%} ± {rec_sd:.1%} | "
            f"{win_mean:.1%} ± {win_sd:.1%} | {inflation} | {len(diffs)} |"
        )
    lines.append("")
    lines.append(
        "Accuracy columns are mean ± standard deviation across runs. The inflation "
        "column is the mean PAIRED difference ± its standard error: each pair is one "
        "split seed and one training seed, so both arms saw the same recordings and "
        "the same initialisation. Comparing the two means directly would fold in the "
        "between-seed scatter, which is the largest source of variation here and "
        "cancels in the pairing."
    )
    return "\n".join(lines)


def paired_differences(runs: list[Run]) -> list[float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small grid, for a smoke test")
    parser.add_argument(
        "--sweep",
        choices=("snr", "stride"),
        default="snr",
        help="snr: vary noise at the default stride. stride: vary overlap at a fixed SNR.",
    )
    args = parser.parse_args()

    # 15 pairs per cell. A first pass used 3 x 2 = 6 and the paired standard
    # error came out near 6 pp against an effect around 10 pp -- the direction
    # was clear, the magnitude was not. Seed scatter is the dominant noise
    # source in this project, so the fix is more seeds.
    split_seeds = [42, 7, 1234, 2026, 99]
    train_seeds = [0, 1, 2]

    if args.sweep == "stride":
        cells = [(STRIDE_NOISE, s) for s in STRIDES]
        summary = summarise_strides
        stem = "leakage_stride"
    else:
        cells = [(n, None) for n in NOISE_LEVELS]
        summary = summarise
        stem = "leakage"

    if args.quick:
        cells = cells[:2]
        split_seeds = [42]
        train_seeds = [0]

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if args.quick else ""
    runs_path = ARTIFACTS / f"{stem}_runs{suffix}.json"
    table_path = ARTIFACTS / f"{stem}_table{suffix}.md"

    check_environment(runs_path)

    def checkpoint(runs: list[Run]) -> None:
        """Persist after every run so an interruption keeps what it earned."""
        runs_path.write_text(json.dumps([asdict(r) for r in runs], indent=2), encoding="utf-8")
        table_path.write_text(summary(runs) + "\n", encoding="utf-8")

    print(
        f"{args.sweep} sweep: {len(cells)} cells x {len(split_seeds)} split seeds x "
        f"{len(train_seeds)} train seeds x 2 strategies "
        f"= {len(cells) * len(split_seeds) * len(train_seeds) * 2} runs",
        flush=True,
    )
    runs = run_grid(cells, split_seeds, train_seeds, checkpoint=checkpoint)

    table = summary(runs)
    print()
    print(table)
    checkpoint(runs)
    print(f"\nwrote {runs_path.name} and {table_path.name}")


if __name__ == "__main__":
    main()
