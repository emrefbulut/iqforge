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

1. The deliberately wrong split lives in `iqforge.measurement`, not in the CLI.
   `iqforge build` refuses to split at the window level; proving why that
   refusal is right requires doing the wrong thing on purpose, and that
   capability must not ship to users. This script only prepares the recordings.

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
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_example as gen  # noqa: E402

from iqforge.measurement import (
    DEFAULT_SPLIT_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    BuildSpec,
    Run,
    check_environment,
    guard_artifact_rows,
    measure_cell_via_cli,  # noqa: E402
    summarise_snr_table,
    summarise_stride_table,
)

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

#: Window length held fixed for the stride sweep, so overlap is the only thing
#: that moves.
#: The published grids' seed lists. Fixed before the run and written into the
#: table, so the sample size is a stated choice rather than whatever the code
#: happened to default to.
SPLIT_SEEDS = DEFAULT_SPLIT_SEEDS
TRAIN_SEEDS = DEFAULT_TRAIN_SEEDS

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


def generate_recordings(noise_sigma: float, out_dir: Path) -> None:
    """Write the experiment's recordings at a given noise level."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for plan in gen._build_plans(RECORDS_PER_CELL):
        gen.write_record(plan, gen.build_signal(plan, noise_sigma=noise_sigma), out_dir)


def _spec(stride: int | None) -> BuildSpec:
    return BuildSpec(
        window=WINDOW,
        stride=stride,
        split=SPLIT_RATIOS,
        labels="annotations",
        balance_by="core:freq_lower_edge",
        assert_offsets_shared=True,
    )


def _cell_runs(records: Path, spec: BuildSpec) -> list[Run]:
    """Every run for one synthetic cell, through the shipped command.

    `--force` is required here and nowhere else in these scripts. The synthetic
    recordings carry no SigMF annotations that the audit can turn into class
    axes, so the refuse path classifies them as category 6 ("cannot split") on
    the audit's view even though `build` splits them correctly with
    `--labels annotations`. The override is safe for this generator and this
    generator only; it is not a pattern to copy to real captures.
    """
    return measure_cell_via_cli(
        records,
        spec,
        split_seeds=SPLIT_SEEDS,
        train_seeds=TRAIN_SEEDS,
        force=True,
        force_reason=(
            "synthetic fixtures have no audit-visible class axis; build splits them "
            "correctly with --labels annotations"
        ),
    )


def run_grid(cells: list[tuple[float, int | None]]) -> list[Run]:
    """Prepare recordings per noise level, then measure through the CLI."""
    work = Path(tempfile.mkdtemp(prefix="iqforge-leakage-"))
    try:
        prepared: dict[float, Path] = {}
        runs: list[Run] = []
        for noise, stride in cells:
            if noise not in prepared:
                records = work / f"records_{noise:.2f}"
                generate_recordings(noise, records)
                prepared[noise] = records
            snr = burst_snr_db(noise)
            for run in _cell_runs(prepared[noise], _spec(stride)):
                run.noise_sigma = noise
                run.snr_db = snr
                run.stride = stride
                runs.append(run)
        return runs
    finally:
        shutil.rmtree(work, ignore_errors=True)


def summarise_strides(runs: list[Run]) -> str:
    """Stride-sweep table. Overlap is the mechanism."""
    return summarise_stride_table(
        runs,
        window=WINDOW,
        caption=(
            f"Window fixed at {WINDOW} samples; noise fixed at sigma={STRIDE_NOISE} "
            f"({burst_snr_db(STRIDE_NOISE):+.1f} dB burst SNR), the region where the SNR "
            "sweep found the largest inflation. Inflation is the mean paired difference "
            "± its standard error."
        ),
    )


def summarise(runs: list[Run]) -> str:
    """SNR-sweep table."""
    return summarise_snr_table(
        runs,
        caption=(
            "Accuracy columns are mean ± standard deviation across runs. The inflation "
            "column is the mean PAIRED difference ± its standard error: each pair is one "
            "split seed and one training seed, so both arms saw the same recordings and "
            "the same initialisation. Comparing the two means directly would fold in the "
            "between-seed scatter, which is the largest source of variation here and "
            "cancels in the pairing."
        ),
    )


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

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if args.quick else ""
    runs_path = ARTIFACTS / f"{stem}_runs{suffix}.json"
    table_path = ARTIFACTS / f"{stem}_table{suffix}.md"

    pairs = len(SPLIT_SEEDS.split(",")) * len(TRAIN_SEEDS.split(","))
    planned = len(cells) * pairs * 2

    check_environment(runs_path)
    # Checked against the plan, before anything is trained. Guarding inside a
    # per-run checkpoint would fire on the first partial write instead, and
    # guarding after the grid would waste the whole run before saying so.
    guard_artifact_rows(runs_path, planned)

    print(
        f"{args.sweep} sweep: {len(cells)} cells x {pairs} seed pairs x 2 strategies "
        f"= {planned} runs (via `iqforge measure-leakage`)",
        flush=True,
    )
    runs = run_grid(cells)

    table = summary(runs)
    print()
    print(table)
    guard_artifact_rows(runs_path, len(runs))
    guard_artifact_rows(runs_path, len(runs))
    runs_path.write_text(json.dumps([asdict(r) for r in runs], indent=2), encoding="utf-8")
    table_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nwrote {runs_path.name} and {table_path.name}")


if __name__ == "__main__":
    main()
