"""The stride sweep on LoRaIQ, where the classes are propagation environments.

The third run of the same experiment. §2 and §3 measured it on synthetic
signals; `leakage_real.py` repeated the stride half on DASH7 and reproduced the
zero-overlap null but could not size the effect, because that dataset's task is
a step function with no region of partial competence (methodology §6).

LoRaIQ is the first assessed dataset `iqforge audit` returns `unknown` for: no
single measurable axis separates the classes, the best reaching 91% against a
chance of 47%. That is the regime the measurement needs.

Three things differ from `leakage_real.py`:

1. **No noise is added.** DASH7 needed it because the raw task sat at 100% in
   both arms. Here the honest task is already hard, so injecting noise would
   only move it towards chance.

2. **Recordings are grouped by transmission.** One LoRa frame is heard by up to
   four rooftop receivers at the same instant, so it is four files and one
   event. Splitting them independently puts the same instant of radio on both
   sides of the split, which is a leak recording-level splitting cannot see --
   the unit of independence is the transmission. `--group-by` holds them
   together and `iqforge audit` confirms it: without it, 271 of 465
   air-time-sharing pairs land in different splits; with it, none do.

3. **A fixed segment is taken around each frame.** Recordings run from 15 244
   to 13.3 M samples, so windowing them whole would spend the entire budget on
   the longest few and weight the dataset by file length rather than by class.
   The segment is centred on the frame the index CSV points at, so the signal
   is inside it for every recording rather than clipped for the ones whose
   frame starts late.

Usage:
    uv run python scripts/leakage_loraiq.py --check    # audit only, no training
    uv run python scripts/leakage_loraiq.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sigmf import SigMFFile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from leakage_experiment import (  # noqa: E402
    EPOCHS,
    Run,
    build_window_level,
    check_environment,
    current_environment,
    paired_differences,
    train_once,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

SCRATCH = Path(
    "C:/Users/Emre/AppData/Local/Temp/claude/C--Users-Emre-Desktop-project/"
    "0edfda65-7ea5-4ec7-85c8-a30cc5d9358b/scratchpad/scan"
)
DEFAULT_SOURCE = SCRATCH / "loraiq"
DEFAULT_INDEX = SCRATCH / "loraiq.csv"
DEFAULT_LABELS = SCRATCH / "loraiq_labels.csv"
DEFAULT_GROUPS = SCRATCH / "loraiq_groups.csv"

WINDOW = 1024
SPLIT_RATIOS = "0.6,0.2,0.2"

#: Samples per recording, the length of the shortest one in the set. Every
#: recording contributes the same number of windows, so the dataset is weighted
#: by class rather than by how long a file happens to be.
SEGMENT = 15_244

#: Samples of lead-in kept before the frame the index points at.
LEAD_IN = 2_048

#: Same ladder as the synthetic and DASH7 sweeps, so the three are comparable.
#: Run order puts the three anchors first -- no overlap, the tool's default,
#: near-total overlap -- so an interruption still leaves the shape of the curve.
STRIDES = (1024, 512, 128, 768, 256)


def frame_offsets(index_csv: Path) -> dict[str, int]:
    """Sample offset of the first frame in each recording, from the index CSV."""
    offsets: dict[str, int] = {}
    with index_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row["sigmf_file"].replace("./sigmfs/", "") + ".sigmf-meta"
            start = int(float(row["sigmf_file_offset"]))
            offsets[key] = min(offsets.get(key, start), start)
    return offsets


def prepare(source: Path, out_dir: Path, offsets: dict[str, int]) -> int:
    """Write a fixed-length segment of every recording, layout preserved."""
    from iqforge.io import load

    written = 0
    for meta in sorted(source.rglob("*.sigmf-meta")):
        record_id = meta.relative_to(source).as_posix()
        rec = load(meta)
        start = max(0, offsets.get(record_id, 0) - LEAD_IN)
        start = min(start, max(0, rec.num_samples - SEGMENT))
        take = min(SEGMENT, rec.num_samples - start)
        if take < WINDOW:
            continue
        segment = rec.read(start, take).astype(np.complex64)

        target = out_dir / meta.parent.relative_to(source)
        target.mkdir(parents=True, exist_ok=True)
        data_path = target / f"{meta.stem}.sigmf-data"
        interleaved = np.empty(segment.size * 2, dtype=np.float32)
        interleaved[0::2] = segment.real
        interleaved[1::2] = segment.imag
        interleaved.tofile(data_path)

        handle = SigMFFile(
            data_file=str(data_path),
            global_info={
                SigMFFile.DATATYPE_KEY: "cf32_le",
                SigMFFile.SAMPLE_RATE_KEY: rec.sample_rate,
                SigMFFile.VERSION_KEY: "1.0.0",
                SigMFFile.DESCRIPTION_KEY: f"LoRaIQ segment from {start}",
            },
        )
        capture = {SigMFFile.START_INDEX_KEY: 0}
        if rec.center_frequency is not None:
            capture[SigMFFile.FREQUENCY_KEY] = rec.center_frequency
        if rec.capture_datetime is not None:
            capture[SigMFFile.DATETIME_KEY] = rec.capture_datetime
        handle.add_capture(0, metadata=capture)
        handle.tofile(str(target / f"{meta.stem}.sigmf-meta"))
        written += 1
    return written


def build_recording_level(
    records: Path, out: Path, seed: int, stride: int, labels: Path, groups: Path
) -> None:
    """Build the honest split through the CLI, and refuse to proceed on a warning.

    The grouping is not optional here. Without it the split separates recordings
    that are the same instant of radio, which is a leak this experiment would
    otherwise be measuring on top of the one it is trying to isolate.
    """
    if out.exists():
        shutil.rmtree(out)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "iqforge",
            "build",
            str(records),
            "-o",
            str(out),
            "--labels",
            "csv",
            "--label-file",
            str(labels),
            "--group-by",
            f"csv:{groups}",
            "--window",
            str(WINDOW),
            "--stride",
            str(stride),
            "--split",
            SPLIT_RATIOS,
            "--seed",
            str(seed),
        ],  # fmt: skip
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if "warning" in output.lower():
        raise RuntimeError(f"iqforge warned while building (seed {seed}):\n{output.strip()}")


def run(
    records: Path,
    strides: tuple[int, ...],
    split_seeds: list[int],
    train_seeds: list[int],
    labels: Path,
    groups: Path,
    checkpoint: Callable[[list[Run]], None] | None = None,
) -> list[Run]:
    """Sweep the stride, holding everything else fixed."""
    runs: list[Run] = []
    environment = current_environment()
    work = Path(tempfile.mkdtemp(prefix="iqforge-loraiq-"))
    total = len(strides) * len(split_seeds) * len(train_seeds) * 2
    done = 0
    started = time.time()
    try:
        for stride in strides:
            for split_seed in split_seeds:
                cell = f"{stride}_{split_seed}"
                rec_ds, win_ds = work / f"r_{cell}", work / f"w_{cell}"
                build_recording_level(records, rec_ds, split_seed, stride, labels, groups)
                build_window_level(rec_ds, win_ds, split_seed)
                for strategy, dataset in (("recording-level", rec_ds), ("window-level", win_ds)):
                    for train_seed in train_seeds:
                        t0 = time.time()
                        acc, train_acc, n_train, n_test = train_once(dataset, train_seed)
                        done += 1
                        runs.append(
                            Run(
                                noise_sigma=0.0,
                                snr_db=float("nan"),
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
                        if checkpoint is not None:
                            checkpoint(runs)
                        rate = (time.time() - started) / done
                        print(
                            f"  [{done:3d}/{total}] stride={stride:4d} {strategy:15s} "
                            f"split={split_seed} train={train_seed}  test={acc:6.2%} "
                            f"train={train_acc:6.2%}  ({time.time() - t0:.0f}s/run, "
                            f"{n_train}/{n_test} windows, "
                            f"~{rate * (total - done) / 60:.0f} min left)",
                            flush=True,
                        )
                shutil.rmtree(rec_ds, ignore_errors=True)
                shutil.rmtree(win_ds, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\ntotal wall time {(time.time() - started) / 60:.1f} min for {done} runs")
    return runs


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarise(runs: list[Run]) -> str:
    """Stride-sweep table. Stride 1024 is the row that carries the claim."""
    lines = [
        "| stride | overlap | windows/rec | recording-level | window-level | inflation (paired) | n |",  # noqa: E501
        "|---|---|---|---|---|---|---|",
    ]
    for stride in sorted({r.stride for r in runs if r.stride is not None}, reverse=True):
        at = [r for r in runs if r.stride == stride]
        rec = [r.test_accuracy for r in at if r.strategy == "recording-level"]
        win = [r.test_accuracy for r in at if r.strategy == "window-level"]
        diffs = paired_differences(at)
        if not rec or not win:
            continue
        mean_diff = statistics.mean(diffs) if diffs else float("nan")
        stderr = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        lines.append(
            f"| {stride} | {1.0 - stride / WINDOW:.0%} | {(SEGMENT - WINDOW) // stride + 1} | "
            f"{statistics.mean(rec):.1%} ± {_sd(rec):.1%} | "
            f"{statistics.mean(win):.1%} ± {_sd(win):.1%} | "
            f"**{mean_diff * 100:+.1f} pp** ± {stderr * 100:.1f} | {len(diffs)} |"
        )
    lines.append("")
    lines.append(
        f"LoRaIQ, class = propagation environment (drone_los, drone_nlos, "
        f"pedestrian_partial_los, pedestrian_nlos, indoor), 312 recordings over 13 "
        f"capture sessions, grouped by transmission id so simultaneous receptions stay "
        f"in one split. Window fixed at {WINDOW} samples over a {SEGMENT}-sample segment "
        f"centred on each frame; no noise added. Overlap is the only thing that moves "
        f"between rows. Inflation is the mean paired difference ± its standard error."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--stem", default="leakage_loraiq")
    parser.add_argument(
        "--check", action="store_true", help="Prepare and audit only, train nothing"
    )
    args = parser.parse_args()

    for path in (args.source, args.index, args.labels, args.groups):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    prepared = Path(tempfile.mkdtemp(prefix="iqforge-loraiq-src-"))
    t0 = time.time()
    count = prepare(args.source, prepared, frame_offsets(args.index))
    print(f"prepared {count} recordings in {time.time() - t0:.0f}s -> {prepared}", flush=True)

    if args.check:
        subprocess.run([sys.executable, "-m", "iqforge", "audit", str(prepared)], check=False)
        return

    split_seeds, train_seeds = [42, 7, 1234, 2026, 99], [0, 1, 2]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    runs_path = ARTIFACTS / f"{args.stem}_runs.json"
    table_path = ARTIFACTS / f"{args.stem}_table.md"

    check_environment(runs_path)

    def checkpoint(runs: list[Run]) -> None:
        """Persist after every run so an interruption keeps what it earned."""
        runs_path.write_text(json.dumps([asdict(r) for r in runs], indent=2), encoding="utf-8")
        table_path.write_text(summarise(runs) + "\n", encoding="utf-8")

    print(
        f"{len(STRIDES)} strides x {len(split_seeds)} split x {len(train_seeds)} train x 2 "
        f"= {len(STRIDES) * len(split_seeds) * len(train_seeds) * 2} runs, {EPOCHS} epochs each",
        flush=True,
    )
    try:
        runs = run(
            prepared, STRIDES, split_seeds, train_seeds, args.labels, args.groups, checkpoint
        )
    finally:
        shutil.rmtree(prepared, ignore_errors=True)

    table = summarise(runs)
    print()
    print(table)
    checkpoint(runs)
    print(f"\nwrote {runs_path.name} and {table_path.name}")


if __name__ == "__main__":
    main()
