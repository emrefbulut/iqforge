"""The leakage measurement, on a real capture instead of synthetic signals.

Same paired design as `leakage_experiment.py`: the same windows, the same model,
the same seeds, and only the split method differing. What changes is where the
recordings come from.

Source is the DASH7 cabled set (Zenodo 10961311, CC BY 4.0, USRP B210,
ci16_le, 7.68 MS/s). Class is the Lo-Rate channel -- CH0, CH93, CH186 -- which
is a carrier offset inside one captured band, with ten independent recorder runs
per channel.

Three decisions decide whether the number means anything:

1. **Noise is added to the continuous signal, before windowing.** Adding it per
   window would give two overlapping windows independent noise in the samples
   they share, which destroys the very correlation that causes the leak. The
   measurement would come out clean for the wrong reason.

2. **The segment sits inside a packet.** The class lives in the carrier, and the
   carrier is only on air while a packet transmits: measured over a whole
   recording, 6.8% of it carries signal and the other 93% is noise floor, where
   no window carries class information at all. Packet timing differs between
   runs, so each recording's first packet is located rather than assumed.

3. **SNR is set per recording**, against that recording's own segment power. Any
   power difference between channels would otherwise be a second, easier route
   to the answer than the carrier offset the task is supposed to be about.

Usage:
    uv run python scripts/leakage_real.py --pilot
    uv run python scripts/leakage_real.py --sweep stride
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

#: Where the extracted DASH7 cabled tree lives; override with --source.
DEFAULT_SOURCE = Path(
    "C:/Users/Emre/AppData/Local/Temp/claude/C--Users-Emre-Desktop-project/"
    "0edfda65-7ea5-4ec7-85c8-a30cc5d9358b/scratchpad/dash7/extracted/cabled"
)

WINDOW, STRIDE = 1024, 512
SPLIT_RATIOS = "0.6,0.2,0.2"

#: Samples taken from inside one packet. A packet runs 207 872 samples (27.1 ms)
#: so this sits comfortably within it, and yields 199 windows per recording at
#: stride 512 -- enough to train on without the grid taking days.
SEGMENT = 102_400

#: Target SNRs in dB, signal against added noise, measured per recording.
#: `None` is the recording as captured, which the pilot exists to check for a
#: ceiling.
#:
#: The pilot found this grid useless: raw and 0 dB both sat at 100/100. The
#: channels occupy 19.5 kHz of a 7.68 MHz capture, so there is ~26 dB of
#: processing gain and "0 dB wideband" is +26 dB where the signal lives. A
#: follow-up probe from -15 to -40 dB found the usable band, and found it 7 dB
#: wide -- see `--sweep stride` below for what was run instead.
SNR_LEVELS: tuple[float | None, ...] = (None, 6.0, 3.0, 0.0, -3.0, -6.0)

#: Strides for the overlap sweep, at a fixed SNR. Same design as the synthetic
#: sweep in `leakage_experiment.STRIDES`, and for the same reason: overlap is the
#: MECHANISM. At stride 1024 the windows are disjoint, so a window-level split
#: can only ever put *different* samples on both sides and the inflation must be
#: zero; at stride 128 each window shares 7/8 of itself with its neighbour.
#:
#: The order is the run order, not the reading order -- the table is sorted by
#: stride however this list is arranged. It puts the three anchors first (no
#: overlap, the tool's default, near-total overlap) so that an interruption
#: still leaves the shape of the curve, and it front-loads the cheap cells:
#: window count per recording runs 100 / 133 / 199 / 397 / 793 as the stride
#: halves, and training time with it.
STRIDES = (1024, 512, 128, 768, 256)

#: Wideband SNR held fixed for the stride sweep.
#:
#: Chosen from the bracket probe (`artifacts/leakage_real_probe_table.md`,
#: `..._cliff_table.md`), on one criterion: the honest arm has to be stable and
#: strictly between chance and the ceiling, or the paired difference measures
#: baseline scatter instead of overlap. At -19 dB the recording-level arm gave
#: 43.7% ± 0.6 -- 10 points above the 33.3% chance line with 56 points of
#: headroom left. At -15 and -17 dB that arm is bimodal (53.5% ± 25.6), and at
#: -21 dB it is pinned at chance (33.7% ± 0.5), where any inflation would be
#: memorisation of a task the model cannot do at all -- a real effect, but not
#: the one a user would ever be in a position to misread.
STRIDE_SNR_DB = -19.0


def find_first_packet(path: Path, search: int = 12_000_000, block: int = 1024) -> int:
    """Sample index where the recording's first packet starts.

    A power envelope with a threshold set between the floor and the peak. The
    first few blocks are ignored: the recorder leaves a short transient at
    sample 0 that is not a packet.
    """
    from iqforge.io import load

    samples = load(path).read(0, search)
    n = samples.size // block
    power = (np.abs(samples[: n * block].reshape(n, block)) ** 2).mean(axis=1)
    db = 10 * np.log10(power + 1e-20)
    above = db > db.min() + (db.max() - db.min()) * 0.35
    above[:4] = False
    if not above.any():
        raise RuntimeError(f"no packet found in the first {search:,} samples of {path}")
    return int(np.argmax(above)) * block


def prepare(source: Path, out_dir: Path, snr_db: float | None, rng_seed: int) -> None:
    """Write a noised, packet-aligned copy of every recording.

    The directory layout is preserved, so `--labels dirname --dirname-level 2`
    still reads the channel as the class.
    """
    from iqforge.io import load

    rng = np.random.default_rng(rng_seed)
    for meta in sorted(source.rglob("*.sigmf-meta")):
        rec = load(meta)
        start = find_first_packet(meta)
        # A little way in, so the packet's leading edge is not in the first window.
        segment = rec.read(start + 4096, SEGMENT).astype(np.complex64)

        if snr_db is not None:
            power = float(np.mean(np.abs(segment) ** 2))
            sigma = np.sqrt(power / (2.0 * 10 ** (snr_db / 10.0)))
            noise = rng.standard_normal(segment.size) + 1j * rng.standard_normal(segment.size)
            segment = (segment + sigma * noise).astype(np.complex64)

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
                SigMFFile.DESCRIPTION_KEY: f"DASH7 cabled packet segment, snr={snr_db}",
            },
        )
        handle.add_capture(0, metadata={SigMFFile.FREQUENCY_KEY: rec.center_frequency})
        handle.tofile(str(target / f"{meta.stem}.sigmf-meta"))


def build_recording_level(records: Path, out: Path, seed: int, stride: int = STRIDE) -> None:
    """Build the honest split through the CLI, and refuse to proceed on a warning."""
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
            "dirname",
            "--dirname-level",
            "2",
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
    cells: list[tuple[float | None, int]],
    split_seeds: list[int],
    train_seeds: list[int],
    checkpoint: Callable[[list[Run]], None] | None = None,
) -> list[Run]:
    """Run the experiment over a list of `(wideband SNR, stride)` cells.

    Args:
        cells: One cell per row of the eventual table. The SNR sweep varies the
            first element and holds the stride at the tool's default; the stride
            sweep does the opposite. One loop serves both so the two cannot
            drift apart in how they prepare, build or train.
        split_seeds: Seeds for the recording-level split.
        train_seeds: Seeds for weight init and batch order.
        checkpoint: Called with the runs so far after every run. The stride
            sweep takes hours; the pilot already lost a formatted table to a
            crash that came after all twelve trainings had finished, and only
            survived because the JSON was written first. Nothing here is worth
            re-earning.

    Noised recordings are prepared once per distinct SNR and reused across every
    stride at that SNR, so the stride sweep varies overlap over one fixed noise
    realisation and nothing else.
    """
    runs: list[Run] = []
    environment = current_environment()
    work = Path(tempfile.mkdtemp(prefix="iqforge-real-"))
    total = len(cells) * len(split_seeds) * len(train_seeds) * 2
    done = 0
    started = time.time()
    prepared: dict[str, Path] = {}
    try:
        for snr, stride in cells:
            tag = "raw" if snr is None else f"{snr:+.0f}"
            if tag not in prepared:
                records = work / f"rec_{tag}"
                t0 = time.time()
                prepare(SOURCE, records, snr, rng_seed=1234)
                prepared[tag] = records
                print(f"  prepared {tag} in {time.time() - t0:.0f}s", flush=True)
            records = prepared[tag]

            for split_seed in split_seeds:
                cell = f"{tag}_{stride}_{split_seed}"
                rec_ds, win_ds = work / f"r_{cell}", work / f"w_{cell}"
                build_recording_level(records, rec_ds, split_seed, stride=stride)
                build_window_level(rec_ds, win_ds, split_seed)
                for strategy, dataset in (("recording-level", rec_ds), ("window-level", win_ds)):
                    for train_seed in train_seeds:
                        t1 = time.time()
                        acc, train_acc, n_train, n_test = train_once(dataset, train_seed)
                        done += 1
                        runs.append(
                            Run(
                                noise_sigma=float("nan") if snr is None else snr,
                                snr_db=float("inf") if snr is None else snr,
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
                            f"  [{done:3d}/{total}] snr={tag:>4} stride={stride:4d} "
                            f"{strategy:15s} split={split_seed} train={train_seed}  "
                            f"test={acc:6.2%} train={train_acc:6.2%}  "
                            f"({time.time() - t1:.0f}s/run, {n_train}/{n_test} windows, "
                            f"~{rate * (total - done) / 60:.0f} min left)",
                            flush=True,
                        )
                shutil.rmtree(rec_ds, ignore_errors=True)
                shutil.rmtree(win_ds, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\ntotal wall time {(time.time() - started) / 60:.1f} min for {done} runs")
    return runs


#: The channels occupy 19.5 kHz of a 7.68 MHz capture, measured. Detecting one
#: against the others therefore comes with about 26 dB of processing gain, so a
#: wideband SNR of 0 dB is still roughly +26 dB where the signal actually lives.
#: Both figures are reported: the wideband one is what the noise was set to, the
#: in-band one is what the task sees.
OCCUPIED_BW_HZ = 19_500.0
SAMPLE_RATE_HZ = 7_680_000.0
PROCESSING_GAIN_DB = 10.0 * np.log10(SAMPLE_RATE_HZ / OCCUPIED_BW_HZ)


def summarise_real(runs: list[Run]) -> str:
    """Results table, grouped by SNR.

    Not `leakage_experiment.summarise`: that groups by `noise_sigma`, and the
    unnoised level has no sigma. Grouping on a NaN silently matches nothing,
    which is how the first pilot run finished its twelve trainings and then
    crashed formatting them.
    """

    def stats(subset: list[Run]) -> tuple[float, float]:
        values = [r.test_accuracy for r in subset]
        if not values:
            return float("nan"), 0.0
        return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)

    lines = [
        "| wideband SNR | in-band SNR | recording-level | window-level | inflation (paired) | n |",
        "|---|---|---|---|---|---|",
    ]
    for snr in sorted({r.snr_db for r in runs}, reverse=True):
        at = [r for r in runs if r.snr_db == snr]
        rec_mean, rec_sd = stats([r for r in at if r.strategy == "recording-level"])
        win_mean, win_sd = stats([r for r in at if r.strategy == "window-level"])
        diffs = paired_differences(at)
        if diffs:
            mean_diff = statistics.mean(diffs)
            stderr = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
            inflation = f"**{mean_diff * 100:+.1f} pp** ± {stderr * 100:.1f}"
        else:
            inflation = "-"
        wide = "raw" if math.isinf(snr) else f"{snr:+.0f} dB"
        inband = "-" if math.isinf(snr) else f"{snr + PROCESSING_GAIN_DB:+.1f} dB"
        lines.append(
            f"| {wide} | {inband} | {rec_mean:.1%} ± {rec_sd:.1%} | "
            f"{win_mean:.1%} ± {win_sd:.1%} | {inflation} | {len(diffs)} |"
        )
    lines.append("")
    lines.append(
        f"Wideband SNR is what the added noise was set to, over the full "
        f"{SAMPLE_RATE_HZ / 1e6:.2f} MHz capture. The channels occupy "
        f"{OCCUPIED_BW_HZ / 1e3:.1f} kHz, so the task sees about "
        f"{PROCESSING_GAIN_DB:.0f} dB more than that -- which is why the raw recordings, "
        f"and 0 dB wideband, are both at the ceiling."
    )
    return "\n".join(lines)


def summarise_real_strides(runs: list[Run]) -> str:
    """Stride-sweep table: inflation against overlap, at one fixed SNR.

    The row that carries the argument is stride 1024. There the windows are
    disjoint, so a window-level split separates samples that were never shared
    and the inflation has to be zero. If it is not, the effect measured in the
    other rows is something other than overlap.
    """
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
            # Checkpointing writes this table mid-cell, before the second arm has
            # run. Leave the row out until both sides exist.
            continue
        mean_diff = statistics.mean(diffs) if diffs else float("nan")
        stderr = (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        per_rec = (SEGMENT - WINDOW) // stride + 1
        lines.append(
            f"| {stride} | {1.0 - stride / WINDOW:.0%} | {per_rec} | "
            f"{statistics.mean(rec):.1%} ± {_sd(rec):.1%} | "
            f"{statistics.mean(win):.1%} ± {_sd(win):.1%} | "
            f"**{mean_diff * 100:+.1f} pp** ± {stderr * 100:.1f} | {len(diffs)} |"
        )
    lines.append("")
    lines.append(
        f"DASH7 cabled, class = Lo-Rate channel, 10 independent recorder runs each. "
        f"Window fixed at {WINDOW} samples; added noise fixed at {STRIDE_SNR_DB:+.0f} dB "
        f"wideband ({STRIDE_SNR_DB + PROCESSING_GAIN_DB:+.0f} dB in-band), the one probed "
        f"point where the honest arm sits stably between the 33.3% chance line and the "
        f"ceiling. Overlap is the only thing that moves between rows: the noise realisation "
        f"is identical, because the recordings are noised once and re-windowed per stride. "
        f"Inflation is the mean paired difference ± its standard error."
    )
    return "\n".join(lines)


def _sd(values: list[float]) -> float:
    """Standard deviation, 0 for a single sample."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="One SNR pair, 3 seed pairs")
    parser.add_argument(
        "--sweep",
        choices=("snr", "stride"),
        default="snr",
        help="snr: vary added noise at the default stride. stride: vary overlap at a fixed SNR.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--snr",
        help="Comma-separated wideband SNRs in dB, overriding the grid; 'raw' for no noise",
    )
    parser.add_argument("--split-seeds", default="42,7,1234,2026,99")
    parser.add_argument("--train-seeds", default="0,1,2")
    parser.add_argument("--stem", default=None, help="Artifact file stem")
    args = parser.parse_args()

    global SOURCE
    SOURCE = args.source
    if not SOURCE.exists():
        raise SystemExit(f"source not found: {SOURCE}")

    split_seeds = [int(s) for s in args.split_seeds.split(",")]
    train_seeds = [int(s) for s in args.train_seeds.split(",")]
    summary: Callable[[list[Run]], str] = summarise_real

    if args.sweep == "stride":
        cells = [(STRIDE_SNR_DB, s) for s in STRIDES]
        summary = summarise_real_strides
        stem = args.stem or "leakage_real_stride"
    elif args.snr:
        levels: tuple[float | None, ...] = tuple(
            None if p.strip() == "raw" else float(p) for p in args.snr.split(",")
        )
        cells = [(level, STRIDE) for level in levels]
        stem = args.stem or "leakage_real_probe"
    elif args.pilot:
        cells = [(None, STRIDE), (0.0, STRIDE)]
        split_seeds, train_seeds = [42, 7, 1234], [0]
        stem = "leakage_real_pilot"
    else:
        cells = [(level, STRIDE) for level in SNR_LEVELS]
        stem = "leakage_real"

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    runs_path = ARTIFACTS / f"{stem}_runs.json"
    table_path = ARTIFACTS / f"{stem}_table.md"

    check_environment(runs_path)

    def checkpoint(runs: list[Run]) -> None:
        """Persist after every run so an interruption keeps what it earned."""
        runs_path.write_text(json.dumps([asdict(r) for r in runs], indent=2), encoding="utf-8")
        table_path.write_text(summary(runs) + "\n", encoding="utf-8")

    print(
        f"{args.sweep} sweep: {len(cells)} cells x {len(split_seeds)} split x "
        f"{len(train_seeds)} train x 2 "
        f"= {len(cells) * len(split_seeds) * len(train_seeds) * 2} runs, {EPOCHS} epochs each",
        flush=True,
    )
    runs = run(cells, split_seeds, train_seeds, checkpoint=checkpoint)

    table = summary(runs)
    print()
    print(table)
    checkpoint(runs)
    print(f"\nwrote {runs_path.name} and {table_path.name}")


SOURCE = DEFAULT_SOURCE

if __name__ == "__main__":
    main()
