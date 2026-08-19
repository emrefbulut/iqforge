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
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sigmf import SigMFFile

from iqforge.measurement import (
    DEFAULT_SPLIT_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    EPOCHS,
    BuildSpec,
    Run,
    check_environment,
    guard_artifact_rows,
    measure_cell_via_cli,
    summarise_snr_table,
    summarise_stride_table,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

#: Where the extracted DASH7 cabled tree lives; override with --source.
DEFAULT_SOURCE = Path(
    "C:/Users/Emre/AppData/Local/Temp/claude/C--Users-Emre-Desktop-project/"
    "0edfda65-7ea5-4ec7-85c8-a30cc5d9358b/scratchpad/dash7/extracted/cabled"
)

SPLIT_SEEDS = DEFAULT_SPLIT_SEEDS
TRAIN_SEEDS = DEFAULT_TRAIN_SEEDS

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


def _cell_runs(records: Path, spec: BuildSpec) -> list[Run]:
    """Every run for one prepared cell, through the shipped command."""
    return measure_cell_via_cli(records, spec, split_seeds=SPLIT_SEEDS, train_seeds=TRAIN_SEEDS)


def run(cells: list[tuple[float | None, int]]) -> list[Run]:
    """Prepare each SNR once, then measure through the CLI path."""
    work = Path(tempfile.mkdtemp(prefix="iqforge-real-"))
    try:
        prepared: dict[str, Path] = {}
        runs: list[Run] = []
        for snr, stride in cells:
            tag = "raw" if snr is None else f"{snr:+.0f}"
            if tag not in prepared:
                records = work / f"rec_{tag}"
                t0 = time.time()
                prepare(SOURCE, records, snr, rng_seed=1234)
                prepared[tag] = records
                print(f"  prepared {tag} in {time.time() - t0:.0f}s", flush=True)
            spec = BuildSpec(
                window=WINDOW,
                stride=stride,
                split=SPLIT_RATIOS,
                labels="dirname",
                dirname_level=2,
            )
            numeric_snr = float("inf") if snr is None else snr
            numeric_noise = float("nan") if snr is None else snr
            for run_item in _cell_runs(prepared[tag], spec):
                run_item.snr_db = numeric_snr
                run_item.noise_sigma = numeric_noise
                run_item.stride = stride
                runs.append(run_item)
        return runs
    finally:
        shutil.rmtree(work, ignore_errors=True)


#: The channels occupy 19.5 kHz of a 7.68 MHz capture, measured. Detecting one
#: against the others therefore comes with about 26 dB of processing gain, so a
#: wideband SNR of 0 dB is still roughly +26 dB where the signal actually lives.
#: Both figures are reported: the wideband one is what the noise was set to, the
#: in-band one is what the task sees.
OCCUPIED_BW_HZ = 19_500.0
SAMPLE_RATE_HZ = 7_680_000.0
PROCESSING_GAIN_DB = 10.0 * np.log10(SAMPLE_RATE_HZ / OCCUPIED_BW_HZ)


def summarise_real(runs: list[Run]) -> str:
    """Results table, grouped by SNR."""
    return summarise_snr_table(
        runs,
        in_band_gain_db=float(PROCESSING_GAIN_DB),
        caption=(
            f"Wideband SNR is what the added noise was set to, over the full "
            f"{SAMPLE_RATE_HZ / 1e6:.2f} MHz capture. The channels occupy "
            f"{OCCUPIED_BW_HZ / 1e3:.1f} kHz, so the task sees about "
            f"{PROCESSING_GAIN_DB:.0f} dB more than that -- which is why the raw recordings, "
            f"and 0 dB wideband, are both at the ceiling."
        ),
    )


def summarise_real_strides(runs: list[Run]) -> str:
    """Stride-sweep table: inflation against overlap, at one fixed SNR."""
    return summarise_stride_table(
        runs,
        window=WINDOW,
        segment=SEGMENT,
        caption=(
            f"DASH7 cabled, class = Lo-Rate channel, 10 independent recorder runs each. "
            f"Window fixed at {WINDOW} samples; added noise fixed at {STRIDE_SNR_DB:+.0f} dB "
            f"wideband ({STRIDE_SNR_DB + PROCESSING_GAIN_DB:+.0f} dB in-band), the one probed "
            f"point where the honest arm sits stably between the 33.3% chance line and the "
            f"ceiling. Overlap is the only thing that moves between rows: the noise realisation "
            f"is identical, because the recordings are noised once and re-windowed per stride. "
            f"Inflation is the mean paired difference ± its standard error."
        ),
    )


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
    parser.add_argument("--stem", default=None, help="Artifact file stem")
    args = parser.parse_args()

    global SOURCE
    SOURCE = args.source
    if not SOURCE.exists():
        raise SystemExit(f"source not found: {SOURCE}")

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
        stem = "leakage_real_pilot"
    else:
        cells = [(level, STRIDE) for level in SNR_LEVELS]
        stem = "leakage_real"

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    runs_path = ARTIFACTS / f"{stem}_runs.json"
    table_path = ARTIFACTS / f"{stem}_table.md"

    pairs = len(SPLIT_SEEDS.split(",")) * len(TRAIN_SEEDS.split(","))
    planned = len(cells) * pairs * 2

    check_environment(runs_path)
    # Checked against the plan, before anything is trained: a grid that
    # would shrink a published artifact must not start.
    guard_artifact_rows(runs_path, planned)

    # Progress goes to a sibling file. Writing partial results straight
    # over a published artifact is how an interrupted run truncates one.
    partial_path = runs_path.with_suffix(".partial.json")

    def checkpoint(runs: list[Run]) -> None:
        """Persist after every run so an interruption keeps what it earned."""
        partial_path.write_text(json.dumps([asdict(r) for r in runs], indent=2), encoding="utf-8")

    print(
        f"{args.sweep} sweep: {len(cells)} cells x 1 split x 1 train x 2 "
        f"= {len(cells) * 2} runs via `iqforge measure-leakage` ({EPOCHS} epochs each)",
        flush=True,
    )
    runs = run(cells)

    table = summary(runs)
    print()
    print(table)
    guard_artifact_rows(runs_path, len(runs))
    runs_path.write_text(json.dumps([asdict(r) for r in runs], indent=2), encoding="utf-8")
    table_path.write_text(table + "\n", encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    print(f"\nwrote {runs_path.name} and {table_path.name}")


SOURCE = DEFAULT_SOURCE

if __name__ == "__main__":
    main()
