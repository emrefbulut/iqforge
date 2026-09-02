"""Parity gate: does the shipped command still produce the published grids?

Every table in `docs/methodology.md` is quoted from a file in `artifacts/`, and
every one of those files was written by an experiment script that now measures
through `iqforge measure-leakage`. This script re-measures selected cells and
compares them against what is on disk.

What it compares, and why all three:

1. **How many runs came back.** A grid reduced from 15 seed pairs to 1
   reproduces the first pair exactly. That is what happened during the
   migration, and a value-only comparison called it a pass.
2. **Which seed pairs came back.** Fifteen runs from the wrong fifteen seeds
   is the same count and a different measurement.
3. **What each run measured.** Bit-exact, per row, matched by
   (strategy, split seed, train seed) rather than by position.

The seed lists are deliberately NOT passed on the command line. The published
grids were measured at the command's defaults, so the defaults are part of what
is under test -- passing them would turn the sample-size check into a tautology.

This is a deliberate, expensive run, not a test. Expect a couple of hours.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "scripts"))

import leakage_experiment as synthetic  # noqa: E402
import leakage_loraiq as loraiq  # noqa: E402
import leakage_real as real  # noqa: E402

#: Fields compared per row. Accuracies catch a changed computation; window
#: counts catch a changed split, which a changed accuracy might not.
COMPARED = ("test_accuracy", "train_accuracy", "train_windows", "test_windows")

#: Three cells per table rather than one. One cell cannot distinguish "the code
#: path is intact" from "this particular cell happens to agree".
#:
#: For the stride tables these are the three widest strides. Every cell
#: exercises the same code path, so the cheap ones verify it as well as the
#: expensive ones -- and the cost is not close: at stride 128 a LoRaIQ cell is
#: 78 minutes against 11 at stride 1024 (`artifacts/leakage_loraiq.log`).
#: Widen these tuples to reach the dense end; it costs hours, not minutes.
SYNTHETIC_NOISE = (0.08, 0.17, 0.25)
SYNTHETIC_STRIDES = (1024, 768, 512)
DASH7_STRIDES = (1024, 768, 512)
LORAIQ_STRIDES = (1024, 768, 512)


def artifact_rows(name: str, **match: Any) -> list[dict[str, Any]]:
    """Every recorded run in `artifacts/<name>` whose fields equal `match`."""
    rows = json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
    return [r for r in rows if all(r.get(k) == v for k, v in match.items())]


def compare_cell(
    name: str, measured: list[dict[str, Any]], recorded: list[dict[str, Any]]
) -> list[str]:
    """Compare one cell against its recorded runs; return every failure found.

    Returns a list rather than raising so a run reports every cell that
    disagrees instead of only the first. An empty list is a pass.
    """
    failures: list[str] = []
    if not recorded:
        return [f"{name}: no recorded runs matched this cell; the selector is wrong"]

    if len(measured) != len(recorded):
        failures.append(
            f"{name}: measured {len(measured)} runs, the artifact holds "
            f"{len(recorded)}. Sample size is part of the result: a grid cut to "
            f"one seed pair reproduces the first pair exactly and is still a "
            f"different measurement."
        )

    def key(row: dict[str, Any]) -> tuple[str, int, int]:
        return (str(row["strategy"]), int(row["split_seed"]), int(row["train_seed"]))

    measured_keys = {key(r) for r in measured}
    recorded_keys = {key(r) for r in recorded}
    if measured_keys != recorded_keys:
        missing = sorted(recorded_keys - measured_keys)
        unexpected = sorted(measured_keys - recorded_keys)
        failures.append(
            f"{name}: the seed pairs do not match. "
            f"not measured: {missing[:6]}{' ...' if len(missing) > 6 else ''}; "
            f"not in the artifact: {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}"
        )

    index = {key(r): r for r in measured}
    for expected in recorded:
        got = index.get(key(expected))
        if got is None:
            continue  # already reported by the seed-pair check
        for field in COMPARED:
            if got[field] != expected[field]:
                failures.append(
                    f"{name} {key(expected)} {field}: measured {got[field]!r}, "
                    f"artifact {expected[field]!r}"
                )
    return failures


def measure(records: Path, extra: list[str]) -> list[dict[str, Any]]:
    """Run the shipped command over one cell and return every run it produced."""
    command = [
        sys.executable, "-m", "iqforge", "measure-leakage", str(records),
        "--format", "json", *extra,
    ]  # fmt: skip
    done = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if done.returncode != 0:
        raise RuntimeError(f"measure-leakage failed for {records}:\n{done.stdout}\n{done.stderr}")
    measurement = json.loads(done.stdout)["measurement"]
    rows = measurement.get("rows")
    if rows is None:
        raise RuntimeError(
            "measure-leakage returned no 'rows'. A payload that carries only a "
            "summary cannot be checked for sample size, which is the thing this "
            "gate exists to check."
        )
    return list(rows)


def _synthetic_snr(failures: list[str]) -> None:
    """`artifacts/leakage_runs.json` -- noise sweep at the tool's default stride."""
    for noise in SYNTHETIC_NOISE:
        recorded = artifact_rows("leakage_runs.json", noise_sigma=noise)
        work = Path(tempfile.mkdtemp(prefix="parity-snr-"))
        try:
            records = work / "records"
            synthetic.generate_recordings(noise, records)
            measured = measure(
                records,
                [
                    "--split",
                    synthetic.SPLIT_RATIOS,
                    # The synthetic labels live in annotations, which the folder
                    # audit's single-window probe cannot see, so preflight is
                    # inconclusive on data that is fine.
                    "--force",
                    "--labels",
                    "annotations",
                    "--balance-by",
                    "core:freq_lower_edge",
                ],  # fmt: skip
            )
            failures += compare_cell(f"synthetic-snr noise={noise}", measured, recorded)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _synthetic_stride(failures: list[str]) -> None:
    """`artifacts/leakage_stride_runs.json` -- stride sweep at one noise level."""
    work = Path(tempfile.mkdtemp(prefix="parity-synth-stride-"))
    try:
        records = work / "records"
        synthetic.generate_recordings(synthetic.STRIDE_NOISE, records)
        for stride in SYNTHETIC_STRIDES:
            recorded = artifact_rows("leakage_stride_runs.json", stride=stride)
            measured = measure(
                records,
                [
                    "--split",
                    synthetic.SPLIT_RATIOS,
                    "--force",
                    "--labels",
                    "annotations",
                    "--window",
                    str(synthetic.WINDOW),
                    "--stride",
                    str(stride),
                    "--balance-by",
                    "core:freq_lower_edge",
                ],  # fmt: skip
            )
            failures += compare_cell(f"synthetic-stride stride={stride}", measured, recorded)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _dash7_stride(failures: list[str]) -> None:
    """`artifacts/leakage_real_stride_runs.json` -- DASH7 cabled, noised once."""
    if real.DEFAULT_SOURCE is None:
        raise SystemExit(f"DASH7 source not configured: set {real.ENV_DASH7}")
    if not real.DEFAULT_SOURCE.exists():
        raise SystemExit(f"DASH7 source not found: {real.DEFAULT_SOURCE}")
    work = Path(tempfile.mkdtemp(prefix="parity-dash7-"))
    try:
        records = work / "records"
        real.prepare(real.DEFAULT_SOURCE, records, real.STRIDE_SNR_DB, rng_seed=1234)
        for stride in DASH7_STRIDES:
            recorded = artifact_rows("leakage_real_stride_runs.json", stride=stride)
            measured = measure(
                records,
                [
                    "--split",
                    real.SPLIT_RATIOS,
                    "--window",
                    str(real.WINDOW),
                    "--stride",
                    str(stride),
                    "--labels",
                    "dirname",
                    "--dirname-level",
                    "2",
                ],  # fmt: skip
            )
            failures += compare_cell(f"dash7-stride stride={stride}", measured, recorded)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _loraiq_stride(failures: list[str]) -> None:
    """`artifacts/leakage_loraiq_runs.json` -- grouped so receptions stay together."""
    inputs = (
        loraiq.DEFAULT_SOURCE,
        loraiq.DEFAULT_INDEX,
        loraiq.DEFAULT_LABELS,
        loraiq.DEFAULT_GROUPS,
    )
    for path in inputs:
        if path is None:
            raise SystemExit(f"LoRaIQ inputs not configured: set {loraiq.ENV_SOURCE}")
        if not path.exists():
            raise SystemExit(f"LoRaIQ input not found: {path}")
    work = Path(tempfile.mkdtemp(prefix="parity-loraiq-"))
    try:
        records = work / "records"
        loraiq.prepare(loraiq.DEFAULT_SOURCE, records, loraiq.frame_offsets(loraiq.DEFAULT_INDEX))
        for stride in LORAIQ_STRIDES:
            recorded = artifact_rows("leakage_loraiq_runs.json", stride=stride)
            measured = measure(
                records,
                [
                    "--split",
                    loraiq.SPLIT_RATIOS,
                    "--window",
                    str(loraiq.WINDOW),
                    "--stride",
                    str(stride),
                    "--labels",
                    "csv",
                    "--label-file",
                    str(loraiq.DEFAULT_LABELS),
                    "--group-by",
                    f"csv:{loraiq.DEFAULT_GROUPS}",
                ],  # fmt: skip
            )
            failures += compare_cell(f"loraiq-stride stride={stride}", measured, recorded)
    finally:
        shutil.rmtree(work, ignore_errors=True)


TABLES = (
    ("leakage_runs.json", _synthetic_snr),
    ("leakage_stride_runs.json", _synthetic_stride),
    ("leakage_real_stride_runs.json", _dash7_stride),
    ("leakage_loraiq_runs.json", _loraiq_stride),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=[name for name, _ in TABLES],
        default=[name for name, _ in TABLES],
        help=(
            "Which published tables to check. Defaults to all four. A full run "
            "takes hours, so this exists to resume one that was interrupted -- "
            "a partial run is not a gate, and the verdict line says which "
            "tables it covered."
        ),
    )
    selected = parser.parse_args().tables

    failures: list[str] = []
    started = time.time()
    for name, check in TABLES:
        if name not in selected:
            continue
        print(f"\n=== {name}", flush=True)
        before = len(failures)
        cell_start = time.time()
        check(failures)
        added = len(failures) - before
        verdict = "OK" if added == 0 else f"{added} failure(s)"
        print(
            f"=== {name}: {verdict} ({(time.time() - cell_start) / 60:.1f} min)",
            flush=True,
        )

    covered = ", ".join(selected)
    print(f"\ntotal wall time {(time.time() - started) / 60:.1f} min over: {covered}")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for line in failures:
            print(f"  {line}")
        raise SystemExit("PARITY_GATE_FAILED")
    print("\nPARITY_GATE_PASSED")


if __name__ == "__main__":
    main()
