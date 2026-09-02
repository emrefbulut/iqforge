from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "scripts"))

import leakage_experiment as synthetic  # noqa: E402
import leakage_loraiq as loraiq  # noqa: E402
import leakage_real as real  # noqa: E402


def _first_pair(path: Path) -> tuple[dict, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    rec = next(r for r in rows if r["strategy"] == "recording-level")
    win = next(r for r in rows if r["strategy"] == "window-level")
    return rec, win


def _measure(path: Path, extra: list[str]) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "iqforge",
        "measure-leakage",
        str(path),
        "--format",
        "json",
        *extra,
    ]
    done = subprocess.run(
        cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if done.returncode != 0:
        raise RuntimeError(f"measure-leakage failed for {path}:\n{done.stdout}\n{done.stderr}")
    return json.loads(done.stdout)["measurement"]


def _cmp(name: str, got: dict, expect: dict) -> None:
    checks = {
        "test_accuracy": float(got["test_accuracy"]) == float(expect["test_accuracy"]),
        "train_accuracy": float(got["train_accuracy"]) == float(expect["train_accuracy"]),
        "train_windows": int(got["train_windows"]) == int(expect["train_windows"]),
        "test_windows": int(got["test_windows"]) == int(expect["test_windows"]),
    }
    for key, ok in checks.items():
        print(
            f"{name} {key}: got={got[key]} expected={expect[key]} {'MATCH' if ok else 'MISMATCH'}"
        )
    if not all(checks.values()):
        raise SystemExit(f"{name} comparison failed")


def main() -> None:
    # 1) synthetic SNR table sample: noise=0.08, default stride (512)
    exp_rec, exp_win = _first_pair(ARTIFACTS / "leakage_runs.json")
    work = Path(tempfile.mkdtemp(prefix="p5-synth-"))
    try:
        records = work / "records"
        synthetic.generate_recordings(0.08, records)
        measured = _measure(
            records,
            [
                "--split",
                synthetic.SPLIT_RATIOS,
                "--force",
                "--labels",
                "annotations",
                "--balance-by",
                "core:freq_lower_edge",
            ],
        )
        _cmp("synthetic-snr recording", measured["recording_level"], exp_rec)
        _cmp("synthetic-snr window", measured["window_level"], exp_win)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # 2) synthetic stride table sample: noise=0.17, stride=1024
    exp_rec, exp_win = _first_pair(ARTIFACTS / "leakage_stride_runs.json")
    work = Path(tempfile.mkdtemp(prefix="p5-synth-stride-"))
    try:
        records = work / "records"
        synthetic.generate_recordings(0.17, records)
        measured = _measure(
            records,
            [
                "--split",
                synthetic.SPLIT_RATIOS,
                "--force",
                "--labels",
                "annotations",
                "--window",
                "1024",
                "--stride",
                "1024",
                "--balance-by",
                "core:freq_lower_edge",
            ],
        )
        _cmp("synthetic-stride recording", measured["recording_level"], exp_rec)
        _cmp("synthetic-stride window", measured["window_level"], exp_win)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # 3) DASH7 real stride table sample: snr=-19, stride=1024
    if real.DEFAULT_SOURCE is None:
        raise SystemExit(f"DASH7 source not configured: set {real.ENV_DASH7}")
    if not real.DEFAULT_SOURCE.exists():
        raise SystemExit(f"DASH7 source not found: {real.DEFAULT_SOURCE}")
    exp_rec, exp_win = _first_pair(ARTIFACTS / "leakage_real_stride_runs.json")
    work = Path(tempfile.mkdtemp(prefix="p5-real-"))
    try:
        records = work / "records"
        real.prepare(real.DEFAULT_SOURCE, records, -19.0, rng_seed=1234)
        measured = _measure(
            records,
            [
                "--split",
                real.SPLIT_RATIOS,
                "--window",
                "1024",
                "--stride",
                "1024",
                "--labels",
                "dirname",
                "--dirname-level",
                "2",
            ],
        )
        _cmp("real-stride recording", measured["recording_level"], exp_rec)
        _cmp("real-stride window", measured["window_level"], exp_win)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # 4) LoRaIQ table sample: stride=1024
    for path in (
        loraiq.DEFAULT_SOURCE,
        loraiq.DEFAULT_INDEX,
        loraiq.DEFAULT_LABELS,
        loraiq.DEFAULT_GROUPS,
    ):
        if path is None:
            raise SystemExit(f"LoRaIQ inputs not configured: set {loraiq.ENV_SOURCE}")
        if not path.exists():
            raise SystemExit(f"LoRaIQ input not found: {path}")
    exp_rec, exp_win = _first_pair(ARTIFACTS / "leakage_loraiq_runs.json")
    work = Path(tempfile.mkdtemp(prefix="p5-loraiq-"))
    try:
        records = work / "records"
        loraiq.prepare(loraiq.DEFAULT_SOURCE, records, loraiq.frame_offsets(loraiq.DEFAULT_INDEX))
        measured = _measure(
            records,
            [
                "--split",
                loraiq.SPLIT_RATIOS,
                "--window",
                "1024",
                "--stride",
                "1024",
                "--labels",
                "csv",
                "--label-file",
                str(loraiq.DEFAULT_LABELS),
                "--group-by",
                f"csv:{loraiq.DEFAULT_GROUPS}",
            ],
        )
        _cmp("loraiq recording", measured["recording_level"], exp_rec)
        _cmp("loraiq window", measured["window_level"], exp_win)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("ALL_SAMPLE_COMPARISONS_MATCH")


if __name__ == "__main__":
    main()
