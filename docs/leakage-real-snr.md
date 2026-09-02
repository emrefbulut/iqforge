# DASH7 leakage SNR curve — locked plan, not a result

This file is the experiment ROADMAP Now item 1 still needs. It has **not** been
run. There is no `artifacts/leakage_real_runs.json` and no
`artifacts/leakage_real_table.md`. Do not quote this document as a measured
curve.

The published +13.6 pp table is synthetic BPSK/QPSK
(`scripts/leakage_experiment.py` → `artifacts/leakage_runs.json`). DASH7 already
has the stride half of the same paired design
(`uv run python scripts/leakage_real.py --sweep stride` →
`artifacts/leakage_real_stride_runs.json`). The missing half is the SNR sweep:
same pairing, same model, same 20 epochs, only added wideband noise moving.

`iqforge measure-leakage` has no `--sweep snr`. Noise is injected here, before
windowing, for the reasons in `scripts/leakage_real.py`.

## Seeds — lock *n* before any run

Leave the script's seed lists alone. They are
`iqforge.measurement.DEFAULT_SPLIT_SEEDS` and `DEFAULT_TRAIN_SEEDS`:

- split: `42,7,1234,2026,99`
- train: `0,1,2`
- **n = 15 pairs per cell** (5 × 3), both arms → 30 runs per SNR

The script has no seed flags. Do not edit those constants after seeing results.
Do not enlarge *n* if the table is inconclusive (CONTRIBUTING convention 5).
The progress line currently prints `1 split x 1 train`; `_cell_runs` still uses
these 15 pairs, and `guard_artifact_rows` plans 180 runs. Trust the constants,
not that line.
Reconnaissance already used smaller *n* on purpose: probe n = 1
(`artifacts/leakage_real_probe_*`), cliff n = 2
(`artifacts/leakage_real_cliff_*`). Those files are not this grid.

## SNR list — also locked, and not the module default

Do **not** run the script with no `--snr`. `SNR_LEVELS` in
`scripts/leakage_real.py` is `(raw, +6, +3, 0, −3, −6)` dB wideband. The pilot
already put raw and 0 dB at 100/100 on both arms; methodology §6.4 explains the
~26 dB processing gain that makes that band a ceiling.

The list below is taken only from those published reconnaissance files, before
any n = 15 SNR run:

`-15,-17,-19,-21,-22,-28` (wideband dB)

`--stem leakage_real` is required. `--snr` without `--stem` defaults to
`leakage_real_probe` and would replace the published n = 1 reconnaissance.

Window 1024, stride 512, split `0.6,0.2,0.2`, labels `dirname` level 2 — the
script's existing SNR path, not a second implementation.

CPU only. Do not enable CUDA for this table.

## Wait until both are true

1. **`IQFORGE_DASH7` names an extracted `cabled` tree that exists.** The script
   has no fallback path. A zip still downloading is not this variable. Point
   `--source` at the same tree if you prefer a flag over the environment.
2. **`scripts/parity_gate.py` is not using the CPU.** The gate re-measures
   published tables. Do not start this grid while that process is running.

## Command (after both waits)

```bash
uv run python scripts/leakage_real.py --sweep snr --snr -15,-17,-19,-21,-22,-28 --stem leakage_real
```

Writes, if it finishes:

- `artifacts/leakage_real_runs.json`
- `artifacts/leakage_real_table.md`

Interrupted progress is `artifacts/leakage_real_runs.partial.json` and is
removed on a clean finish. `guard_artifact_rows` will refuse to shrink a
published file; these two names are currently absent, so the first complete
write is the one that creates them.

## Expected hours

Six cells × 15 pairs × 2 arms = **180 runs**. The published DASH7 stride-512
cell in `artifacts/leakage_real_stride.log` was about 33 s/run (3582/1194
windows, 20 epochs). 180 × 33 s is about **1.6 hours** on CPU — the same
wall-time methodology §6.4 already gave for a six-cell DASH7 SNR grid at this
window count. That is scaled from those timings, not a measurement of this
grid.

## What the table cannot be

Methodology §6.4: on this dataset the task is a cliff (solved to chance in
about 7 dB), not a graded SNR curve. Running the paired design still produces
the artifact ROADMAP asked for. It does not make DASH7 into the synthetic
BPSK/QPSK experiment. If the n = 15 table is inconclusive, that is the finding.

## Hardware capture

No RTL-SDR, HackRF, or ADALM-Pluto is present, so an own-device capture
(ROADMAP Now item 2) is blocked until a receiver is attached; this recipe does
not invent or reuse files to stand in for that recording.
