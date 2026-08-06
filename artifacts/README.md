# artifacts/

Output of verification runs, kept rather than regenerated. Each file records
what the tool actually produced at some point, so a later change that alters
these is visible as a diff instead of being invisible.

Nothing here is an input. Deleting a file loses the record of a run; the
commands to reproduce each are below.

## Phase 2 — inspection

| File | What it is |
|---|---|
| `spectrogram_bpsk_01.png`, `spectrogram_qpsk_01.png` | matplotlib spectrograms, the independent check that the terminal inspector agrees with a reference implementation |
| `inspect_*.ansi.txt`, `inspect_*.svg` | the terminal inspector's own output, captured with colour |
| `verify_bpsk_01.txt`, `verify_qpsk_01.txt` | numeric verification: the reference tone sits at exactly +100 kHz, and annotation time ranges line up with the power envelope |

```bash
uv run python scripts/verify_spectrogram.py
uv run python scripts/capture_terminal.py
```

## Phase 4 — training

| File | What it is |
|---|---|
| `train_seed_grid.{json,md,log}` | 5 split seeds x 3 training seeds, showing that split-seed scatter dominates training-seed scatter |
| `leakage_audit.log` | recording disjointness, window twinning, offset and burst-position independence on a built dataset |

```bash
uv run python scripts/run_seed_grid.py
uv run python scripts/audit_leakage.py <dataset_dir> --source examples
```

## Split guarantee — leakage measurement

| File | What it is |
|---|---|
| `leakage_runs.json` | every run: SNR, strategy, split seed, training seed, accuracies, split sizes |
| `leakage_table.md` | the summary table, recording-level vs window-level accuracy per SNR |
| `leakage_*_quick.*` | the small smoke grid, kept because it is what the `--quick` path produces |

```bash
uv run python scripts/leakage_experiment.py --quick   # smoke
uv run python scripts/leakage_experiment.py           # full grid
```

The experiment generates its own recordings at each noise level; `examples/` is
never touched.
