# iqforge — Project Specification

This document is an implementation guide. Apply phases in order. At the end of each phase, run the command in the Verification section and confirm you get the expected output. Do not proceed to the next phase until the current one is verified.

---

## 1. What the project is

`iqforge` is a command-line tool that converts raw RF recordings captured with an SDR (SigMF format) into labeled datasets usable for machine learning.

**Problem it solves:** Today there are tools in the RF domain that generate synthetic data (TorchSig) and tools that inspect recordings (IQEngine), but there is no maintained tool that takes a real recording you have and turns it into a trainable PyTorch `Dataset`. iqforge fills that gap.

**Design principle:** Be fully compatible with the SigMF standard. Do not invent your own format. Compatibility with the existing ecosystem (IQEngine, GNU Radio, TorchSig) is this project's most important feature.

---

## 2. Scope

### IN v0
- Reading SigMF recording pairs (`.sigmf-meta` + `.sigmf-data`)
- Metadata inspection command
- Terminal spectrogram display
- Splitting recordings into fixed-length windows
- Labeling (from three sources: SigMF annotation, directory name, CSV file)
- Stratified train/val/test split
- Writing dataset to disk + reading as PyTorch `Dataset` class
- Smoke test with a small baseline CNN

### NOT in v0 — do not build these
- Capture from live SDR hardware (RTL-SDR, HackRF, etc.)
- Signal transmission
- Demodulation or protocol decoding
- Web interface
- Synthetic signal generation
- Training loop, checkpoint management, hyperparameter search
- Cloud storage integration

If one of these items seems tempting, do not do it. Out of scope.

**`scripts/` is not subject to this list.** Tools there are development utilities: helpers used to build and verify the project, not packaged features. They are not included in the wheel (`[tool.hatch.build.targets.wheel]` packages only `src/iqforge`), are not accessible from the CLI, and are not present in the user's installation.

This distinction matters especially for `scripts/make_example.py`: it generates synthetic signals, i.e. it does the thing forbidden in the list above. The prohibition applies to **user-facing** capabilities — iqforge does not generate synthetic data, it processes existing recordings. Example recordings, however, are the test data required by §7 and are generated once and pinned in the repository.

The same rationale applies to `scripts/run_seed_grid.py`: "training loop and hyperparameter search" is out of scope, but writing a measurement script that reports Phase 4 results is not out of scope.

---

## 3. Technology choices

These are decided; do not change them:

| Field | Choice | Note |
|---|---|---|
| Python | 3.11+ | |
| Package management | `uv` | `pyproject.toml`, PEP 621 |
| CLI | `typer` | type-hint based, automatic `--help` |
| Terminal output | `rich` | tables, color, spectrogram |
| SigMF I/O | `sigmf` (sigmf-python) | **do not write your own parser** |
| Numerics | `numpy`, `scipy` | `scipy.signal` for STFT |
| ML | `torch` | **optional dependency**: `iqforge[torch]` |
| Test | `pytest` | |
| Lint/format | `ruff` | |

`torch` must be optional. `build` and `inspect` commands must work without torch installed; only `iqforge.IQForgeDataset` and `train` require torch.

---

## 4. Command interface

```
iqforge info <path>
    Prints SigMF recording metadata as a readable table.
    Sample rate, center frequency, data type, sample count, duration,
    hardware info, annotation list.

iqforge inspect <path> [--start N] [--samples N] [--nfft 1024]
    Draws a spectrogram in the terminal. Also a power plot along the time axis.
    --start: which sample to start from
    --samples: how many samples to show (default 262144)

iqforge build <input> -o <output_dir>
              [--window 1024] [--stride 512]
              [--labels {annotations,dirname,csv}] [--label-file <path>]
              [--exclude-label <label>] [--split 0.7,0.15,0.15] [--seed 42]
              [--balance-by <sigmf field>]
              [--repr {iq2ch,complex,magphase}] [--normalize/--no-normalize]
    --exclude-label: annotations with this label are completely ignored during
              labeling. Repeatable. Default: `ref_tone`. See 5.3 for details.
    --balance-by: the value of the named SigMF field is spread across splits
              while preserving class stratification. Prevents systematic
              distribution of a nuisance variable across splits.
              See 5.6 for details.
    <input> can be a single .sigmf-meta file OR a directory containing multiple
    recordings. If a directory, scans recursively.
    Output: shard files + manifest.json inside <output_dir>

iqforge stats <dataset_dir>
    Summary of the built dataset: class distribution, window count,
    split sizes, disk usage.

iqforge train <dataset_dir> [--epochs 10] [--batch-size 64]
    Trains a simple baseline CNN. The goal is not accuracy records,
    but proving the dataset is actually trainable.
```

---

## 5. Data flow and technical details

### 5.1 SigMF reading

Required supported data types: `cf32_le`, `ci16_le`, `ci8`.
If you see another `core:datatype`, **do not guess silently** — give an explicit error message: which type was found, which ones are supported.

All samples are represented in memory as `complex64`. When converting from integer types, divide by the full-scale value (`32768.0` for `ci16_le`, `128.0` for `ci8`).

Use `numpy.memmap` for large files; do not load the entire file into memory.

### 5.2 Windowing

The recording is split into sliding windows of length `--window` with step `--stride`.
The incomplete window at the end is discarded (no padding).

Window count = `floor((N - window) / stride) + 1`

### 5.3 Labeling

Three sources, selected with `--labels`:

- `annotations`: from the `annotations` array in SigMF metadata. Each annotation contains `core:sample_start` and `core:sample_count`. A window's label is determined by which annotation range the window's center falls into. The label value is taken from the `core:label` field. Windows that fall into no range are discarded (default) — make this changeable with `--keep-unlabeled`.
- `dirname`: the name of the directory containing the recording becomes the label. This is the common layout in device classification datasets (AirID, ORACLE).
- `csv`: CSV provided with `--label-file`. Columns: `filename,label`.

**Note — overlapping annotations in time and `--exclude-label`.**
The `annotations` rule above looks only at the time axis. If two signals are frequency-separated but overlap in time (e.g. in the `examples/sample` recording, `ref_tone` running throughout alongside `bpsk`/`qpsk` bursts) a window falls into multiple annotation ranges and this rule cannot say which is intended.

This ambiguity is **not resolved** with a heuristic like "narrowest range wins." Such a rule appears to guess the correct answer while hiding the fact that the tool never uses the frequency dimension; it silently produces wrong labels on another recording. Instead the problem is handled explicitly: labels specified with `--exclude-label` are completely ignored during labeling. The default value is `ref_tone`, because the reference tone in the bundled example recording is not a class but a measurement reference.

If a window still falls into multiple annotation ranges after applying `--exclude-label`, it is considered unlabelable and discarded; do not silently pick one. How many windows were discarded for this reason must be reported in `build` output.

Frequency-aware labeling (splitting into time-frequency tiles using `core:freq_lower_edge`/`core:freq_upper_edge`) is out of scope for v0.

### 5.4 Representation (`--repr`)

- `iq2ch` (default): `(2, window)` float32. Channel 0 = I (real), channel 1 = Q (imaginary). Most common format in PyTorch.
- `complex`: `(window,)` complex64. Raw form preserved.
- `magphase`: `(2, window)` float32. Channel 0 = magnitude, channel 1 = phase (radians).

### 5.5 Normalization

On by default. Each window is normalized to unit power separately:

```
x = x / sqrt(mean(|x|^2))
```

For zero-power windows, avoid division errors; return zero.

### 5.6 Split

Stratified by label. Deterministic with `--seed`.

**Important:** Windows from the same recording file must go to the same split. Split by recording, not by window. Otherwise neighboring windows end up in both train and test sets and accuracy is artificially inflated. This rule must not be violated.

**If recording-based splitting is not possible, `build` MUST ERROR.** Silently falling back to window-based splitting is forbidden. Silent fallback produces a dataset that appears to work correctly but has inflated test accuracy — this is the most harmful output the tool can produce, because the error is not noticed by looking at the results.

Cases that must error:

- The input contains only one recording file (no second recording to split into splits).
- A class has too few recording files to fill every non-empty split with the requested split ratios.

The error message must state both the problem and the solution. Example:

```
Cannot perform stratified recording-based split: class 'bpsk' has only 1 recording
file; at least 3 required for a 0.7/0.15/0.15 split.

Per SPEC §5.6, windows from the same recording must go to the same split; falling
back to window-based splitting artificially inflates test accuracy.

Do one of the following:
  - provide more recording files per class (use a directory input)
  - reduce --split ratios, e.g. --split 0.5,0.25,0.25
  - produce only a training set with a single recording: --split 1.0,0,0
```

`--split 1.0,0,0` is an explicit escape hatch for users who want empty val/test; no error is raised because this is a deliberate choice.

**Nuisance variable balancing (`--balance-by`).**
Stratifying by class alone is not enough. A variable that carries no information about class (carrier frequency, receiver hardware, recording day) can be distributed systematically across splits; then class distribution looks perfect while the model is evaluated under a condition it never saw in training, and results are misleading.

`--balance-by <field>` takes a SigMF key. The value is looked up first in the raw dict of the annotation that labels the recording, then in the `global` section; thus the mechanism is not specific to synthetic data and works for any SigMF field (`core:freq_lower_edge`, `core:hw`, extension keys…).

Per-class split recording counts do not change — stratification is preserved. What changes is which recording goes to which split: recordings are processed group by group in round-robin fashion, and each recording is placed in the split where its group is least represented. Group counters are shared across classes so splits complement each other.

Balancing may not hold structurally (if group count exceeds the smallest split, if the field is missing on some recordings, or if every recording falls into a separate group). In that case `build` prints a **WARNING** and continues — not an error, because the split is still valid and recording-based; the user can knowingly accept the remaining skew.

**The ratios decide whether balancing is possible at all.** This is not obvious and is worth stating: when a split's recording count is an exact multiple of a group's size, whole groups land in a single split and no group is shared. With 48 recordings in 4 offset groups of 12, a `0.5/0.25/0.25` split (24/12/12) gives `val` and `test` one whole group each — for every seed. The within-split invariant still holds (a constant group cannot predict the label), so `--balance-by` has not failed; what remains is a *cross-split* problem, and `leakage_warnings` is what reports it. Move to `0.6/0.2/0.2` (28/10/10) and all four groups are shared, silently and for every seed. Measured, not assumed: `scripts/leakage_experiment.py` asserts the property directly from the manifest rather than trusting the absence of a warning, because a partially confounded split does not warn.

Carrier offset per recording is stored in `manifest.json` in the `carrier_offset_hz` field and shown in `stats` output both per recording and as a split summary; imbalance is visible even without `--balance-by`.

### 5.7 Disk format

```
<output_dir>/
  manifest.json
  train/shard_0000.npy
  train/shard_0001.npy
  val/shard_0000.npy
  test/shard_0000.npy
```

Each shard is at most 256 MB. `manifest.json` contents:

```json
{
  "iqforge_version": "0.1.0",
  "created": "ISO8601 timestamp",
  "config": { "window": 1024, "stride": 512, "repr": "iq2ch", "normalize": true, "seed": 42 },
  "label_map": { "device_a": 0, "device_b": 1 },
  "source_files": ["...sigmf-meta paths..."],
  "splits": {
    "train": { "shards": ["train/shard_0000.npy"], "labels": [0,0,1,...], "count": 12000 },
    "val":   { ... },
    "test":  { ... }
  }
}
```

Labels are kept in the manifest; do not write them to a separate file.

### 5.8 PyTorch interface

```python
from iqforge import IQForgeDataset

train = IQForgeDataset("out/", split="train")
x, y = train[0]  # x: torch.Tensor (2, 1024) float32, y: int
len(train)
train.label_map  # {"device_a": 0, ...}
```

Must be a subclass of `torch.utils.data.Dataset`, reading shards lazily with memmap.

---

## 6. Terminal spectrogram

Compute with `scipy.signal.stft`, then draw to the terminal.

Drawing method: use Unicode half-block character (`▀`). Each character carries two vertical pixels — top half foreground color, bottom half background color. `rich` supports this. Works in every terminal.

Color scale: viridis-like, in dB. Lower/upper bounds automatic (5th and 99th percentile).

Axis labels: time (seconds) on horizontal, frequency (MHz, around center frequency) on vertical. Computed using `core:sample_rate` and `core:frequency` from metadata.

Kitty/iTerm graphics protocol not in v0. To be added later.

---

## 7. File structure

```
iqforge/
  pyproject.toml
  README.md
  CONTRIBUTING.md
  CITATION.cff
  LICENSE                       (MIT)
  SPEC.md                       (this document)
  .gitignore
  .github/
    workflows/ci.yml            lint + test (3.11, 3.12) + torch + wheel
    ISSUE_TEMPLATE/bug_report.yml
  src/iqforge/
    __init__.py                 exports load() and (lazy) IQForgeDataset
    cli.py                      typer app: info/inspect/build/stats/train
    io.py                       SigMF reading, data type conversion
    windowing.py                windowing, normalization, representations
    labeling.py                 three label sources, --balance-by field reading
    splitting.py                stratified recording-based split, leakage warnings
    storage.py                  shard write/read, manifest
    dataset.py                  IQForgeDataset (torch)
    training.py                 baseline training loop (torch)
    display.py                  terminal spectrogram
    models.py                   baseline CNN
  tests/
    conftest.py                 shared fixtures
    helpers.py                  synthetic SigMF recording generator
    test_io.py
    test_windowing.py
    test_labeling.py
    test_splitting.py
    test_storage.py
    test_display.py
    test_dataset.py             (skipped if torch absent)
    test_models.py              (skipped if torch absent)
  scripts/                      development tools, not packaged — see §2
    make_example.py             generates example recordings
    verify_spectrogram.py       Phase 2 verification, produces PNG
    capture_terminal.py         saves inspect output with colors
    run_seed_grid.py            Phase 4 seed grid
    audit_leakage.py            leakage audit
    demo.sh                     demo recording command sequence
  docs/
    banner.svg                  README header image
    banner.png                  fallback if SVG does not render
    make_banner.py              banner generator
    demo.md                     asciinema/agg recording instructions
  artifacts/                    persistent outputs of phase verifications
  examples/                     16 recordings: bpsk_01…bpsk_08, qpsk_01…qpsk_08
    bpsk_01.sigmf-meta
    bpsk_01.sigmf-data
    ...
```

You generate the example recordings in `examples/`: each short, single-modulation, with annotations, total under 6 MB. These files are critical — the user must be able to try the tool without hardware.

**Structure: 2 classes × 4 carrier offsets × 2 recordings = 16 recordings.** Each recording 32768 samples (0.032 s), total 4.19 MB, 40 labeled windows per recording (1024/512 windowing), total 640.

All three numbers are mandatory:

- **Multiple files:** §5.6 requires recording-based splitting. With a single file this rule cannot be tested on example data, and `build` could silently fall back to window-based splitting.
- **At least 3 recordings per class:** so the 0.7/0.15/0.15 split can fill all three splits non-empty.
- **2 recordings per (class, offset) cell:** §5.6's within-split independence guarantee distributes a offset's recordings round by round. With a single recording in a cell no round can be formed, all recordings for that offset land in the same split, and train and test share no offset; the model is always tested on unseen carrier and accuracy sticks at chance level, and the Phase 4 verification gate measures nothing. This has been measured: in a single-recording setup 12 of 15 runs gave exactly 50%.

What varies recording to recording: noise seed, symbol sequence, burst time position, carrier offset. What stays fixed: bandwidth (86.4 kHz), burst duration (20480 samples), average power. Each class uses every offset and every burst start equally often.

---

## 8. Phases and verification

### Phase 1 — Skeleton + SigMF reading + `info`
Set up: `pyproject.toml`, package structure, `io.py`, only `info` in `cli.py`.
Generate synthetic example recording in `examples/`.

**Verification:**
```
uv run iqforge info examples/sample.sigmf-meta
```
Sample rate, center frequency, data type, and sample count should look correct.
`tests/test_io.py` must pass.

### Phase 2 — `inspect` terminal spectrogram
**Verification:**
```
uv run iqforge inspect examples/sample.sigmf-meta
```
Spectrogram should appear in the terminal and known frequency components of the synthetic signal should appear in the right places. Also write a small verification script that plots the same data to PNG with matplotlib (`scripts/verify_spectrogram.py`) and verify both images show the same structure.

### Phase 3 — `build` and `stats`
Windowing, labeling, splitting, shard writing, manifest.

**Verification:**
```
uv run iqforge build examples/ -o /tmp/ds --balance-by core:freq_lower_edge
uv run iqforge stats /tmp/ds
```
Class distribution should be balanced, window count should match the formula, `manifest.json` should match the schema. Running twice with the same `--seed` should produce identical splits.

Input is the `examples/` directory, not a single file: §5.6 requires recording-based splitting, which cannot be tested with a single file (and `build` correctly errors in that case).

Why `--balance-by` is needed: in example recordings carrier offset carries no information about class but can be distributed systematically across splits. With stratification by class alone, `--seed 42` gave train four positive offsets and val and test four negative offsets — classes balanced but a distribution shift. The "Carrier offset distribution" table in `stats` output makes this visible; each split should contain both negative and positive offsets.

### Phase 4 — `IQForgeDataset` + `train`
**Verification:**
```
uv run --extra torch iqforge train /tmp/ds --epochs 20
```
Training accuracy on synthetic data should exceed 90%. If it does not, there is a bug in the data pipeline — stop and find why; do not tune hyperparameters.

**Why 20 epochs.** Training accuracy with measured values (split seed 11, training seed 0):

| epoch | train | val | test |
|---|---|---|---|
| 5  | 65.4% | 50.0% | 52.5% |
| 10 | 84.0% | 81.2% | 67.5% |
| 20 | 99.0% | 100%  | 95.0% |

The curve is monotonic; at 5 and 10 epochs the model has not yet converged — this is not a pipeline bug. The 90% threshold is met at 20 epochs.

**Expected test accuracy: 90–100%.** Measured (5 splits × 3 training seeds, 20 epochs): mean **98.4% ± 2.8%**, range 91.25%–100%.

This is above the 75–95% band targeted during setup. The reason is not leakage but task ease:

- In-band SNR ≈ 18 dB (burst power 0.0484, noise power 0.0008).
- Each window carries 1024 samples = 64 symbols; a classical BPSK/QPSK discriminator also achieves ~100% under these conditions.
- Test recordings share the same carrier offset as training (§7), so what is measured is modulation discrimination, not carrier generalization.

High accuracy has been audited with `scripts/audit_leakage.py`: recording disjointness is maintained, no duplicate windows across splits (>0.999 similarity: 0 pairs), carrier offset independent of label in every split (deviation 0).

**Measurement resolution is limited.** Test split is 2 recordings / 80 windows; one window is 1.25% and with one recording per class no recording-level attribute independence can be shown statistically. Accuracy differences of a few percentage points should not be read as meaningful.

**Seed protocol.** Split seed (`build --seed`) and training seed (`train --seed`) are separate and must not be confused: the former determines dataset contents, the latter only weight initialization and batch order. Phase 4 results are reported with a 5 split × 3 training seed grid (`scripts/run_seed_grid.py`, outputs in `artifacts/train_seed_grid.*`).

### Phase 5 — Packaging and documentation
README (installation, 3-command quick start, example output), MIT license, GitHub Actions CI (lint + test, Python 3.11 and 3.12).

**Verification:**
```
uv build
uv tool install --from dist/iqforge-0.1.0-py3-none-any.whl iqforge
iqforge info examples/bpsk_01.sigmf-meta
```
(`pipx install dist/iqforge-0.1.0-py3-none-any.whl` also works instead of `uv tool install`; both install the wheel in an isolated environment and put the command on PATH. Since the repo already uses `uv`, verification was done with `uv tool`.)
Must install and run in a clean environment.

---

## 9. Code quality rules

- Type hints on all public functions
- Docstring: what it does + parameters (Google style)
- Error messages should be user-facing and actionable.
  Bad: `ValueError: invalid datatype`
  Good: `Unsupported data type 'cf64_le'. Supported types: cf32_le, ci16_le, ci8.`
- **Do not assume silently.** If an expected field is missing in metadata, error or warn explicitly; do not invent defaults. Do not continue if sample rate is missing.
- `rich` progress bar for long operations
- Tests for every module. Tests should run on synthetic data, require no network access.
- If you are unsure about something in the SigMF specification, do not guess — use the API provided by the `sigmf` library.

---

## 10. Do not go outside this specification

If you have feature suggestions, ask before implementing. Scope creep is this project's biggest risk. Do not change phase order, do not skip verification steps.
