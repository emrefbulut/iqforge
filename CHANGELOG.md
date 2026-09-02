# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, interfaces may change between minor releases. The
dataset written by `build` carries its own `manifest_schema` number so a reader
can tell whether the format it is looking at is one it understands.

## [Unreleased]

### Added

- **`--device cpu|cuda` on `iqforge measure-leakage`**, defaulting to `cpu`.
  CUDA is opt-in for new measurements and is never selected just because a GPU
  is present; `cuda` errors when torch reports none. Published tables, the
  parity gate, and `artifacts/*.json` stay on CPU. `TrainingResult.environment`
  already recorded the device; the measurement path now stamps the device that
  was actually requested.
- `iqforge measure-leakage` now accepts `--balance-by`, so the command path can
  run the same nuisance-balancing setup that the published synthetic measurement
  tables used.
- `TrainingResult.environment` now also records the **numpy, scipy and sigmf**
  versions. Windowing and normalisation run on numpy, the spectrogram on scipy,
  the reader on sigmf; none of those were in the environment dict, so two tables
  measured on the same device with different numeric stacks could not be told
  apart. The sweep scripts still refuse to extend a checkpoint whose environment
  does not match, and that comparison now covers the new fields.
- `docs/methodology.md` §6 numbers the five assessed datasets (6.1 AirID, 6.2
  Vega-C, 6.3 DASH7 `ds_indoor`, 6.4 DASH7 `ds_indoor_cabled`, 6.5 LoRaIQ) so a
  later command can cite `category 4` rather than a paragraph.
- `iqforge audit` reports **shared timestamp**, the sibling of shared air time.
  Shared air time looks for intersecting capture intervals; this looks for the
  same `core:datetime` value repeating across classes and landing as a different
  set in each split. That is the Vega-C pattern (methodology §6.2): five
  satellites, three shared stamps, no interval overlap, a recording-level split
  that puts a different pass in each bin. A single stamp across the whole set is
  still a placeholder (`examples/` does not false-positive).
- `iqforge.measurement` is the paired leakage-measurement core: one `BuildSpec`,
  recording-level build, window-level re-deal, paired training, paired
  statistics. No training CLI yet. The three experiment scripts now call it;
  dataset-specific `prepare` stays in `scripts/`. The LoRaIQ bit-exact cell
  (stride 1024 / split 42 / train 0) is the acceptance gate and is skipped in
  CI when the recordings are not present; published tables are reproduced from
  the recorded run files.
- `iqforge measure-leakage` is the refuse path: it runs `audit`, classifies the
  result into six categories (methodology §6.1–§6.4 plus remaining leaks and
  unsplittable sets), estimates the work a paired cell would do, and stops.
  This version does not train. `--force` overrides a refusal and puts the
  overridden category in the header (`FORCED PAST audit VERDICT 'ceiling'`).
  LoRaIQ-like simultaneous receptions are not refused when `--group-by` holds
  them together.

### Changed

- **The experiment scripts and their tests no longer carry a hardcoded path.**
  `scripts/leakage_real.py`, `scripts/leakage_loraiq.py`, `tests/test_preflight.py`
  and `tests/test_measurement.py` all fell back to an absolute path inside one
  developer's temporary directory. That path resolved on exactly one machine,
  so the scripts ran there and were unrunnable everywhere else, and the LoRaIQ
  tests passed there and skipped silently everywhere else -- while
  `docs/methodology.md` opened with "everything below is reproducible from this
  repository". The datasets are now named by `IQFORGE_DASH7` and
  `IQFORGE_LORAIQ` (plus optional `IQFORGE_LORAIQ_INDEX` / `_LABELS` /
  `_GROUPS`) with no fallback, every path keeps its command-line flag, and both
  the scripts' errors and the tests' skip reasons name the variable to set
  instead of saying "not on this machine". `docs/methodology.md` §Reproducing
  now shows the variables. A new `tests/test_repo_hygiene.py` scans `src/`,
  `scripts/` and `tests/` for committed machine paths so this cannot return
  quietly; a line that only looks like one opts out with an inline
  `not-a-machine-path` marker.
- **`audit` reports the Vega-C pattern as RISK rather than LEAK.** The status
  was wrong about what kind of failure it names. Every other LEAK in this tool
  means the same material is on both sides of a split -- overlapping air time,
  identical data, one recording in two bins. The Vega-C pattern is the
  opposite: a *different* pass, with its own Doppler, elevation and SNR, in
  each split. That is distribution shift, which is what `docs/methodology.md`
  §6.2 already called it, and the two failures move a score in opposite
  directions -- leakage inflates it, shift depresses it. Filing one under the
  other costs `LEAK` its meaning, and the value of these statuses is that they
  mean something precise.
  Consequence is unchanged where it matters: `measure-leakage` still refuses
  the set as category 2, and `audit --strict` still exits non-zero. What
  changes is that `audit` without `--strict` no longer exits 1 on it, which is
  correct -- nothing there is proven to leak.
- **A result measured once no longer prints an uncertainty.** A standard error
  over one seed pair is zero by definition, and `inflation=+53.8 pp +/- 0.0`
  states the strongest possible confidence exactly where the evidence is
  weakest. `measure-leakage` now prints `(uncertainty not estimated, n=1)`
  instead, the JSON payload carries `stderr_pp: null` rather than `0.0`, and
  the markdown tables leave the `±` off a single-pair row and label its
  inflation `(not estimated)`. Rows with a real sample are unchanged, so every
  published table under `artifacts/` renders exactly as before.
- **sigmf 1.13.0 fixed the upstream bug this project works around**
  ([sigmf-python#159](https://github.com/sigmf/sigmf-python/issues/159), fixed
  by [#160](https://github.com/sigmf/sigmf-python/pull/160)): constructing a
  `SigMFFile` no longer rewrites the caller's metadata dict. The tripwire in
  `tests/test_io.py` fired on the release and now records 1.13.0 as verified
  non-mutating, measured rather than assumed -- the accessor the fix added is
  called `declared_version`, not the `__original_version` the pull request
  described. Nothing in `iqforge` changes: `load()` reads `core:version` out of
  the parsed JSON before handing the dict over, which is correct under both
  behaviours, so the workaround became redundant rather than wrong. The
  `sigmf>=1.11.1` floor deliberately stays where it is. `iqforge info` still
  prints `1.0.0 (file); 1.2.6 (reader)`, because 1.13.0 kept normalising
  `core:version` inside the handle's own copy -- confirmed against the three
  public captures the upstream report cites.
- The three leakage scripts (`scripts/leakage_experiment.py`,
  `scripts/leakage_real.py`, `scripts/leakage_loraiq.py`) now execute measured
  cells through `iqforge measure-leakage --format json` instead of keeping a
  second direct measurement path. Dataset-specific preparation remains in
  scripts; pairing/training stays in the command path.
- **The migration gate now compares sample size, not only values.**
  `scripts/parity_gate.py` (was `scripts/_phase5_sample_checks.py`) re-measures
  three cells per published table instead of one and checks three things per
  cell: how many runs came back, which seed pairs they came from, and every
  run's accuracies and window counts, matched by seed rather than by position.
  The version it replaces read one recording-level row and one window-level row
  per table. Every assertion it made was true of a grid that had been cut from
  15 seed pairs to 1, because the surviving pair reproduced exactly -- so the
  gate reported a pass on the change it existed to catch. It also stopped
  running at all when the JSON payload grew a `rows` list, since it read
  `recording_level`/`window_level` keys that no longer exist. The synthetic
  cells run forced preflight intentionally, because their annotation labels are
  not visible to folder-audit's single-window probe.
- SPEC §5.10 now describes the refuse path that shipped, not a constraint on a
  command that did not exist. The command is read-only on the user's recordings
  and will not grow a `--sweep snr` flag. Adding noise requires writing altered
  copies and is dataset-specific (DASH7: signal on air 6.8% of the time, ~26 dB
  of processing gain). SNR injection stays a preparation recipe in `scripts/`;
  the command measures the folder that recipe produces.

## [0.4.0] — 2026-08-19

### Fixed

- **Correctness: `--labels csv` and `--group-by csv:` identified recordings by
  file name, not by path.** In a nested directory layout the same file name
  repeats under every directory, so rows for different recordings collided and
  the surviving one won. Measured on a public capture set: a 312-row label table
  collapsed to 47 distinct keys and **310 of 312 recordings came out carrying a
  single label**. Nothing warned — every recording was labelled, every window
  was labelled, the class counts looked plausible, and the split satisfied every
  constraint the tool checks.

  **Affects `0.1.0`, `0.2.0` and `0.3.0`** (`--labels csv` since `0.1.0`,
  `--group-by csv:` since `0.2.0`). A **flat** directory of uniquely named
  recordings is unaffected, because there the file name is already a unique key
  — which is why the repository's own fixtures never caught it.

  If you built a dataset from a CSV over a nested layout with an affected
  version, check it:

  ```bash
  iqforge audit <dataset-or-folder>
  ```

  The report's `classes:` line shows the distribution, and `label source` says
  whether every label the table declares survived the lookup.

  Both paths now match the value as written — normally the path relative to the
  input directory — and fall back to the bare name only when that name is
  unambiguous, refusing with a message naming the fix when it is not.

- `build --labels csv` now **errors** when the labels the CSV gives the
  recordings being built do not all survive into the dataset, naming the ones
  that were lost. No threshold and no notion of "too imbalanced": a rare-event
  dataset with 200 background against 1 event passes silently, while 4 declared
  labels arriving as 1 does not. Compared after `--exclude-label` and after
  unlabelled windows are dropped, so neither can look like a collapse.
- `iqforge audit` carried the same file-name bug against its own manifest, which
  left the class-axis checks silent and made the shared-air-time lookup compare
  `None` to `None` — reporting 465 unchecked pairs as a pass, inside the one
  command written on the principle that an unexamined area must never read as
  one.
- `iqforge audit` read `core:datetime` from `global`; SigMF puts it in
  `captures`. The capture-time confound check reported NOT CHECKED on every
  conforming recording — including a public set whose two classes were recorded
  a week apart.
- `iqforge audit` ranked class axes by raw score, so an axis measurable on only
  one class scored 100% against a chance of 100% and was reported as the reason
  a task sits at the ceiling. Axes are ranked by margin over chance now, and the
  `unknown` verdict prints that margin.
- `iqforge audit` treated a constant `core:datetime` as evidence of simultaneous
  capture. `examples/` stamps all 16 recordings `2024-01-01T00:00:00Z`, which
  read literally is sixteen simultaneous captures; a single distinct timestamp
  across a whole set is now reported as a placeholder.
- A folder audit no longer stops at the first unreadable recording. One
  `cf16_le` file in a 330-file set produced no report at all; it now reports how
  many could not be opened and audits the rest, with the denominator on its own
  row so every other check is read against it.
- Findings identify recordings by path relative to the audited root. A capture
  set with a `3.sigmf-meta` under every session and receiver produced findings
  reading `3.sigmf-meta / 3.sigmf-meta`, which named two files and pointed at
  neither.

### Added

- `iqforge audit` reports **shared air time**: recordings whose capture
  intervals intersect must land in the same split. One transmission heard by
  four receivers is four files and one event, and recording-level splitting does
  not help — the unit of independence is the transmission, not the file. This is
  how `--group-by` gets verified rather than assumed.
- **`iqforge audit --labels csv`** (with `--label-file`), so a dataset labelled
  from a table can be assessed *before* it is built, which is what auditing a
  folder is for. Same relative-path matching as `build`; a collapsed table is
  reported as a proven finding rather than refused, along with any recordings
  the table omits.
- `build --labels annotations` warns when every window lands in one class. That
  source has no declared label set to compare against, so discovery finding a
  single label is the closest analogue of a collapse.
- `iqforge audit` and `iqforge stats` always print the class distribution and
  the **chance line** — what a constant predictor would score. Not a check and
  not a status: imbalance is a property of the input, and a rare-event dataset
  is a normal thing to build. There is deliberately no threshold on skew.
- **`--group-by collection`**, reading `core:collection` — the field SigMF
  already has for this, whose own specification example is "channels from a
  phased array". `iqforge audit` gains a `collection members` row reporting
  whether a declared collection's members landed in one split.

  Documented as a **hint, not a proof**, and the code enforces the distinction:
  the audit row is never better than `PASS/sample`, because a Collection asserts
  that recordings are *related* and not that they are statistically dependent —
  a collection of "everything in my paper" and one of "four simultaneous
  receptions of one frame" are the same object to a reader of the format. A
  recording can also declare at most one collection, so nested grouping levels
  cannot all be expressed.
- `Recording.capture_datetime`, read from the first capture segment.
- **`iqforge train --device auto|cpu|cuda`**, defaulting to `cpu`. The default
  is a reproducibility promise: cuDNN selects kernels by heuristic, so the same
  seed on the same GPU can pick a different reduction order between runs. `cuda`
  errors rather than falling back when none is present, and warns that its
  numbers are not bit-comparable with CPU runs. The `[torch]` extra still
  installs a CPU wheel; CONTRIBUTING documents installing a CUDA build.
- `TrainingResult.environment` records the device, torch version and CUDA
  version, and the sweep scripts stamp every run with it and **refuse to extend
  a checkpoint measured on a different device**. A grid resumed on another
  device yields a table whose rows are not comparable and whose output would not
  show it.
- `docs/methodology.md` gains three cases: LoRaIQ as a fifth assessed dataset,
  a disqualification that is a property of a dataset *paired with a class
  definition* rather than of the data, and an acquisition method that was
  technically correct per file and invalid in aggregate.

### Changed

- **`sigmf` floor raised to `>=1.11.1`**, from `>=1.2.1`. The old floor claimed
  compatibility with releases nobody here had run: `uv.lock` is not committed,
  so CI resolves fresh and therefore tests exactly one version — the newest.
  1.11.1 and 1.12.0 are the two that have actually been exercised. No upper
  bound is added: capping a library's dependency propagates into every
  environment that installs it, and the compensating control is a test that
  reports a behaviour change rather than a pin that blocks users.
- The upstream `core:version` report
  ([sigmf-python#159](https://github.com/sigmf/sigmf-python/issues/159)) was
  accepted and fixed in [#160](https://github.com/sigmf/sigmf-python/pull/160),
  approved and slated for `v1.13.0`. `sigmf 1.12.0` still mutates, so the
  workaround stays; `load()` reads `core:version` before handing the dict over
  and is correct either way. The test that watches for this now distinguishes
  "new version, behaviour unchanged" from "new version, behaviour changed", and
  a stand-in for the fixed library is exercised so the forward path is tested
  rather than assumed.

## [0.3.0] — 2026-08-12

### Added

- **`iqforge audit`** reports leakage risk and whether a leakage measurement is
  possible at all, on a built dataset or on a folder of recordings, without
  training anything. It checks recording disjointness, cross-split window
  overlap, which measurable axis separates the classes, the processing gain
  available to that separation, and — on a folder — whether two recordings
  claim the same air time or ship identical data.

  It has no "clean" status. Findings are `LEAK`, `RISK`, `PASS/proof`,
  `PASS/sample` or `NOT CHECKED`, the summary counts unchecked areas separately
  from passes, and the list of what was not checked cannot be suppressed. The
  report is fixed-width ASCII so it can be quoted unaltered; `--format json`
  carries the same fields. Exit code 1 on `LEAK`, and on `RISK` too with
  `--strict`.

  Overlap is decided from sample-index ranges rather than content similarity,
  and in a dataset `iqforge` built it is settled by proof: windows can only
  overlap within a recording, so recording disjointness makes cross-split
  overlap impossible without reading a single window.

### Changed

- `docs/methodology.md` §3 adds the stride sweep repeated on a real capture,
  and §6 adds a fourth assessed dataset. `scripts/audit_leakage.py` now
  documents that its cosine-similarity twinning check is blind to offset
  overlap; `iqforge audit` does not inherit it.

## [0.2.0] — 2026-08-10

### Added

- **`--group-by`** keeps recordings that are not independent of each other in
  the same split. The counterpart to `--balance-by`, which spreads a nuisance
  variable across splits; this holds related recordings together. Two schemes:
  `path:<regex>` over the recording's relative path, and `csv:<file>` with
  `recording,group` columns. Recordings sharing a key become one indivisible
  unit, and the unit replaces the recording as the thing allocated, so
  stratification and balancing both operate over units.
- **`--dirname-level`** chooses which ancestor directory `--labels dirname`
  reads. `1` (the default, unchanged) is the recording's own directory, `2` its
  parent — for layouts that group recordings twice, such as `CH0/rec1/`.
- A warning when `--labels dirname` produces labels that look like run numbers
  and another directory level offers real class names. It fires only when both
  hold, so numbered classes such as `device_01` or `snr_10` stay quiet.
- A warning when an annotation claims samples the data file does not contain
  (`sample_start + sample_count` past the end). The sample count comes from the
  file size, so this is a flat contradiction between metadata and data.
- A diagnostic when `--labels annotations` finds nothing: how many annotations
  were scanned, how many carry `core:label`, and which other keys hold text,
  with an example. Behaviour is unchanged — `core:label` is still the only
  field read, and the message says so.
- **`manifest_schema`** in `manifest.json`, starting at `1`. `iqforge_version`
  moves on every release whether the format changed or not, so it cannot tell a
  reader whether the shape of the file is one it understands. A manifest with a
  schema newer than the reader is refused rather than read optimistically; an
  older one, or one written before the field existed, is read normally.
- [`docs/methodology.md`](docs/methodology.md): how the split claims were
  measured, what the numbers do not cover, the silent failures found along the
  way, and why the leakage measurement could not be repeated on real data.
- Real-capture verification. The `ci8` and `ci16_le` paths were checked against
  the raw bytes of three public recordings from the GNU Radio SigMF collection
  (exact ÷128 and ÷32768, correct I/Q order). On a 40 MS/s cellular capture,
  annotated frequency bands measured 31.8 dB above unannotated ones.

### Fixed

- `iqforge info` reported the reader's SigMF version rather than the
  recording's. `SigMFFile(metadata=...)` mutates the dict it is given and
  overwrites `core:version` with the spec version the installed library
  implements, so three real captures declaring `1.0.0` were shown as `1.2.6`.
  The version is now read before the dict is handed over, and both are shown
  when they differ. Reported upstream as
  [sigmf/sigmf-python#159](https://github.com/sigmf/sigmf-python/issues/159).

## [0.1.0] — 2026-08-10

First release.

### Added

- SigMF reading for `cf32_le`, `ci16_le` and `ci8`; anything else is an explicit
  error rather than a silent guess. Large files are memory-mapped.
- `iqforge info` — metadata and annotations as a table.
- `iqforge inspect` — spectrogram and power-over-time in the terminal.
- `iqforge build` — fixed-length windowing with configurable stride (no padding,
  no partial windows), labelling from SigMF annotations, directory names or a
  CSV, and sharded export with a manifest recording the config, label map,
  source files and split assignment. Same seed, same bytes.
- **Recording-level stratified splitting.** Every window of a recording lands in
  one split. When the requested ratios cannot be satisfied that way, `build`
  stops with an error rather than falling back to window-level splitting.
- `--balance-by <sigmf-field>` spreads a nuisance variable across splits while
  keeping class stratification exact, and warns when that is not structurally
  possible.
- `--exclude-label` for annotations that are not classes; `ref_tone` by default.
- `iqforge stats` — class distribution, per-recording carrier offsets and a
  per-split summary.
- `IQForgeDataset` (`torch.utils.data.Dataset`) and `iqforge train`, a baseline
  CNN for checking that a dataset is trainable. `torch` is an optional extra;
  `info`, `inspect`, `build` and `stats` work without it.
- 16 example recordings, so the whole pipeline runs without hardware.

[Unreleased]: https://github.com/emrefbulut/iqforge/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/emrefbulut/iqforge/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/emrefbulut/iqforge/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/emrefbulut/iqforge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/emrefbulut/iqforge/releases/tag/v0.1.0
