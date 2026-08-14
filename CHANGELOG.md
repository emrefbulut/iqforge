# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, interfaces may change between minor releases. The
dataset written by `build` carries its own `manifest_schema` number so a reader
can tell whether the format it is looking at is one it understands.

## [Unreleased]

### Fixed

- **`--labels csv` and `--group-by csv:` identified recordings by file name.**
  In a nested layout the same name repeats under every directory, so a 312-row
  label table collapsed to 47 keys and 310 of 312 recordings came out with one
  label, silently. Both now match the value as written — normally the path
  relative to the input directory — and fall back to the bare name only when it
  is unambiguous, refusing with a message that names the fix when it is not. A
  flat layout is unaffected.
- `iqforge audit` had the same bug against its own manifest, which left the
  class-axis checks silent and made the split lookup compare `None` to `None`.
- `iqforge audit` read `core:datetime` from `global`; SigMF puts it in
  `captures`. The capture-time confound check reported NOT CHECKED on every
  conforming recording.
- `iqforge audit` ranked class axes by raw score, so an axis measurable on one
  class only scored 100% against a chance of 100% and was reported as the reason
  a task sits at the ceiling. Axes are ranked by margin over chance now.
- A folder audit no longer stops at the first unreadable recording; it reports
  how many could not be opened and audits the rest.

### Added

- `iqforge audit` reports **shared air time**: recordings whose capture
  intervals intersect must land in the same split. One transmission heard by
  four receivers is four files and one event, and recording-level splitting
  does not help — the unit of independence is the transmission. This is how
  `--group-by` gets verified rather than assumed.
- `iqforge audit --labels csv` (with `--label-file`), so a dataset labelled from
  a table can be assessed before it is built — which is what auditing a folder
  is for. It applies the same relative-path matching as `build` and reports a
  collapsed table as a proven finding rather than refusing, plus any recordings
  the table does not list.
- `docs/methodology.md` §6 gains a fifth dataset, LoRaIQ, which publishes
  acquisition provenance the other four withhold — and narrows the section's
  claim from "nobody records it" to "there is no standard place to put it".

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

[Unreleased]: https://github.com/emrefbulut/iqforge/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/emrefbulut/iqforge/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/emrefbulut/iqforge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/emrefbulut/iqforge/releases/tag/v0.1.0
