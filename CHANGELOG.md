# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, interfaces may change between minor releases. The
dataset written by `build` carries its own `manifest_schema` number so a reader
can tell whether the format it is looking at is one it understands.

## [Unreleased]

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

[Unreleased]: https://github.com/emrefbulut/iqforge/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/emrefbulut/iqforge/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/emrefbulut/iqforge/releases/tag/v0.1.0
