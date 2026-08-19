# Claude Code Transition Handoff

Date: 2026-08-19  
Branch: `cursor/phase-1-independent-gaps-38eb`

This document is the operator-facing handoff for continuing leakage-measurement work with Claude Code on this branch. It is command-oriented and intentionally does not duplicate checklist details already tracked in `docs/phase5-phase6-checklist.md`.

## 1) Completed phases and commit refs

- **Phase 1 baseline (refuse-only command):** `e56f171`  
  Added `iqforge measure-leakage` as a refuse path that runs `audit`, classifies, and stops.
- **Phase 2 measurement core extraction:** `febbf0e`  
  Pulled paired leakage measurement into shared library path (`src/iqforge/measurement.py`).
- **Phase 4 execution enablement:** `40b55e8`  
  Enabled Phase 4 measurement execution and constrained sweep modes.
- **Phase 5/6 prep artifact (planning-only):** `5bebd0f`  
  Added `docs/phase5-phase6-checklist.md` for pending migration/doc sync tasks.

Quick check:

```bash
git log --oneline --decorate -n 12
git show --stat 40b55e8
git show --stat 5bebd0f
```

## 2) Key decisions to preserve

- **No `--sweep snr` in `iqforge measure-leakage`.**  
  This is deliberate and documented in `SPEC.md` §5.10 and `CHANGELOG.md` `[Unreleased]`.
- **Reason:** adding SNR/noise requires writing altered recordings and dataset-specific preparation choices, which breaks command read-only guarantees and can fail silently.
- **Contract:** `iqforge measure-leakage` remains read-only against user recordings, runs `audit`, classifies/estimates, and (in this branch state) can run paired measurement via the current command path where applicable.
- **LoRaIQ-specific handling:** simultaneous receptions must be grouped (`--group-by`), and LoRaIQ-like cases held together by grouping are not treated as a structural leak refusal.

Quick check:

```bash
rg --line-number -- "--sweep snr|read-only|measure-leakage" SPEC.md CHANGELOG.md README.md
```

## 3) Run/verify paths (LoRaIQ and non-LoRaIQ)

### A. CLI sanity and audit/refusal path

```bash
uv run iqforge measure-leakage <recordings_or_dataset> --format json
uv run iqforge measure-leakage <recordings_or_dataset> --format json --force
```

What to confirm:

- JSON output includes audit context and measurement/refusal fields.
- `--force` keeps override reason visible in header/output metadata.

### B. Non-LoRaIQ path (synthetic + DASH7 real)

Synthetic experiment (fast smoke):

```bash
uv run python scripts/leakage_experiment.py --quick
```

Synthetic fuller run:

```bash
uv run python scripts/leakage_experiment.py
```

DASH7 path (real capture script entrypoints):

```bash
uv run python scripts/leakage_real.py --pilot
uv run python scripts/leakage_real.py --sweep stride
```

### C. LoRaIQ path

Pre-check/audit-only:

```bash
uv run python scripts/leakage_loraiq.py --check
```

Run sweep:

```bash
uv run python scripts/leakage_loraiq.py
```

What to confirm:

- Grouped transmission behavior is active in LoRaIQ path (`--group-by`/group CSV flow).
- Result artifacts update under `artifacts/` without changing source recordings in place.

## 4) Known caveats

- Local tree currently includes untracked working files from active migration/testing work; keep docs-only commits scoped.
- Several workflow scripts default to local scratch/dataset paths and may require `--source`/input overrides on another machine.
- LoRaIQ full reproduction depends on local dataset/index CSV availability and is intentionally not CI-portable.
- `docs/phase5-phase6-checklist.md` is prep-only and not proof of migration completion.

## 5) Next actions (operator order)

1. Keep implementation ownership in active files (`src/iqforge/cli.py`, `scripts/leakage_experiment.py`, `scripts/leakage_loraiq.py`, `scripts/leakage_real.py`).
2. After migration behavior stabilizes, update behavior-dependent docs in one pass: `README.md`, `SPEC.md` (§4, §5.10), `docs/methodology.md`, `CHANGELOG.md`.
3. Re-run targeted command checks above and capture concrete output deltas for PR body.
4. Use `.github/pull_request_template.md` when preparing PR narrative and verification notes.
5. Keep this handoff doc as the branch entrypoint; keep detailed prep checklist in `docs/phase5-phase6-checklist.md`.
