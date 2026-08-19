# Local + GitHub Handoff Status (Phase 1 Independent Stream)

Date: 2026-08-19  
Branch: `cursor/phase-1-independent-gaps-38eb`

## Goal

Make the branch easy to hand off as a **locally readable** and **GitHub-readable** workstream without blocking ongoing migration implementation.

## Current state snapshot

- Branch exists and is ahead of remote.
- Migration implementation is in progress in code files and should continue independently.
- Prep checklist exists at `docs/phase5-phase6-checklist.md`.
- This stream adds only additive handoff/readability docs and no migration logic.

## Remaining gaps vs handoff goal

### Local handoff readability

- [x] There is a concrete migration-prep checklist (`docs/phase5-phase6-checklist.md`).
- [x] There is now a branch-level handoff status note (this file).
- [ ] Final migration outcome summary is still pending (must be authored after implementation stabilizes).
- [ ] Explicit validation evidence for final migration behavior is pending (must come from migration worker outputs).

### GitHub / PR readability

- [x] README/SPEC/Methodology already explain current shipped behavior and boundaries.
- [x] PR readability template/checklist is added in `.github/pull_request_template.md`.
- [ ] Final PR body must include concrete before/after behavior excerpts once migration is complete.
- [ ] Final PR should link artifacts proving outcome (logs/tables) produced by migration worker.

### Clean local state

- [ ] Working tree is not clean (expected while migration worker is active).
- [ ] Post-migration branch should be restaged and rechecked before push/PR creation.

## Coordination notes for the main migration worker

1. Keep ownership of implementation files:
   - `src/iqforge/cli.py`
   - `scripts/leakage_experiment.py`
   - `scripts/leakage_loraiq.py`
   - `scripts/leakage_real.py`
2. Use this stream's docs as handoff framing only; do not treat them as migration completion proof.
3. Before opening/updating PR, fill the template checklist with:
   - exact command/output deltas
   - refusal/force-path behavior confirmation
   - any docs text synchronized after final behavior lands

## Suggested finalization checklist (after migration lands)

- [ ] Re-run targeted tests/commands for the migrated path.
- [ ] Update `CHANGELOG.md` `[Unreleased]` with final behavior wording.
- [ ] Refresh docs sections that are migration-dependent (`README.md`, `SPEC.md` 4/5.10, `docs/methodology.md` reproducing block as needed).
- [ ] Confirm clean `git status` (except intentional tracked artifacts).
- [ ] Prepare PR description using `.github/pull_request_template.md`.
