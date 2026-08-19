# Phase 5/6 Prep Checklist (No Migration Execution)

Purpose: prep-only planning artifact for upcoming Phase 5 migration work and Phase 6 documentation updates. This file does **not** execute migration and does not change measurement logic or tests.

Date checked: 2026-08-19  
Branch checked: `cursor/phase-1-independent-gaps-38eb`

---

## Scope guardrails

- No migration execution in this prep step.
- No changes to `src/iqforge/measurement.py` behavior.
- No changes to LoRaIQ tests or test logic.

---

## A) Phase 5 migration checklist (code/doc touchpoints only)

### A1. CLI behavior and help text

- [ ] **`src/iqforge/cli.py` (`measure-leakage` command block)**  
  Must be aligned to final Phase 5 behavior and messaging:
  - command docstring/help must describe shipped behavior precisely
  - output/exit-code semantics must match intended migration outcome
  - flags and defaults must remain internally consistent (`--sweep`, `--force`, `--format`, etc.)

  **Current readiness:** **NOT READY** (migration still pending as planned).  
  **Evidence:** `measure-leakage` command exists and still carries transition-sensitive wording; this is the main migration surface.

### A2. Refuse-path vs measurement split of responsibilities

- [ ] **`src/iqforge/preflight.py` module contract and messages**  
  Must reflect the post-migration boundary between classification/refusal and measurement execution.

  **Current readiness:** **NOT READY** (needs final migration alignment).  
  **Evidence:** module-level text still states "This module does not train..." and positions itself as refuse-path logic.

- [ ] **`src/iqforge/measurement.py` API + invariants review**  
  Must remain the paired-measurement engine, with no regressions in paired setup assumptions.

  **Current readiness:** **READY FOR MIGRATION INPUT** (implementation present; no prep edits needed now).  
  **Evidence:** file already contains paired measurement core (`BuildSpec`, recording-level build, window-level re-deal, paired summaries).

### A3. Audit/measurement contract consistency

- [ ] **`src/iqforge/audit.py` handoff wording to measurement path**  
  Ensure audit "next step" guidance remains accurate after migration.

  **Current readiness:** **PARTIALLY READY**.  
  **Evidence:** audit already references `measure-leakage` handoff, but final wording will need one migration pass review.

---

## B) Phase 6 docs checklist (required files/sections)

### B1. README updates

- [ ] **`README.md`**
  - installation text and quickstart alignment with migrated behavior
  - example output snippets aligned with final command behavior
  - any wording that still says "This version does not train" updated if behavior changes

  **Current readiness:** **NOT READY** (will need Phase 6 doc refresh).  
  **Evidence:** README currently contains measure-leakage wording that references non-training behavior.

### B2. SPEC updates (required sections)

- [ ] **`SPEC.md` section 4 ("Command interface")**
  - `iqforge measure-leakage` command block text and semantics
  - flags/help and outcome descriptions

  **Current readiness:** **NOT READY** (expected to change with migration).

- [ ] **`SPEC.md` section 5.10 ("Leakage measurement")**
  - narrative around refuse categories, command behavior, and constraints
  - keep explicit rationale for read-only guarantees where applicable

  **Current readiness:** **NOT READY** (expected to change with migration).

### B3. Methodology reproducing lines

- [ ] **`docs/methodology.md` (`## Reproducing` section)**
  - reproduction commands and text must match post-migration command behavior
  - ensure no stale reproduction path remains

  **Current readiness:** **NOT READY** (migration-dependent wording may need update).

### B4. Changelog

- [ ] **`CHANGELOG.md` (`## [Unreleased]`)**
  - add concise entries for Phase 5 migration behavior changes
  - add Phase 6 doc synchronization notes

  **Current readiness:** **READY** (target section exists and is active).  
  **Evidence:** `## [Unreleased]` section is present.

### B5. Release notes

- [ ] **`docs/release-notes/<next-version>.md` (new file)**
  - summarize migration outcome and doc updates
  - include user-facing upgrade notes, if any

  **Current readiness:** **NOT READY** (next release note file not created yet).  
  **Evidence:** existing notes are up to `docs/release-notes/v0.4.0.md`; no next-version draft file yet.

---

## C) Repository readiness summary

- Phase 5 migration touchpoints are identified and isolated; no migration executed in this prep step.
- Phase 6 required docs surfaces are identified with explicit section-level targets.
- Most required updates are intentionally pending (not-ready) because they depend on the upcoming migration outcome.
- `CHANGELOG.md` is already structurally ready (`[Unreleased]` exists) and can receive entries when migration lands.

---

## D) Quick validation for this prep artifact

- This artifact is documentation-only (`docs/`).
- No source, measurement logic, or test files are modified by this prep artifact.
- Therefore this prep change introduces no new test requirements by itself.
