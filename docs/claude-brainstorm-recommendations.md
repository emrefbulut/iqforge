# Claude Brainstorm Recommendations

## Current State Summary

- PR #1 is merged into `main` and the local `main` branch is synced to `origin/main`.
- The project now has a strong leakage-safety core: recording-level split guarantees, explicit refuse-path preflight (`measure-leakage`), and high test depth around audit/preflight/measurement logic.
- Runtime verification is mixed: `pytest` passes fully on current `main` (`302 passed`), but CI/lint currently fails due style/import issues in recent scripts and one undefined name.
- Documentation and scientific framing are a project strength, but there is growing complexity across `cli.py`, audit/preflight/measurement modules, and long-lived methodology scripts.

## Prioritized Gaps

### P0 (Immediate)

1. **CI reliability is broken for merged main**
   - `ruff` currently fails on `scripts/_phase5_sample_checks.py`, `scripts/leakage_loraiq.py`, `scripts/leakage_real.py`, and import order in `src/iqforge/cli.py`.
   - Result: PR checks can be red while tests are green, reducing trust in branch protection.

2. **CLI module size/complexity risk**
   - `src/iqforge/cli.py` now combines many concerns (build, audit, measure orchestration, rendering, validation).
   - This raises regression risk when changing UX or adding new commands.

3. **Measurement path reproducibility coupling**
   - Parts of real-data workflows still rely on machine-local paths and environment assumptions in tests/scripts.
   - This can cause reproducibility drift between local runs and CI.

### P1 (Near-Term)

1. **Performance/scalability observability gaps**
   - Large dataset runs have progress output but limited structured performance telemetry (I/O throughput, memory peak, shard write rates, per-stage timing).
   - Hard to reason about scaling behavior before users hit bottlenecks.

2. **CLI UX for decision/action loops**
   - `audit` and `measure-leakage` are principled but dense; users may need clearer "do this next" command scaffolding for each refusal category.
   - JSON schemas for automation-friendly outputs could be formalized/versioned.

3. **Methodology-to-code contract hardening**
   - Many scientific constraints are encoded in prose + tests; fewer machine-checked invariants at CLI boundary for every mode.
   - Opportunity for stricter contract tests per category and per sweep mode.

### P2 (Strategic)

1. **Architecture extraction**
   - Split command orchestration from domain kernels (audit/preflight/measurement) into smaller service modules.
   - Enables safer feature additions and easier third-party integration.

2. **Extended RF realism evaluation**
   - Current logic handles major known leakage patterns well, but multi-emitter, dense-spectrum, and hardware-domain shift benchmarking can be expanded.

3. **Plugin/extension surface**
   - A formal extension point for custom leakage checks and domain-specific groupings could widen adoption without destabilizing core behavior.

## Concrete Proposal List

1. **Stabilize main branch checks**
   - Fix current lint/import errors and ensure `gh` branch protection gates on passing CI.
   - Add a small "CI smoke" job to run `ruff + pytest -q` before merge decisions.

2. **Refactor CLI incrementally**
   - Extract command handlers into focused modules (`commands/build.py`, `commands/audit.py`, `commands/measure.py`) while preserving external CLI contract.
   - Keep shared formatting helpers in one place; isolate policy decisions from rendering.

3. **Add performance instrumentation**
   - Emit optional JSON trace (`--profile-json`) for build/measure stages: elapsed seconds, windows processed, write MB/s, and memory snapshot.
   - Add one benchmark fixture in CI-nightly (not PR) to detect regressions.

4. **Tighten methodology invariants in tests**
   - Add explicit contract tests for:
     - refusal category precedence;
     - `--force` output guarantees;
     - sweep mode constraints;
     - deterministic dataset identity for fixed seed/config.

5. **Improve actionability of preflight output**
   - For each refusal category, print one minimal unblock checklist and one recommended command template.
   - Keep ASCII report contract unchanged; enrich JSON with structured remediation fields.

## Suggested Next-Phase Roadmap

### Phase A — Mainline Reliability
- Repair lint/import failures currently on `main`.
- Re-run full CI and enforce branch protection to block red merges.
- Ship patch release focused on reliability only.

### Phase B — CLI/Internal Architecture
- Introduce handler-level module split with no behavior changes.
- Add regression harness that snapshots representative CLI outputs.
- Prepare internal API boundaries for future SDK/automation usage.

### Phase C — Measurement Ops and Scale
- Add optional profiling output and benchmark fixtures.
- Validate performance over larger synthetic + real workloads.
- Document tuning guidance for window/stride/grouping at scale.

### Phase D — Methodology Expansion
- Expand evaluation matrix for dense-spectrum and multi-emitter captures.
- Add stronger domain-shift checks and publish acceptance criteria updates.

## Gemini Master Prompt Adaptation for Claude

The provided Gemini master prompt should be used as a **structured critic and planner**, not as an unconditional executor. Claude should challenge assumptions against this repository's hard constraints:

- leakage-safety over convenience,
- no silent fallback behavior,
- explicit "not checked" reporting,
- reproducible paired comparisons.

### How Claude Should Use It Here

1. **Preserve invariant hierarchy**
   - Never recommend shortcuts that weaken recording-level disjointness guarantees or refusal semantics.

2. **Translate broad ideas into testable deltas**
   - Every recommendation should include:
     - affected module(s),
     - expected user-visible behavior,
     - test impact,
     - rollback strategy.

3. **Cross-check against methodology and CI reality**
   - Validate proposals against `docs/methodology.md`, existing refusal categories, and current CI constraints.

4. **Prefer minimal-risk sequencing**
   - Reliability and reproducibility fixes before feature expansion.

### Suggested Prompt Wrapper (Use with Gemini Prompt Content)

Use the user-provided Gemini master prompt as core context, then prepend:

```text
Project context override:
- Repo: iqforge
- Primary goal: leakage-safe SDR-to-PyTorch dataset pipeline
- Non-negotiables: no silent fallback, explicit not-checked reporting, reproducible split/measurement behavior
- Current priority: stabilize mainline CI + preserve methodology correctness while proposing next-phase improvements

When proposing changes:
1) classify as reliability / architecture / performance / methodology,
2) assign priority P0/P1/P2,
3) include concrete file-level change plan and tests,
4) state regression risks explicitly,
5) reject any idea that weakens leakage safeguards.
```

### Note on Prompt Embedding

The exact Gemini master prompt text was not found in-repo. Paste the exact user-provided prompt under this section when handing off to Claude so the wrapper above can be applied verbatim.
