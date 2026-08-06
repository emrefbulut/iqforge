# Roadmap

Identity stays fixed: **real SigMF → leak-safe, reproducible PyTorch dataset.**

This file uses **Now / Next / Later**, not calendar months. Ship on a calendar
when something is ready; do not wait for hardware or a big feature to cut a
release.

---

## Now

Done recently (keep green):

- [x] Encoding-safe CLI on non-UTF-8 locales (redirect / pipe)
- [x] Single-sourced version (`__version__` → `pyproject.toml`)
- [x] Publish workflow gated behind tests + Trusted Publishing docs
- [x] Honest status: no fake `v0.1.0` claim until tagged
- [x] Contributor history cleaned; tool-local ignore kept out of the repo
- [x] **Leakage measurement.** `scripts/leakage_experiment.py`, 180 runs, in the
      README. Window-level splitting inflates reported accuracy by up to
      **+13.6 pp**; the effect vanishes above +3 dB, where both splits sit at the
      ceiling, and peaks where the signal is marginal. The wrong split lives in
      the script, never in the CLI.

Do next, in this order:

1. **Publish `0.1.0`.** Chore, not a level-up — removes install friction.
   Follow [docs/publishing.md](docs/publishing.md) (pending Trusted Publisher on
   PyPI, then tag `v0.1.0`).
2. **Run the pipeline on a public real SigMF recording** (SigMF examples /
   IQEngine-hosted captures). Validates `ci16_le` / `ci8` without buying
   hardware. Document what worked and what did not.
3. **Repeat the leakage measurement on that real recording.** The current number
   is synthetic BPSK/QPSK; the same curve on a real capture is what turns it
   from an illustration into a result worth publishing.

---

## Next

After Now is done — still reliability-first:

- [ ] Hardware capture on **one** device (RTL-SDR / HackRF / Pluto) — parallel to
      public-file work, not a blocker for it. One device proves one path, not all
      integer/I/Q conventions.
- [ ] Frequency-aware labeling (SPEC §5.3 deferred item)
- [ ] Surface leakage / balance diagnostics in `stats` (or a thin `audit` wrapper
      around the existing script)
- [ ] Find **~3 people who work with real RF data** and watch them use iqforge.
      Missing users is a product gap; more features will not close it.
- [ ] Docs site (CLI + Python API reference) when the surface stops thrashing

Versioning: cut `0.1.x` / `0.2.0` when useful, on a schedule if needed — not
“only when hardware is done.”

---

## Later

Do not start these until Next has external signal or real-data proof:

- Live capture into SigMF (one radio first)
- Kitty / iTerm graphics for `inspect`
- JOSS short paper (`paper.md` + docs/tests) — software review, after real-data
  verification
- IEEE-style **result** paper — only if the leakage experiment is the result;
  that experiment belongs in Now, not here
- Broader hardware matrix (more datatypes / vendors)

### Out of scope (scope traps)

| Temptation | Why not |
|---|---|
| Web UI | Dilutes the CLI identity |
| Own file format | Stay on SigMF |
| Hyperparameter search / training framework | TorchSig / Lightning territory |
| Multi-radio capture at once | Maintenance cost before users |

---

## Research note

The only research direction that is uniquely iqforge’s is **A**: measure how
window-level splits inflate accuracy vs recording-level splits. Options that
compete with TorchSig, IQEngine, or turn iqforge into a training framework are
identity loss.

JOSS reviews the **software**. A conference paper needs a **result**. Same
experiment can feed both; they are not the same deliverable.

---

## Definition of done for “higher level”

| Signal | Meaning |
|---|---|
| On PyPI | Installable (chore) |
| Public + hardware SigMF green | Domain-credible |
| Leakage number in README | Claim is evidenced — **done**, synthetic; real capture still pending |
| 3 external users / issues from real captures | Not building in a vacuum |
| Stable API + JOSS or cited experiment | Research-adjacent tooling |
