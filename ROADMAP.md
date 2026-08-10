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

- [x] **Published `0.1.0`** to PyPI, tagged `v0.1.0`. A chore rather than a
      level-up, as billed: it removes install friction and proves nothing about
      quality.
- [x] **Real SigMF verification with public captures.** Three recordings from
      the GNU Radio collection. `ci8` and `ci16_le` checked against the raw
      bytes (exact ÷128 and ÷32768, correct I/Q order); annotated bands measured
      31.8 dB above unannotated ones. Limits found on real data are in the
      README.

Do next, in this order:

1. **Repeat the leakage measurement on a real recording.** The current number is
   synthetic BPSK/QPSK; the same curve on a real capture is what turns it from
   an illustration into a result worth publishing.
2. **Verification with an own hardware capture.** One device end to end. Public
   files validated the reader; they cannot validate against the conventions of a
   radio nobody here has run.

---

## Next

After Now is done — still reliability-first:

- [ ] Hardware capture on **one** device (RTL-SDR / HackRF / Pluto) — parallel to
      public-file work, not a blocker for it. One device proves one path, not all
      integer/I/Q conventions.
- [ ] Frequency-aware labeling (SPEC §5.3 deferred item)
- [ ] **`--group-by`: keep related recordings together.** Not implemented, and
      not the same thing as `--balance-by`.

      The manifest records which recording went to which split, so the split is
      auditable. What it cannot express is a dependency *between* recordings —
      that two of them came from one acquisition and must not be separated.
      `--balance-by` does the opposite by design: it spreads a nuisance variable
      **across** splits. There is no way to say "these two are twins, keep them
      on the same side".

      Found by trying to use real data, not by reading the code. In DASH7
      `ds_indoor` two recordings at the same location, same channel, 43 seconds
      apart are separate recorder runs by every structural test and the same
      channel realisation physically; splitting them apart leaks. In AirID the
      ~140 transmissions inside one burst are slices of one continuous capture.
      Both are cases where the honest unit of independence is coarser than the
      file, and the tool currently has no vocabulary for it. See
      [docs/methodology.md](docs/methodology.md) §6.

      Shape it would take: a key read the same way `--balance-by` reads one —
      any SigMF field, or a path component — with the splitter treating each
      group as indivisible. Note the interaction: grouping reduces the number of
      independent units, so the SPEC §5.6 error for "not enough recordings per
      class" gets easier to hit, which is correct behaviour and should be said
      out loud rather than worked around.
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
