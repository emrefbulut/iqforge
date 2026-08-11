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
- [x] **`--group-by`: keep related recordings together.** Shipped with two
      schemes, `path:<regex>` and `csv:<file>`. Recordings sharing a key become
      one indivisible unit; the unit replaces the recording as the thing being
      allocated, and no unit can span two splits.

      Motivated by real data rather than by reading the code. In DASH7
      `ds_indoor` two recordings at the same location and channel, 43 seconds
      apart, are separate recorder runs by every structural test and the same
      channel realisation physically. In AirID the transmissions inside one
      burst are slices of one continuous capture. See
      [docs/methodology.md](docs/methodology.md) §6.

- [ ] **`--group-by` by SigMF field.** Deliberately not in the first release.

      The obvious third scheme would read a metadata key, the way
      `--balance-by` does. It was left out because it would have solved none of
      the three datasets that motivated the feature: DASH7 keeps the location
      in a directory and the channel in the file name while every file declares
      the same centre frequency, AirID encodes the burst in the file name, and
      the Vega-C recordings carry their session as a timestamp. Shipping a
      scheme that answers no known case is surface area without users.

      Worth adding when a dataset turns up that does record its acquisition in
      metadata — a session UUID, `core:hw` for the receiver. The resolution
      logic already exists in `annotation_field_value`, so it is a small
      addition once there is a reason.
- [ ] Surface leakage / balance diagnostics in `stats` (or a thin `audit` wrapper
      around the existing script)

      **Do not wrap the twinning check as it stands.** `scripts/audit_leakage.py`
      looks for leaked windows by cosine similarity between flattened test and
      train windows, and that instrument cannot see the leak it was written for.
      Flattening lays a window out by position; two windows offset by half a
      stride hold the same samples at *different* positions, so the dot product
      compares sample *k* of one against sample *k+512* of the other and scores
      them no higher than two unrelated windows. The check finds duplicates
      (offset exactly 0) and nothing else, while every real case is a partial
      overlap. Found when it returned 0.091 for both arms of a real-data probe
      and was briefly read as evidence of no overlap.

      The right test needs no threshold and no content at all: carry
      `(record_id, start_sample)` per window, and report any train/test pair
      from the same recording whose `[start, start + window)` intervals
      intersect. Exact, O(n log n) by sorting, and it answers the question that
      was actually being asked. Content similarity answers a different one —
      *are these two windows alike* — which is not the same as *do these two
      windows share samples*, and only the second one is leakage.
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
