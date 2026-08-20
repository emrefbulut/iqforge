# Contributing

Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/emrefbulut/iqforge
cd iqforge
uv sync --group dev
```

`torch` is optional — `info`, `inspect`, `build`, `stats`, and `audit` run
without it. To work on `IQForgeDataset` or `train`:

```bash
uv sync --group dev --extra torch
```

## Tests and lint

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests that need `torch` skip themselves when it isn't installed, so a run
without the extra should still be fully green.

Every module has a matching test file. Tests use synthetic data and never touch
the network.

## Contribution flow

`main` is protected: it takes no direct pushes, and a pull request merges only
once the five CI jobs (`lint`, `test (3.11)`, `test (3.12)`, `test-torch`,
`build`) are green. Approving review is deliberately not required — this is a
one-maintainer project, and a required review would leave nothing mergeable.

```bash
git switch -c fix/short-name
uv run pytest && uv run ruff check . && uv run ruff format --check .
git commit -am "Say what the change does, not which files it moved"
git push -u origin fix/short-name
gh pr create
```

One branch per logical change, and one pull request per branch. A pull request
that fixes three unrelated things cannot be reverted for one of them.

## Training on a GPU

`iqforge train` runs on the **CPU** by default, and that default is deliberate
rather than an oversight. The README promises the same seed gives the same
bytes; cuDNN selects kernels by heuristic, so the same seed on the same GPU can
pick a different reduction order between runs. The paired experiments in
[docs/methodology.md](docs/methodology.md) depend on it even more strongly —
their whole design is "everything identical except the split assignment", which
stops being true the moment two rows land on different devices.

The `[torch]` extra installs a CPU-only wheel and **stays that way**. To use a
GPU, install a CUDA build yourself first, then the package without the extra:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv sync --group dev
```

Then:

```bash
uv run iqforge train dataset/ --device cuda
```

`--device auto` picks CUDA when it is available and CPU otherwise; `--device
cuda` errors rather than falling back, because a run that silently used a
different device than it was asked for is worse than one that stops. A CUDA run
prints a warning that its numbers are not bit-comparable with CPU runs, and
`TrainingResult.environment` records the device, torch version, CUDA version,
and the numpy / scipy / sigmf versions windowing, normalisation and reading
depend on, so a results file can be checked later. The sweep scripts in `scripts/`
refuse to extend a checkpoint that was measured on a different environment.

## Engineering conventions

These are the rules this project already runs on. They are written down here
because every one of them was learned by breaking it first, and because a
contributor cannot follow a rule that only exists in a reviewer's head.

**1. Never fall back silently.** If a guarantee cannot be met, stop and say so.
A plausible wrong answer is worse than a refusal, because a refusal gets fixed
and a wrong number gets published. `build` refuses to split when the requested
ratios cannot be satisfied at the recording level rather than dealing windows
out individually (SPEC §5.6); `load()` names an unsupported `core:datatype`
instead of guessing a width; `read_manifest` refuses a `manifest_schema` newer
than it understands, because "this field is missing" and "this field is missing
and it mattered" are indistinguishable from inside. When you add a code path
that cannot deliver what its caller expects, make it raise.

**2. A passing test does not prove it can fail.** After writing a test, break
the code it covers on purpose and confirm the test goes red. Do not commit the
mutation — the point is the check, not the change. This project has three
tests that only became real after that step: the spectrogram test that pins
`+100 kHz` power above `-100 kHz` (it passed against an I/Q swap until the
ratio assertion replaced a presence assertion), the CLI error path (rich reads
bracketed text as a style tag and deletes it, so an unescaped message silently
lost content), and the audit report's width test, which compared line lengths
against the `WIDTH` constant itself — raising `WIDTH` to 200 kept it green.
That one is now pinned to the literal `78`.

**3. Do not claim what you did not measure.** Any number this project produces
carries the conditions it was produced under: device, library versions
(`torch`, `numpy`, `scipy`, `sigmf` — windowing and normalisation are numpy
arithmetic and the spectrogram is scipy, so either can move a result without a
line of this repository changing), the seed count, and an interval rather than
a bare point estimate. A dependency floor is a claim too: `sigmf>=1.11.1` is
the oldest release actually exercised, not the oldest that might work.

**4. Protect published artifacts.** Files under `artifacts/` are quoted by
`docs/methodology.md` and `README.md`. If a change regenerates one, the sample
size and run count must not fall. **Value equality is not a sufficient
acceptance criterion**: a grid reduced from 15 seed pairs to 1 reproduces the
first pair exactly and is a different measurement. An acceptance check has to
compare the shape of the result, not only its first row.

**5. Do not enlarge a sample after seeing the result.** The seed count is fixed
before the run and written into the report. When a measurement comes back
inconclusive, that is the finding; buying significance by continuing to sample
lets the result choose the stopping rule. The intermediate rows of the LoRaIQ
stride sweep were left at n = 15 for this reason, and the tables say so.

**6. Report your own mistakes.** If you found and fixed an error along the way,
say so in the output rather than quietly shipping the corrected version. The
value of `docs/methodology.md` §7 is entirely in the failures it lists, and
each of those entries exists because someone wrote down a mistake they could
have deleted instead.

**7. Ask before changing scope.** A proposal is presented with its reasoning
before it is implemented, not alongside the implementation. This applies most
strongly when the change would relax a guarantee — those are the changes least
likely to be noticed in review and most likely to matter.

**8. Nothing session- or machine-specific is committed.** No absolute paths, no
temporary directories, no local configuration. A script that reproduces a
published number has to be runnable by someone who is not you; point it at data
through an environment variable or a flag and skip with a stated reason when the
data is absent.

## The most useful bug report

**If `iqforge` misreads a recording from your hardware, that is the single most
valuable issue you can file.**

The `cf32_le` path is exercised end to end by the example recordings, and the
integer paths (`ci16_le`, `ci8`) have been checked against the raw bytes of
public captures — exact ÷32768 and ÷128, correct I/Q order
([methodology §5](docs/methodology.md)). That covers three recordings from two
publishers, both of which happened to agree with the reading here.

What it does not cover is your radio. Signed vs. unsigned interpretation, full
scale (`2^(n-1)` vs `2^(n-1) - 1`) and hardware-specific I/Q ordering are all
places a vendor can differ, and a file that disagrees with `iqforge` loads
without complaint and looks like noise.

Please attach the `.sigmf-meta` file if you can share it — metadata alone is
often enough to reproduce the problem, and it contains no signal data. The
[bug report template](.github/ISSUE_TEMPLATE/bug_report.yml) asks for it along
with your `iqforge info` output.

## Scope

`iqforge` turns existing recordings into datasets. Live SDR capture, signal
transmission, demodulation, and synthetic signal generation are out of scope —
see [SPEC.md](SPEC.md) §2. If you want to propose something that crosses that
line, open an issue first so we can talk about it before you write code.

One rule is not negotiable: windows from the same recording must never appear
in more than one split, and the tool must error rather than silently fall back
to window-level splitting. See [SPEC.md](SPEC.md) §5.6.
