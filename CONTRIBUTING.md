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
`TrainingResult.environment` records the device, torch version and CUDA version
so a results file can be checked later. The sweep scripts in `scripts/` refuse
to extend a checkpoint that was measured on a different device.

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
