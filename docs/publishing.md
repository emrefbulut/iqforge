# Publishing iqforge

This document covers the pre-release checklist, GitHub release notes, PyPI upload,
and repository metadata for `v0.1.0`.

## Pre-release checklist

Run these from the repository root before tagging or publishing:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv build
```

Optional but recommended:

```bash
# Wheel smoke test (clean install)
uv tool install --from dist/iqforge-0.1.0-py3-none-any.whl iqforge
iqforge info examples/bpsk_01.sigmf-meta
```

Confirm:

- [ ] Version in `pyproject.toml` matches the tag (`0.1.0`)
- [ ] `CITATION.cff` version matches; update `date-released` to the actual release date
- [ ] README status and roadmap reflect PyPI state
- [ ] CI is green on `main`
- [ ] `PYPI_API_TOKEN` secret is set in GitHub (for automated publish)

## GitHub Release notes (v0.1.0)

Use this template when creating the release on GitHub:

```markdown
## iqforge v0.1.0 — first alpha

Turn SDR captures (SigMF) into leak-safe PyTorch datasets.

### Highlights

- SigMF reading (`cf32_le`, `ci16_le`, `ci8`) with memory-mapped large files
- Terminal spectrogram inspector
- Windowing, labelling (annotations / dirname / CSV), recording-level stratified splits
- Sharded dataset export with manifest
- `IQForgeDataset` and baseline CNN training command
- 16 example recordings for hardware-free testing

### Install

From source (PyPI not yet published):

```bash
git clone https://github.com/emrefbulut/iqforge
cd iqforge
uv sync --extra torch
uv run iqforge info examples/bpsk_01.sigmf-meta
```

### Requirements

- Python 3.11+
- `torch` is optional; required only for `IQForgeDataset` and `iqforge train`

### Full changelog

See commit history since initial release.
```

Create the release:

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes-file release-notes.md
```

## PyPI upload

### Manual upload

```bash
uv build
uv run twine upload dist/*
```

You need a PyPI account and API token. Create one at https://pypi.org/manage/account/token/
with scope limited to the `iqforge` project.

### Automated upload

The [publish workflow](../.github/workflows/publish.yml) runs on `workflow_dispatch`
or when a `v*` tag is pushed. It uses the `PYPI_API_TOKEN` repository secret.

After the first successful publish, update README.md to remove the "not published yet"
note and check off the PyPI item in the roadmap.

## Wheel smoke test

After building or installing from PyPI:

```bash
uv build
uv tool install --from dist/iqforge-0.1.0-py3-none-any.whl iqforge

iqforge info examples/bpsk_01.sigmf-meta
iqforge build examples/ -o /tmp/iqforge-smoke --balance-by core:freq_lower_edge
iqforge stats /tmp/iqforge-smoke
```

All commands should exit 0. The wheel does not bundle `examples/`; run smoke tests
from a clone or pass an absolute path to your own SigMF files.

## GitHub repository metadata

Set the description and topics:

```bash
gh repo edit emrefbulut/iqforge \
  --description "Turn SDR captures (SigMF) into leak-safe PyTorch datasets"

gh repo edit emrefbulut/iqforge \
  --add-topic python \
  --add-topic pytorch \
  --add-topic sdr \
  --add-topic sigmf \
  --add-topic machine-learning \
  --add-topic signal-processing \
  --add-topic dataset \
  --add-topic rf
```

Verify:

```bash
gh repo view emrefbulut/iqforge --json description,repositoryTopics
```
