# Publishing iqforge

Pre-release checklist, PyPI setup, GitHub release, and repository metadata.

A PyPI version number is permanent. A bad upload can be yanked, but the number
can never be reused — `0.1.0` would be burned and the fix would have to ship as
`0.1.1`. Everything below exists to keep that from happening.

## Pre-release checklist

Run these from the repository root before tagging:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv build
```

Confirm:

- [ ] `__version__` in `src/iqforge/__init__.py` matches the tag you are about to
      push (`0.1.0` → `v0.1.0`). This is the only place the version is written;
      `pyproject.toml` reads it from there and `tests/test_packaging.py` checks
      they agree.
- [ ] `CITATION.cff`: `version` matches, and `date-released` is filled in with
      the actual release date (it is intentionally absent until then)
- [ ] README status line and roadmap no longer say "not on PyPI yet"
- [ ] CI is green on `main` — the publish workflow re-runs these checks and will
      refuse to upload otherwise, but finding out before you tag is cheaper
- [ ] PyPI Trusted Publisher is configured (below)

## PyPI setup — Trusted Publishing

There is no API token to create or store. PyPI verifies the workflow's OIDC
identity instead, so nothing long-lived can leak from the repository secrets.

Because `iqforge` does not exist on PyPI yet, register a **pending** publisher:

1. Go to https://pypi.org/manage/account/publishing/
2. Under "Add a new pending publisher", fill in:
   - PyPI Project Name: `iqforge`
   - Owner: `emrefbulut`
   - Repository name: `iqforge`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. Save.

The environment name must match `environment: pypi` in
[`.github/workflows/publish.yml`](../.github/workflows/publish.yml). GitHub
creates that environment on first use; you can optionally add a required
reviewer to it under *Settings → Environments* so a publish waits for your
approval.

After the first successful upload the pending publisher becomes a normal one.

## Releasing

The publish workflow triggers on any `v*` tag. It runs lint, the full test
suite, and a check that the tag matches `__version__` before it builds or
uploads anything.

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes-file docs/release-notes/v0.1.0.md
```

Watch the run:

```bash
gh run watch
```

If the verify job fails, delete the tag before retrying — a tag that never
published is not a release:

```bash
git tag -d v0.1.0 && git push origin :refs/tags/v0.1.0
```

### Manual upload

Only needed if the workflow is unavailable. Requires an API token, which
Trusted Publishing otherwise makes unnecessary:

```bash
uv build
uv publish --token pypi-...
```

## Wheel smoke test

```bash
uv build
uv tool install --from dist/iqforge-0.1.0-py3-none-any.whl iqforge

iqforge info examples/bpsk_01.sigmf-meta
iqforge build examples/ -o /tmp/iqforge-smoke --balance-by core:freq_lower_edge
iqforge stats /tmp/iqforge-smoke
```

All commands should exit 0. The wheel does not bundle `examples/`, so run these
from a clone or point them at your own SigMF files. The sdist does bundle them
(~5.7 MB) on purpose: a reviewer who downloads the tarball can run the whole
pipeline without hardware.

## GitHub repository metadata

Set the description and topics:

```bash
gh repo edit emrefbulut/iqforge \
  --description "Turn SDR captures (SigMF) into leak-safe PyTorch datasets"

gh repo edit emrefbulut/iqforge \
  --add-topic python --add-topic pytorch --add-topic sdr --add-topic sigmf \
  --add-topic machine-learning --add-topic signal-processing \
  --add-topic dataset --add-topic rf
```

Verify:

```bash
gh repo view emrefbulut/iqforge --json description,repositoryTopics
```

Upload `docs/banner.png` under *Settings → Social preview*. That image is what
renders when the repository link is shared on LinkedIn, Slack or X; without it
GitHub falls back to a generic card. It must be a PNG — the SVG is rejected.
