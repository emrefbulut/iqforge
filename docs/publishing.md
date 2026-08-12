# Publishing iqforge

Pre-release checklist, PyPI setup, GitHub release, and repository metadata.

A PyPI version number is permanent. A bad upload can be yanked, but the number
can never be reused — `0.2.0` would be burned and the fix would have to ship as
`0.2.1`. Everything below exists to keep that from happening.

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
      push (`0.2.0` → `v0.2.0`). This is the only place the version is written;
      `pyproject.toml` reads it from there and `tests/test_packaging.py` checks
      they agree.
- [ ] `CITATION.cff`: `version` matches, and `date-released` is filled in with
      the actual release date (it is intentionally absent until then)
- [ ] `CHANGELOG.md` has a dated section for this version and an empty
      `[Unreleased]` above it
- [ ] `docs/release-notes/v<version>.md` exists — `gh release create` reads it
- [ ] README status line names the version being released
- [ ] CI is green on `main` — the publish workflow re-runs these checks and will
      refuse to upload otherwise, but finding out before you tag is cheaper
- [ ] PyPI Trusted Publisher is configured (below)

## PyPI setup — Trusted Publishing

There is no API token to create or store. PyPI verifies the workflow's OIDC
identity instead, so nothing long-lived can leak from the repository secrets.

This was configured for the `0.1.0` release and stays in place; there is nothing
to redo per release. To check or change it, go to the project's *Publishing*
settings on PyPI. The publisher is:

- Owner: `emrefbulut`
- Repository name: `iqforge`
- Workflow name: `publish.yml`
- Environment name: `pypi`

(For a brand-new project the same values are entered under "Add a new pending
publisher" before the first upload exists.)

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
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
gh release create v0.2.0 --title "v0.2.0" --notes-file docs/release-notes/v0.2.0.md
```

Watch the run:

```bash
gh run watch
```

If the verify job fails, delete the tag before retrying — a tag that never
published is not a release:

```bash
git tag -d v0.2.0 && git push origin :refs/tags/v0.2.0
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
uv tool install --from dist/iqforge-0.2.0-py3-none-any.whl iqforge

iqforge info examples/bpsk_01.sigmf-meta
iqforge build examples/ -o /tmp/iqforge-smoke --balance-by core:freq_lower_edge
iqforge stats /tmp/iqforge-smoke
iqforge audit /tmp/iqforge-smoke
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
