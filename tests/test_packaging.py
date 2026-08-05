"""Tests for what the built package claims about itself.

The version reaches users through three paths: the wheel metadata (what pip
resolves), `iqforge version` (what a bug report quotes), and the dataset
manifest (what makes a build reproducible). They must agree, or a released
artifact misidentifies itself and the disagreement is invisible until someone
tries to reproduce a result.
"""

from __future__ import annotations

from importlib.metadata import version as installed_version

from typer.testing import CliRunner

from iqforge import __version__
from iqforge.cli import app

runner = CliRunner()


def test_installed_metadata_matches_the_package_constant() -> None:
    """`pyproject.toml` must not carry a second, drifting version string.

    The version is declared `dynamic` and read from `__init__.py`. If someone
    reintroduces a static `version = "..."` in `pyproject.toml`, the two can
    disagree and this test is what notices.
    """
    assert installed_version("iqforge") == __version__


def test_version_command_reports_the_package_version() -> None:
    """`iqforge version` prints exactly what the package says it is."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.output
