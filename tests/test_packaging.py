"""Tests for what the built package claims about itself.

The version reaches users through three paths: the wheel metadata (what pip
resolves), `iqforge version` (what a bug report quotes), and the dataset
manifest (what makes a build reproducible). They must agree, or a released
artifact misidentifies itself and the disagreement is invisible until someone
tries to reproduce a result.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqforge import __version__
from iqforge.cli import app

runner = CliRunner()

#: A recording that carries annotations, so the table containing `→` gets drawn.
EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "bpsk_01.sigmf-meta"


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


def _run_cli(tmp_path: Path, *args: str, encoding: str) -> subprocess.CompletedProcess[bytes]:
    """Run the CLI in a subprocess with stdout redirected to a file.

    A subprocess is the only honest way to test this: the failure depends on
    what encoding Python picks for a *real* pipe at interpreter startup, and
    CliRunner replaces the streams with objects that accept any string.
    """
    env = {**os.environ, "PYTHONIOENCODING": encoding}
    out = tmp_path / "captured.txt"
    with out.open("wb") as handle:
        return subprocess.run(
            [sys.executable, "-m", "iqforge", *args],
            stdout=handle,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )


@pytest.mark.parametrize("encoding", ["cp1254", "cp1252", "ascii", "utf-8"])
def test_output_survives_redirection_under_any_locale(tmp_path: Path, encoding: str) -> None:
    """Redirecting output must not crash on a non-UTF-8 locale.

    `iqforge info` prints `→` in the annotation table. Piped to a file on a
    Turkish Windows install, Python encoded as cp1254 and raised
    UnicodeEncodeError — while the identical command printed to the terminal
    worked, which is what kept it hidden.
    """
    result = _run_cli(tmp_path, "info", str(EXAMPLE), encoding=encoding)

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"UnicodeEncodeError" not in result.stderr


def test_inspect_survives_redirection_under_a_non_utf8_locale(tmp_path: Path) -> None:
    """The spectrogram is built entirely from block-drawing characters."""
    result = _run_cli(tmp_path, "inspect", str(EXAMPLE), encoding="cp1254")

    assert result.returncode == 0, result.stderr.decode(errors="replace")
