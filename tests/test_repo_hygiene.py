"""Nothing session- or machine-specific is committed (CONTRIBUTING convention 8).

This is the convention with no natural failure mode: a hardcoded absolute path
works perfectly on the machine that wrote it, and the tests that depend on it
pass there. Everywhere else they skip, quietly, and a skip that nobody can act
on is a skip that nobody reads. `docs/methodology.md` claimed the published
numbers were reproducible from this repository for as long as two of the three
experiment scripts could only find their recordings inside one developer's
temporary directory.

So the check has to be mechanical rather than a review habit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from helpers import loraiq_paths, loraiq_skip_reason

ROOT = Path(__file__).resolve().parent.parent

#: Trees whose contents ship or are meant to be run by someone else.
SCANNED = ("src", "scripts", "tests")

#: An absolute path anchored at a Windows drive or a POSIX home/tmp root.
#: Narrow on purpose: `/` alone matches every URL and every docstring.
MACHINE_PATH = re.compile(
    r"""(?ix)
    (?: ["'] )                      # only inside a string literal
    (?:
        [A-Z]:[/\\]                 # C:/ or C:\
      | /home/[a-z0-9._-]+/         # /home/someone/
      | /Users/[a-z0-9._-]+/        # /Users/someone/
      | /tmp/[a-z0-9._-]+/          # /tmp/something/
      | /var/folders/               # macOS temp
    )
    """
)


#: Opt-out marker for a line that only *looks* like a machine path -- synthetic
#: data in a formatting test, for instance. Deliberately an inline comment
#: rather than a file allowlist: the exemption is then visible at the line it
#: applies to, and a reviewer reads it next to the string it excuses.
ALLOW_MARKER = "not-a-machine-path"


def _python_files() -> list[Path]:
    files: list[Path] = []
    for tree in SCANNED:
        files.extend(sorted((ROOT / tree).rglob("*.py")))
    return [f for f in files if "__pycache__" not in f.parts]


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_source_file_hardcodes_a_machine_path(path: Path) -> None:
    """A path that resolves on one machine is a setting, not a default.

    Data too large to commit is configured through an environment variable or
    a flag, and its absence is reported as a named, actionable skip -- never
    papered over with a fallback that happens to exist locally.
    """
    offenders = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if MACHINE_PATH.search(line) and ALLOW_MARKER not in line
    ]
    assert not offenders, "machine-specific path(s) committed:\n" + "\n".join(offenders)


def test_the_loraiq_skip_reason_names_the_variable_to_set() -> None:
    """ "Not on this machine" is not a reason anyone can act on.

    The reason has to distinguish "you never configured this" from "you
    configured it and mistyped the path", because those need opposite fixes.
    """
    reason = loraiq_skip_reason()
    if reason is None:
        assert loraiq_paths() is not None
        pytest.skip("LoRaIQ data is configured here, so there is no skip reason to inspect")
    assert "IQFORGE_LORAIQ" in reason, reason
    assert "not in this repository" in reason or "not a directory" in reason or "missing" in reason


def test_the_resolver_reports_a_mistyped_path_differently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set-but-wrong must not read the same as never-set."""
    monkeypatch.setenv("IQFORGE_LORAIQ", str(ROOT / "no-such-directory"))
    reason = loraiq_skip_reason()

    assert reason is not None
    assert "is not a directory" in reason, reason
    assert loraiq_paths() is None


def test_the_resolver_finds_nothing_when_the_variable_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No fallback path. This is the regression the module comment describes."""
    for var in (
        "IQFORGE_LORAIQ",
        "IQFORGE_LORAIQ_INDEX",
        "IQFORGE_LORAIQ_LABELS",
        "IQFORGE_LORAIQ_GROUPS",
    ):
        monkeypatch.delenv(var, raising=False)

    assert loraiq_paths() is None
    reason = loraiq_skip_reason()
    assert reason is not None
    assert "IQFORGE_LORAIQ is not set" in reason


def test_the_experiment_scripts_expose_no_default_data_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing configured, the scripts must resolve to None, not a guess."""
    import importlib
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    for var in ("IQFORGE_DASH7", "IQFORGE_LORAIQ"):
        monkeypatch.delenv(var, raising=False)

    real = importlib.reload(importlib.import_module("leakage_real"))
    loraiq = importlib.reload(importlib.import_module("leakage_loraiq"))

    assert real.DEFAULT_SOURCE is None
    assert loraiq.DEFAULT_SOURCE is None
    assert loraiq.DEFAULT_INDEX is None
    assert loraiq.DEFAULT_LABELS is None
    assert loraiq.DEFAULT_GROUPS is None
