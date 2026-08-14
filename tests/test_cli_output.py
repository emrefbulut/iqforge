"""Tests for what the CLI actually puts on screen.

These check the rendered output, not the source string. `rich` reads anything
in square brackets as a style tag and deletes it, so a message that looks right
in the source can reach the user mangled — `pip install 'iqforge[torch]'` was
printed as `pip install 'iqforge'` until this was caught.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from helpers import write_record
from iqforge import TORCH_REQUIRED
from iqforge.cli import _render_annotations, _render_overview, app
from iqforge.io import load

runner = CliRunner()

#: Modules that import torch and must be re-imported when torch is masked.
TORCH_DEPENDENT = ("iqforge.dataset", "iqforge.models", "iqforge.training")


def _flat(output: str) -> str:
    """Undo rich's line wrapping so assertions do not depend on terminal width.

    The console width cannot be controlled from here: neither
    `CliRunner(env=...)` nor `monkeypatch.setenv("COLUMNS", ...)` reaches the
    Console the CLI builds at import time — measured, not assumed. Rich keeps
    the space it wraps on and hard-breaks long tokens without inserting
    anything, so dropping the newlines reconstructs the original line in both
    cases.
    """
    return output.replace("\n", "")


@pytest.fixture
def without_torch(monkeypatch: pytest.MonkeyPatch):
    """Make `import torch` fail, as it does in an install without the extra."""
    monkeypatch.setitem(sys.modules, "torch", None)
    for name in TORCH_DEPENDENT:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return None


def _rendered(markup: str, width: int = 200) -> str:
    """Return what `rich` would actually print for this markup string."""
    console = Console(width=width, force_terminal=False, no_color=True)
    console.begin_capture()
    console.print(markup)
    return console.end_capture()


def test_torch_hint_names_the_extra_in_the_source() -> None:
    """The shared message names the extra, so both users of it can say so."""
    message = TORCH_REQUIRED.format(what="`iqforge train`")

    assert "pip install 'iqforge[torch]'" in message
    assert "uv sync --extra torch" in message


def test_unescaped_markup_would_eat_the_extra() -> None:
    """Pin down the failure mode this whole module exists for.

    Printed without escaping, rich reads `[torch]` as a style tag and drops it -
    the user is told to `pip install 'iqforge'`, which does not install torch.
    If rich ever stops doing this, the escaping can be reconsidered.
    """
    rendered = _rendered(TORCH_REQUIRED.format(what="x"))

    assert "[torch]" not in rendered
    assert "pip install 'iqforge'" in rendered


def test_train_without_torch_prints_an_installable_command(
    without_torch: None, tmp_path: Path
) -> None:
    """`iqforge train` without torch tells the user exactly what to install.

    This is the regression test: it reads what reached the screen, not the
    source string.
    """
    result = runner.invoke(app, ["train", str(tmp_path)])

    assert result.exit_code == 1
    flat = _flat(result.output)
    assert "[torch]" in flat
    assert "pip install 'iqforge[torch]'" in flat
    assert "uv sync --extra torch" in flat


def test_importing_the_dataset_without_torch_explains_itself(without_torch: None) -> None:
    """`import iqforge.dataset` explains itself instead of a bare import error."""
    with pytest.raises(ModuleNotFoundError) as exc:
        import iqforge.dataset  # noqa: F401

    message = str(exc.value)
    assert "IQForgeDataset" in message
    assert "pip install 'iqforge[torch]'" in message


def test_lazy_attribute_without_torch_explains_itself(without_torch: None) -> None:
    """`from iqforge import IQForgeDataset` gets the same explanation."""
    import iqforge

    with pytest.raises(ModuleNotFoundError, match=r"iqforge\[torch\]"):
        iqforge.IQForgeDataset  # noqa: B018


@pytest.mark.parametrize("module", TORCH_DEPENDENT)
def test_every_torch_module_explains_itself(without_torch: None, module: str) -> None:
    """Every torch-dependent module names the install command."""
    with pytest.raises(ModuleNotFoundError, match=r"iqforge\[torch\]"):
        __import__(module)


def _render_table(table: object, width: int = 200) -> str:
    """Render a rich renderable on a console this test controls.

    The tables go through the CLI's own console in real use, and its width
    cannot be set from a test. A narrow terminal makes rich TRUNCATE cells with
    an ellipsis, which no post-processing can undo — and truncation is not what
    these tests are about. Rendering the table directly keeps the subject to
    the one thing under test: whether the escaping preserved the brackets.
    """
    console = Console(width=width, force_terminal=False, no_color=True)
    console.begin_capture()
    console.print(table)
    return console.end_capture()


def test_metadata_with_brackets_survives_info(tmp_path: Path) -> None:
    """Bracketed text in the metadata must reach the screen intact.

    A recorder name like `GNU Radio [3.10]` is ordinary; rich would delete the
    bracketed part unless it is escaped.
    """
    samples = (np.arange(64) + 1j * np.arange(64)).astype(np.complex64)
    meta = write_record(tmp_path, samples, name="capture")
    text = meta.read_text(encoding="utf-8")
    text = text.replace(
        '"core:version": "1.0.0"',
        '"core:version": "1.0.0", "core:hw": "SDR [rev2]", "core:recorder": "GNU Radio [3.10]"',
    )
    meta.write_text(text, encoding="utf-8")

    assert runner.invoke(app, ["info", str(meta)]).exit_code == 0
    rendered = _render_table(_render_overview(load(meta)))

    assert "SDR [rev2]" in rendered
    assert "GNU Radio [3.10]" in rendered


def test_bracketed_label_survives_info(tmp_path: Path) -> None:
    """An annotation label containing brackets is printed as written."""
    samples = (np.arange(64) + 1j * np.arange(64)).astype(np.complex64)
    meta = write_record(
        tmp_path,
        samples,
        name="capture",
        annotations=[{"core:sample_start": 0, "core:sample_count": 32, "core:label": "wifi[ch6]"}],
    )

    assert runner.invoke(app, ["info", str(meta)]).exit_code == 0
    rendered = _render_table(_render_annotations(load(meta)))

    assert "wifi[ch6]" in rendered


def test_error_message_with_brackets_survives(tmp_path: Path) -> None:
    """A file name with brackets is shown in full in the error."""
    missing = tmp_path / "capture[1].sigmf-meta"

    result = runner.invoke(app, ["info", str(missing)])

    assert result.exit_code == 1
    assert "capture[1].sigmf-meta" in _flat(result.output)


def test_build_refuses_a_label_csv_whose_names_collapse(tmp_path):
    """The LoRaIQ regression, reconstructed at fixture scale.

    Four recordings under four session directories all named 3.sigmf-meta, a
    CSV that labels each one differently, and a lookup that used to reduce the
    key to the bare name. The build produced a one-class dataset and printed
    nothing; now it stops and names what was lost.
    """
    labels = []
    for session, label in (("s1", "alpha"), ("s2", "beta"), ("s3", "gamma"), ("s4", "delta")):
        write_record(
            tmp_path / "recs" / session,
            np.exp(2j * np.pi * 0.05 * np.arange(8192)).astype(np.complex64),
            name="3",
        )
        labels.append((f"{session}/3.sigmf-meta", label))

    # A table keyed by bare name, as a user would write it for a flat layout.
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "filename,label\n" + "".join(f"{Path(k).name},{v}\n" for k, v in labels), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "build",
            str(tmp_path / "recs"),
            "-o",
            str(tmp_path / "ds"),
            "--labels",
            "csv",
            "--label-file",
            str(csv_path),
            "--split",
            "1.0,0,0",
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "distinct label" in result.output
    assert "does not identify recordings uniquely" in result.output


def test_build_accepts_the_same_table_written_as_relative_paths(tmp_path):
    """The fix the error message tells the user to apply."""
    rows = []
    for session, label in (("s1", "alpha"), ("s2", "beta")):
        write_record(
            tmp_path / "recs" / session,
            np.exp(2j * np.pi * 0.05 * np.arange(8192)).astype(np.complex64),
            name="3",
        )
        rows.append((f"{session}/3.sigmf-meta", label))
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "filename,label\n" + "".join(f"{k},{v}\n" for k, v in rows), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "build",
            str(tmp_path / "recs"),
            "-o",
            str(tmp_path / "ds"),
            "--labels",
            "csv",
            "--label-file",
            str(csv_path),
            "--split",
            "1.0,0,0",
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["label_map"]) == {"alpha", "beta"}
