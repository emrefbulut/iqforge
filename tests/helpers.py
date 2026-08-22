"""Shared helper for building SigMF recordings in tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from iqforge.io import SUPPORTED_DATATYPES


def write_record(
    directory: Path,
    samples: np.ndarray,
    datatype: str = "cf32_le",
    *,
    name: str = "rec",
    sample_rate: float | None = 1_024_000.0,
    center_freq: float | None = 100_000_000.0,
    annotations: list[dict] | None = None,
    capture_extra: dict | None = None,
) -> Path:
    """Write a SigMF recording pair by hand and return the metadata path.

    Args:
        directory: Target directory; created if missing.
        samples: Complex samples.
        datatype: The `core:datatype` value. An unsupported value may be passed
            to exercise error paths; the data is then written as `cf32_le`.
        name: File name without extension.
        sample_rate: `core:sample_rate`; the field is omitted when None.
        center_freq: `core:frequency`; the field is omitted when None.
        annotations: Raw annotation dictionaries.
        capture_extra: Extra keys for the capture segment, such as
            `core:datetime` -- which SigMF puts here rather than in
            `global`.

    Returns:
        Path of the written `.sigmf-meta` file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    np_dtype, full_scale = SUPPORTED_DATATYPES.get(datatype, ("<f4", 1.0))
    interleaved = np.empty(samples.size * 2, dtype=np.float64)
    interleaved[0::2] = samples.real
    interleaved[1::2] = samples.imag
    (interleaved * full_scale).astype(np_dtype).tofile(directory / f"{name}.sigmf-data")

    global_info: dict = {"core:datatype": datatype, "core:version": "1.0.0"}
    if sample_rate is not None:
        global_info["core:sample_rate"] = sample_rate

    capture: dict = {"core:sample_start": 0}
    if center_freq is not None:
        capture["core:frequency"] = center_freq
    capture.update(capture_extra or {})

    meta_path = directory / f"{name}.sigmf-meta"
    meta_path.write_text(
        json.dumps(
            {"global": global_info, "captures": [capture], "annotations": annotations or []}
        ),
        encoding="utf-8",
    )
    return meta_path


# --------------------------------------------------------------------------
# Research datasets that are not in the repository
# --------------------------------------------------------------------------

#: Directory of prepared LoRaIQ recordings. Companion CSVs are looked up
#: alongside it unless their own variables are set.
ENV_LORAIQ = "IQFORGE_LORAIQ"
ENV_LORAIQ_INDEX = "IQFORGE_LORAIQ_INDEX"
ENV_LORAIQ_LABELS = "IQFORGE_LORAIQ_LABELS"
ENV_LORAIQ_GROUPS = "IQFORGE_LORAIQ_GROUPS"


def _env_path(var: str) -> Path | None:
    raw = os.environ.get(var)
    return Path(raw) if raw else None


def loraiq_paths() -> tuple[Path, Path, Path, Path] | None:
    """`(source, index, labels, groups)`, or None when the data is absent.

    There is deliberately no fallback path. Both callers used to carry an
    absolute path into one developer's temporary directory, which meant the
    tests passed there and skipped everywhere else -- while
    `docs/methodology.md` claimed the results were reproducible from this
    repository. A path that resolves on exactly one machine is a machine
    setting, not a default; `loraiq_skip_reason` says which setting is missing.
    """
    source = _env_path(ENV_LORAIQ)
    if source is None or not source.is_dir():
        return None
    index = _env_path(ENV_LORAIQ_INDEX) or source.parent / "loraiq.csv"
    labels = _env_path(ENV_LORAIQ_LABELS) or source.parent / "loraiq_labels.csv"
    groups = _env_path(ENV_LORAIQ_GROUPS) or source.parent / "loraiq_groups.csv"
    if not all(p.exists() for p in (index, labels, groups)):
        return None
    return source, index, labels, groups


def loraiq_skip_reason() -> str | None:
    """Why the LoRaIQ tests cannot run here, or None when they can.

    Named rather than generic: "LoRaIQ recordings are not on this machine" is
    indistinguishable from "you set the variable and mistyped the path", and a
    skip nobody can act on is a skip nobody notices.
    """
    source = _env_path(ENV_LORAIQ)
    if source is None:
        return (
            f"{ENV_LORAIQ} is not set. Point it at a directory of prepared LoRaIQ "
            f"recordings to run this; the dataset is a public download and is not "
            f"in this repository."
        )
    if not source.is_dir():
        return f"{ENV_LORAIQ} is set to '{source}', which is not a directory."
    missing = [
        f"{var}={path}"
        for var, path in (
            (ENV_LORAIQ_INDEX, _env_path(ENV_LORAIQ_INDEX) or source.parent / "loraiq.csv"),
            (
                ENV_LORAIQ_LABELS,
                _env_path(ENV_LORAIQ_LABELS) or source.parent / "loraiq_labels.csv",
            ),
            (
                ENV_LORAIQ_GROUPS,
                _env_path(ENV_LORAIQ_GROUPS) or source.parent / "loraiq_groups.csv",
            ),
        )
        if not path.exists()
    ]
    if missing:
        return (
            f"{ENV_LORAIQ} resolves, but these companion files are missing: "
            f"{', '.join(missing)}. They are looked up alongside the recordings "
            f"unless their own variables are set."
        )
    return None
