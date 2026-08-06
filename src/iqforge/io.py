"""Reading SigMF recordings and converting sample datatypes.

Metadata parsing is delegated to the `sigmf` (sigmf-python) library; this module
only exposes the raw samples as `complex64` in a memory-friendly way.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from sigmf import SigMFFile

#: Supported `core:datatype` values -> (numpy dtype, full-scale divisor)
SUPPORTED_DATATYPES: dict[str, tuple[str, float]] = {
    "cf32_le": ("<f4", 1.0),
    "ci16_le": ("<i2", 32768.0),
    "ci8": ("i1", 128.0),
}

META_EXT = ".sigmf-meta"
DATA_EXT = ".sigmf-data"


class IQForgeError(Exception):
    """An error meant to be shown to the user."""


@dataclass(frozen=True)
class Annotation:
    """A single SigMF annotation.

    Attributes:
        sample_start: Sample index where the annotation begins.
        sample_count: Number of samples the annotation spans.
        label: The `core:label` field, or None.
        freq_lower_edge: Lower frequency edge in Hz, or None.
        freq_upper_edge: Upper frequency edge in Hz, or None.
        description: The `core:description` field, or None.
        raw: The annotation's raw SigMF dictionary. Gives access to fields not
            parsed above, including extension keys; `--balance-by` uses this.
    """

    sample_start: int
    sample_count: int
    label: str | None = None
    freq_lower_edge: float | None = None
    freq_upper_edge: float | None = None
    description: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sample_end(self) -> int:
        """Sample index just past the end of the annotation."""
        return self.sample_start + self.sample_count


@dataclass
class Recording:
    """An opened SigMF recording pair (`.sigmf-meta` + `.sigmf-data`).

    Sample data is read lazily through `numpy.memmap`; the file is never loaded
    into memory in one piece.
    """

    meta_path: Path
    data_path: Path
    datatype: str
    sample_rate: float
    center_frequency: float | None
    num_samples: int
    annotations: list[Annotation]
    global_info: dict[str, Any]

    #: `core:version` exactly as written in the file, before the sigmf library
    #: touches it. The library replaces the field with the spec version IT
    #: implements, so `global_info["core:version"]` reports the reader, not the
    #: recording -- measured against real captures that declare 1.0.0 and are
    #: reported as 1.2.6. Anyone debugging a compatibility problem needs the
    #: number the writer actually put there. `None` if the file omits it.
    declared_version: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Length of the recording in seconds."""
        return self.num_samples / self.sample_rate

    def read(self, start: int = 0, count: int | None = None) -> np.ndarray:
        """Read `complex64` samples from the recording.

        Args:
            start: Sample index to start reading from.
            count: Number of samples to read; None reads to the end.

        Returns:
            A one-dimensional `complex64` array.

        Raises:
            IQForgeError: If `start` lies outside the recording.
        """
        if start < 0 or start > self.num_samples:
            raise IQForgeError(
                f"Start index {start} is outside the recording. Valid range: 0..{self.num_samples}."
            )
        available = self.num_samples - start
        n = available if count is None else min(count, available)
        if n <= 0:
            return np.empty(0, dtype=np.complex64)

        np_dtype, full_scale = SUPPORTED_DATATYPES[self.datatype]
        raw = np.memmap(
            self.data_path,
            dtype=np_dtype,
            mode="r",
            offset=start * 2 * np.dtype(np_dtype).itemsize,
            shape=(n * 2,),
        )
        interleaved = np.asarray(raw, dtype=np.float32)
        if full_scale != 1.0:
            interleaved = interleaved / np.float32(full_scale)
        return (interleaved[0::2] + 1j * interleaved[1::2]).astype(np.complex64)


def _resolve_paths(path: str | Path) -> tuple[Path, Path]:
    """Derive the metadata and data file paths from the given path."""
    p = Path(path)
    if p.suffix == META_EXT:
        meta = p
    elif p.suffix == DATA_EXT:
        meta = p.with_suffix(META_EXT)
    else:
        meta = Path(str(p) + META_EXT)

    if not meta.exists():
        raise IQForgeError(
            f"SigMF metadata file not found: {meta}. "
            f"Pass a '{META_EXT}' file or a recording name without an extension."
        )
    data = meta.with_suffix(DATA_EXT)
    if not data.exists():
        raise IQForgeError(
            f"SigMF data file not found: {data}. "
            f"'{data.name}' must sit in the same directory as '{meta.name}'."
        )
    return meta, data


def load(path: str | Path) -> Recording:
    """Open a SigMF recording and validate its metadata.

    Args:
        path: A `.sigmf-meta` file, a `.sigmf-data` file, or a recording name
            without an extension.

    Returns:
        The opened `Recording`.

    Raises:
        IQForgeError: If a file is missing, the datatype is unsupported, or a
            required metadata field is absent.
    """
    meta_path, data_path = _resolve_paths(path)

    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise IQForgeError(
            f"'{meta_path.name}' is not valid JSON: {exc}. "
            "A SigMF metadata file must be a UTF-8 encoded JSON object."
        ) from exc

    # Must happen BEFORE SigMFFile: the library mutates the dict it is handed,
    # overwriting core:version with the spec version it implements. Reading it
    # afterwards -- from `raw` or from the handle -- returns the reader's
    # version for every file. Verified against captures that declare 1.0.0.
    declared_version = _declared_version(raw)

    # Schema validation is left to the sigmf library. This module reads the
    # samples itself via memmap, so the data file is never bound to the library.
    try:
        handle = SigMFFile(metadata=raw)
    except Exception as exc:  # sigmf raises several different error types
        raise IQForgeError(f"Could not read SigMF metadata ({meta_path}): {exc}") from exc

    global_info = dict(handle.get_global_info())

    datatype = global_info.get("core:datatype")
    if datatype is None:
        raise IQForgeError(
            f"'{meta_path.name}' has no required 'core:datatype' field. "
            f"Supported: {', '.join(SUPPORTED_DATATYPES)}."
        )
    if datatype not in SUPPORTED_DATATYPES:
        raise IQForgeError(
            f"Unsupported datatype '{datatype}'. Supported: {', '.join(SUPPORTED_DATATYPES)}."
        )

    sample_rate = global_info.get("core:sample_rate")
    if sample_rate is None:
        raise IQForgeError(
            f"'{meta_path.name}' has no 'core:sample_rate'. Without a sample rate the "
            "time and frequency axes cannot be computed; add the field to the metadata."
        )

    np_dtype, _ = SUPPORTED_DATATYPES[datatype]
    bytes_per_sample = 2 * np.dtype(np_dtype).itemsize
    file_bytes = data_path.stat().st_size
    if file_bytes % bytes_per_sample != 0:
        raise IQForgeError(
            f"The size of '{data_path.name}' ({file_bytes} bytes) is not a whole multiple "
            f"of {bytes_per_sample} bytes per sample for '{datatype}'. The file may be corrupt."
        )
    num_samples = file_bytes // bytes_per_sample

    center_frequency: float | None = None
    captures = handle.get_captures()
    if captures:
        freq = captures[0].get("core:frequency")
        center_frequency = float(freq) if freq is not None else None

    annotations = [
        Annotation(
            sample_start=int(a["core:sample_start"]),
            sample_count=int(a.get("core:sample_count", 0)),
            label=a.get("core:label"),
            freq_lower_edge=a.get("core:freq_lower_edge"),
            freq_upper_edge=a.get("core:freq_upper_edge"),
            description=a.get("core:description"),
            raw=MappingProxyType(dict(a)),
        )
        for a in handle.get_annotations()
    ]
    annotations.sort(key=lambda a: (a.sample_start, a.sample_count))

    return Recording(
        meta_path=meta_path,
        data_path=data_path,
        datatype=datatype,
        sample_rate=float(sample_rate),
        center_frequency=center_frequency,
        num_samples=num_samples,
        annotations=annotations,
        global_info=global_info,
        declared_version=declared_version,
    )


def _declared_version(raw: dict[str, Any]) -> str | None:
    """Read `core:version` straight from the parsed file.

    Deliberately reads `raw` rather than the sigmf handle: the library
    overwrites the field with the spec version it implements, so going through
    it would return the reader's version for every file ever written.
    """
    global_section = raw.get("global")
    if not isinstance(global_section, dict):
        return None
    version = global_section.get("core:version")
    return str(version) if version is not None else None
