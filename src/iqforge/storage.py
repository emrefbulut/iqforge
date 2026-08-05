"""Writing and reading shard files, and manifest.json."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from iqforge.io import IQForgeError

MANIFEST_NAME = "manifest.json"

#: Upper bound on the size of one shard file (SPEC §5.7).
SHARD_MAX_BYTES = 256 * 1024 * 1024


class ShardWriter:
    """Writes a split's windows into shards of at most `SHARD_MAX_BYTES`."""

    def __init__(self, root: Path, split: str, max_bytes: int = SHARD_MAX_BYTES) -> None:
        """Prepare the writer.

        Args:
            root: Dataset root directory.
            split: Split name (`train`, `val`, `test`).
            max_bytes: Upper bound per shard.
        """
        self.root = root
        self.split = split
        self.max_bytes = max_bytes
        self.shards: list[str] = []
        self.labels: list[int] = []
        self._buffer: list[np.ndarray] = []
        self._buffered_bytes = 0

    def add(self, windows: np.ndarray, labels: list[int]) -> None:
        """Queue a batch of windows and their labels.

        Args:
            windows: An `(n, ...)` representation array.
            labels: `n` integer labels.
        """
        if windows.shape[0] == 0:
            return
        item_bytes = windows.nbytes // windows.shape[0]
        if self._buffered_bytes + windows.nbytes > self.max_bytes and self._buffer:
            self.flush()
        self._buffer.append(windows)
        self._buffered_bytes += windows.nbytes
        self.labels.extend(labels)

        # Flush immediately if even a single batch exceeds the limit.
        if self._buffered_bytes >= self.max_bytes - item_bytes:
            self.flush()

    def flush(self) -> None:
        """Write the queued windows into a new shard file."""
        if not self._buffer:
            return
        split_dir = self.root / self.split
        split_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self.split}/shard_{len(self.shards):04d}.npy"
        np.save(self.root / name, np.concatenate(self._buffer, axis=0))
        self.shards.append(name)
        self._buffer.clear()
        self._buffered_bytes = 0

    @property
    def count(self) -> int:
        """Total windows written to this split."""
        return len(self.labels)


def write_manifest(
    root: Path,
    *,
    version: str,
    config: dict[str, Any],
    label_map: dict[str, int],
    source_files: list[str],
    splits: dict[str, dict[str, Any]],
) -> Path:
    """Write the `manifest.json` file.

    Args:
        root: Dataset root directory.
        version: iqforge version.
        config: The build parameters used.
        label_map: Label -> integer mapping.
        source_files: Paths of the source `.sigmf-meta` files.
        splits: Split name -> `{shards, labels, count, records}`.

    Returns:
        Path of the written manifest.
    """
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "iqforge_version": version,
        "created": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "config": config,
        "label_map": label_map,
        "source_files": source_files,
        "splits": splits,
    }
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_manifest(root: Path) -> dict[str, Any]:
    """Read a dataset's manifest.

    Raises:
        IQForgeError: If the directory or manifest is missing, or the JSON is
            malformed.
    """
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        raise IQForgeError(
            f"'{root}' is not an iqforge dataset: {MANIFEST_NAME} not found. "
            "Run `iqforge build <input> -o <dir>` first."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IQForgeError(f"'{path}' is not valid JSON: {exc}") from exc


def dataset_size_bytes(root: Path) -> int:
    """Return the total size of the dataset on disk, in bytes."""
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
