"""`IQForgeDataset` - connects a built dataset to PyTorch.

This module needs `torch`. The `info`, `inspect`, `build` and `stats` commands
work without it, which is why `iqforge/__init__.py` imports this lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqforge import _require_torch

_require_torch("IQForgeDataset")

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from iqforge.io import IQForgeError  # noqa: E402
from iqforge.splitting import SPLIT_NAMES  # noqa: E402
from iqforge.storage import read_manifest  # noqa: E402


class IQForgeDataset(Dataset):
    """A `torch.utils.data.Dataset` over the output of `iqforge build`.

    Shard files are read lazily through `numpy.memmap`: the dataset is never
    loaded into memory as a whole, only the windows actually touched are paged
    in.

    Attributes:
        root: The dataset directory.
        split: `train`, `val` or `test`.
        label_map: Label -> integer mapping.
        manifest: The raw manifest dictionary.
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        """Open the dataset.

        Args:
            root: A directory containing `manifest.json`.
            split: Which split to read.

        Raises:
            IQForgeError: If the directory is not a dataset, the split name is
                invalid, or the split is empty.
        """
        self.root = Path(root)
        self.split = split
        self.manifest: dict[str, Any] = read_manifest(self.root)

        if split not in SPLIT_NAMES:
            raise IQForgeError(f"Unknown split '{split}'. Supported: {', '.join(SPLIT_NAMES)}.")

        entry = self.manifest["splits"][split]
        self.label_map: dict[str, int] = self.manifest["label_map"]
        self._labels: list[int] = entry["labels"]
        self._shard_names: list[str] = entry["shards"]

        if entry["count"] == 0:
            raise IQForgeError(
                f"The '{split}' split is empty. Check the split with "
                f"`iqforge stats {self.root}`; the --split ratios may not have "
                "allocated any recording to it."
            )

        self._shards: list[np.memmap | None] = [None] * len(self._shard_names)
        self._offsets: list[int] = []
        total = 0
        for name in self._shard_names:
            self._offsets.append(total)
            total += int(np.load(self.root / name, mmap_mode="r").shape[0])

        if total != len(self._labels):
            raise IQForgeError(
                f"'{split}' is corrupt: the shards hold {total} windows but the manifest "
                f"lists {len(self._labels)} labels."
            )

    def __len__(self) -> int:
        """Number of windows in the split."""
        return len(self._labels)

    @property
    def classes(self) -> list[str]:
        """Label names, in integer order."""
        return [name for name, _ in sorted(self.label_map.items(), key=lambda kv: kv[1])]

    def _shard(self, index: int) -> np.memmap:
        """Open a shard lazily and cache it."""
        if self._shards[index] is None:
            self._shards[index] = np.load(self.root / self._shard_names[index], mmap_mode="r")
        return self._shards[index]  # type: ignore[return-value]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return one window and its label.

        Args:
            index: Window index.

        Returns:
            `(x, y)` where `x` is a `(2, window)` float32 tensor or a
            `(window,)` complex64 tensor depending on the representation, and
            `y` is the integer label.
        """
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} is out of range (0..{len(self) - 1}).")

        shard_index = int(np.searchsorted(self._offsets, index, side="right")) - 1
        row = index - self._offsets[shard_index]
        window = np.asarray(self._shard(shard_index)[row])
        return torch.from_numpy(window.copy()), self._labels[index]

    def __repr__(self) -> str:
        """Short summary."""
        return (
            f"IQForgeDataset(root='{self.root}', split='{self.split}', "
            f"n={len(self)}, classes={self.classes})"
        )
