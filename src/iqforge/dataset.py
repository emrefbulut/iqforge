"""`IQForgeDataset` — kurulmuş veri setini PyTorch'a bağlar.

Bu modül `torch` gerektirir. `info`, `inspect`, `build` ve `stats` komutları
torch olmadan çalışır; bu yüzden `iqforge/__init__.py` bu modülü tembel yükler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from iqforge.io import IQForgeError
from iqforge.splitting import SPLIT_NAMES
from iqforge.storage import read_manifest


class IQForgeDataset(Dataset):
    """`iqforge build` çıktısını okuyan `torch.utils.data.Dataset`.

    Shard dosyaları `numpy.memmap` ile tembel okunur: veri setinin tamamı
    belleğe alınmaz, yalnızca erişilen pencereler sayfalanır.

    Attributes:
        root: Veri seti klasörü.
        split: `train`, `val` veya `test`.
        label_map: Etiket -> tamsayı eşlemesi.
        manifest: Ham manifest sözlüğü.
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        """Veri setini açar.

        Args:
            root: `manifest.json` içeren klasör.
            split: Okunacak split.

        Raises:
            IQForgeError: Klasör veri seti değilse, split adı geçersizse veya
                split boşsa.
        """
        self.root = Path(root)
        self.split = split
        self.manifest: dict[str, Any] = read_manifest(self.root)

        if split not in SPLIT_NAMES:
            raise IQForgeError(
                f"Bilinmeyen split '{split}'. Desteklenenler: {', '.join(SPLIT_NAMES)}."
            )

        entry = self.manifest["splits"][split]
        self.label_map: dict[str, int] = self.manifest["label_map"]
        self._labels: list[int] = entry["labels"]
        self._shard_names: list[str] = entry["shards"]

        if entry["count"] == 0:
            raise IQForgeError(
                f"'{split}' split'i boş. `iqforge stats {self.root}` ile bölmeyi kontrol edin; "
                "--split oranları bu split'e kayıt ayırmamış olabilir."
            )

        self._shards: list[np.memmap | None] = [None] * len(self._shard_names)
        self._offsets: list[int] = []
        total = 0
        for name in self._shard_names:
            self._offsets.append(total)
            total += int(np.load(self.root / name, mmap_mode="r").shape[0])

        if total != len(self._labels):
            raise IQForgeError(
                f"'{split}' bozuk: shard'larda {total} pencere var ama manifest "
                f"{len(self._labels)} etiket listeliyor."
            )

    def __len__(self) -> int:
        """Split'teki pencere sayısı."""
        return len(self._labels)

    @property
    def classes(self) -> list[str]:
        """Etiket adları, tamsayı sırasına göre."""
        return [name for name, _ in sorted(self.label_map.items(), key=lambda kv: kv[1])]

    def _shard(self, index: int) -> np.memmap:
        """Bir shard'ı tembel açar ve önbelleğe alır."""
        if self._shards[index] is None:
            self._shards[index] = np.load(self.root / self._shard_names[index], mmap_mode="r")
        return self._shards[index]  # type: ignore[return-value]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Tek bir pencereyi ve etiketini verir.

        Args:
            index: Pencere indisi.

        Returns:
            `(x, y)` — `x` temsile göre `(2, window)` float32 veya `(window,)`
            complex64 tensör; `y` tamsayı etiket.
        """
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"İndis {index} aralık dışında (0..{len(self) - 1}).")

        shard_index = int(np.searchsorted(self._offsets, index, side="right")) - 1
        row = index - self._offsets[shard_index]
        window = np.asarray(self._shard(shard_index)[row])
        return torch.from_numpy(window.copy()), self._labels[index]

    def __repr__(self) -> str:
        """Kısa özet."""
        return (
            f"IQForgeDataset(root='{self.root}', split='{self.split}', "
            f"n={len(self)}, classes={self.classes})"
        )
