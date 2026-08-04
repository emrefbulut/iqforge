"""Shard dosyalarının yazılması/okunması ve manifest.json."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from iqforge.io import IQForgeError

MANIFEST_NAME = "manifest.json"

#: Bir shard dosyasının üst sınırı (SPEC §5.7).
SHARD_MAX_BYTES = 256 * 1024 * 1024


class ShardWriter:
    """Bir split'in pencerelerini en fazla `SHARD_MAX_BYTES` boyutlu shard'lara yazar."""

    def __init__(self, root: Path, split: str, max_bytes: int = SHARD_MAX_BYTES) -> None:
        """Yazıcıyı hazırlar.

        Args:
            root: Veri seti kök klasörü.
            split: Split adı (`train`, `val`, `test`).
            max_bytes: Shard başına üst sınır.
        """
        self.root = root
        self.split = split
        self.max_bytes = max_bytes
        self.shards: list[str] = []
        self.labels: list[int] = []
        self._buffer: list[np.ndarray] = []
        self._buffered_bytes = 0

    def add(self, windows: np.ndarray, labels: list[int]) -> None:
        """Bir parti pencereyi ve etiketlerini kuyruğa ekler.

        Args:
            windows: `(n, ...)` şeklinde temsil dizisi.
            labels: `n` uzunluğunda tamsayı etiket listesi.
        """
        if windows.shape[0] == 0:
            return
        item_bytes = windows.nbytes // windows.shape[0]
        if self._buffered_bytes + windows.nbytes > self.max_bytes and self._buffer:
            self.flush()
        self._buffer.append(windows)
        self._buffered_bytes += windows.nbytes
        self.labels.extend(labels)

        # Tek parti bile sınırı aşıyorsa hemen boşalt.
        if self._buffered_bytes >= self.max_bytes - item_bytes:
            self.flush()

    def flush(self) -> None:
        """Kuyruktaki pencereleri yeni bir shard dosyasına yazar."""
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
        """Bu split'e yazılmış toplam pencere sayısı."""
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
    """`manifest.json` dosyasını yazar.

    Args:
        root: Veri seti kök klasörü.
        version: iqforge sürümü.
        config: Kullanılan build parametreleri.
        label_map: Etiket -> tamsayı eşlemesi.
        source_files: Kaynak `.sigmf-meta` yolları.
        splits: Split adı -> `{shards, labels, count, records}`.

    Returns:
        Yazılan manifest yolu.
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
    """Veri setinin manifest'ini okur.

    Raises:
        IQForgeError: Klasör veya manifest yoksa, ya da JSON bozuksa.
    """
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        raise IQForgeError(
            f"'{root}' bir iqforge veri seti değil: {MANIFEST_NAME} bulunamadı. "
            "Önce `iqforge build <girdi> -o <klasör>` çalıştırın."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IQForgeError(f"'{path}' geçerli JSON değil: {exc}") from exc


def dataset_size_bytes(root: Path) -> int:
    """Veri setinin diskte kapladığı toplam baytı verir."""
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())
