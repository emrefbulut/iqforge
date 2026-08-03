"""sigkit.storage testleri."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigkit.io import SigkitError
from sigkit.storage import (
    MANIFEST_NAME,
    ShardWriter,
    dataset_size_bytes,
    read_manifest,
    write_manifest,
)


def _batch(n: int, window: int = 64) -> np.ndarray:
    return np.zeros((n, 2, window), dtype=np.float32)


def test_writer_produces_one_shard_when_under_limit(tmp_path: Path) -> None:
    """Sınırın altında kalan veri tek shard'a yazılmalı."""
    writer = ShardWriter(tmp_path, "train")
    writer.add(_batch(10), list(range(10)))
    writer.flush()

    assert writer.shards == ["train/shard_0000.npy"]
    assert writer.count == 10
    assert np.load(tmp_path / "train/shard_0000.npy").shape == (10, 2, 64)


def test_writer_rolls_over_at_the_size_limit(tmp_path: Path) -> None:
    """Shard boyut sınırı aşılınca yeni dosyaya geçilmeli (SPEC §5.7)."""
    item_bytes = _batch(1).nbytes
    writer = ShardWriter(tmp_path, "train", max_bytes=item_bytes * 4)

    for i in range(9):
        writer.add(_batch(1), [i])
    writer.flush()

    assert len(writer.shards) > 1, "sınır aşıldığı halde tek shard yazılmış"
    assert writer.count == 9
    total = sum(np.load(tmp_path / s).shape[0] for s in writer.shards)
    assert total == 9
    for shard in writer.shards:
        assert (tmp_path / shard).stat().st_size <= item_bytes * 4 + 256  # +npy başlığı


def test_writer_preserves_order_and_labels(tmp_path: Path) -> None:
    """Pencereler ve etiketler eklenme sırasını korumalı."""
    writer = ShardWriter(tmp_path, "val")
    for i in range(5):
        batch = np.full((1, 2, 4), i, dtype=np.float32)
        writer.add(batch, [i * 10])
    writer.flush()

    stacked = np.concatenate([np.load(tmp_path / s) for s in writer.shards], axis=0)

    assert writer.labels == [0, 10, 20, 30, 40]
    assert [int(stacked[i, 0, 0]) for i in range(5)] == [0, 1, 2, 3, 4]


def test_empty_writer_creates_no_files(tmp_path: Path) -> None:
    """Hiç pencere eklenmemişse shard dosyası oluşturulmamalı."""
    writer = ShardWriter(tmp_path, "test")
    writer.flush()

    assert writer.shards == []
    assert writer.count == 0
    assert not (tmp_path / "test").exists()


def test_manifest_round_trip(tmp_path: Path) -> None:
    """Yazılan manifest aynı içerikle geri okunmalı."""
    write_manifest(
        tmp_path,
        version="0.1.0",
        config={"window": 1024, "stride": 512, "repr": "iq2ch", "normalize": True, "seed": 42},
        label_map={"bpsk": 0, "qpsk": 1},
        source_files=["examples/bpsk_01.sigmf-meta"],
        splits={
            "train": {"shards": ["train/shard_0000.npy"], "labels": [0, 1], "count": 2},
            "val": {"shards": [], "labels": [], "count": 0},
            "test": {"shards": [], "labels": [], "count": 0},
        },
    )

    manifest = read_manifest(tmp_path)

    assert manifest["sigkit_version"] == "0.1.0"
    assert manifest["label_map"] == {"bpsk": 0, "qpsk": 1}
    assert manifest["config"]["window"] == 1024
    assert manifest["splits"]["train"]["count"] == 2
    assert manifest["created"].endswith("Z")


def test_manifest_is_utf8_and_human_readable(tmp_path: Path) -> None:
    """Manifest girintili ve Türkçe karakterleri kaçırmadan yazılmalı."""
    write_manifest(
        tmp_path,
        version="0.1.0",
        config={"not": "ölçüm"},
        label_map={},
        source_files=[],
        splits={},
    )
    raw = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8")

    assert "ölçüm" in raw
    assert "\n  " in raw
    assert json.loads(raw)["config"]["not"] == "ölçüm"


def test_missing_manifest_points_at_the_fix(tmp_path: Path) -> None:
    """Manifest yoksa hata ne yapılacağını söylemeli."""
    with pytest.raises(SigkitError) as exc:
        read_manifest(tmp_path)
    assert "sigkit build" in str(exc.value)


def test_corrupt_manifest_is_reported(tmp_path: Path) -> None:
    """Bozuk JSON sessizce yutulmamalı."""
    (tmp_path / MANIFEST_NAME).write_text("{bozuk", encoding="utf-8")
    with pytest.raises(SigkitError, match="geçerli JSON değil"):
        read_manifest(tmp_path)


def test_dataset_size_counts_every_file(tmp_path: Path) -> None:
    """Disk kullanımı alt klasörlerdeki dosyaları da saymalı."""
    (tmp_path / "train").mkdir()
    (tmp_path / "train/shard_0000.npy").write_bytes(b"x" * 100)
    (tmp_path / MANIFEST_NAME).write_bytes(b"y" * 50)

    assert dataset_size_bytes(tmp_path) == 150
