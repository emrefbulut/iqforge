"""iqforge.dataset testleri. torch kurulu değilse atlanır."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="IQForgeDataset için torch gerekli")

from iqforge.dataset import IQForgeDataset  # noqa: E402
from iqforge.io import IQForgeError  # noqa: E402
from iqforge.storage import ShardWriter, write_manifest  # noqa: E402


def _make_dataset(
    root: Path,
    *,
    train_rows: int = 6,
    test_rows: int = 2,
    shard_max_bytes: int | None = None,
    window: int = 16,
) -> Path:
    """Elle küçük bir veri seti kurar; satır i'nin ilk örneği i olur."""
    splits: dict[str, dict] = {}
    for split, rows in (("train", train_rows), ("val", 0), ("test", test_rows)):
        writer = (
            ShardWriter(root, split, max_bytes=shard_max_bytes)
            if shard_max_bytes
            else ShardWriter(root, split)
        )
        for i in range(rows):
            batch = np.full((1, 2, window), float(i), dtype=np.float32)
            writer.add(batch, [i % 2])
        writer.flush()
        splits[split] = {
            "shards": writer.shards,
            "labels": writer.labels,
            "count": writer.count,
            "records": [],
        }

    write_manifest(
        root,
        version="0.1.0",
        config={"window": window, "stride": 8, "repr": "iq2ch", "normalize": True, "seed": 42},
        label_map={"bpsk": 0, "qpsk": 1},
        source_files=[],
        splits=splits,
    )
    return root


def test_length_and_labels_follow_the_manifest(tmp_path: Path) -> None:
    """Uzunluk ve etiketler manifest ile birebir olmalı."""
    _make_dataset(tmp_path)
    data = IQForgeDataset(tmp_path, split="train")

    assert len(data) == 6
    assert [int(data[i][1]) for i in range(6)] == [0, 1, 0, 1, 0, 1]
    assert data.label_map == {"bpsk": 0, "qpsk": 1}
    assert data.classes == ["bpsk", "qpsk"]


def test_item_is_a_float32_tensor_of_the_right_shape(tmp_path: Path) -> None:
    """`x` `(2, window)` float32 tensör olmalı (SPEC §5.8)."""
    _make_dataset(tmp_path, window=32)
    x, y = IQForgeDataset(tmp_path, split="train")[0]

    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 32)
    assert x.dtype == torch.float32
    assert isinstance(y, int)


def test_rows_are_returned_in_manifest_order_across_shards(tmp_path: Path) -> None:
    """Birden fazla shard'a bölünmüş veri sırayı korumalı."""
    _make_dataset(
        tmp_path, train_rows=7, shard_max_bytes=np.zeros((1, 2, 16), np.float32).nbytes * 2
    )
    data = IQForgeDataset(tmp_path, split="train")

    assert len(data) == 7
    assert [float(data[i][0][0, 0]) for i in range(7)] == [0, 1, 2, 3, 4, 5, 6]


def test_negative_and_out_of_range_indices(tmp_path: Path) -> None:
    """Negatif indis sondan saymalı, aralık dışı indis hata vermeli."""
    _make_dataset(tmp_path, train_rows=4)
    data = IQForgeDataset(tmp_path, split="train")

    assert float(data[-1][0][0, 0]) == 3.0
    with pytest.raises(IndexError):
        data[4]


def test_empty_split_is_rejected_with_a_hint(tmp_path: Path) -> None:
    """Boş split sessizce sıfır uzunluk dönmemeli, ne yapılacağını söylemeli."""
    _make_dataset(tmp_path)
    with pytest.raises(IQForgeError) as exc:
        IQForgeDataset(tmp_path, split="val")
    assert "iqforge stats" in str(exc.value)


def test_unknown_split_lists_the_valid_ones(tmp_path: Path) -> None:
    """Geçersiz split adı desteklenenleri listelemeli."""
    _make_dataset(tmp_path)
    with pytest.raises(IQForgeError) as exc:
        IQForgeDataset(tmp_path, split="deneme")
    assert "train" in str(exc.value) and "test" in str(exc.value)


def test_label_count_mismatch_is_detected(tmp_path: Path) -> None:
    """Shard satır sayısı ile etiket sayısı uyuşmazsa sessizce devam edilmemeli."""
    _make_dataset(tmp_path, train_rows=4)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["labels"] = [0, 1, 0]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IQForgeError, match="bozuk"):
        IQForgeDataset(tmp_path, split="train")


def test_works_with_a_torch_dataloader(tmp_path: Path) -> None:
    """Standart DataLoader ile batch'lenebilmeli."""
    from torch.utils.data import DataLoader

    _make_dataset(tmp_path, train_rows=6, window=8)
    loader = DataLoader(IQForgeDataset(tmp_path, split="train"), batch_size=4)
    batches = list(loader)

    assert [tuple(x.shape) for x, _ in batches] == [(4, 2, 8), (2, 2, 8)]
    assert torch.cat([y for _, y in batches]).tolist() == [0, 1, 0, 1, 0, 1]


def test_shards_are_memory_mapped_not_loaded(tmp_path: Path) -> None:
    """Shard'lar memmap ile açılmalı; veri seti belleğe kopyalanmamalı."""
    _make_dataset(tmp_path, train_rows=4)
    data = IQForgeDataset(tmp_path, split="train")
    data[0]

    assert isinstance(data._shard(0), np.memmap)
