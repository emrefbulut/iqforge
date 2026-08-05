"""Tests for iqforge.storage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from iqforge.io import IQForgeError
from iqforge.storage import (
    MANIFEST_NAME,
    ShardWriter,
    dataset_size_bytes,
    read_manifest,
    write_manifest,
)


def _batch(n: int, window: int = 64) -> np.ndarray:
    return np.zeros((n, 2, window), dtype=np.float32)


def test_writer_produces_one_shard_when_under_limit(tmp_path: Path) -> None:
    """Data below the limit goes into a single shard."""
    writer = ShardWriter(tmp_path, "train")
    writer.add(_batch(10), list(range(10)))
    writer.flush()

    assert writer.shards == ["train/shard_0000.npy"]
    assert writer.count == 10
    assert np.load(tmp_path / "train/shard_0000.npy").shape == (10, 2, 64)


def test_writer_rolls_over_at_the_size_limit(tmp_path: Path) -> None:
    """Crossing the shard size limit starts a new file (SPEC §5.7)."""
    item_bytes = _batch(1).nbytes
    writer = ShardWriter(tmp_path, "train", max_bytes=item_bytes * 4)

    for i in range(9):
        writer.add(_batch(1), [i])
    writer.flush()

    assert len(writer.shards) > 1, "the limit was exceeded but only one shard was written"
    assert writer.count == 9
    total = sum(np.load(tmp_path / s).shape[0] for s in writer.shards)
    assert total == 9
    for shard in writer.shards:
        assert (tmp_path / shard).stat().st_size <= item_bytes * 4 + 256  # + npy header


def test_writer_preserves_order_and_labels(tmp_path: Path) -> None:
    """Windows and labels keep the order they were added in."""
    writer = ShardWriter(tmp_path, "val")
    for i in range(5):
        batch = np.full((1, 2, 4), i, dtype=np.float32)
        writer.add(batch, [i * 10])
    writer.flush()

    stacked = np.concatenate([np.load(tmp_path / s) for s in writer.shards], axis=0)

    assert writer.labels == [0, 10, 20, 30, 40]
    assert [int(stacked[i, 0, 0]) for i in range(5)] == [0, 1, 2, 3, 4]


def test_empty_writer_creates_no_files(tmp_path: Path) -> None:
    """No windows added means no shard file is created."""
    writer = ShardWriter(tmp_path, "test")
    writer.flush()

    assert writer.shards == []
    assert writer.count == 0
    assert not (tmp_path / "test").exists()


def test_manifest_round_trip(tmp_path: Path) -> None:
    """A written manifest reads back with the same contents."""
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

    assert manifest["iqforge_version"] == "0.1.0"
    assert manifest["label_map"] == {"bpsk": 0, "qpsk": 1}
    assert manifest["config"]["window"] == 1024
    assert manifest["splits"]["train"]["count"] == 2
    assert manifest["created"].endswith("Z")


def test_manifest_is_utf8_and_human_readable(tmp_path: Path) -> None:
    """The manifest is indented and keeps non-ASCII characters intact."""
    write_manifest(
        tmp_path,
        version="0.1.0",
        config={"note": "µV/√Hz"},
        label_map={},
        source_files=[],
        splits={},
    )
    raw = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8")

    assert "µV/√Hz" in raw
    assert "\n  " in raw
    assert json.loads(raw)["config"]["note"] == "µV/√Hz"


def test_missing_manifest_points_at_the_fix(tmp_path: Path) -> None:
    """A missing manifest must say what to do about it."""
    with pytest.raises(IQForgeError) as exc:
        read_manifest(tmp_path)
    assert "iqforge build" in str(exc.value)


def test_corrupt_manifest_is_reported(tmp_path: Path) -> None:
    """Malformed JSON must not be swallowed."""
    (tmp_path / MANIFEST_NAME).write_text("{broken", encoding="utf-8")
    with pytest.raises(IQForgeError, match="not valid JSON"):
        read_manifest(tmp_path)


def test_dataset_size_counts_every_file(tmp_path: Path) -> None:
    """Disk usage must include files in subdirectories."""
    (tmp_path / "train").mkdir()
    (tmp_path / "train/shard_0000.npy").write_bytes(b"x" * 100)
    (tmp_path / MANIFEST_NAME).write_bytes(b"y" * 50)

    assert dataset_size_bytes(tmp_path) == 150
