"""Fixtures shared across the test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from helpers import write_record as _write_record
from iqforge.io import Recording, load


@pytest.fixture
def make_recording() -> Callable[..., Recording]:
    """Factory that writes a synthetic recording and returns it opened."""

    def _make(directory: Path, samples: np.ndarray, **kwargs) -> Recording:
        return load(_write_record(directory, samples, **kwargs))

    return _make


@pytest.fixture
def noise() -> Callable[..., np.ndarray]:
    """Factory producing deterministic complex noise."""

    def _make(n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)

    return _make
