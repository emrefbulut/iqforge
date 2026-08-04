"""Testlerin paylaştığı SigMF kayıt üretme yardımcısı."""

from __future__ import annotations

import json
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
) -> Path:
    """Testler için elle bir SigMF kayıt çifti yazar ve meta yolunu döndürür.

    Args:
        directory: Yazılacak klasör; yoksa oluşturulur.
        samples: Kompleks örnekler.
        datatype: `core:datatype` değeri. Desteklenmeyen bir değer verilebilir;
            hata yollarını sınamak için veri `cf32_le` gibi yazılır.
        name: Uzantısız dosya adı.
        sample_rate: `core:sample_rate`; None ise alan hiç yazılmaz.
        center_freq: `core:frequency`; None ise alan hiç yazılmaz.
        annotations: Ham annotation sözlükleri.

    Returns:
        Yazılan `.sigmf-meta` yolu.
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

    meta_path = directory / f"{name}.sigmf-meta"
    meta_path.write_text(
        json.dumps(
            {"global": global_info, "captures": [capture], "annotations": annotations or []}
        ),
        encoding="utf-8",
    )
    return meta_path
