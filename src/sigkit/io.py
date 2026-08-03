"""SigMF kayıtlarının okunması ve veri tipi dönüşümleri.

Metadata ayrıştırması `sigmf` (sigmf-python) kütüphanesine bırakılır; bu modül
yalnızca ham örnek verisini `complex64` olarak, bellek dostu biçimde sunar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sigmf import SigMFFile

#: Desteklenen `core:datatype` değerleri -> (numpy dtype, tam ölçek böleni)
SUPPORTED_DATATYPES: dict[str, tuple[str, float]] = {
    "cf32_le": ("<f4", 1.0),
    "ci16_le": ("<i2", 32768.0),
    "ci8": ("i1", 128.0),
}

META_EXT = ".sigmf-meta"
DATA_EXT = ".sigmf-data"


class SigkitError(Exception):
    """sigkit'in kullanıcıya gösterilebilir hataları."""


@dataclass(frozen=True)
class Annotation:
    """Tek bir SigMF annotation kaydı.

    Attributes:
        sample_start: Annotation'ın başladığı örnek indisi.
        sample_count: Annotation'ın kapsadığı örnek sayısı.
        label: `core:label` alanı; yoksa None.
        freq_lower_edge: Alt frekans sınırı (Hz); yoksa None.
        freq_upper_edge: Üst frekans sınırı (Hz); yoksa None.
        description: `core:description` alanı; yoksa None.
    """

    sample_start: int
    sample_count: int
    label: str | None = None
    freq_lower_edge: float | None = None
    freq_upper_edge: float | None = None
    description: str | None = None

    @property
    def sample_end(self) -> int:
        """Annotation'ın bittiği (dahil olmayan) örnek indisi."""
        return self.sample_start + self.sample_count


@dataclass
class Recording:
    """Açılmış bir SigMF kayıt çifti (`.sigmf-meta` + `.sigmf-data`).

    Örnek verisi `numpy.memmap` üzerinden tembel okunur; dosyanın tamamı
    belleğe alınmaz.
    """

    meta_path: Path
    data_path: Path
    datatype: str
    sample_rate: float
    center_frequency: float | None
    num_samples: int
    annotations: list[Annotation]
    global_info: dict[str, Any]

    @property
    def duration_seconds(self) -> float:
        """Kaydın saniye cinsinden süresi."""
        return self.num_samples / self.sample_rate

    def read(self, start: int = 0, count: int | None = None) -> np.ndarray:
        """Kayıttan `complex64` örnekler okur.

        Args:
            start: Okumaya başlanacak örnek indisi.
            count: Okunacak örnek sayısı; None ise kaydın sonuna kadar.

        Returns:
            `complex64` tipinde tek boyutlu dizi.

        Raises:
            SigkitError: `start` kayıt sınırlarının dışındaysa.
        """
        if start < 0 or start > self.num_samples:
            raise SigkitError(
                f"Başlangıç indisi {start} kayıt sınırlarının dışında. "
                f"Geçerli aralık: 0..{self.num_samples}."
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
    """Verilen yoldan meta ve data dosya yollarını türetir."""
    p = Path(path)
    if p.suffix == META_EXT:
        meta = p
    elif p.suffix == DATA_EXT:
        meta = p.with_suffix(META_EXT)
    else:
        meta = Path(str(p) + META_EXT)

    if not meta.exists():
        raise SigkitError(
            f"SigMF metadata dosyası bulunamadı: {meta}. "
            f"Bir '{META_EXT}' dosyası veya uzantısız kayıt adı verin."
        )
    data = meta.with_suffix(DATA_EXT)
    if not data.exists():
        raise SigkitError(
            f"SigMF veri dosyası bulunamadı: {data}. "
            f"'{meta.name}' ile aynı klasörde '{data.name}' bulunmalı."
        )
    return meta, data


def load(path: str | Path) -> Recording:
    """Bir SigMF kaydını açar ve metadata'sını doğrular.

    Args:
        path: `.sigmf-meta` dosyası, `.sigmf-data` dosyası veya uzantısız kayıt adı.

    Returns:
        Açılmış `Recording`.

    Raises:
        SigkitError: Dosya yoksa, veri tipi desteklenmiyorsa veya zorunlu
            metadata alanları eksikse.
    """
    meta_path, data_path = _resolve_paths(path)

    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SigkitError(
            f"'{meta_path.name}' geçerli JSON değil: {exc}. "
            "SigMF metadata dosyası UTF-8 kodlu bir JSON nesnesi olmalı."
        ) from exc

    # Şema doğrulaması sigmf kütüphanesine bırakılır; örnek verisini bu modül
    # memmap ile kendi okur, bu yüzden veri dosyası kütüphaneye bağlanmaz.
    try:
        handle = SigMFFile(metadata=raw)
    except Exception as exc:  # sigmf çeşitli hata tipleri fırlatabilir
        raise SigkitError(f"SigMF metadata okunamadı ({meta_path}): {exc}") from exc

    global_info = dict(handle.get_global_info())

    datatype = global_info.get("core:datatype")
    if datatype is None:
        raise SigkitError(
            f"'{meta_path.name}' içinde zorunlu 'core:datatype' alanı yok. "
            f"Desteklenenler: {', '.join(SUPPORTED_DATATYPES)}."
        )
    if datatype not in SUPPORTED_DATATYPES:
        raise SigkitError(
            f"Desteklenmeyen veri tipi '{datatype}'. "
            f"Desteklenenler: {', '.join(SUPPORTED_DATATYPES)}."
        )

    sample_rate = global_info.get("core:sample_rate")
    if sample_rate is None:
        raise SigkitError(
            f"'{meta_path.name}' içinde 'core:sample_rate' yok. "
            "Örnekleme hızı olmadan zaman/frekans ekseni hesaplanamaz; "
            "metadata'ya bu alanı ekleyin."
        )

    np_dtype, _ = SUPPORTED_DATATYPES[datatype]
    bytes_per_sample = 2 * np.dtype(np_dtype).itemsize
    file_bytes = data_path.stat().st_size
    if file_bytes % bytes_per_sample != 0:
        raise SigkitError(
            f"'{data_path.name}' boyutu ({file_bytes} bayt) '{datatype}' için örnek "
            f"başına {bytes_per_sample} bayta tam bölünmüyor. Dosya bozuk olabilir."
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
    )
