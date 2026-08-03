"""sigkit.io testleri. Tümü sentetik veriyle çalışır, ağ erişimi gerektirmez."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigkit.io import SUPPORTED_DATATYPES, SigkitError, load

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample.sigmf-meta"


def write_record(
    tmp_path: Path,
    samples: np.ndarray,
    datatype: str = "cf32_le",
    *,
    name: str = "rec",
    sample_rate: float | None = 1_000_000.0,
    center_freq: float | None = 100_000_000.0,
    annotations: list[dict] | None = None,
) -> Path:
    """Testler için elle bir SigMF kayıt çifti yazar ve meta yolunu döndürür."""
    np_dtype, full_scale = SUPPORTED_DATATYPES.get(datatype, ("<f4", 1.0))
    interleaved = np.empty(samples.size * 2, dtype=np.float64)
    interleaved[0::2] = samples.real
    interleaved[1::2] = samples.imag
    (interleaved * full_scale).astype(np_dtype).tofile(tmp_path / f"{name}.sigmf-data")

    global_info: dict = {"core:datatype": datatype, "core:version": "1.0.0"}
    if sample_rate is not None:
        global_info["core:sample_rate"] = sample_rate

    capture: dict = {"core:sample_start": 0}
    if center_freq is not None:
        capture["core:frequency"] = center_freq

    meta = {
        "global": global_info,
        "captures": [capture],
        "annotations": annotations or [],
    }
    meta_path = tmp_path / f"{name}.sigmf-meta"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return meta_path


@pytest.fixture
def tone() -> np.ndarray:
    """Küçük, deterministik bir kompleks test sinyali."""
    n = np.arange(4096)
    return (0.5 * np.exp(2j * np.pi * 0.05 * n)).astype(np.complex64)


@pytest.mark.parametrize("datatype", ["cf32_le", "ci16_le", "ci8"])
def test_roundtrip_all_supported_datatypes(tmp_path: Path, tone: np.ndarray, datatype: str) -> None:
    """Desteklenen üç veri tipi de complex64 olarak doğru ölçekte geri okunur."""
    meta = write_record(tmp_path, tone, datatype)
    rec = load(meta)

    assert rec.datatype == datatype
    assert rec.num_samples == tone.size
    samples = rec.read()
    assert samples.dtype == np.complex64
    tolerance = {"cf32_le": 1e-6, "ci16_le": 1e-4, "ci8": 1e-2}[datatype]
    assert np.allclose(samples, tone, atol=tolerance)


def test_metadata_fields(tmp_path: Path, tone: np.ndarray) -> None:
    """Örnekleme hızı, merkez frekans, örnek sayısı ve süre doğru okunur."""
    meta = write_record(tmp_path, tone, sample_rate=2_000_000.0, center_freq=915e6)
    rec = load(meta)

    assert rec.sample_rate == 2_000_000.0
    assert rec.center_frequency == 915e6
    assert rec.num_samples == 4096
    assert rec.duration_seconds == pytest.approx(4096 / 2_000_000.0)


def test_partial_read(tmp_path: Path, tone: np.ndarray) -> None:
    """start/count ile kısmi okuma doğru dilimi verir ve sınırı aşmaz."""
    meta = write_record(tmp_path, tone)
    rec = load(meta)

    chunk = rec.read(start=1000, count=256)
    assert chunk.size == 256
    assert np.allclose(chunk, tone[1000:1256], atol=1e-6)

    assert rec.read(start=4000, count=10_000).size == 96
    assert rec.read(start=rec.num_samples).size == 0

    with pytest.raises(SigkitError, match="sınırlarının dışında"):
        rec.read(start=-1)


def test_unsupported_datatype_is_explicit(tmp_path: Path, tone: np.ndarray) -> None:
    """Desteklenmeyen veri tipi sessizce tahmin edilmez; mesaj eylem içerir."""
    meta = write_record(tmp_path, tone, datatype="cf64_le")
    with pytest.raises(SigkitError) as exc:
        load(meta)
    message = str(exc.value)
    assert "cf64_le" in message
    assert "cf32_le" in message and "ci16_le" in message and "ci8" in message


def test_missing_sample_rate_is_an_error(tmp_path: Path, tone: np.ndarray) -> None:
    """Örnekleme hızı yoksa varsayılan uydurulmaz, hata verilir."""
    meta = write_record(tmp_path, tone, sample_rate=None)
    with pytest.raises(SigkitError, match="core:sample_rate"):
        load(meta)


def test_missing_files_are_reported(tmp_path: Path, tone: np.ndarray) -> None:
    """Eksik meta/veri dosyaları için ayrı ayrı açık hata verilir."""
    with pytest.raises(SigkitError, match="metadata dosyası bulunamadı"):
        load(tmp_path / "yok.sigmf-meta")

    meta = write_record(tmp_path, tone)
    meta.with_suffix(".sigmf-data").unlink()
    with pytest.raises(SigkitError, match="veri dosyası bulunamadı"):
        load(meta)


def test_truncated_data_file_is_rejected(tmp_path: Path, tone: np.ndarray) -> None:
    """Örnek boyutuna bölünmeyen veri dosyası bozuk sayılır."""
    meta = write_record(tmp_path, tone)
    data = meta.with_suffix(".sigmf-data")
    data.write_bytes(data.read_bytes()[:-3])
    with pytest.raises(SigkitError, match="bozuk"):
        load(meta)


def test_annotations_are_parsed_and_sorted(tmp_path: Path, tone: np.ndarray) -> None:
    """Annotation'lar okunur ve başlangıç indisine göre sıralanır."""
    meta = write_record(
        tmp_path,
        tone,
        annotations=[
            {"core:sample_start": 2000, "core:sample_count": 500, "core:label": "b"},
            {
                "core:sample_start": 0,
                "core:sample_count": 1000,
                "core:label": "a",
                "core:freq_lower_edge": 99e6,
                "core:freq_upper_edge": 101e6,
            },
        ],
    )
    rec = load(meta)

    assert [a.label for a in rec.annotations] == ["a", "b"]
    assert rec.annotations[0].sample_end == 1000
    assert rec.annotations[0].freq_upper_edge == 101e6


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/sample.sigmf-meta üretilmemiş")
def test_example_record_metadata() -> None:
    """Paketle gelen sentetik kayıt beklenen metadata'ya sahip."""
    rec = load(EXAMPLE)

    assert rec.datatype == "cf32_le"
    assert rec.sample_rate == 1_024_000.0
    assert rec.center_frequency == 2_450_000_000.0
    assert rec.num_samples == 512_000
    assert rec.duration_seconds == pytest.approx(0.5)
    assert rec.data_path.stat().st_size < 5_000_000
    assert {a.label for a in rec.annotations} == {"ref_tone", "bpsk", "qpsk"}


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/sample.sigmf-meta üretilmemiş")
def test_example_reference_tone_is_exactly_plus_100_khz() -> None:
    """Referans ton, merkez frekanstan tam +100 kHz'te olmalı.

    Sonraki fazların (özellikle spektrogram doğrulamasının) dayandığı sabit budur.
    """
    rec = load(EXAMPLE)
    ref = next(a for a in rec.annotations if a.label == "ref_tone")
    assert ref.sample_start == 0
    assert ref.sample_count == rec.num_samples
    assert (ref.freq_lower_edge + ref.freq_upper_edge) / 2 - rec.center_frequency == 100_000.0

    # Modülasyonlu burstlerin olmadığı, yalnız tonun bulunduğu bir bölge seç.
    quiet = rec.read(start=470_000, count=32_768)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(quiet)))
    freqs = np.fft.fftshift(np.fft.fftfreq(quiet.size, d=1.0 / rec.sample_rate))
    peak_offset = freqs[int(np.argmax(spectrum))]

    assert peak_offset == pytest.approx(100_000.0, abs=rec.sample_rate / quiet.size)
