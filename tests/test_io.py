"""iqforge.io testleri. Tümü sentetik veriyle çalışır, ağ erişimi gerektirmez."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import write_record
from iqforge.io import Annotation, IQForgeError, Recording, load

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.sigmf-meta"))


def _burst(rec: Recording) -> Annotation:
    """Kaydın tek modülasyonlu burst annotation'ını verir (ref_tone hariç)."""
    bursts = [a for a in rec.annotations if a.label != "ref_tone"]
    assert len(bursts) == 1, f"{rec.meta_path.name}: tam olarak bir burst bekleniyordu"
    return bursts[0]


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

    with pytest.raises(IQForgeError, match="sınırlarının dışında"):
        rec.read(start=-1)


def test_unsupported_datatype_is_explicit(tmp_path: Path, tone: np.ndarray) -> None:
    """Desteklenmeyen veri tipi sessizce tahmin edilmez; mesaj eylem içerir."""
    meta = write_record(tmp_path, tone, datatype="cf64_le")
    with pytest.raises(IQForgeError) as exc:
        load(meta)
    message = str(exc.value)
    assert "cf64_le" in message
    assert "cf32_le" in message and "ci16_le" in message and "ci8" in message


def test_missing_sample_rate_is_an_error(tmp_path: Path, tone: np.ndarray) -> None:
    """Örnekleme hızı yoksa varsayılan uydurulmaz, hata verilir."""
    meta = write_record(tmp_path, tone, sample_rate=None)
    with pytest.raises(IQForgeError, match="core:sample_rate"):
        load(meta)


def test_missing_files_are_reported(tmp_path: Path, tone: np.ndarray) -> None:
    """Eksik meta/veri dosyaları için ayrı ayrı açık hata verilir."""
    with pytest.raises(IQForgeError, match="metadata dosyası bulunamadı"):
        load(tmp_path / "yok.sigmf-meta")

    meta = write_record(tmp_path, tone)
    meta.with_suffix(".sigmf-data").unlink()
    with pytest.raises(IQForgeError, match="veri dosyası bulunamadı"):
        load(meta)


def test_truncated_data_file_is_rejected(tmp_path: Path, tone: np.ndarray) -> None:
    """Örnek boyutuna bölünmeyen veri dosyası bozuk sayılır."""
    meta = write_record(tmp_path, tone)
    data = meta.with_suffix(".sigmf-data")
    data.write_bytes(data.read_bytes()[:-3])
    with pytest.raises(IQForgeError, match="bozuk"):
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


@pytest.mark.skipif(not EXAMPLES, reason="examples/ kayıtları üretilmemiş")
def test_example_set_has_eight_records_balanced_by_class() -> None:
    """Örnek veri seti dört bpsk + dört qpsk kayıttan oluşmalı.

    Kayıt sayısı SPEC §5.6'nın kayıt bazında bölme kuralı için kritiktir: sınıf
    başına dört kayıt, 0.7/0.15/0.15 bölmesinin üç split'i de doldurmasına yeter.
    """
    labels = [_burst(load(p)).label for p in EXAMPLES]

    assert len(EXAMPLES) == 8
    assert labels.count("bpsk") == 4
    assert labels.count("qpsk") == 4
    assert len({p.stem for p in EXAMPLES}) == 8, "kayıt adları benzersiz olmalı"


@pytest.mark.skipif(not EXAMPLES, reason="examples/ kayıtları üretilmemiş")
def test_example_records_share_metadata_and_fit_size_budget() -> None:
    """Her kayıt aynı temel metadata'ya sahip ve toplam 5 MB'ın altında."""
    total = 0
    for path in EXAMPLES:
        rec = load(path)
        assert rec.datatype == "cf32_le"
        assert rec.sample_rate == 1_024_000.0
        assert rec.center_frequency == 2_450_000_000.0
        assert rec.num_samples == 65_536
        assert rec.duration_seconds == pytest.approx(0.064)
        assert {a.label for a in rec.annotations} == {"ref_tone", _burst(rec).label}
        total += rec.data_path.stat().st_size + rec.meta_path.stat().st_size

    assert total < 5_000_000, f"örnek veri seti 5 MB'ı aşıyor: {total / 1e6:.2f} MB"


@pytest.mark.skipif(not EXAMPLES, reason="examples/ kayıtları üretilmemiş")
def test_example_bursts_are_equal_in_bandwidth_and_duration() -> None:
    """Tüm burstler aynı bant genişliğinde ve aynı sürede olmalı.

    Sınıflar yalnızca modülasyonla ayrışmalı; bant genişliği veya süre farkı
    sınıflandırıcıya kısayol verir.
    """
    widths = {_burst(load(p)).freq_upper_edge - _burst(load(p)).freq_lower_edge for p in EXAMPLES}
    counts = {_burst(load(p)).sample_count for p in EXAMPLES}

    assert widths == {86_400.0}
    assert counts == {40_960}


@pytest.mark.skipif(not EXAMPLES, reason="examples/ kayıtları üretilmemiş")
def test_carrier_offset_carries_no_class_information() -> None:
    """İki sınıf aynı taşıyıcı ofset havuzunu kullanmalı.

    Aksi halde ağ modülasyonu değil taşıyıcı frekansını öğrenir ve Faz 4'teki
    doğruluk ölçümü anlamsızlaşır.
    """
    by_class: dict[str, set[float]] = {}
    for path in EXAMPLES:
        rec = load(path)
        a = _burst(rec)
        centre = (a.freq_lower_edge + a.freq_upper_edge) / 2 - rec.center_frequency
        by_class.setdefault(a.label, set()).add(round(centre))

    assert by_class["bpsk"] == by_class["qpsk"], (
        f"taşıyıcı ofsetleri sınıflar arasında farklı: {by_class}"
    )
    assert len(by_class["bpsk"]) == 4


@pytest.mark.skipif(not EXAMPLES, reason="examples/ kayıtları üretilmemiş")
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_reference_tone_is_exactly_plus_100_khz(path: Path) -> None:
    """Her kayıtta referans ton merkez frekanstan tam +100 kHz'te olmalı.

    Sonraki fazların (özellikle spektrogram doğrulamasının) dayandığı sabit budur.
    """
    rec = load(path)
    ref = next(a for a in rec.annotations if a.label == "ref_tone")
    assert ref.sample_start == 0
    assert ref.sample_count == rec.num_samples
    assert (ref.freq_lower_edge + ref.freq_upper_edge) / 2 - rec.center_frequency == 100_000.0

    # Burstün bittiği, yalnız tonun bulunduğu sessiz kuyruk.
    quiet_start = _burst(rec).sample_end
    quiet = rec.read(start=quiet_start, count=rec.num_samples - quiet_start)
    assert quiet.size >= 8192, "ton ölçümü için yeterli sessiz bölge yok"

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(quiet)))
    freqs = np.fft.fftshift(np.fft.fftfreq(quiet.size, d=1.0 / rec.sample_rate))
    bin_width = rec.sample_rate / quiet.size

    # Tepe İŞARETLİ olarak +100 kHz'te olmalı. I/Q yer değiştirirse (x -> j*conj(x))
    # ton -100 kHz'e taşınır; bu yüzden |frekans| değil, işaretli frekans kontrol edilir.
    peak_offset = freqs[int(np.argmax(spectrum))]
    assert peak_offset == pytest.approx(100_000.0, abs=bin_width)
    assert peak_offset > 0, f"Referans ton negatif frekansta bulundu ({peak_offset:.0f} Hz)"

    # Ayna bin (-100 kHz) belirgin biçimde zayıf olmalı: I/Q takasına karşı
    # tek başına yeterli olan asıl kontrol budur.
    power_plus = spectrum[int(np.argmin(np.abs(freqs - 100_000.0)))]
    power_minus = spectrum[int(np.argmin(np.abs(freqs + 100_000.0)))]
    assert power_plus > 100.0 * power_minus, (
        f"+100 kHz / -100 kHz güç oranı yetersiz: {power_plus / power_minus:.1f}x"
    )
