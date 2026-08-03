"""examples/ içindeki sentetik SigMF örnek kaydını üretir.

Bu script Faz 1'de **bir kez** çalıştırılır; üretilen `examples/sample.sigmf-*`
dosyaları sonraki fazlarda sabit referans kabul edilir ve yeniden üretilmez.

Kaydın içeriği:
  * Sürekli saf ton — merkez frekanstan tam +100 kHz kaymış. Faz 2'deki
    spektrogram doğrulaması bu bilinen sinyali referans alır.
  * BPSK bursti  — merkez frekanstan -200 kHz kaymış, 0.025 s … 0.225 s.
  * QPSK bursti  — merkez frekanstan +250 kHz kaymış, 0.2625 s … 0.4625 s.
  * Düşük seviyeli kompleks Gauss gürültüsü.

İki burst **yalnızca modülasyon türüyle** ayrışır: sembol hızı (64 kBd), RRC
roll-off (beta=0.35), dolayısıyla işgal edilen bant genişliği (86.4 kHz), süre
(204800 örnek) ve ortalama güç ikisinde de aynıdır. Yalnızca taşıyıcı ofseti
farklıdır. Böylece sınıflandırıcı bant genişliği veya güç gibi kestirme
ipuçlarıyla değil, gerçekten takımyıldız yapısıyla ayırmak zorunda kalır.

Örnekleme hızı bilerek 1.024 MHz seçilmiştir: 1024 noktalı FFT'de bin aralığı
tam 1 kHz olur, böylece +100 kHz'lik referans ton tam olarak +100. bine düşer.

Kullanım:
    python scripts/make_example.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import sigmf
from sigmf import SigMFFile

SEED = 20240101
SAMPLE_RATE = 1_024_000.0  # Hz
CENTER_FREQ = 2_450_000_000.0  # Hz
NUM_SAMPLES = 512_000  # 0.5 s

REF_TONE_OFFSET = 100_000.0  # Hz — bilinen referans sinyal
REF_TONE_AMPLITUDE = 0.35

# İki burst için ortak parametreler: aynı sembol hızı ve aynı roll-off, yani
# aynı bant genişliği; aynı süre; aynı ortalama güç. Tek fark modülasyon türü.
SYMBOL_RATE = 64_000.0  # Bd
RRC_BETA = 0.35
BURST_RMS = 0.30  # her iki burst de bu ortalama güce normalize edilir
BURST_COUNT = 204_800  # örnek — her iki burst için aynı
BURST_RAMP = 512  # örnek — zarf yükselme/düşme süresi

#: İşgal edilen bant genişliği (Hz); annotation frekans sınırları buradan gelir.
OCCUPIED_BW = SYMBOL_RATE * (1.0 + RRC_BETA)

BPSK_OFFSET = -200_000.0
BPSK_START = 25_600

QPSK_OFFSET = 250_000.0
QPSK_START = 268_800

NOISE_SIGMA = 0.02
PEAK_TARGET = 0.9

OUT_DIR = Path(__file__).resolve().parent.parent / "examples"
OUT_NAME = "sample"


def rrc_taps(sps: int, span: int = 8, beta: float = 0.35) -> np.ndarray:
    """Kök yükseltilmiş kosinüs (RRC) darbe şekillendirme katsayıları üretir.

    Args:
        sps: Sembol başına örnek sayısı.
        span: Filtrenin kaç sembol boyunca uzandığı.
        beta: Roll-off faktörü.

    Returns:
        Enerjisi 1'e normalize edilmiş float64 katsayı dizisi.
    """
    n = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=np.float64)
    t = n / sps
    taps = np.empty_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            taps[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(
                np.pi * ti * (1.0 + beta)
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps**2))


def _shaped_burst(symbols: np.ndarray, sps: int, length: int) -> np.ndarray:
    """Sembolleri RRC ile şekillendirip istenen uzunlukta bir burst döndürür.

    Burst, kenarları yumuşatılmış zarfla çarpılır ve ortalama gücü tam olarak
    `BURST_RMS**2` olacak şekilde normalize edilir. Böylece farklı modülasyonlar
    birebir aynı güçte olur.

    Args:
        symbols: Kompleks sembol dizisi.
        sps: Sembol başına örnek sayısı.
        length: İstenen burst uzunluğu (örnek).

    Returns:
        `length` uzunluğunda, ortalama gücü `BURST_RMS**2` olan complex128 dizi.
    """
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    shaped = np.convolve(upsampled, rrc_taps(sps, beta=RRC_BETA), mode="same")
    if len(shaped) < length:
        shaped = np.pad(shaped, (0, length - len(shaped)))
    burst = shaped[:length] * _envelope(length, BURST_RAMP)
    return burst * (BURST_RMS / np.sqrt(np.mean(np.abs(burst) ** 2)))


def _envelope(length: int, ramp: int) -> np.ndarray:
    """Burst kenarlarını yumuşatan kosinüs rampalı zarf üretir."""
    env = np.ones(length, dtype=np.float64)
    r = np.hanning(2 * ramp)
    env[:ramp] = r[:ramp]
    env[-ramp:] = r[ramp:]
    return env


def build_signal() -> np.ndarray:
    """Sentetik kaydın kompleks örneklerini üretir."""
    rng = np.random.default_rng(SEED)
    t = np.arange(NUM_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    x = np.zeros(NUM_SAMPLES, dtype=np.complex128)

    # Referans ton: merkez frekanstan tam +100 kHz, kaydın tamamı boyunca.
    x += REF_TONE_AMPLITUDE * np.exp(2j * np.pi * REF_TONE_OFFSET * t)

    # Her iki burst de aynı sembol hızında, aynı uzunlukta ve aynı güçte.
    sps = int(round(SAMPLE_RATE / SYMBOL_RATE))
    n_sym = BURST_COUNT // sps + 8

    # BPSK: takımyıldız {-1, +1}
    bits = rng.integers(0, 2, n_sym)
    bpsk = _shaped_burst(2.0 * bits - 1.0 + 0j, sps, BURST_COUNT)
    seg = slice(BPSK_START, BPSK_START + BURST_COUNT)
    x[seg] += bpsk * np.exp(2j * np.pi * BPSK_OFFSET * t[seg])

    # QPSK: takımyıldız {e^{j(pi/4 + k*pi/2)}}
    quad = rng.integers(0, 4, n_sym)
    qpsk = _shaped_burst(np.exp(1j * (np.pi / 4.0 + quad * np.pi / 2.0)), sps, BURST_COUNT)
    seg = slice(QPSK_START, QPSK_START + BURST_COUNT)
    x[seg] += qpsk * np.exp(2j * np.pi * QPSK_OFFSET * t[seg])

    # Gürültü tabanı
    x += NOISE_SIGMA * (rng.standard_normal(NUM_SAMPLES) + 1j * rng.standard_normal(NUM_SAMPLES))

    return (x * (PEAK_TARGET / np.max(np.abs(x)))).astype(np.complex64)


def write_record(samples: np.ndarray, out_dir: Path, name: str) -> Path:
    """Örnekleri ve metadata'yı SigMF kayıt çifti olarak diske yazar.

    Args:
        samples: `complex64` örnekler.
        out_dir: Çıktı klasörü.
        name: Uzantısız kayıt adı.

    Returns:
        Yazılan `.sigmf-meta` dosyasının yolu.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{name}.sigmf-data"
    meta_path = out_dir / f"{name}.sigmf-meta"

    interleaved = np.empty(samples.size * 2, dtype=np.float32)
    interleaved[0::2] = samples.real
    interleaved[1::2] = samples.imag
    interleaved.tofile(data_path)

    meta = SigMFFile(
        data_file=str(data_path),
        global_info={
            sigmf.DATATYPE_KEY: "cf32_le",
            sigmf.SAMPLE_RATE_KEY: SAMPLE_RATE,
            sigmf.AUTHOR_KEY: "sigkit",
            sigmf.DESCRIPTION_KEY: (
                "sigkit sentetik örnek kaydı. Sürekli referans ton (merkez +100 kHz), "
                "BPSK ve QPSK burstleri, düşük seviyeli AWGN."
            ),
            sigmf.HW_KEY: "synthetic (scripts/make_example.py)",
            sigmf.RECORDER_KEY: "sigkit scripts/make_example.py",
            sigmf.VERSION_KEY: "1.0.0",
        },
    )
    meta.add_capture(
        0,
        metadata={
            sigmf.FREQUENCY_KEY: CENTER_FREQ,
            sigmf.DATETIME_KEY: dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
            .isoformat()
            .replace("+00:00", "Z"),
        },
    )

    meta.add_annotation(
        0,
        NUM_SAMPLES,
        metadata={
            sigmf.LABEL_KEY: "ref_tone",
            sigmf.FREQ_LOWER_EDGE_KEY: CENTER_FREQ + REF_TONE_OFFSET - 500.0,
            sigmf.FREQ_UPPER_EDGE_KEY: CENTER_FREQ + REF_TONE_OFFSET + 500.0,
            sigmf.COMMENT_KEY: (
                "Bilinen referans sinyal: merkez frekanstan tam +100000 Hz kaymış saf ton, "
                "kaydın tamamı boyunca sürekli. Spektrogram doğrulaması bunu kullanır."
            ),
        },
    )
    burst_comment = (
        f"{SYMBOL_RATE / 1e3:.0f} kBd, RRC beta={RRC_BETA}, "
        f"bant genişliği {OCCUPIED_BW / 1e3:.1f} kHz, ortalama güç {BURST_RMS**2:.4f}"
    )
    for label, offset, start in (
        ("bpsk", BPSK_OFFSET, BPSK_START),
        ("qpsk", QPSK_OFFSET, QPSK_START),
    ):
        meta.add_annotation(
            start,
            BURST_COUNT,
            metadata={
                sigmf.LABEL_KEY: label,
                sigmf.FREQ_LOWER_EDGE_KEY: CENTER_FREQ + offset - OCCUPIED_BW / 2.0,
                sigmf.FREQ_UPPER_EDGE_KEY: CENTER_FREQ + offset + OCCUPIED_BW / 2.0,
                sigmf.COMMENT_KEY: f"{label.upper()}, {burst_comment}",
            },
        )

    meta.tofile(str(meta_path), skip_validate=False, pretty=True)
    return meta_path


def main() -> None:
    """Sentetik kaydı üretir ve özetini yazdırır.

    Kayıt zaten varsa hiçbir şey yapmaz: örnek kayıt Faz 1'de bir kez üretilir ve
    sonraki fazlarda sabit kalır. Bilinçli olarak yeniden üretmek için `--force`.
    """
    force = "--force" in sys.argv
    existing = OUT_DIR / f"{OUT_NAME}.sigmf-meta"
    if existing.exists() and not force:
        print(f"{existing} zaten var — üretim atlandı.")
        print("Bilerek yeniden üretmek için: python scripts/make_example.py --force")
        return
    if force:
        for suffix in (".sigmf-meta", ".sigmf-data"):
            (OUT_DIR / f"{OUT_NAME}{suffix}").unlink(missing_ok=True)

    samples = build_signal()
    meta_path = write_record(samples, OUT_DIR, OUT_NAME)
    data_path = meta_path.with_suffix(".sigmf-data")
    print(f"yazıldı: {meta_path}")
    print(f"yazıldı: {data_path} ({data_path.stat().st_size / 1e6:.2f} MB)")
    print(f"örnek sayısı: {samples.size}, süre: {samples.size / SAMPLE_RATE:.4f} s")
    print(f"referans ton: merkez {REF_TONE_OFFSET:+.0f} Hz")
    print(
        f"burstler: {SYMBOL_RATE / 1e3:.0f} kBd, "
        f"bant genişliği {OCCUPIED_BW / 1e3:.1f} kHz, {BURST_COUNT} örnek (her ikisi de)"
    )
    for label, start in (("bpsk", BPSK_START), ("qpsk", QPSK_START)):
        seg = samples[start : start + BURST_COUNT]
        t0, t1 = start / SAMPLE_RATE, (start + BURST_COUNT) / SAMPLE_RATE
        print(
            f"  {label}: örnek {start}..{start + BURST_COUNT} "
            f"({t0:.4f}..{t1:.4f} s), segment gücü {np.mean(np.abs(seg) ** 2):.6f} "
            f"(ton ve gürültü dahil — ikisinde de aynı)"
        )


if __name__ == "__main__":
    main()
