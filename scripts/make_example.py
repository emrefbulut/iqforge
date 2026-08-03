"""examples/ içindeki sentetik SigMF örnek kaydını üretir.

Bu script Faz 1'de **bir kez** çalıştırılır; üretilen `examples/sample.sigmf-*`
dosyaları sonraki fazlarda sabit referans kabul edilir ve yeniden üretilmez.

Kaydın içeriği:
  * Sürekli saf ton — merkez frekanstan tam +100 kHz kaymış. Faz 2'deki
    spektrogram doğrulaması bu bilinen sinyali referans alır.
  * BPSK bursti  — merkez frekanstan -200 kHz kaymış, 0.025 s … 0.225 s.
  * QPSK bursti  — merkez frekanstan +250 kHz kaymış, 0.2625 s … 0.4625 s.
  * Düşük seviyeli kompleks Gauss gürültüsü.

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

BPSK_OFFSET = -200_000.0
BPSK_START, BPSK_COUNT = 25_600, 204_800
BPSK_SYMBOL_RATE = 64_000.0

QPSK_OFFSET = 250_000.0
QPSK_START, QPSK_COUNT = 268_800, 204_800
QPSK_SYMBOL_RATE = 128_000.0

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


def _shaped_burst(
    symbols: np.ndarray, sps: int, length: int, rng: np.random.Generator
) -> np.ndarray:
    """Sembolleri RRC ile şekillendirip istenen uzunlukta bir burst döndürür."""
    del rng  # şekillendirme deterministik; semboller çağıran tarafından üretilir
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    shaped = np.convolve(upsampled, rrc_taps(sps), mode="same")
    if len(shaped) < length:
        shaped = np.pad(shaped, (0, length - len(shaped)))
    return shaped[:length]


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

    # BPSK bursti
    sps = int(round(SAMPLE_RATE / BPSK_SYMBOL_RATE))
    n_sym = BPSK_COUNT // sps + 8
    bits = rng.integers(0, 2, n_sym)
    bpsk = _shaped_burst(2.0 * bits - 1.0 + 0j, sps, BPSK_COUNT, rng)
    bpsk *= _envelope(BPSK_COUNT, 512)
    seg = slice(BPSK_START, BPSK_START + BPSK_COUNT)
    x[seg] += 0.55 * bpsk * np.exp(2j * np.pi * BPSK_OFFSET * t[seg])

    # QPSK bursti
    sps = int(round(SAMPLE_RATE / QPSK_SYMBOL_RATE))
    n_sym = QPSK_COUNT // sps + 8
    quad = rng.integers(0, 4, n_sym)
    qpsk_syms = np.exp(1j * (np.pi / 4.0 + quad * np.pi / 2.0))
    qpsk = _shaped_burst(qpsk_syms, sps, QPSK_COUNT, rng)
    qpsk *= _envelope(QPSK_COUNT, 512)
    seg = slice(QPSK_START, QPSK_START + QPSK_COUNT)
    x[seg] += 0.55 * qpsk * np.exp(2j * np.pi * QPSK_OFFSET * t[seg])

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
    meta.add_annotation(
        BPSK_START,
        BPSK_COUNT,
        metadata={
            sigmf.LABEL_KEY: "bpsk",
            sigmf.FREQ_LOWER_EDGE_KEY: CENTER_FREQ + BPSK_OFFSET - BPSK_SYMBOL_RATE * 0.675,
            sigmf.FREQ_UPPER_EDGE_KEY: CENTER_FREQ + BPSK_OFFSET + BPSK_SYMBOL_RATE * 0.675,
            sigmf.COMMENT_KEY: "BPSK, 64 kBd, RRC beta=0.35",
        },
    )
    meta.add_annotation(
        QPSK_START,
        QPSK_COUNT,
        metadata={
            sigmf.LABEL_KEY: "qpsk",
            sigmf.FREQ_LOWER_EDGE_KEY: CENTER_FREQ + QPSK_OFFSET - QPSK_SYMBOL_RATE * 0.675,
            sigmf.FREQ_UPPER_EDGE_KEY: CENTER_FREQ + QPSK_OFFSET + QPSK_SYMBOL_RATE * 0.675,
            sigmf.COMMENT_KEY: "QPSK, 128 kBd, RRC beta=0.35",
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


if __name__ == "__main__":
    main()
