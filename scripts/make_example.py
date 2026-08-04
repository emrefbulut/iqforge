"""examples/ içindeki sentetik SigMF örnek kayıtlarını üretir.

Bu script bir kez çalıştırılır; üretilen `examples/*.sigmf-*` dosyaları sonraki
fazlarda sabit referans kabul edilir ve yeniden üretilmez.

Neden birden fazla kayıt: SPEC §5.6 "aynı kayıt dosyasından gelen pencereler
aynı split'e gitmeli" diyor. Tek bir kayıt dosyası varsa bu kural sınanamaz ve
`build` sessizce pencere bazlı bölmeye düşme riski taşır. Bu yüzden veri seti
sekiz ayrı kayıt çiftinden oluşur: dört BPSK, dört QPSK.

Her kayıtta:
  * Sürekli saf ton — merkez frekanstan tam +100 kHz kaymış (`ref_tone`).
    Faz 2'deki spektrogram doğrulaması bu bilinen sinyali referans alır.
  * Tek bir modülasyonlu burst (`bpsk` VEYA `qpsk`), kaydın kimliğini belirler.
  * Düşük seviyeli kompleks Gauss gürültüsü.

Sınıflar arasında yalnızca modülasyon türü değişir. Kısayol ipuçlarını kapatmak
için şunlar iki sınıfta da birebir aynıdır:
  * sembol hızı (64 kBd) ve RRC roll-off (0.35) → aynı bant genişliği (86.4 kHz)
  * burst süresi (40960 örnek) ve ortalama gücü
  * taşıyıcı ofset havuzu — her sınıf aynı dört ofseti birer kez kullanır, yani
    taşıyıcı frekansı sınıf hakkında hiçbir bilgi taşımaz
Kayıttan kayda değişenler: gürültü tohumu, sembol dizisi, burst zaman konumu ve
taşıyıcı ofseti.

Örnekleme hızı bilerek 1.024 MHz seçilmiştir: 1024 noktalı FFT'de bin aralığı
tam 1 kHz olur, böylece +100 kHz'lik referans ton tam olarak +100. bine düşer.

Kullanım:
    python scripts/make_example.py          # varsa atlar
    python scripts/make_example.py --force  # bilerek yeniden üretir
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sigmf
from sigmf import SigMFFile

BASE_SEED = 20240101
SAMPLE_RATE = 1_024_000.0  # Hz
CENTER_FREQ = 2_450_000_000.0  # Hz
NUM_SAMPLES = 65_536  # kayıt başına 0.064 s

REF_TONE_OFFSET = 100_000.0  # Hz — bilinen referans sinyal
REF_TONE_AMPLITUDE = 0.25

# Her iki sınıf için ortak burst parametreleri.
SYMBOL_RATE = 64_000.0  # Bd
RRC_BETA = 0.35
BURST_RMS = 0.22  # her burst bu ortalama güce normalize edilir
BURST_COUNT = 40_960  # örnek — her kayıtta aynı
BURST_RAMP = 512  # örnek — zarf yükselme/düşme süresi

#: İşgal edilen bant genişliği (Hz); annotation frekans sınırları buradan gelir.
OCCUPIED_BW = SYMBOL_RATE * (1.0 + RRC_BETA)

NOISE_SIGMA = 0.02

OUT_DIR = Path(__file__).resolve().parent.parent / "examples"


@dataclass(frozen=True)
class RecordPlan:
    """Üretilecek tek bir kaydın parametreleri.

    Attributes:
        name: Uzantısız dosya adı; kaydın kimliğini taşır (ör. `bpsk_01`).
        modulation: `"bpsk"` veya `"qpsk"`.
        carrier_offset: Burstün merkez frekansa göre ofseti (Hz).
        burst_start: Burstün başladığı örnek indisi.
        seed: Bu kayda özgü rastgelelik tohumu.
    """

    name: str
    modulation: str
    carrier_offset: float
    burst_start: int
    seed: int


def _build_plans() -> list[RecordPlan]:
    """Sekiz kaydın planını üretir.

    İki sınıf aynı taşıyıcı ofset havuzunu ve aynı burst başlangıç havuzunu
    kullanır; yalnızca eşleşmeleri farklıdır. Böylece ne taşıyıcı frekansı ne de
    burst konumu sınıf hakkında bilgi taşır.
    """
    offsets = (-280_000.0, -180_000.0, 180_000.0, 280_000.0)
    bpsk_starts = (4_096, 12_288, 8_192, 16_384)
    qpsk_starts = (12_288, 4_096, 16_384, 8_192)

    plans: list[RecordPlan] = []
    for modulation, starts in (("bpsk", bpsk_starts), ("qpsk", qpsk_starts)):
        for i, (offset, start) in enumerate(zip(offsets, starts, strict=True), start=1):
            plans.append(
                RecordPlan(
                    name=f"{modulation}_{i:02d}",
                    modulation=modulation,
                    carrier_offset=offset,
                    burst_start=start,
                    seed=BASE_SEED + len(plans),
                )
            )
    return plans


PLANS = _build_plans()


def rrc_taps(sps: int, span: int = 8, beta: float = RRC_BETA) -> np.ndarray:
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


def _envelope(length: int, ramp: int) -> np.ndarray:
    """Burst kenarlarını yumuşatan kosinüs rampalı zarf üretir."""
    env = np.ones(length, dtype=np.float64)
    r = np.hanning(2 * ramp)
    env[:ramp] = r[:ramp]
    env[-ramp:] = r[ramp:]
    return env


def _shaped_burst(symbols: np.ndarray, sps: int, length: int) -> np.ndarray:
    """Sembolleri RRC ile şekillendirip istenen uzunlukta bir burst döndürür.

    Burst, kenarları yumuşatılmış zarfla çarpılır ve ortalama gücü tam olarak
    `BURST_RMS**2` olacak şekilde normalize edilir. Böylece farklı modülasyonlar
    ve farklı kayıtlar birebir aynı güçte olur.

    Args:
        symbols: Kompleks sembol dizisi.
        sps: Sembol başına örnek sayısı.
        length: İstenen burst uzunluğu (örnek).

    Returns:
        `length` uzunluğunda, ortalama gücü `BURST_RMS**2` olan complex128 dizi.
    """
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    shaped = np.convolve(upsampled, rrc_taps(sps), mode="same")
    if len(shaped) < length:
        shaped = np.pad(shaped, (0, length - len(shaped)))
    burst = shaped[:length] * _envelope(length, BURST_RAMP)
    return burst * (BURST_RMS / np.sqrt(np.mean(np.abs(burst) ** 2)))


def _symbols(modulation: str, count: int, rng: np.random.Generator) -> np.ndarray:
    """Verilen modülasyona ait rastgele takımyıldız sembolleri üretir.

    Raises:
        ValueError: Modülasyon tanınmıyorsa.
    """
    if modulation == "bpsk":
        return 2.0 * rng.integers(0, 2, count) - 1.0 + 0j
    if modulation == "qpsk":
        return np.exp(1j * (np.pi / 4.0 + rng.integers(0, 4, count) * np.pi / 2.0))
    raise ValueError(f"Bilinmeyen modülasyon '{modulation}'. Desteklenenler: bpsk, qpsk.")


def build_signal(plan: RecordPlan) -> np.ndarray:
    """Tek bir kaydın kompleks örneklerini üretir."""
    rng = np.random.default_rng(plan.seed)
    t = np.arange(NUM_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    x = np.zeros(NUM_SAMPLES, dtype=np.complex128)

    # Referans ton: merkez frekanstan tam +100 kHz, kaydın tamamı boyunca.
    x += REF_TONE_AMPLITUDE * np.exp(2j * np.pi * REF_TONE_OFFSET * t)

    sps = int(round(SAMPLE_RATE / SYMBOL_RATE))
    burst = _shaped_burst(_symbols(plan.modulation, BURST_COUNT // sps + 8, rng), sps, BURST_COUNT)
    seg = slice(plan.burst_start, plan.burst_start + BURST_COUNT)
    x[seg] += burst * np.exp(2j * np.pi * plan.carrier_offset * t[seg])

    x += NOISE_SIGMA * (rng.standard_normal(NUM_SAMPLES) + 1j * rng.standard_normal(NUM_SAMPLES))
    return x.astype(np.complex64)


def write_record(plan: RecordPlan, samples: np.ndarray, out_dir: Path) -> Path:
    """Örnekleri ve metadata'yı SigMF kayıt çifti olarak diske yazar.

    Args:
        plan: Kaydın parametreleri.
        samples: `complex64` örnekler.
        out_dir: Çıktı klasörü.

    Returns:
        Yazılan `.sigmf-meta` dosyasının yolu.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{plan.name}.sigmf-data"
    meta_path = out_dir / f"{plan.name}.sigmf-meta"

    interleaved = np.empty(samples.size * 2, dtype=np.float32)
    interleaved[0::2] = samples.real
    interleaved[1::2] = samples.imag
    interleaved.tofile(data_path)

    meta = SigMFFile(
        data_file=str(data_path),
        global_info={
            sigmf.DATATYPE_KEY: "cf32_le",
            sigmf.SAMPLE_RATE_KEY: SAMPLE_RATE,
            sigmf.AUTHOR_KEY: "iqforge",
            sigmf.DESCRIPTION_KEY: (
                f"iqforge sentetik örnek kaydı '{plan.name}'. Sürekli referans ton "
                f"(merkez +100 kHz), tek {plan.modulation.upper()} bursti, düşük seviyeli AWGN."
            ),
            sigmf.HW_KEY: "synthetic (scripts/make_example.py)",
            sigmf.RECORDER_KEY: "iqforge scripts/make_example.py",
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
                "kaydın tamamı boyunca sürekli. Spektrogram doğrulaması bunu kullanır. "
                "Bir sınıf değil ölçüm referansıdır; etiketlemede --exclude-label ile dışlanır."
            ),
        },
    )
    meta.add_annotation(
        plan.burst_start,
        BURST_COUNT,
        metadata={
            sigmf.LABEL_KEY: plan.modulation,
            sigmf.FREQ_LOWER_EDGE_KEY: CENTER_FREQ + plan.carrier_offset - OCCUPIED_BW / 2.0,
            sigmf.FREQ_UPPER_EDGE_KEY: CENTER_FREQ + plan.carrier_offset + OCCUPIED_BW / 2.0,
            sigmf.COMMENT_KEY: (
                f"{plan.modulation.upper()}, {SYMBOL_RATE / 1e3:.0f} kBd, RRC beta={RRC_BETA}, "
                f"bant genişliği {OCCUPIED_BW / 1e3:.1f} kHz, "
                f"ortalama güç {BURST_RMS**2:.4f}, taşıyıcı {plan.carrier_offset / 1e3:+.0f} kHz"
            ),
        },
    )

    meta.tofile(str(meta_path), skip_validate=False, pretty=True)
    return meta_path


def main() -> None:
    """Tüm kayıtları üretir ve özetini yazdırır.

    Kayıtlardan herhangi biri zaten varsa hiçbir şey yapmaz: örnek veri seti bir
    kez üretilir ve sabit kalır. Bilinçli yeniden üretim için `--force`.
    """
    force = "--force" in sys.argv
    existing = [p for p in PLANS if (OUT_DIR / f"{p.name}.sigmf-meta").exists()]
    if existing and not force:
        print(f"{len(existing)} kayıt zaten var ({OUT_DIR}) — üretim atlandı.")
        print("Bilerek yeniden üretmek için: python scripts/make_example.py --force")
        return
    if force:
        for stale in sorted(OUT_DIR.glob("*.sigmf-*")):
            stale.unlink()

    total_bytes = 0
    print(f"{'kayıt':<10} {'modülasyon':<11} {'taşıyıcı':>11} {'burst (s)':>18} {'güç':>9}")
    for plan in PLANS:
        samples = build_signal(plan)
        meta_path = write_record(plan, samples, OUT_DIR)
        total_bytes += meta_path.with_suffix(".sigmf-data").stat().st_size

        seg = samples[plan.burst_start : plan.burst_start + BURST_COUNT]
        t0 = plan.burst_start / SAMPLE_RATE
        t1 = (plan.burst_start + BURST_COUNT) / SAMPLE_RATE
        print(
            f"{plan.name:<10} {plan.modulation:<11} {plan.carrier_offset / 1e3:>+8.0f} kHz "
            f"{t0:>8.4f}…{t1:<8.4f} {np.mean(np.abs(seg) ** 2):>9.6f}"
        )

    print(
        f"\n{len(PLANS)} kayıt, kayıt başına {NUM_SAMPLES} örnek "
        f"({NUM_SAMPLES / SAMPLE_RATE:.4f} s), toplam {total_bytes / 1e6:.2f} MB"
    )
    print(f"referans ton: merkez {REF_TONE_OFFSET:+.0f} Hz (her kayıtta)")
    print(f"burst bant genişliği: {OCCUPIED_BW / 1e3:.1f} kHz (her kayıtta)")


if __name__ == "__main__":
    main()
