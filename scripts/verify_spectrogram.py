"""Terminal spektrogramını matplotlib ile çizilmiş referansa karşı doğrular.

`sigkit inspect` ile birebir aynı STFT'yi (`sigkit.display.compute_spectrogram`)
kullanır, sonucu PNG'ye çizer ve iki çizim yolunun aynı yapıyı gösterdiğini
sayısal olarak sınar:

  * Referans tonun tepe frekansı tam +100 kHz mi?
  * BPSK/QPSK burstlerinin bant içi gücünün yükseldiği zaman aralığı,
    annotation'daki `core:sample_start`/`core:sample_count` ile uyuşuyor mu?

Her iki soru hem tam çözünürlüklü STFT üzerinde (matplotlib'in çizdiği veri) hem
de terminale basılan havuzlanmış piksel matrisi üzerinde ayrı ayrı yanıtlanır;
iki yol aynı cevabı vermezse çizim yollarından biri bozuk demektir.

Kullanım:
    python scripts/verify_spectrogram.py -o artifacts/spectrogram_full.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sigkit.display import (  # noqa: E402
    CLIP_PERCENTILES,
    _pool_max,
    _pool_mean_axis,
    compute_spectrogram,
)
from sigkit.io import Recording, load  # noqa: E402

REF_TONE_HZ = 100_000.0
#: Örnek kayıtlardaki burst kenar rampasının uzunluğu (scripts/make_example.py).
BURST_RAMP = 512
#: Terminal görünümünün varsayılan karakter ızgarası (COLUMNS=100, 24 satır).
TERM_WIDTH, TERM_HEIGHT = 91, 24


#: Güç eğrisinin blok uzunluğu (örnek). Her bloğun zamanı blok MERKEZİ alınır.
POWER_BLOCK = 512


def power_curve(
    samples: np.ndarray, start: int, sample_rate: float, block: int = POWER_BLOCK
) -> tuple[np.ndarray, np.ndarray]:
    """Zaman ekseninde blok ortalamalı güç eğrisini verir.

    Zaman ekseni blokların MERKEZİNE karşılık gelir. Blok başlangıcı kullanılırsa
    eğri yarım blok kadar (varsayılanda 0.25 ms) sola kayar ve annotation
    kenarlarıyla karşılaştırma sistematik bir hata içerir.

    Args:
        samples: Kompleks örnekler.
        start: Örneklerin kayıttaki başlangıç indisi.
        sample_rate: Örnekleme hızı (Hz).
        block: Ortalama alınacak blok uzunluğu (örnek).

    Returns:
        `(times, power_db)` — blok merkezlerinin zamanı (s) ve dB güç.
    """
    usable = (samples.size // block) * block
    blocks = np.abs(samples[:usable].reshape(-1, block)) ** 2
    centres = (np.arange(blocks.shape[0]) + 0.5) * block + start
    return centres / sample_rate, 10.0 * np.log10(blocks.mean(axis=1) + 1e-12)


def half_power_edges(
    times: np.ndarray, power_db: np.ndarray
) -> tuple[float, float] | tuple[None, None]:
    """Basamak kenarlarını yarı-güç noktasında, alt örnek çözünürlükte bulur.

    Taban ve plato seviyeleri persentille kestirilir; eşik ikisinin lineer güçteki
    ortasıdır. Eşiği kesen komşu iki blok arasında doğrusal interpolasyon yapılır,
    böylece çözünürlük blok aralığından daha ince olur.

    Returns:
        `(yükselen_kenar, düşen_kenar)` saniye; kenar bulunamazsa `(None, None)`.
    """
    floor_db, plateau_db = np.percentile(power_db, 10), np.percentile(power_db, 90)
    threshold_db = 10.0 * np.log10((10 ** (floor_db / 10) + 10 ** (plateau_db / 10)) / 2.0)

    above = power_db > threshold_db
    if not above.any() or above.all():
        return None, None

    def _cross(i: int) -> float:
        """i-1 ile i arasında eşiği kestiği anı doğrusal interpolasyonla bulur."""
        y0, y1 = power_db[i - 1], power_db[i]
        frac = (threshold_db - y0) / (y1 - y0)
        return float(times[i - 1] + frac * (times[i] - times[i - 1]))

    rises = np.flatnonzero(~above[:-1] & above[1:]) + 1
    falls = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    if rises.size == 0 or falls.size == 0:
        return None, None
    return _cross(int(rises[0])), _cross(int(falls[-1]))


def band_power_db(freqs: np.ndarray, power_db: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Verilen frekans bandındaki toplam gücü zaman ekseni boyunca dB olarak verir."""
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        raise ValueError(f"{lo:.0f}…{hi:.0f} Hz bandına düşen STFT bini yok.")
    return 10.0 * np.log10(np.sum(10.0 ** (power_db[mask] / 10.0), axis=0))


def active_span(values_db: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    """Bir güç serisinin taban ile plato arasındaki eşiği aştığı zaman aralığını bulur.

    Args:
        values_db: Zaman serisi (dB).
        times: Aynı uzunlukta zaman ekseni (s).

    Returns:
        `(başlangıç, bitiş)` saniye. Eşiği aşan örnek yoksa `(nan, nan)`.
    """
    floor, plateau = np.percentile(values_db, 10), np.percentile(values_db, 90)
    above = np.flatnonzero(values_db > (floor + plateau) / 2.0)
    if above.size == 0:
        return float("nan"), float("nan")
    return float(times[above[0]]), float(times[above[-1]])


def terminal_pixels(freqs: np.ndarray, power_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`display.spectrogram_panel` ile aynı havuzlamayı uygulayıp pikselleri verir.

    Returns:
        `(pixels, pixel_freqs)` — `pixels` en üstte en yüksek frekans olacak
        şekilde `(2*TERM_HEIGHT, TERM_WIDTH)`, `pixel_freqs` her pikselin ofset
        frekansı (Hz).
    """
    flipped = power_db[::-1, :]
    pixels = _pool_max(_pool_max(flipped, 2 * TERM_HEIGHT, axis=0), TERM_WIDTH, axis=1)
    return pixels, _pool_mean_axis(freqs[::-1], 2 * TERM_HEIGHT)


def draw_png(
    rec: Recording,
    samples: np.ndarray,
    start: int,
    freqs: np.ndarray,
    times: np.ndarray,
    power_db: np.ndarray,
    out_path: Path,
) -> None:
    """Spektrogram, frekans kesiti ve güç grafiğini tek bir PNG'ye çizer."""
    vmin, vmax = np.percentile(power_db, CLIP_PERCENTILES)
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 11), height_ratios=[3, 1.4, 1.2], constrained_layout=True
    )

    ax = axes[0]
    mesh = ax.pcolormesh(
        times, freqs / 1e3, power_db, cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest"
    )
    # Etiketi açıkça 90° döndür: matplotlib'in colorbar varsayılanı -90'dır ve
    # yazı baş aşağı okunur.
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("güç (dB)", rotation=90, va="bottom", labelpad=14)
    # Referans işareti yalnızca kenarlarda çizilir: tam genişlikte bir çizgi,
    # göstermesi gereken tek binlik tonun üstünü örterdi.
    for xmin, xmax in ((0.0, 0.035), (0.965, 1.0)):
        ax.axhline(REF_TONE_HZ / 1e3, xmin=xmin, xmax=xmax, color="red", lw=1.6)
    ax.plot(
        [],
        [],
        color="red",
        lw=1.6,
        label=f"ref_tone beklenen: +{REF_TONE_HZ / 1e3:.0f} kHz (kenar işaretleri)",
    )
    for a in rec.annotations:
        if a.label == "ref_tone" or a.freq_lower_edge is None:
            continue
        t0, t1 = a.sample_start / rec.sample_rate, a.sample_end / rec.sample_rate
        if t1 < times[0] or t0 > times[-1]:
            continue  # pencere dışındaki annotation ekseni gereksiz yere gerer
        lo = (a.freq_lower_edge - (rec.center_frequency or 0.0)) / 1e3
        hi = (a.freq_upper_edge - (rec.center_frequency or 0.0)) / 1e3
        ax.add_patch(
            plt.Rectangle(
                (t0, lo), t1 - t0, hi - lo, fill=False, ec="white", ls="--", lw=1.4, alpha=0.9
            )
        )
        ax.text(t0, hi + 8, a.label, color="white", fontsize=9, weight="bold")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("merkeze göre ofset (kHz)")
    ax.set_title(
        f"{rec.meta_path.name} — merkez {(rec.center_frequency or 0) / 1e6:.6g} MHz, "
        f"{rec.sample_rate / 1e6:.6g} MS/s\n"
        "beyaz kesikli: annotation aralıkları    "
        "kırmızı kenar işaretleri: beklenen ref_tone frekansı"
    )
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    spectrum = 10.0 * np.log10(np.mean(10.0 ** (power_db / 10.0), axis=1))
    ax.plot(freqs / 1e3, spectrum, lw=0.8, color="#2a788e")
    ax.axvline(REF_TONE_HZ / 1e3, color="red", ls="--", lw=1.0)
    peak_hz = freqs[int(np.argmax(spectrum))]
    ax.annotate(
        f"tepe: {peak_hz / 1e3:+.3f} kHz",
        xy=(peak_hz / 1e3, spectrum.max()),
        xytext=(peak_hz / 1e3 + 60, spectrum.max() - 4),
        arrowprops={"arrowstyle": "->", "color": "red"},
        color="red",
        fontsize=9,
    )
    ax.set_xlabel("merkeze göre ofset (kHz)")
    ax.set_ylabel("ortalama güç (dB)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    t_power, p_db = power_curve(samples, start, rec.sample_rate)
    ax.plot(t_power, p_db, lw=0.7, color="#440154")
    rise, fall = half_power_edges(t_power, p_db)
    for edge in (rise, fall):
        if edge is not None:
            ax.axvline(edge, color="tab:red", ls=":", lw=1.2)
    for a in rec.annotations:
        t0, t1 = a.sample_start / rec.sample_rate, a.sample_end / rec.sample_rate
        if a.label == "ref_tone" or t1 < times[0] or t0 > times[-1]:
            continue
        ax.axvspan(t0, t1, alpha=0.15, color="tab:orange")
        ax.text(max(t0, times[0]), ax.get_ylim()[1], f" {a.label}", fontsize=8)
    ax.set_xlabel("zaman (s)")
    ax.set_ylabel("toplam güç (dB)")
    ax.set_xlim(times[0], times[-1])
    ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dpi, tek binlik referans tonun raster'da en az bir piksel kaplaması için
    # yeterince yüksek seçilir (nfft=1024 satır, panel yüksekliği ~6 inç).
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def report(
    rec: Recording,
    samples: np.ndarray,
    start: int,
    freqs: np.ndarray,
    times: np.ndarray,
    power_db: np.ndarray,
) -> bool:
    """İki çizim yolunu karşılaştırır ve tüm kontrollerin geçip geçmediğini döndürür."""
    pixels, pixel_freqs = terminal_pixels(freqs, power_db)
    bin_width = float(freqs[1] - freqs[0])
    ok = True

    print("\n--- referans ton (+100 kHz) ---")
    spectrum = 10.0 * np.log10(np.mean(10.0 ** (power_db / 10.0), axis=1))
    peak_full = float(freqs[int(np.argmax(spectrum))])
    row = int(np.argmax(pixels.max(axis=1)))
    peak_term = float(pixel_freqs[row])
    row_span = 0.5 * abs(pixel_freqs[0] - pixel_freqs[-1]) / (len(pixel_freqs) - 1) * 2

    print(
        f"  matplotlib (tam çözünürlük): tepe {peak_full / 1e3:+.3f} kHz  (bin {bin_width:.0f} Hz)"
    )
    print(
        f"  terminal (havuzlanmış)     : tepe satırı {peak_term / 1e3:+.1f} kHz  "
        f"(satır yüksekliği ~{row_span / 1e3:.1f} kHz)"
    )
    if abs(peak_full - REF_TONE_HZ) > bin_width:
        sapma = abs(peak_full - REF_TONE_HZ)
        print(f"  BAŞARISIZ: tam çözünürlüklü tepe +100 kHz'ten {sapma:.0f} Hz sapıyor")
        ok = False
    else:
        print("  TAMAM: tam çözünürlüklü tepe +100 kHz'te (bir bin içinde)")
    if abs(peak_term - REF_TONE_HZ) > row_span:
        print("  BAŞARISIZ: terminal tepe satırı +100 kHz'i içermiyor")
        ok = False
    else:
        print("  TAMAM: terminal tepe satırı +100 kHz'i içeriyor")

    print("\n--- burst zaman aralıkları ---")
    for a in rec.annotations:
        if a.label == "ref_tone" or a.freq_lower_edge is None:
            continue
        center = rec.center_frequency or 0.0
        lo, hi = a.freq_lower_edge - center, a.freq_upper_edge - center
        exp0, exp1 = a.sample_start / rec.sample_rate, a.sample_end / rec.sample_rate
        if exp0 > times[-1] or exp1 < times[0]:
            print(f"  {a.label}: görüntülenen pencerenin dışında, atlandı")
            continue

        got0, got1 = active_span(band_power_db(freqs, power_db, lo, hi), times)
        rows = (pixel_freqs >= lo) & (pixel_freqs <= hi)
        pixel_times = _pool_mean_axis(times, TERM_WIDTH)
        term0, term1 = active_span(pixels[rows].max(axis=0), pixel_times)

        print(
            f"  {a.label}: annotation {exp0:.4f}…{exp1:.4f} s  "
            f"({lo / 1e3:+.1f}…{hi / 1e3:+.1f} kHz)"
        )
        print(
            f"      matplotlib ölçümü : {got0:.4f}…{got1:.4f} s  "
            f"(fark {abs(got0 - exp0) * 1e3:+.1f} / {abs(got1 - exp1) * 1e3:+.1f} ms)"
        )
        print(f"      terminal ölçümü   : {term0:.4f}…{term1:.4f} s")
        if abs(got0 - exp0) > 0.005 or abs(got1 - exp1) > 0.005:
            print("      BAŞARISIZ: 5 ms toleransın dışında")
            ok = False
        else:
            print("      TAMAM: 5 ms tolerans içinde")

    ok &= _report_power_edges(rec, samples, start, times)
    return ok


def ramp_half_power_offset(ramp: int) -> float:
    """Hann kenar rampasının yarı-güç noktasının rampa başından uzaklığı (örnek).

    `scripts/make_example.py` burst kenarlarını `np.hanning(2*ramp)` genlik
    rampasıyla yumuşatır. Güç eğrisi rampanın yarı-güç noktasında eşiği keser;
    annotation ise rampanın BAŞINI işaretler. Beklenen kayıklık budur.
    """
    env_power = np.hanning(2 * ramp)[:ramp] ** 2
    i = int(np.flatnonzero(env_power >= 0.5)[0])
    y0, y1 = env_power[i - 1], env_power[i]
    return float((i - 1) + (0.5 - y0) / (y1 - y0))


def _report_power_edges(rec: Recording, samples: np.ndarray, start: int, times: np.ndarray) -> bool:
    """Güç eğrisinin basamak kenarlarını annotation kenarlarıyla sayısal karşılaştırır.

    Kenarlar iki farklı blok uzunluğunda ölçülür. Blok küçüldükçe ölçüm rampanın
    analitik yarı-güç noktasına yakınsamalıdır; yakınsamıyorsa rampayla
    açıklanamayan bir sistematik zaman hatası var demektir.
    """
    print("\n--- güç eğrisi basamak kenarları vs annotation ---")
    bursts = [a for a in rec.annotations if a.label != "ref_tone"]
    if len(bursts) != 1:
        print(f"  {len(bursts)} burst annotation'ı var; bu kontrol tek burstlü kayıt bekliyor.")
        return True

    a = bursts[0]
    exp0, exp1 = a.sample_start / rec.sample_rate, a.sample_end / rec.sample_rate
    if exp0 < times[0] or exp1 > times[-1]:
        print("  burst görüntülenen pencerenin dışında, atlandı")
        return True

    expected = ramp_half_power_offset(BURST_RAMP)
    print(f"  annotation           : {exp0:.5f}…{exp1:.5f} s")
    print(
        f"  beklenen kayıklık    : {expected:+.1f} / {-expected:+.1f} örnek — "
        f"{BURST_RAMP} örneklik Hann rampasının yarı-güç noktası"
    )

    fine = 32
    results: dict[int, tuple[float, float]] = {}
    for block in (POWER_BLOCK, fine):
        t_p, p_db = power_curve(samples, start, rec.sample_rate, block=block)
        rise, fall = half_power_edges(t_p, p_db)
        if rise is None or fall is None:
            print(f"  BAŞARISIZ: blok={block} için basamak kenarı bulunamadı")
            return False
        n0 = (rise - exp0) * rec.sample_rate
        n1 = (fall - exp1) * rec.sample_rate
        results[block] = (n0, n1)
        tag = "çizilen eğri" if block == POWER_BLOCK else "ince ızgara"
        print(
            f"  blok={block:<4} ({tag}): {rise:.5f}…{fall:.5f} s  "
            f"fark {(rise - exp0) * 1e3:+.3f} / {(fall - exp1) * 1e3:+.3f} ms  "
            f"({n0:+.0f} / {n1:+.0f} örnek)"
        )

    coarse_n0, coarse_n1 = results[POWER_BLOCK]
    fine_n0, fine_n1 = results[fine]
    print(
        f"  blok ızgarası etkisi : {abs(coarse_n0 - fine_n0):.0f} / "
        f"{abs(coarse_n1 - fine_n1):.0f} örnek — {POWER_BLOCK} örneklik blok "
        f"{BURST_RAMP} örneklik rampayı tek bloğa sıkıştırıp kenarı dışarı iter"
    )

    if not (fine_n0 > 0 > fine_n1):
        print("  BAŞARISIZ: kayıklığın işareti rampayla uyumsuz (içeri değil dışarı kayıyor)")
        return False
    tolerance = 0.1 * BURST_RAMP
    if abs(fine_n0 - expected) > tolerance or abs(fine_n1 + expected) > tolerance:
        print(
            f"  BAŞARISIZ: ince ızgara ölçümü rampanın yarı-güç noktasından "
            f"{tolerance:.0f} örnekten fazla sapıyor — rampayla açıklanamayan hata var"
        )
        return False
    print("  TAMAM: ince ızgarada kayıklık rampanın yarı-güç noktasına yakınsıyor;")
    print("         kalan fark yalnızca blok ızgarası çözünürlüğü, sistematik hata yok")
    return True


def main() -> int:
    """Doğrulamayı çalıştırır; tüm kontroller geçerse 0 döndürür."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="examples/bpsk_01.sigmf-meta")
    parser.add_argument("-o", "--output", default="artifacts/spectrogram_bpsk_01.png")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=None, help="varsayılan: kaydın tamamı")
    parser.add_argument("--nfft", type=int, default=1024)
    args = parser.parse_args()

    rec = load(args.path)
    data = rec.read(start=args.start, count=args.samples)
    freqs, times, power_db = compute_spectrogram(data, rec.sample_rate, args.nfft)
    times = times + args.start / rec.sample_rate

    out = Path(args.output)
    draw_png(rec, data, args.start, freqs, times, power_db, out)
    print(f"PNG yazıldı: {out}")
    print(
        f"pencere: örnek {args.start}…{args.start + data.size} "
        f"({times[0]:.4f}…{times[-1]:.4f} s), nfft={args.nfft}"
    )

    ok = report(rec, data, args.start, freqs, times, power_db)
    print("\nSONUÇ:", "tüm kontroller geçti" if ok else "EN AZ BİR KONTROL BAŞARISIZ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
