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
#: Terminal görünümünün varsayılan karakter ızgarası (COLUMNS=100, 24 satır).
TERM_WIDTH, TERM_HEIGHT = 91, 24


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
    fig.colorbar(mesh, ax=ax, label="güç (dB)", pad=0.01)
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
    block = 512
    usable = (samples.size // block) * block
    p = np.abs(samples[:usable].reshape(-1, block)) ** 2
    t_power = start / rec.sample_rate + np.arange(p.shape[0]) * block / rec.sample_rate
    ax.plot(t_power, 10.0 * np.log10(p.mean(axis=1) + 1e-12), lw=0.7, color="#440154")
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


def report(rec: Recording, freqs: np.ndarray, times: np.ndarray, power_db: np.ndarray) -> bool:
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
    return ok


def main() -> int:
    """Doğrulamayı çalıştırır; tüm kontroller geçerse 0 döndürür."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="examples/sample.sigmf-meta")
    parser.add_argument("-o", "--output", default="artifacts/spectrogram_full.png")
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

    ok = report(rec, freqs, times, power_db)
    print("\nSONUÇ:", "tüm kontroller geçti" if ok else "EN AZ BİR KONTROL BAŞARISIZ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
