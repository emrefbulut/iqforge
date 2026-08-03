"""Terminal spektrogramı ve zaman ekseninde güç grafiği.

Çizim yöntemi Unicode üst yarı blok karakteridir (`▀`): her karakter hücresi iki
dikey piksel taşır — üst yarı ön plan rengiyle, alt yarı arka plan rengiyle
boyanır. Böylece grafik protokolü olmayan terminallerde de çalışır.
"""

from __future__ import annotations

import numpy as np
from rich.color import Color
from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from scipy import signal

from sigkit.io import Recording

#: matplotlib'in viridis paletinden 32 noktada örneklenmiş RGB çapaları.
#: Runtime'da matplotlib bağımlılığı olmasın diye gömülüdür.
VIRIDIS: tuple[tuple[int, int, int], ...] = (
    (68, 1, 84), (71, 13, 96), (72, 24, 106), (72, 35, 116),
    (71, 46, 124), (69, 56, 130), (66, 65, 134), (62, 74, 137),
    (58, 84, 140), (54, 93, 141), (50, 101, 142), (46, 109, 142),
    (43, 117, 142), (40, 125, 142), (37, 132, 142), (34, 140, 141),
    (31, 148, 140), (30, 156, 137), (32, 163, 134), (37, 171, 130),
    (46, 179, 124), (58, 186, 118), (72, 193, 110), (88, 199, 101),
    (108, 205, 90), (127, 211, 78), (147, 215, 65), (168, 219, 52),
    (192, 223, 37), (213, 226, 26), (234, 229, 26), (253, 231, 37),
)  # fmt: skip

UPPER_HALF_BLOCK = "▀"
SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"

#: Renk ölçeğinin alt/üst sınırı için kullanılan persentiller.
CLIP_PERCENTILES = (5.0, 99.0)

FREQ_LABEL_WIDTH = 9


def colormap(values: np.ndarray) -> np.ndarray:
    """[0, 1] aralığındaki değerleri viridis benzeri RGB renklere çevirir.

    Args:
        values: [0, 1] aralığında, herhangi bir şekle sahip dizi.

    Returns:
        `values.shape + (3,)` şeklinde uint8 RGB dizisi.
    """
    anchors = np.asarray(VIRIDIS, dtype=np.float64)
    pos = np.clip(values, 0.0, 1.0) * (len(anchors) - 1)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, len(anchors) - 1)
    frac = (pos - lo)[..., None]
    rgb = anchors[lo] * (1.0 - frac) + anchors[hi] * frac
    return np.round(rgb).astype(np.uint8)


def compute_spectrogram(
    samples: np.ndarray, sample_rate: float, nfft: int = 1024
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT ile spektrogramı dB ölçeğinde hesaplar.

    Args:
        samples: Kompleks örnekler.
        sample_rate: Örnekleme hızı (Hz).
        nfft: FFT uzunluğu.

    Returns:
        `(freqs, times, power_db)` üçlüsü. `freqs` artan sırada, merkez frekansa
        göre ofset (Hz); `times` saniye; `power_db` `(len(freqs), len(times))`.

    Raises:
        ValueError: Örnek sayısı `nfft`'ten azsa.
    """
    if samples.size < nfft:
        raise ValueError(
            f"Spektrogram için en az {nfft} örnek gerekli, {samples.size} örnek verildi. "
            f"--nfft değerini küçültün veya --samples değerini artırın."
        )

    freqs, times, zxx = signal.stft(
        samples,
        fs=sample_rate,
        nperseg=nfft,
        noverlap=nfft // 2,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    freqs = np.fft.fftshift(freqs)
    zxx = np.fft.fftshift(zxx, axes=0)
    power_db = 20.0 * np.log10(np.abs(zxx) + 1e-12)
    return freqs, times, power_db


def _bucket_starts(n_in: int, n_out: int) -> np.ndarray:
    """`n_in` uzunluğundaki ekseni `n_out` kovaya bölen başlangıç indislerini verir."""
    return (np.arange(n_out) * n_in) // n_out


def _pool_max(arr: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Ekseni `n_out` uzunluğuna indirger; küçültürken maksimum alır.

    Maksimum kullanılır çünkü dar bantlı bir ton ortalama alınırken komşu
    gürültü binlerine karışıp kaybolur; maksimum onu görünür tutar.
    """
    n_in = arr.shape[axis]
    if n_out >= n_in:
        idx = np.clip((np.arange(n_out) * n_in) // n_out, 0, n_in - 1)
        return np.take(arr, idx, axis=axis)
    return np.maximum.reduceat(arr, _bucket_starts(n_in, n_out), axis=axis)


def _pool_mean_axis(values: np.ndarray, n_out: int) -> np.ndarray:
    """Tek boyutlu bir ekseni `n_out` kovaya indirger; her kovanın ortalamasını verir."""
    n_in = values.size
    if n_out >= n_in:
        idx = np.clip((np.arange(n_out) * n_in) // n_out, 0, n_in - 1)
        return values[idx]
    starts = _bucket_starts(n_in, n_out)
    sums = np.add.reduceat(values, starts)
    counts = np.diff(np.append(starts, n_in))
    return sums / counts


def _format_offset_mhz(hz: float) -> str:
    """Merkez frekansa göre ofseti MHz cinsinde işaretli biçimlendirir."""
    return f"{hz / 1e6:+.3f}"


def spectrogram_panel(
    freqs: np.ndarray,
    times: np.ndarray,
    power_db: np.ndarray,
    *,
    width: int,
    height: int,
) -> Text:
    """Spektrogramı yarım blok karakterleriyle çizilmiş metin olarak döndürür.

    Args:
        freqs: Merkez frekansa göre ofset frekans ekseni (Hz), artan sırada.
        times: Zaman ekseni (s).
        power_db: dB ölçeğinde güç, `(freq, time)`.
        width: Çizim alanının karakter genişliği (eksen etiketleri hariç).
        height: Çizim alanının karakter yüksekliği; dikey çözünürlük 2×height.

    Returns:
        Renkli `rich.text.Text`.
    """
    # Üst satır en yüksek frekans olsun diye frekans eksenini ters çevir.
    flipped = power_db[::-1, :]
    freqs_desc = freqs[::-1]

    pixels = _pool_max(_pool_max(flipped, 2 * height, axis=0), width, axis=1)
    pixel_freqs = _pool_mean_axis(freqs_desc, 2 * height)

    # Persentiller havuzlama SONRASI, yani gerçekten ekrana basılan değerler
    # üzerinden hesaplanır. Tam çözünürlüklü diziden hesaplanırsa max-havuzlanmış
    # piksellerin neredeyse tamamı üst sınırın üstünde kalır ve görüntü tekdüze olur.
    vmin, vmax = np.percentile(pixels, CLIP_PERCENTILES)
    if vmax <= vmin:
        vmax = vmin + 1.0
    rgb = colormap((pixels - vmin) / (vmax - vmin))

    text = Text()
    for row in range(height):
        top, bottom = rgb[2 * row], rgb[2 * row + 1]
        label = _format_offset_mhz((pixel_freqs[2 * row] + pixel_freqs[2 * row + 1]) / 2.0)
        text.append(f"{label:>{FREQ_LABEL_WIDTH - 1}} ", style="dim")
        for col in range(width):
            style = Style(
                color=Color.from_rgb(*top[col].tolist()),
                bgcolor=Color.from_rgb(*bottom[col].tolist()),
            )
            text.append(UPPER_HALF_BLOCK, style=style)
        text.append("\n")

    text.append(_time_axis(times, width))
    return text


def _time_axis(times: np.ndarray, width: int) -> Text:
    """Yatay zaman eksenini (tik çizgisi + etiketler) çizer."""
    n_ticks = max(2, min(6, width // 12))
    cols = [round(i * (width - 1) / (n_ticks - 1)) for i in range(n_ticks)]
    labels = [f"{times[round(c * (times.size - 1) / (width - 1))]:.3f}" for c in cols]

    ruler = [" "] * width
    for c in cols:
        ruler[c] = "┬"
    axis = Text(" " * FREQ_LABEL_WIDTH + "".join(ruler) + "\n", style="dim")

    line = [" "] * (width + len(labels[-1]))
    for c, label in zip(cols, labels, strict=True):
        pos = min(max(c - len(label) // 2, 0), width - 1)
        line[pos : pos + len(label)] = list(label)
    axis.append(" " * FREQ_LABEL_WIDTH + "".join(line).rstrip() + " s\n", style="dim")
    return axis


def power_panel(samples: np.ndarray, times: np.ndarray, width: int) -> Text:
    """Zaman ekseninde güç grafiğini tek satırlık sparkline olarak çizer.

    Args:
        samples: Kompleks örnekler.
        times: Spektrogramın zaman ekseni (yalnızca uzunluk hizalaması için).
        width: Sparkline genişliği (karakter).

    Returns:
        Renkli `rich.text.Text`.
    """
    del times
    usable = (samples.size // width) * width
    if usable == 0:
        return Text()
    power = np.abs(samples[:usable].reshape(width, -1)) ** 2
    power_db = 10.0 * np.log10(power.mean(axis=1) + 1e-12)

    lo, hi = power_db.min(), power_db.max()
    if hi <= lo:
        hi = lo + 1.0
    levels = np.clip(
        ((power_db - lo) / (hi - lo) * (len(SPARK_BLOCKS) - 1)).round().astype(int),
        0,
        len(SPARK_BLOCKS) - 1,
    )

    text = Text(f"{'güç':>{FREQ_LABEL_WIDTH - 1}} ", style="dim")
    text.append("".join(SPARK_BLOCKS[i] for i in levels), style="bright_cyan")
    text.append(f"\n{'':>{FREQ_LABEL_WIDTH}}{lo:.1f} … {hi:.1f} dB\n", style="dim")
    return text


def render_inspect(
    rec: Recording,
    samples: np.ndarray,
    start: int,
    nfft: int,
    *,
    width: int,
    height: int,
) -> RenderableType:
    """`sigkit inspect` çıktısının tamamını (başlık + spektrogram + güç) üretir.

    Args:
        rec: Açılmış kayıt.
        samples: Görüntülenecek kompleks örnekler.
        start: Örneklerin kayıttaki başlangıç indisi.
        nfft: FFT uzunluğu.
        width: Toplam kullanılabilir karakter genişliği.
        height: Spektrogramın karakter yüksekliği.

    Returns:
        `rich` ile yazdırılabilir bir nesne.
    """
    freqs, times, power_db = compute_spectrogram(samples, rec.sample_rate, nfft)
    times = times + start / rec.sample_rate
    # Son sütun boş bırakılır: tam genişlikte bir satır bazı terminallerde
    # (ve rich'in legacy Windows modunda) otomatik olarak alt satıra sarar.
    plot_width = max(16, width - FREQ_LABEL_WIDTH - 1)

    center_mhz = (rec.center_frequency or 0.0) / 1e6
    header = Text.assemble(
        (rec.meta_path.name, "bold"),
        ("  merkez ", "dim"),
        (f"{center_mhz:.6g} MHz", "cyan"),
        ("  hız ", "dim"),
        (f"{rec.sample_rate / 1e6:.6g} MS/s", "cyan"),
        ("  örnek ", "dim"),
        (f"{start}…{start + samples.size}", "cyan"),
        ("  nfft ", "dim"),
        (f"{nfft}", "cyan"),
        ("\ndikey: merkeze göre ofset (MHz)   yatay: zaman (s)\n", "dim"),
    )
    return Group(
        header,
        spectrogram_panel(freqs, times, power_db, width=plot_width, height=height),
        power_panel(samples, times, plot_width),
    )
