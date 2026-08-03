# sigkit

SDR ile yakalanmış ham RF kayıtlarını (SigMF) makine öğrenmesinde kullanılabilir
etiketli veri setlerine çeviren komut satırı aracı.

> Geliştirme aşamasında — Faz 1 (iskelet + SigMF okuma + `info`) tamamlandı.
> Ayrıntılı README Faz 5'te yazılacak; yol haritası için [SPEC.md](SPEC.md).

## Hızlı deneme

```bash
uv sync --group dev
uv run sigkit info examples/sample.sigmf-meta
```

`examples/sample.sigmf-meta` paketle gelen sentetik kayıttır; donanım gerektirmez.
İçinde iki modülasyonlu burst (BPSK, QPSK) ve merkez frekanstan tam **+100 kHz**
kaymış sürekli bir referans ton bulunur. Referans ton ayrı bir annotation
(`ref_tone`) olarak işaretlidir ve sonraki fazlardaki doğrulamaların dayanağıdır.

İki burst yalnızca modülasyon türüyle ayrışır: sembol hızı (64 kBd), bant
genişliği (86.4 kHz), süre ve ortalama güç ikisinde de aynıdır.

## Bilinen sınırlamalar

- **`ci16_le` ve `ci8` dönüşüm yolları gerçek veriyle doğrulanmadı.** Bu iki
  veri tipinin `complex64`'e dönüşümü (sırasıyla 32768.0 ve 128.0 tam ölçek
  böleni) yalnızca `tests/test_io.py` içindeki sentetik gidiş-dönüş testleriyle
  sınanmıştır: test verisi aynı varsayımla yazılıp aynı varsayımla okunur, yani
  test kendi kabulünü doğrular. Gerçek bir SDR'dan (RTL-SDR `ci8`, USRP/HackRF
  `ci16_le`) alınmış bir kayıtla karşılaştırma henüz yapılmadı. Özellikle şunlar
  açık: tamsayı örneklerin işaretli/işaretsiz yorumu, ölçeklemenin `2^(n-1)` mi
  `2^(n-1) - 1` mi olması gerektiği ve donanıma özgü I/Q sırası. `cf32_le` yolu
  örnek kayıtla uçtan uca doğrulanmıştır.
- **Zaman bazlı etiketleme frekansta ayrık sinyalleri ayıramaz.** Ayrıntı ve
  `--exclude-label` ile nasıl ele alındığı: [SPEC.md](SPEC.md) §5.3.

## Lisans

MIT (Faz 5'te eklenecek).
