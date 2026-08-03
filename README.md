# sigkit

SDR ile yakalanmış ham RF kayıtlarını (SigMF) makine öğrenmesinde kullanılabilir
etiketli veri setlerine çeviren komut satırı aracı.

> Geliştirme aşamasında — Faz 3 (`info`, `inspect`, `build`, `stats`) tamamlandı.
> `SigkitDataset` ve `train` Faz 4'te gelecek.
> Ayrıntılı README Faz 5'te yazılacak; yol haritası için [SPEC.md](SPEC.md).

## Hızlı deneme

```bash
uv sync --group dev
uv run sigkit info examples/bpsk_01.sigmf-meta
uv run sigkit inspect examples/bpsk_01.sigmf-meta
uv run sigkit build examples/ -o /tmp/ds
uv run sigkit stats /tmp/ds
```

`inspect` spektrogramı Unicode yarım blok karakteriyle çizer ve tüm bilgiyi
**renkte** taşır. Çıktıyı bir dosyaya yönlendirirseniz `rich` renkleri kapatır ve
geriye tekdüze bir blok kalır; renkli kaydetmek için
`scripts/capture_terminal.py` kullanın.

## Örnek veri seti

`examples/` sekiz sentetik kayıt çifti içerir; donanım gerektirmez:
`bpsk_01…bpsk_04` ve `qpsk_01…qpsk_04`, her biri 65536 örnek (0.064 s),
toplam 4.2 MB.

Her kayıtta merkez frekanstan tam **+100 kHz** kaymış sürekli bir referans ton
(`ref_tone` annotation'ı) ve tek bir modülasyonlu burst vardır. Referans ton bir
sınıf değil ölçüm referansıdır; etiketlemede `--exclude-label` ile dışlanır
([SPEC.md](SPEC.md) §5.3).

**Neden sekiz ayrı dosya:** SPEC §5.6 pencereleri kayıt bazında bölmeyi
zorunlu kılar. Tek dosya olsaydı bu kural örnek veriyle sınanamazdı. Sınıf
başına dört kayıt, 0.7/0.15/0.15 bölmesinin üç split'i de doldurmasına yeter.

**Kısayol ipuçları kapatıldı.** İki sınıf yalnızca modülasyon türüyle ayrışır;
şunlar ikisinde de birebir aynıdır: sembol hızı (64 kBd), bant genişliği
(86.4 kHz), burst süresi (40960 örnek), ortalama güç ve taşıyıcı ofset havuzu
(−280/−180/+180/+280 kHz — her sınıf dördünü de birer kez kullanır, yani
taşıyıcı frekansı sınıf hakkında bilgi taşımaz).

## Kayıt bazında bölme

`build`, pencereleri **kayıt bazında** böler: bir kaydın tüm pencereleri aynı
split'e gider. Pencere bazlı bölme komşu pencereleri hem eğitime hem teste
düşürür ve test doğruluğunu yapay olarak şişirir.

Bu kural uygulanamıyorsa `build` **hata verir**, sessizce pencere bazlı bölmeye
düşmez. Tek kayıtla çalışmak istiyorsanız açık kaçış yolu `--split 1.0,0,0`:

```bash
uv run sigkit build examples/bpsk_01.sigmf-meta -o /tmp/tek --split 1.0,0,0
```

Hangi kaydın hangi split'e düştüğü hem `build` çıktısında hem `stats`'ta
isim isim listelenir ve `manifest.json` içinde saklanır.

## Doğrulama çıktıları

`artifacts/` altındaki dosyalar faz doğrulamalarının kalıcı kayıtlarıdır:

| Dosya | İçerik |
|---|---|
| `inspect_{bpsk,qpsk}_01.{ansi.txt,svg}` | `sigkit inspect` çıktısı, renkleriyle |
| `spectrogram_{bpsk,qpsk}_01.png` | Aynı verinin matplotlib karşılığı |
| `verify_{bpsk,qpsk}_01.txt` | `scripts/verify_spectrogram.py` sayısal karşılaştırma raporu |

`.ansi.txt` dosyaları terminalde `cat` ile, `.svg` dosyaları tarayıcıda görüntülenir.

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
