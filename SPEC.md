# sigkit — Proje Spesifikasyonu

Bu doküman Claude Code için yazılmıştır. Fazları sırayla uygula. Her fazın sonunda
"Doğrulama" bölümündeki komutu çalıştır ve beklenen çıktıyı aldığından emin ol.
Bir faz doğrulanmadan bir sonrakine geçme.

---

## 1. Proje nedir

`sigkit`, SDR ile yakalanmış ham RF kayıtlarını (SigMF formatı) makine öğrenmesinde
kullanılabilir etiketli veri setlerine çeviren bir komut satırı aracıdır.

**Çözdüğü problem:** Bugün RF alanında sentetik veri üreten araçlar (TorchSig) ve
kayıt inceleyen araçlar (IQEngine) var, ama "elimdeki gerçek kaydı alıp PyTorch'ta
eğitilebilir bir `Dataset` haline getir" adımını yapan bakımlı bir araç yok.
sigkit bu boşluğu doldurur.

**Tasarım ilkesi:** SigMF standardıyla tam uyumlu ol. Kendi format icat etme.
Mevcut ekosistemle (IQEngine, GNU Radio, TorchSig) uyumluluk bu projenin en
önemli özelliğidir.

---

## 2. Kapsam

### v0'da VAR
- SigMF kayıt çiftlerini (`.sigmf-meta` + `.sigmf-data`) okuma
- Metadata inceleme komutu
- Terminalde spektrogram görüntüleme
- Kaydı sabit uzunlukta pencerelere bölme
- Etiketleme (üç kaynaktan: SigMF annotation, klasör adı, CSV dosyası)
- Katmanlı (stratified) train/val/test bölme
- Diske veri seti yazma + PyTorch `Dataset` sınıfı olarak okuma
- Küçük bir baseline CNN ile duman testi

### v0'da YOK — bunları yapma
- Canlı SDR donanımından yakalama (RTL-SDR, HackRF vb.)
- Sinyal gönderimi
- Demodülasyon veya protokol çözme
- Web arayüzü
- Sentetik sinyal üretimi
- Eğitim döngüsü, checkpoint yönetimi, hyperparameter arama
- Bulut depolama entegrasyonu

Bu maddelerden birini yapmak cazip gelirse yapma. Kapsam dışı.

---

## 3. Teknoloji seçimleri

Bunlar karar verilmiştir, değiştirme:

| Alan | Seçim | Not |
|---|---|---|
| Python | 3.11+ | |
| Paket yönetimi | `uv` | `pyproject.toml`, PEP 621 |
| CLI | `typer` | type hint tabanlı, otomatik `--help` |
| Terminal çıktısı | `rich` | tablo, renk, spektrogram |
| SigMF I/O | `sigmf` (sigmf-python) | **kendi parser'ını yazma** |
| Sayısal | `numpy`, `scipy` | STFT için `scipy.signal` |
| ML | `torch` | **opsiyonel bağımlılık**: `sigkit[torch]` |
| Test | `pytest` | |
| Lint/format | `ruff` | |

`torch` opsiyonel olmalı. `build` ve `inspect` komutları torch kurulu olmadan
çalışmalı; sadece `sigkit.SigkitDataset` ve `train` torch gerektirir.

---

## 4. Komut arayüzü

```
sigkit info <path>
    SigMF kaydının metadata'sını okunabilir tablo olarak yazdırır.
    Örnekleme hızı, merkez frekans, veri tipi, örnek sayısı, süre,
    donanım bilgisi, annotation listesi.

sigkit inspect <path> [--start N] [--samples N] [--nfft 1024]
    Terminalde spektrogram çizer. Ayrıca zaman ekseninde güç grafiği.
    --start: kaçıncı örnekten başlasın
    --samples: kaç örnek gösterilsin (varsayılan 262144)

sigkit build <input> -o <output_dir>
              [--window 1024] [--stride 512]
              [--labels {annotations,dirname,csv}] [--label-file <path>]
              [--exclude-label <label>] [--split 0.7,0.15,0.15] [--seed 42]
              [--repr {iq2ch,complex,magphase}] [--normalize/--no-normalize]
    --exclude-label: bu etikete sahip annotation'lar etiketleme sırasında hiç
              dikkate alınmaz. Yinelenebilir. Varsayılan: `ref_tone`. Ayrıntı 5.3.
    <input> tek bir .sigmf-meta dosyası VEYA içinde birden fazla kayıt olan
    bir klasör olabilir. Klasörse özyinelemeli tarar.
    Çıktı: <output_dir> içine shard dosyaları + manifest.json

sigkit stats <dataset_dir>
    Kurulmuş veri setinin özeti: sınıf dağılımı, pencere sayısı,
    split boyutları, disk kullanımı.

sigkit train <dataset_dir> [--epochs 10] [--batch-size 64]
    Basit bir baseline CNN eğitir. Amaç doğruluk rekoru değil,
    veri setinin gerçekten eğitilebilir olduğunu kanıtlamak.
```

---

## 5. Veri akışı ve teknik detaylar

### 5.1 SigMF okuma

Desteklenmesi zorunlu veri tipleri: `cf32_le`, `ci16_le`, `ci8`.
Başka bir `core:datatype` görürsen **sessizce tahmin etme** — açık bir hata mesajı
ver: hangi tip bulundu, hangileri destekleniyor.

Tüm örnekler bellekte `complex64` olarak temsil edilir. Tamsayı tiplerinden
dönüştürürken tam ölçek değerine böl (`ci16_le` için 32768.0, `ci8` için 128.0).

Büyük dosyalar için `numpy.memmap` kullan, dosyanın tamamını belleğe alma.

### 5.2 Pencereleme

Kayıt, `--window` uzunluğunda, `--stride` adımıyla kayan pencerelere bölünür.
Sondaki eksik pencere atılır (padding yapma).

Pencere sayısı = `floor((N - window) / stride) + 1`

### 5.3 Etiketleme

Üç kaynak, `--labels` ile seçilir:

- `annotations`: SigMF metadata'sındaki `annotations` dizisinden. Her annotation
  bir `core:sample_start` ve `core:sample_count` içerir. Bir pencerenin etiketi,
  o pencerenin merkezinin hangi annotation aralığına düştüğüyle belirlenir.
  Etiket değeri `core:label` alanından alınır. Hiçbir aralığa düşmeyen pencereler
  atılır (varsayılan) — bunu `--keep-unlabeled` ile değiştirilebilir yap.
- `dirname`: kaydın bulunduğu klasörün adı etiket olur. Cihaz sınıflandırma
  veri setlerinde (AirID, ORACLE) yaygın düzen budur.
- `csv`: `--label-file` ile verilen CSV. Sütunlar: `filename,label`.

**Not — zamanda örtüşen annotation'lar ve `--exclude-label`.**
Yukarıdaki `annotations` kuralı yalnızca zaman eksenine bakar. Frekansta ayrık
ama zamanda örtüşen iki sinyal varsa (örneğin `examples/sample` kaydında kayıt
boyunca süren `ref_tone` ile aynı anda var olan `bpsk`/`qpsk` burstleri) bir
pencere birden fazla annotation aralığına düşer ve bu kural hangisinin
kastedildiğini söyleyemez.

Bu belirsizlik "en dar aralık kazanır" gibi bir heuristic'le **çözülmez.**
Böyle bir kural doğru cevabı tahmin ediyormuş gibi görünür, oysa aracın
frekans boyutunu hiç kullanmadığı gerçeğini gizler; başka bir kayıtta sessizce
yanlış etiket üretir. Bunun yerine sorun açıkça ele alınır: `--exclude-label`
ile belirtilen etiketler etiketleme sırasında hiç dikkate alınmaz. Varsayılan
değer `ref_tone`'dur, çünkü paketle gelen örnek kayıttaki referans ton bir
sınıf değil, ölçüm referansıdır.

Bir pencere `--exclude-label` uygulandıktan sonra hâlâ birden fazla annotation
aralığına düşüyorsa etiketlenemez sayılır ve atılır; sessizce birini seçme.
Kaç pencerenin bu nedenle atıldığı `build` çıktısında raporlanmalı.

Frekans-farkındalıklı etiketleme (`core:freq_lower_edge`/`core:freq_upper_edge`
kullanarak zaman-frekans karolarına ayırma) v0 kapsamı dışındadır.

### 5.4 Temsil (`--repr`)

- `iq2ch` (varsayılan): `(2, window)` şeklinde float32. Kanal 0 = I (gerçek),
  kanal 1 = Q (sanal). PyTorch'ta en yaygın kullanılan biçim.
- `complex`: `(window,)` complex64. Ham hali korunur.
- `magphase`: `(2, window)` float32. Kanal 0 = genlik, kanal 1 = faz (radyan).

### 5.5 Normalizasyon

Varsayılan açık. Her pencere ayrı ayrı birim güce normalize edilir:

```
x = x / sqrt(mean(|x|^2))
```

Sıfır güçlü pencerelerde bölme hatası oluşmasın, sıfır dönsün.

### 5.6 Bölme (split)

Etikete göre katmanlı (stratified). `--seed` ile deterministik.

**Önemli:** Aynı kayıt dosyasından gelen pencereler aynı split'e gitmeli.
Kayıt bazında böl, pencere bazında değil. Aksi halde komşu pencereler hem
eğitim hem test setine düşer ve doğruluk yapay olarak şişer. Bu kural
ihlal edilmemeli.

### 5.7 Disk formatı

```
<output_dir>/
  manifest.json
  train/shard_0000.npy
  train/shard_0001.npy
  val/shard_0000.npy
  test/shard_0000.npy
```

Her shard en fazla 256 MB. `manifest.json` içeriği:

```json
{
  "sigkit_version": "0.1.0",
  "created": "ISO8601 zaman damgası",
  "config": { "window": 1024, "stride": 512, "repr": "iq2ch", "normalize": true, "seed": 42 },
  "label_map": { "device_a": 0, "device_b": 1 },
  "source_files": ["...sigmf-meta yolları..."],
  "splits": {
    "train": { "shards": ["train/shard_0000.npy"], "labels": [0,0,1,...], "count": 12000 },
    "val":   { ... },
    "test":  { ... }
  }
}
```

Etiketler manifest'te tutulur, ayrı dosyaya yazma.

### 5.8 PyTorch arayüzü

```python
from sigkit import SigkitDataset

train = SigkitDataset("out/", split="train")
x, y = train[0]  # x: torch.Tensor (2, 1024) float32, y: int
len(train)
train.label_map  # {"device_a": 0, ...}
```

`torch.utils.data.Dataset` alt sınıfı olmalı, shard'ları memmap ile lazy okumalı.

---

## 6. Terminal spektrogramı

`scipy.signal.stft` ile hesapla, sonra terminale çiz.

Çizim yöntemi: Unicode yarım blok karakteri (`▀`) kullan. Her karakter iki
dikey piksel taşır — üst yarı ön plan rengi, alt yarı arka plan rengi.
`rich` bunu destekler. Böylece her terminalde çalışır.

Renk skalası: viridis benzeri, dB ölçeğinde. Alt/üst sınır otomatik
(persentil 5 ve 99).

Eksen etiketleri: yatayda zaman (saniye), dikeyde frekans (MHz, merkez
frekans etrafında). Metadata'daki `core:sample_rate` ve `core:frequency`
kullanılarak hesaplanır.

Kitty/iTerm grafik protokolü v0'da yok. Sonra eklenecek.

---

## 7. Dosya yapısı

```
sigkit/
  pyproject.toml
  README.md
  LICENSE                 (MIT)
  .github/workflows/ci.yml
  src/sigkit/
    __init__.py           SigkitDataset ve load() dışa aktarılır
    cli.py                typer uygulaması
    io.py                 SigMF okuma, veri tipi dönüşümü
    windowing.py          pencereleme
    labeling.py           üç etiket kaynağı
    splitting.py          katmanlı bölme
    storage.py            shard yazma/okuma, manifest
    dataset.py            SigkitDataset (torch)
    display.py            terminal spektrogram
    models.py             baseline CNN
  tests/
    test_io.py
    test_windowing.py
    test_labeling.py
    test_splitting.py
    test_storage.py
  examples/
    sample.sigmf-meta     küçük sentetik örnek kayıt
    sample.sigmf-data
```

`examples/` içindeki örnek kaydı sen üret: birkaç saniyelik, iki farklı
modülasyonlu sentetik sinyal, annotation'larıyla birlikte, 5 MB'ın altında.
Bu dosya kritik — kullanıcı donanım olmadan aracı deneyebilmeli.

---

## 8. Fazlar ve doğrulama

### Faz 1 — İskelet + SigMF okuma + `info`
Kur: `pyproject.toml`, paket yapısı, `io.py`, `cli.py` içinde sadece `info`.
`examples/` içindeki sentetik örnek kaydı üret.

**Doğrulama:**
```
uv run sigkit info examples/sample.sigmf-meta
```
Örnekleme hızı, merkez frekans, veri tipi ve örnek sayısı doğru görünmeli.
`tests/test_io.py` geçmeli.

### Faz 2 — `inspect` terminal spektrogramı
**Doğrulama:**
```
uv run sigkit inspect examples/sample.sigmf-meta
```
Terminalde spektrogram görünmeli ve sentetik sinyalin bilinen frekans
bileşenleri doğru yerde çıkmalı. Ayrıca aynı veriyi matplotlib ile PNG'ye
çizen küçük bir doğrulama scripti yaz (`scripts/verify_spectrogram.py`) ve
iki görüntünün aynı yapıyı gösterdiğini kontrol et.

### Faz 3 — `build` ve `stats`
Pencereleme, etiketleme, bölme, shard yazma, manifest.

**Doğrulama:**
```
uv run sigkit build examples/sample.sigmf-meta -o /tmp/ds
uv run sigkit stats /tmp/ds
```
Sınıf dağılımı dengeli olmalı, pencere sayısı formülle hesaplananla eşleşmeli,
`manifest.json` şemaya uymalı. Aynı `--seed` ile iki kez çalıştırıldığında
birebir aynı bölme çıkmalı.

### Faz 4 — `SigkitDataset` + `train`
**Doğrulama:**
```
uv run --extra torch sigkit train /tmp/ds --epochs 5
```
Sentetik veride eğitim doğruluğu %90'ın üstüne çıkmalı. Çıkmıyorsa veri
pipeline'ında hata var demektir — durup nedenini bul, hyperparameter oynama.

### Faz 5 — Paketleme ve dokümantasyon
README (kurulum, 3 komutluk hızlı başlangıç, örnek çıktı), MIT lisansı,
GitHub Actions CI (lint + test, Python 3.11 ve 3.12).

**Doğrulama:**
```
uv build
pipx install dist/sigkit-0.1.0-py3-none-any.whl
sigkit info examples/sample.sigmf-meta
```
Temiz bir ortamda kurulup çalışmalı.

---

## 9. Kod kalitesi kuralları

- Tüm public fonksiyonlarda type hint
- Docstring: ne yaptığı + parametreler (Google stili)
- Hata mesajları kullanıcıya yönelik ve eyleme dönük olsun.
  Kötü: `ValueError: invalid datatype`
  İyi: `Desteklenmeyen veri tipi 'cf64_le'. Desteklenenler: cf32_le, ci16_le, ci8.`
- **Sessizce varsayım yapma.** Metadata'da beklenen bir alan yoksa hata ver
  veya açıkça uyar, varsayılan uydurma. Örnekleme hızı yoksa devam etme.
- Uzun işlemlerde `rich` progress bar
- Her modül için test. Testler sentetik veriyle çalışsın, ağ erişimi
  gerektirmesin.
- SigMF spesifikasyonundan emin olmadığın bir şey varsa, tahmin etme —
  `sigmf` kütüphanesinin sunduğu API'yi kullan.

---

## 10. Bu spesifikasyonun dışına çıkma

Ek özellik önerin varsa uygulamadan önce sor. Kapsam genişlemesi bu projenin
en büyük riski. Faz sırasını değiştirme, doğrulama adımlarını atlama.
