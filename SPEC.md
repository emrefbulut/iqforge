# iqforge — Proje Spesifikasyonu

Bu doküman Claude Code için yazılmıştır. Fazları sırayla uygula. Her fazın sonunda
"Doğrulama" bölümündeki komutu çalıştır ve beklenen çıktıyı aldığından emin ol.
Bir faz doğrulanmadan bir sonrakine geçme.

---

## 1. Proje nedir

`iqforge`, SDR ile yakalanmış ham RF kayıtlarını (SigMF formatı) makine öğrenmesinde
kullanılabilir etiketli veri setlerine çeviren bir komut satırı aracıdır.

**Çözdüğü problem:** Bugün RF alanında sentetik veri üreten araçlar (TorchSig) ve
kayıt inceleyen araçlar (IQEngine) var, ama "elimdeki gerçek kaydı alıp PyTorch'ta
eğitilebilir bir `Dataset` haline getir" adımını yapan bakımlı bir araç yok.
iqforge bu boşluğu doldurur.

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

**`scripts/` bu listeye tabi değildir.** Oradaki araçlar geliştirme
araçlarıdır: paketlenen bir özellik değil, projeyi üretmek ve doğrulamak için
kullanılan yardımcılardır. Wheel'e girmezler (`[tool.hatch.build.targets.wheel]`
yalnızca `src/iqforge` paketler), CLI'dan erişilemezler ve kullanıcının
kurulumunda bulunmazlar.

Bu ayrım özellikle `scripts/make_example.py` için önemli: sentetik sinyal
üretiyor, yani yukarıdaki listede yasaklanan işi yapıyor. Yasak, aracın
**kullanıcıya sunduğu** yetenekler içindir — iqforge sentetik veri üretmez,
mevcut kayıtları işler. Örnek kayıtlar ise §7'nin gerektirdiği test verisidir
ve depoya bir kez üretilip sabitlenir.

Aynı gerekçe `scripts/run_seed_grid.py` için de geçerli: "eğitim döngüsü ve
hyperparameter arama" kapsam dışıdır, ama Faz 4 sonuçlarını raporlayan bir
ölçüm scripti yazmak kapsam dışı değildir.

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
| ML | `torch` | **opsiyonel bağımlılık**: `iqforge[torch]` |
| Test | `pytest` | |
| Lint/format | `ruff` | |

`torch` opsiyonel olmalı. `build` ve `inspect` komutları torch kurulu olmadan
çalışmalı; sadece `iqforge.IQForgeDataset` ve `train` torch gerektirir.

---

## 4. Komut arayüzü

```
iqforge info <path>
    SigMF kaydının metadata'sını okunabilir tablo olarak yazdırır.
    Örnekleme hızı, merkez frekans, veri tipi, örnek sayısı, süre,
    donanım bilgisi, annotation listesi.

iqforge inspect <path> [--start N] [--samples N] [--nfft 1024]
    Terminalde spektrogram çizer. Ayrıca zaman ekseninde güç grafiği.
    --start: kaçıncı örnekten başlasın
    --samples: kaç örnek gösterilsin (varsayılan 262144)

iqforge build <input> -o <output_dir>
              [--window 1024] [--stride 512]
              [--labels {annotations,dirname,csv}] [--label-file <path>]
              [--exclude-label <label>] [--split 0.7,0.15,0.15] [--seed 42]
              [--balance-by <sigmf alanı>]
              [--repr {iq2ch,complex,magphase}] [--normalize/--no-normalize]
    --exclude-label: bu etikete sahip annotation'lar etiketleme sırasında hiç
              dikkate alınmaz. Yinelenebilir. Varsayılan: `ref_tone`. Ayrıntı 5.3.
    --balance-by: adı verilen SigMF alanının değeri, sınıf katmanlaması
              korunarak split'lere yayılır. Rahatsız edici değişkenin (nuisance
              variable) split'ler arasında sistematik dağılmasını önler.
              Ayrıntı 5.6.
    <input> tek bir .sigmf-meta dosyası VEYA içinde birden fazla kayıt olan
    bir klasör olabilir. Klasörse özyinelemeli tarar.
    Çıktı: <output_dir> içine shard dosyaları + manifest.json

iqforge stats <dataset_dir>
    Kurulmuş veri setinin özeti: sınıf dağılımı, pencere sayısı,
    split boyutları, disk kullanımı.

iqforge train <dataset_dir> [--epochs 10] [--batch-size 64]
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

**Kayıt bazında bölme yapılamıyorsa `build` HATA VERSİN.** Pencere bazlı
bölmeye sessizce düşmek yasaktır. Sessiz geri düşüş, kullanıcıya doğru
çalışıyormuş gibi görünen ama test doğruluğu şişmiş bir veri seti üretir —
bu, aracın üretebileceği en zararlı çıktıdır, çünkü hata sonuçlara bakarak
fark edilmez.

Hata verilecek durumlar:

- Girdide tek bir kayıt dosyası var (bir split'e ayıracak ikinci kayıt yok).
- Bir sınıfın kayıt sayısı, istenen split oranlarıyla o sınıfa boş olmayan
  her split'i dolduramayacak kadar az.

Hata mesajı hem sorunu hem çözümü söylemeli. Örnek:

```
Kayıt bazında katmanlı bölme yapılamıyor: 'bpsk' sınıfında yalnızca 1 kayıt
dosyası var, 0.7/0.15/0.15 bölmesi için en az 3 gerekli.

SPEC §5.6 gereği aynı kayıttan gelen pencereler aynı split'e gitmeli; pencere
bazlı bölmeye düşmek test doğruluğunu yapay olarak şişirir.

Şunlardan birini yapın:
  - her sınıf için daha fazla kayıt dosyası verin (klasör girdisi kullanın)
  - --split oranlarını azaltın, örn. --split 0.5,0.25,0.25
  - tek kayıtla yalnızca eğitim seti üretin: --split 1.0,0,0
```

`--split 1.0,0,0` boş val/test üretmek isteyen kullanıcı için açık bir kaçış
yoludur; bu bilinçli bir seçim olduğu için hata verilmez.

**Rahatsız edici değişken dengesi (`--balance-by`).**
Sınıfa göre katmanlamak yeterli değildir. Sınıf hakkında hiçbir bilgi taşımayan
bir değişken (taşıyıcı frekansı, alıcı donanımı, kayıt günü) split'ler arasında
sistematik olarak dağılabilir; o zaman sınıf dağılımı kusursuz görünürken model
eğitimde görmediği bir koşulda değerlendirilir ve sonuç yanıltıcı olur.

`--balance-by <alan>` bir SigMF anahtarı alır. Değer sırayla kayda etiketini
veren annotation'ın ham sözlüğünde, sonra `global` bölümünde aranır; böylece
mekanizma sentetik veriye özel değil, herhangi bir SigMF alanı için çalışır
(`core:freq_lower_edge`, `core:hw`, uzantı anahtarları…).

Sınıf başına split kayıt sayıları değişmez — katmanlama bozulmaz. Değişen,
hangi kaydın hangi split'e gittiğidir: kayıtlar grup grup dönüşümlü işlenir ve
her kayıt kendi grubunun en az temsil edildiği split'e yerleştirilir. Grup
sayaçları sınıflar arasında paylaşılır, böylece split'ler birbirini tamamlar.

Dengeleme yapısal olarak tutmayabilir (grup sayısı en küçük split'ten fazlaysa,
alan bazı kayıtlarda yoksa, ya da her kayıt ayrı bir gruba düşüyorsa). Bu
durumda `build` **UYARI** basar ve devam eder — hata değildir, çünkü bölme yine
de geçerli ve kayıt bazlıdır; kullanıcı kalan kaymayı bilerek kabul edebilir.

Taşıyıcı ofseti her kayıt için `manifest.json` içinde `carrier_offset_hz`
alanında saklanır ve `stats` çıktısında hem kayıt bazında hem split özeti
olarak gösterilir; dengesizlik `--balance-by` kullanılmasa da görünür olur.

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
  "iqforge_version": "0.1.0",
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
from iqforge import IQForgeDataset

train = IQForgeDataset("out/", split="train")
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
iqforge/
  pyproject.toml
  README.md
  CONTRIBUTING.md
  CITATION.cff
  LICENSE                       (MIT)
  SPEC.md                       (bu doküman)
  .gitignore
  .github/
    workflows/ci.yml            lint + test (3.11, 3.12) + torch + wheel
    ISSUE_TEMPLATE/bug_report.yml
  src/iqforge/
    __init__.py                 load() ve (tembel) IQForgeDataset dışa aktarılır
    cli.py                      typer uygulaması: info/inspect/build/stats/train
    io.py                       SigMF okuma, veri tipi dönüşümü
    windowing.py                pencereleme, normalizasyon, temsiller
    labeling.py                 üç etiket kaynağı, --balance-by alan okuma
    splitting.py                katmanlı kayıt bazlı bölme, sızıntı uyarıları
    storage.py                  shard yazma/okuma, manifest
    dataset.py                  IQForgeDataset (torch)
    training.py                 baseline eğitim döngüsü (torch)
    display.py                  terminal spektrogram
    models.py                   baseline CNN
  tests/
    conftest.py                 paylaşılan fixture'lar
    helpers.py                  sentetik SigMF kayıt üreticisi
    test_io.py
    test_windowing.py
    test_labeling.py
    test_splitting.py
    test_storage.py
    test_display.py
    test_dataset.py             (torch yoksa atlanır)
    test_models.py              (torch yoksa atlanır)
  scripts/                      geliştirme araçları, pakete girmez — bkz. §2
    make_example.py             örnek kayıtları üretir
    verify_spectrogram.py       Faz 2 doğrulaması, PNG üretir
    capture_terminal.py         inspect çıktısını renkleriyle kaydeder
    run_seed_grid.py            Faz 4 tohum ızgarası
    audit_leakage.py            sızıntı denetimi
    demo.sh                     tanıtım kaydı komut dizisi
  docs/
    banner.svg                  README başlık görseli
    banner.png                  SVG render olmazsa yedek
    make_banner.py              banner üreticisi
    demo.md                     asciinema/agg ile kayıt alma talimatları
  artifacts/                    faz doğrulamalarının kalıcı çıktıları
  examples/                     16 kayıt: bpsk_01…bpsk_08, qpsk_01…qpsk_08
    bpsk_01.sigmf-meta
    bpsk_01.sigmf-data
    ...
```

`examples/` içindeki örnek kayıtları sen üret: her biri kısa, tek modülasyonlu,
annotation'larıyla birlikte, toplamı 6 MB'ın altında. Bu dosyalar kritik —
kullanıcı donanım olmadan aracı deneyebilmeli.

**Yapı: 2 sınıf × 4 taşıyıcı ofset × 2 kayıt = 16 kayıt.** Her kayıt 32768
örnek (0.032 s), toplam 4.19 MB, kayıt başına 40 etiketli pencere (1024/512
pencerelemede), toplam 640.

Üç sayı da zorunlu:

- **Birden fazla dosya:** §5.6 kayıt bazında bölme istiyor. Tek dosyayla bu
  kural örnek veriyle sınanamaz, `build` da sessizce pencere bazlı bölmeye
  düşebilirdi.
- **Sınıf başına en az 3 kayıt:** 0.7/0.15/0.15 bölmesinin üç split'i de boş
  olmayacak şekilde dolabilmesi için.
- **(sınıf, ofset) hücresi başına 2 kayıt:** §5.6'nın split içi bağımsızlık
  garantisi, bir ofsetin kayıtlarını tur tur dağıtır. Hücrede tek kayıt olsaydı
  tur oluşturulamaz, o ofsetin tüm kayıtları aynı split'e düşer ve train ile
  test hiçbir ofseti paylaşamazdı; model her zaman görülmemiş taşıyıcıda
  sınanır, doğruluk şans seviyesine çakılır ve Faz 4 doğrulama kapısı hiçbir
  şey ölçemez. Bu durum ölçülmüştür: tek kayıtlı kurulumda 15 koşunun 12'si
  tam %50 vermiştir.

Kayıttan kayda değişenler: gürültü tohumu, sembol dizisi, burst zaman konumu,
taşıyıcı ofseti. Sabit kalanlar: bant genişliği (86.4 kHz), burst süresi
(20480 örnek), ortalama güç. Her sınıf her ofseti ve her burst başlangıcını
eşit sayıda kullanır.

---

## 8. Fazlar ve doğrulama

### Faz 1 — İskelet + SigMF okuma + `info`
Kur: `pyproject.toml`, paket yapısı, `io.py`, `cli.py` içinde sadece `info`.
`examples/` içindeki sentetik örnek kaydı üret.

**Doğrulama:**
```
uv run iqforge info examples/sample.sigmf-meta
```
Örnekleme hızı, merkez frekans, veri tipi ve örnek sayısı doğru görünmeli.
`tests/test_io.py` geçmeli.

### Faz 2 — `inspect` terminal spektrogramı
**Doğrulama:**
```
uv run iqforge inspect examples/sample.sigmf-meta
```
Terminalde spektrogram görünmeli ve sentetik sinyalin bilinen frekans
bileşenleri doğru yerde çıkmalı. Ayrıca aynı veriyi matplotlib ile PNG'ye
çizen küçük bir doğrulama scripti yaz (`scripts/verify_spectrogram.py`) ve
iki görüntünün aynı yapıyı gösterdiğini kontrol et.

### Faz 3 — `build` ve `stats`
Pencereleme, etiketleme, bölme, shard yazma, manifest.

**Doğrulama:**
```
uv run iqforge build examples/ -o /tmp/ds --balance-by core:freq_lower_edge
uv run iqforge stats /tmp/ds
```
Sınıf dağılımı dengeli olmalı, pencere sayısı formülle hesaplananla eşleşmeli,
`manifest.json` şemaya uymalı. Aynı `--seed` ile iki kez çalıştırıldığında
birebir aynı bölme çıkmalı.

Girdi tek dosya değil `examples/` klasörüdür: §5.6 kayıt bazında bölme
istiyor, tek dosyayla bu kural sınanamaz (ve `build` doğru şekilde hata verir).

`--balance-by` neden gerekli: örnek kayıtlarda taşıyıcı ofseti sınıf hakkında
bilgi taşımaz ama split'ler arasında sistematik olarak dağılabilir. Yalnızca
sınıfa göre katmanlandığında `--seed 42` train'e dört pozitif ofseti, val ve
test'e dört negatif ofseti veriyordu — sınıflar dengeli olduğu halde bir
dağılım kayması. `stats` çıktısındaki "Taşıyıcı ofset dağılımı" tablosu bunu
görünür kılar; her split'te negatif ve pozitif ofsetler birlikte bulunmalı.

### Faz 4 — `IQForgeDataset` + `train`
**Doğrulama:**
```
uv run --extra torch iqforge train /tmp/ds --epochs 20
```
Sentetik veride eğitim doğruluğu %90'ın üstüne çıkmalı. Çıkmıyorsa veri
pipeline'ında hata var demektir — durup nedenini bul, hyperparameter oynama.

**Epoch sayısı neden 20.** Eğitim doğruluğu ölçülen değerlerle (bölme tohumu
11, eğitim tohumu 0):

| epoch | eğitim | val | test |
|---|---|---|---|
| 5  | %65.4 | %50.0 | %52.5 |
| 10 | %84.0 | %81.2 | %67.5 |
| 20 | %99.0 | %100  | %95.0 |

Eğri monoton; 5 ve 10 epoch'ta model henüz yakınsamamıştır, bu bir boru hattı
hatası değildir. %90 eşiği 20 epoch'ta karşılanır.

**Beklenen test doğruluğu: %90–100.** Ölçülen (5 bölme × 3 eğitim tohumu,
20 epoch): ortalama **%98.4 ± %2.8**, aralık %91.25–%100.

Bu, kurulum sırasında hedeflenen %75–95 bandının üstündedir. Nedeni sızıntı
değil, görevin kolay olmasıdır:

- Bant içi SNR ≈ 18 dB (burst gücü 0.0484, gürültü gücü 0.0008).
- Her pencere 1024 örnek = 64 sembol taşır; klasik bir BPSK/QPSK ayırıcısı da
  bu koşulda ~%100 yapar.
- Test kayıtları eğitimle aynı taşıyıcı ofsetini paylaşır (§7), yani ölçülen
  şey modülasyon ayrımıdır, taşıyıcıya genelleme değil.

Yüksek doğruluk `scripts/audit_leakage.py` ile denetlenmiştir: kayıt ayrıklığı
sağlanıyor, split'ler arasında ikiz pencere yok (>0.999 benzerlikte 0 çift),
taşıyıcı ofseti her split'te etiketten bağımsız (sapma 0).

**Ölçüm çözünürlüğü sınırlıdır.** Test split'i 2 kayıt / 80 penceredir; bir
pencere %1.25 eder ve sınıf başına tek kayıt olduğu için kayıt-düzeyi hiçbir
özniteliğin bağımsızlığı istatistiksel olarak gösterilemez. Doğruluk farkları
birkaç puanlık aralıkta anlamlı okunmamalıdır.

**Tohum protokolü.** Bölme tohumu (`build --seed`) ile eğitim tohumu
(`train --seed`) ayrıdır ve karıştırılmamalıdır: ilki veri setinin içeriğini,
ikincisi yalnızca ağırlık ilklendirmesi ile batch sırasını belirler. Faz 4
sonuçları 5 bölme × 3 eğitim tohumu ızgarasıyla raporlanır
(`scripts/run_seed_grid.py`, çıktılar `artifacts/train_seed_grid.*`).

### Faz 5 — Paketleme ve dokümantasyon
README (kurulum, 3 komutluk hızlı başlangıç, örnek çıktı), MIT lisansı,
GitHub Actions CI (lint + test, Python 3.11 ve 3.12).

**Doğrulama:**
```
uv build
pipx install dist/iqforge-0.1.0-py3-none-any.whl
iqforge info examples/sample.sigmf-meta
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
