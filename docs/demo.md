# Tanıtım kaydı nasıl alınır

`scripts/demo.sh` kesme istemez, baştan sona kendi akar. Kayıt yaklaşık
**40 saniye** sürer.

## Gereksinimler

**UTF-8 terminal.** Spektrogram yarım blok karakteri (`▀`), tablolar kutu çizim
karakterleri kullanır. Terminal UTF-8 değilse kayıt bozuk çıkar:

```bash
locale        # LANG ve LC_ALL içinde UTF-8 görünmeli
```

**asciinema** — terminal kaydı alır:

```bash
# Debian/Ubuntu
sudo apt install asciinema
# macOS
brew install asciinema
# her yerde
pipx install asciinema
```

**agg** — kaydı GIF'e çevirir (asciinema'nın resmi aracı, Rust):

```bash
# macOS
brew install agg
# Rust kuruluysa
cargo install --git https://github.com/asciinema/agg
```

Hazır ikili dosyalar: <https://github.com/asciinema/agg/releases>

## 1. Kaydı al

Depo kökünden:

```bash
asciinema rec \
  --cols 100 --rows 34 \
  --idle-time-limit 2 \
  --command "bash scripts/demo.sh" \
  docs/demo.cast
```

- `--cols 100` — script `COLUMNS=100` ihraç eder, ama asciinema'nın da aynı
  genişlikte kaydetmesi gerekir; yoksa oynatımda satırlar sarar.
- `--rows 34` — spektrogram 24 satır + tablolar için yeterli yükseklik.
- `--idle-time-limit 2` — 2 saniyeden uzun boşluklar kısaltılır. `train`
  adımının 14 saniyelik sessiz bekleyişi böylece kaydı şişirmez.
- `--command` — kayıt bittiğinde kabuk otomatik kapanır, "exit" yazman gerekmez.

Kaydı gözden geçir:

```bash
asciinema play docs/demo.cast
```

Beğenmediysen `docs/demo.cast` dosyasını sil ve tekrar al. Script idempotenttir,
`/tmp/ds` ve `/tmp/ds-single` her çalıştırmada baştan temizlenir.

## 2. GIF'e çevir

```bash
agg \
  --font-size 16 \
  --theme asciinema \
  --speed 1.2 \
  --fps-cap 12 \
  --idle-time-limit 2 \
  --last-frame-duration 3 \
  docs/demo.cast docs/demo.gif
```

- `--speed 1.2` — 40 saniyeyi ~33 saniyeye indirir; okunabilirliği bozmaz.
- `--fps-cap 12` — dosya boyutunu belirleyen ana parametre. 30 fps'te GIF
  birkaç on MB olabilir; 12 fps'te birkaç MB'a iner ve terminal kaydında
  fark edilmez.
- `--last-frame-duration 3` — son kare (split hata mesajı) ekranda kalsın,
  GIF başa sarmadan önce okunabilsin.
- `--font-size 16` — GitHub'da README genişliğinde okunaklı.

Tüm seçenekler için `agg --help`; sürümler arasında değişebiliyor.

Boyutu kontrol et:

```bash
ls -lh docs/demo.gif
```

**5 MB'ı geçiyorsa** `--fps-cap 10` veya `--font-size 14` dene. GitHub README'de
10 MB üstü GIF'ler yavaş yükleniyor.

## 3. Nereye konacak

Her ikisini de depoya al:

```
docs/demo.cast   # kaynak kayıt — GIF'i parametre değiştirip yeniden üretmeye yarar
docs/demo.gif    # README'de gösterilen dosya
```

> **`docs/` içindeki diğer görseller.** `banner.svg` README'nin başlığıdır;
> `banner.png` onun yedeğidir — SVG'nin render edilmediği ortamlar (bazı RSS
> okuyucular, e-posta önizlemeleri, GitHub'ın sosyal medya kartı) için durur.
> Referans verilmiyor diye ölü dosya sanılıp silinmesin. İkisi de
> `make_banner.py` ile üretilir.

README'ye eklemek için (banner'ın hemen altına):

```markdown
<p align="center">
  <img src="docs/demo.gif" alt="iqforge demo" width="100%">
</p>
```

## Kayıtta ne görünüyor

| Adım | Komut | Ne gösteriyor |
|---|---|---|
| 1 | `info` | Kayıtta ne var: örnekleme hızı, merkez frekans, veri tipi, annotation'lar |
| 2 | `inspect` | Terminalde spektrogram — +100 kHz referans ton ve BPSK bursti |
| 3 | `build` | Pencereleme, etiketleme, kayıt bazlı bölme, shard yazma |
| 4 | `stats` | Sınıf dağılımı, split başına kayıtlar, taşıyıcı ofset dağılımı |
| 5 | `train` | Baseline CNN, 20 epoch, test doğruluğu |
| 6 | `build` (tek kayıt) | **Asıl mesele:** kayıt bazlı bölme yapılamayınca hata verip durur |

Son adım kasıtlıdır ve çıkış kodu 1'dir; script bunu yutar, kayıt hatasız biter.
