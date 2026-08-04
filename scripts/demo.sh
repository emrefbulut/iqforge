#!/usr/bin/env bash
# iqforge tanıtım kaydı için komut dizisi.
#
# asciinema ile kaydedilmek üzere yazılmıştır; kayıt alma talimatları
# docs/demo.md içindedir. Script kesme istemez, baştan sona kendi akar.
#
# Idempotent: her çalıştırmada çıktı klasörlerini baştan siler, böylece
# kayıt tekrar alındığında ekranda "klasör zaten var" gibi gürültü çıkmaz.

set -uo pipefail

# Terminal genişliği sabit: rich COLUMNS'u okur, böylece tablolar ve
# spektrogram kaydın genişliğinden bağımsız olarak hep aynı görünür.
export COLUMNS=100
export LINES=34

# Spektrogram yarım blok karakteri (U+2580) ve tablolar kutu çizim karakterleri
# kullanır. Çıktı bir dosyaya yönlendirildiğinde Python varsayılan olarak yerel
# kod sayfasına düşer ve bu karakterler kaybolur; UTF-8'i açıkça zorla.
export PYTHONIOENCODING=utf-8

DATASET_DIR="${DATASET_DIR:-/tmp/ds}"
SINGLE_DIR="${SINGLE_DIR:-/tmp/ds-single}"

# Çalıştırıcıyı hızdan yavaşa doğru seç. `uv run` her çağrıda ortamı yeniden
# çözdüğü için komut başına ~2 saniye ekler; altı komutta bu 12 saniye eder ve
# kaydı gereksiz uzatır. Doğrudan binary varsa onu kullan.
if command -v iqforge >/dev/null 2>&1; then
    IQ=(iqforge)
elif [ -x .venv/bin/iqforge ]; then
    IQ=(.venv/bin/iqforge)
elif [ -x .venv/Scripts/iqforge.exe ]; then
    IQ=(.venv/Scripts/iqforge.exe)
else
    IQ=(uv run iqforge)
fi

rm -rf "$DATASET_DIR" "$SINGLE_DIR"

# Komutu ekranda gösterip çalıştırır, ardından verilen kadar duraklar.
run() {
    local pause="$1"
    shift
    printf '\n\033[1;32m$\033[0m \033[1miqforge %s\033[0m\n\n' "$*"
    "${IQ[@]}" "$@"
    sleep "$pause"
}

# 1. Kayıtta ne var?
run 1.5 info examples/bpsk_01.sigmf-meta

# 2. Sinyale bak. Spektrogramın okunması için daha uzun duraklama.
run 4 inspect examples/bpsk_01.sigmf-meta

# 3. Pencerele, etiketle, böl, yaz.
run 1.5 build examples/ -o "$DATASET_DIR" --balance-by core:freq_lower_edge

# 4. Ne kurduk?
run 2.5 stats "$DATASET_DIR"

# 5. Veri seti gerçekten eğitilebilir mi?
run 2 train "$DATASET_DIR" --epochs 20

# 6. Asıl mesele: yapamayacağı şeyi sessizce yapmıyor.
#    Tek kayıtla kayıt bazlı bölme mümkün değil, `build` hata verip duruyor.
printf '\n\033[1;33m# Tek kayıtla kayıt bazlı bölme yapılamaz.\033[0m\n'
printf '\033[1;33m# iqforge sessizce pencere bazlı bölmeye DÜŞMEZ, durur:\033[0m\n'
run 3 build examples/bpsk_01.sigmf-meta -o "$SINGLE_DIR" || true

rm -rf "$SINGLE_DIR"
