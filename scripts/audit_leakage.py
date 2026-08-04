"""Yüksek test doğruluğu gerçek mi, sızıntı mı?

Doğruluk %98'in üstüne çıktığında varsayılan tepki kutlamak değil denetlemek
olmalı. Bu script kurulmuş bir veri setinde dört bağımsız kontrol yapar:

  1. Kayıt ayrıklığı — hiçbir kayıt birden fazla split'te olmamalı (SPEC §5.6).
  2. Pencere ikizliği — test penceresi ile eğitim penceresi arasındaki en yüksek
     kosinüs benzerliği. Aynı kayıttan gelen komşu pencereler %50 örtüştüğü için
     kayıt İÇİNDE yüksek benzerlik normaldir; split'ler ARASINDA yüksek benzerlik
     sızıntı demektir.
  3. Ofset bağımsızlığı — her split'te sınıf x taşıyıcı ofset kontenjansı.
  4. Burst konumu bağımsızlığı — aynı kontrol, bu kez burst başlangıcı için.
     `--balance-by` yalnızca verilen alanı dengeler; başka bir alan sızdırabilir.

Kullanım:
    python scripts/audit_leakage.py <dataset_dir> [--source examples]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from iqforge.io import load


def load_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, list[int]]:
    """Bir split'in tüm pencerelerini ve etiketlerini belleğe alır."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["splits"][split]
    if not entry["shards"]:
        return np.zeros((0, 0)), []
    arrays = [np.load(dataset_dir / s) for s in entry["shards"]]
    stacked = np.concatenate(arrays, axis=0)
    return stacked.reshape(stacked.shape[0], -1), entry["labels"]


#: Bu eşiğin üstündeki benzerlik pratikte "aynı pencere" demektir.
DUPLICATE_THRESHOLD = 0.999


def max_cross_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """İki pencere kümesi arasındaki en yüksek/ortalama benzerlik ve ikiz sayısı.

    Returns:
        `(maks, ortalama, ikiz_sayısı)`. İkiz sayısı, benzerliği
        `DUPLICATE_THRESHOLD` üstünde olan çift sayısıdır; gerçek sızıntının
        imzası budur. Yüksek ama 1'e uzak benzerlik, tüm kayıtlarda ortak olan
        referans tonundan gelir ve sınıf bilgisi taşımaz.
    """
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), 0
    a_unit = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_unit = b / np.linalg.norm(b, axis=1, keepdims=True)
    similarity = np.abs(a_unit @ b_unit.T)
    duplicates = int((similarity > DUPLICATE_THRESHOLD).sum())
    return float(similarity.max()), float(similarity.mean()), duplicates


def within_record_similarity(dataset_dir: Path, split: str) -> float:
    """Aynı split içindeki pencereler arası en yüksek benzerlik (referans değer).

    Adım pencereden küçük olduğu için komşu pencereler örtüşür; bu değer,
    split'ler arası benzerliğin ne kadar yüksek sayılacağına ölçü olur.
    """
    windows, _ = load_split(dataset_dir, split)
    if windows.shape[0] < 2:
        return float("nan")
    unit = windows / np.linalg.norm(windows, axis=1, keepdims=True)
    similarity = np.abs(unit @ unit.T)
    np.fill_diagonal(similarity, 0.0)
    return float(similarity.max())


def contingency_deviation(cells: dict[tuple[str, object], int]) -> float:
    """Bağımsızlıktan en büyük mutlak sapma (0 ise tam bağımsız)."""
    labels = sorted({label for label, _ in cells})
    values = sorted({value for _, value in cells}, key=str)
    n = sum(cells.values())
    if n == 0:
        return 0.0
    worst = 0.0
    for label in labels:
        row = sum(cells.get((label, v), 0) for v in values)
        for value in values:
            column = sum(cells.get((lab, value), 0) for lab in labels)
            worst = max(worst, abs(cells.get((label, value), 0) - row * column / n))
    return worst


def _sort_key(value: object) -> tuple[int, float, str]:
    """Sütun başlıklarını sayısalsa sayı, değilse metin olarak sıralar.

    Ofset başlıkları '+180k' gibi son ek taşıyabildiği için doğrudan float()
    çağırmak hata verir.
    """
    text = str(value)
    try:
        return (0, float(text.rstrip("k")), "")
    except ValueError:
        return (1, 0.0, text)


def print_contingency(title: str, cells: dict[tuple[str, object], int]) -> float:
    """Kontenjans tablosunu basar ve sapmayı döndürür."""
    labels = sorted({label for label, _ in cells})
    values = sorted({value for _, value in cells}, key=_sort_key)
    print(f"    {title}")
    print("      " + "".join(f"{v:>10}" for v in values) + f"{'toplam':>10}")
    for label in labels:
        row = "".join(f"{cells.get((label, v), 0):>10d}" for v in values)
        total = sum(cells.get((label, v), 0) for v in values)
        print(f"      {label:<6}{row}{total:>10d}")
    deviation = contingency_deviation(cells)
    print(f"      -> {'BAĞIMSIZ' if deviation < 1e-9 else f'BAĞIMLI (sapma {deviation:.2f})'}")
    return deviation


def main() -> None:
    """Denetimi çalıştırır."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--source", type=Path, default=Path("examples"))
    args = parser.parse_args()

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    splits = ("train", "val", "test")

    print("1) KAYIT AYRIKLIĞI")
    where: dict[str, list[str]] = defaultdict(list)
    for split in splits:
        for record in manifest["splits"][split]["records"]:
            where[record["id"]].append(split)
    overlapping = {k: v for k, v in where.items() if len(v) > 1}
    print(f"   toplam kayıt: {len(where)}")
    print(f"   birden fazla split'te olan: {overlapping or 'YOK'}")

    print("\n2) PENCERE İKİZLİĞİ (kosinüs benzerliği, mutlak)")
    train_windows, _ = load_split(args.dataset_dir, "train")
    for split in ("val", "test"):
        windows, _ = load_split(args.dataset_dir, split)
        peak, mean, duplicates = max_cross_similarity(windows, train_windows)
        print(
            f"   {split} vs train : maks {peak:.4f}  ortalama {mean:.4f}  "
            f"ikiz (>{DUPLICATE_THRESHOLD}): {duplicates}"
        )
    within = within_record_similarity(args.dataset_dir, "train")
    print(f"   train içi (örtüşen komşu pencereler): maks {within:.4f}")
    print(
        "   not: taban benzerlik yüksektir çünkü her pencere aynı +100 kHz referans\n"
        "        tonunu içerir; sızıntının imzası ~1.0 benzerlik ve ikiz sayısı > 0'dır."
    )

    print("\n3) SINIF x TAŞIYICI OFSET")
    worst_offset = 0.0
    for split in splits:
        records = manifest["splits"][split]["records"]
        if not records:
            continue
        cells: dict[tuple[str, object], int] = defaultdict(int)
        for record in records:
            cells[(record["label"], f"{record['carrier_offset_hz'] / 1e3:+.0f}k")] += 1
        worst_offset = max(
            worst_offset, print_contingency(f"{split} ({len(records)} kayıt)", cells)
        )

    print("\n4) SINIF x BURST BAŞLANGICI")
    starts: dict[str, int] = {}
    for path in sorted(args.source.glob("*.sigmf-meta")):
        rec = load(path)
        burst = next(a for a in rec.annotations if a.label != "ref_tone")
        starts[path.name] = burst.sample_start

    worst_start = 0.0
    for split in splits:
        records = manifest["splits"][split]["records"]
        if not records:
            continue
        cells = defaultdict(int)
        for record in records:
            cells[(record["label"], str(starts[Path(record["id"]).name]))] += 1
        worst_start = max(worst_start, print_contingency(f"{split} ({len(records)} kayıt)", cells))

    offset_verdict = "tam" if worst_offset < 1e-9 else f"sapma {worst_offset:.2f}"
    start_verdict = "tam" if worst_start < 1e-9 else f"sapma {worst_start:.2f}"
    print("\nÖZET")
    print(f"   kayıt sızıntısı          : {'YOK' if not overlapping else 'VAR'}")
    print(f"   ofset bağımsızlığı       : {offset_verdict}")
    print(f"   burst konumu bağımsızlığı: {start_verdict}")
    print(
        "\n   Uyarı: sınıf başına tek kayıt içeren split'lerde (val/test) hiçbir\n"
        "   kayıt-düzeyi özniteliğin bağımsızlığı GÖSTERİLEMEZ — iki kaydı ayıran\n"
        "   her alan sınıfları da 'ayırmış' görünür. n=2'de kontenjans tablosu\n"
        "   dejenere olur; buradaki sapma sızıntı kanıtı değildir."
    )


if __name__ == "__main__":
    main()
