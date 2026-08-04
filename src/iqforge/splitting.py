"""Katmanlı, KAYIT BAZLI train/val/test bölmesi (SPEC §5.6).

Bu modülün tek kuralı var: aynı kayıt dosyasından gelen pencereler aynı split'e
gider. Pencere bazlı bölme, komşu pencereleri hem eğitime hem teste düşürdüğü
için test doğruluğunu yapay olarak şişirir.

Kural uygulanamıyorsa sessizce pencere bazlı bölmeye DÜŞÜLMEZ, hata verilir.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from iqforge.io import IQForgeError

#: Split adları, manifest'teki sırayla.
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitPlan:
    """Hangi kaydın hangi split'e gittiğini tutar.

    Attributes:
        assignment: Kayıt kimliği -> split adı.
        ratios: Kullanılan oranlar.
        seed: Kullanılan tohum.
        groups: Dengeleme kullanıldıysa kayıt kimliği -> rahatsız edici değişken
            grubu; kullanılmadıysa boş.
    """

    assignment: dict[str, str]
    ratios: tuple[float, float, float]
    seed: int
    groups: dict[str, str] = field(default_factory=dict)

    def records_in(self, split: str) -> list[str]:
        """Bir split'e düşen kayıt kimliklerini sıralı olarak verir."""
        return sorted(rid for rid, name in self.assignment.items() if name == split)


def parse_ratios(text: str) -> tuple[float, float, float]:
    """`0.7,0.15,0.15` biçimindeki oran dizgisini çözer.

    Raises:
        IQForgeError: Biçim bozuksa, negatif değer varsa veya toplam 1 değilse.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise IQForgeError(
            f"--split üç değer bekliyor (train,val,test), {len(parts)} verildi: '{text}'. "
            "Örnek: --split 0.7,0.15,0.15"
        )
    try:
        values = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise IQForgeError(
            f"--split sayısal olmalı: '{text}'. Örnek: --split 0.7,0.15,0.15"
        ) from exc

    if any(v < 0 for v in values):
        raise IQForgeError(f"--split değerleri negatif olamaz: '{text}'.")
    if abs(sum(values) - 1.0) > 1e-6:
        raise IQForgeError(f"--split değerlerinin toplamı 1 olmalı, {sum(values):.6g} verildi.")
    return values  # type: ignore[return-value]


def _allocate(n: int, ratios: tuple[float, float, float], active: list[int]) -> list[int]:
    """`n` kaydı oranlara göre paylaştırır; her aktif split'e en az bir kayıt verir.

    Önce standart en-büyük-kalan (largest remainder) yöntemi uygulanır, sonra boş
    kalan aktif split'lere en kalabalık split'ten birer kayıt aktarılır.

    Minimum garantisini baştan ayırmak yerine sonradan uygulamak önemli: baştan
    her split'e birer kayıt verilirse küçük split'ler sistematik olarak şişer
    (10 kayıt, 0.7/0.15/0.15 -> 6/2/2 yerine doğrusu 7/2/1).
    """
    exact = [n * ratios[i] if i in active else 0.0 for i in range(3)]
    counts = [int(value) for value in exact]

    order = sorted(active, key=lambda i: (-(exact[i] - int(exact[i])), i))
    for k in range(n - sum(counts)):
        counts[order[k % len(order)]] += 1

    for i in active:
        if counts[i] == 0:
            donor = max(active, key=lambda j: (counts[j], -j))
            counts[donor] -= 1
            counts[i] += 1
    return counts


def _shuffled(values: list[str], rng: np.random.Generator) -> list[str]:
    """Deterministik karıştırma; girdi önce sıralanır ki sonuç girdi sırasından bağımsız olsun."""
    ordered = sorted(values)
    return [ordered[i] for i in rng.permutation(len(ordered))]


def _assign_balanced(
    by_label: dict[str, list[str]],
    groups: dict[str, str],
    ratios: tuple[float, float, float],
    active: list[int],
    rng: np.random.Generator,
) -> dict[str, str]:
    """Katmanlı bölmeyi, rahatsız edici değişkeni split'lere yayarak yapar.

    Her sınıfın split başına kayıt sayısı `_allocate` ile sabittir — katmanlama
    bozulmaz. Değişen yalnızca HANGİ kaydın hangi split'e gittiğidir.

    Hedef: **bir split'in içinde grup ile etiket bağımsız olsun.** Bunun için
    atama "tur" birimiyle yapılır: bir tur = bir gruptan, her sınıftan birer
    kayıt. Tur bütün olarak tek bir split'e gider, böylece o split içinde grup
    etiketi ele veremez.

    Bu kural şart, çünkü tersi felakettir: gruplar sınıflar arasında
    tamamlayıcı dağıtılırsa (bpsk train'de pozitif ofsetleri, qpsk negatifleri
    alırsa) grup, split İÇİNDE etiketi birebir tahmin eder. Model o kestirmeyi
    öğrenir, eğitimde %100 yapar ve ilişki başka split'te tersine döndüğü için
    testte %0'a düşer — şans seviyesinin de altına.

    Turlar gruplar arasında dönüşümlü işlenir ve her tur, hedefine göre EN ÇOK
    AÇIĞI olan split'e verilir (mutlak kapasiteye değil orana bakılır). Mutlak
    kapasite kullanılsaydı train tamamen dolana kadar hiçbir grup val/test'e
    geçmez, dolayısıyla hiçbir grup train ile val/test arasında paylaşılmazdı;
    o zaman değerlendirme her zaman görülmemiş gruplar üzerinde yapılırdı.

    Grup başına yalnızca bir sınıfın kaydı varsa tur oluşturulamaz; kalanlar
    etiket bazında en açık split'e konur ve `leakage_warnings` durumu bildirir.
    """
    targets: dict[tuple[str, int], int] = {}
    for label, records in by_label.items():
        counts = _allocate(len(records), ratios, active)
        for i in active:
            targets[(label, i)] = counts[i]
    remaining = dict(targets)

    # cell[grup][etiket] = o gruptaki, o etikete sahip kayıtlar
    cell: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for label, records in by_label.items():
        for record_id in records:
            cell[groups[record_id]].setdefault(label, []).append(record_id)
    for buckets in cell.values():
        for label in buckets:
            buckets[label] = _shuffled(buckets[label], rng)

    def _deficit(split: int, labels: list[str]) -> float:
        """Split'in hedefine göre doldurulmamış oranı."""
        total = sum(targets[(label, split)] for label in labels)
        left = sum(remaining[(label, split)] for label in labels)
        return left / total if total else 0.0

    assignment: dict[str, str] = {}
    rotation = _shuffled(list(cell), rng)
    while any(any(cell[group].values()) for group in rotation):
        for group in rotation:
            buckets = cell[group]
            if not any(buckets.values()):
                continue
            present = [label for label, records in buckets.items() if records]
            rounds = {i: min(remaining[(label, i)] for label in present) for i in active}
            candidates = [i for i in active if rounds[i] > 0]

            if candidates:
                best = max(candidates, key=lambda i: (_deficit(i, present), rounds[i], -i))
                for label in present:
                    record_id = buckets[label].pop()
                    assignment[record_id] = SPLIT_NAMES[best]
                    remaining[(label, best)] -= 1
                continue

            # Tam tur oluşturulamıyor: kalanları etiket bazında en açık split'e koy.
            for label in present:
                record_id = buckets[label].pop()
                target = max(
                    (i for i in active if remaining[(label, i)] > 0),
                    key=lambda i: (remaining[(label, i)], -i),
                )
                assignment[record_id] = SPLIT_NAMES[target]
                remaining[(label, target)] -= 1

    return assignment


def stratified_record_split(
    record_labels: dict[str, str],
    ratios: tuple[float, float, float],
    seed: int,
    record_groups: dict[str, str] | None = None,
) -> SplitPlan:
    """Kayıtları etikete göre katmanlı biçimde split'lere dağıtır.

    Bölme KAYIT bazındadır: bir kaydın tüm pencereleri aynı split'e gider.
    Aynı `seed` ile birebir aynı sonucu verir.

    `record_groups` verilirse katmanlama korunarak rahatsız edici değişken de
    split'lere yayılır (bkz. `--balance-by`).

    Args:
        record_labels: Kayıt kimliği -> o kaydın baskın etiketi.
        ratios: `(train, val, test)` oranları.
        seed: Deterministik karıştırma tohumu.
        record_groups: Kayıt kimliği -> dengelenecek grup değeri.

    Returns:
        Kayıt ataması.

    Raises:
        IQForgeError: Bir sınıfın kayıt sayısı, oranların gerektirdiği boş olmayan
            split'leri dolduramayacak kadar azsa.
    """
    if not record_labels:
        raise IQForgeError("Bölünecek kayıt yok: girdide etiketlenebilir pencere bulunamadı.")

    active = [i for i, r in enumerate(ratios) if r > 0]
    by_label: dict[str, list[str]] = defaultdict(list)
    for record_id, label in record_labels.items():
        by_label[label].append(record_id)

    for label in sorted(by_label):
        available = len(by_label[label])
        if available < len(active):
            needed = ", ".join(f"{SPLIT_NAMES[i]}={ratios[i]:g}" for i in active)
            raise IQForgeError(
                f"Kayıt bazında katmanlı bölme yapılamıyor: '{label}' sınıfında yalnızca "
                f"{available} kayıt dosyası var, {'/'.join(f'{r:g}' for r in ratios)} bölmesi "
                f"için en az {len(active)} gerekli ({needed}).\n\n"
                "SPEC §5.6 gereği aynı kayıttan gelen pencereler aynı split'e gitmeli; "
                "pencere bazlı bölmeye düşmek test doğruluğunu yapay olarak şişirir.\n\n"
                "Şunlardan birini yapın:\n"
                "  - her sınıf için daha fazla kayıt dosyası verin (klasör girdisi kullanın)\n"
                "  - --split oranlarını azaltın, örn. --split 0.5,0.25,0.25\n"
                "  - tek kayıtla yalnızca eğitim seti üretin: --split 1.0,0,0"
            )

    rng = np.random.default_rng(seed)

    if record_groups is not None:
        assignment = _assign_balanced(by_label, record_groups, ratios, active, rng)
        return SplitPlan(
            assignment=assignment, ratios=ratios, seed=seed, groups=dict(record_groups)
        )

    assignment = {}
    for label in sorted(by_label):
        shuffled = _shuffled(by_label[label], rng)
        counts = _allocate(len(shuffled), ratios, active)
        cursor = 0
        for split_index in range(3):
            for record_id in shuffled[cursor : cursor + counts[split_index]]:
                assignment[record_id] = SPLIT_NAMES[split_index]
            cursor += counts[split_index]

    return SplitPlan(assignment=assignment, ratios=ratios, seed=seed)


def balance_warnings(plan: SplitPlan, field_name: str) -> list[str]:
    """Dengelemenin ne kadar tuttuğunu denetler ve uyarı metinleri döndürür.

    Dengeleme yapısal olarak imkânsız olabilir (ör. grup sayısı en küçük
    split'ten fazlaysa). Bu durum HATA değildir — bölme yine de geçerlidir —
    ama kullanıcı bilmelidir, çünkü kalan kayma sonuçları etkileyebilir.

    Returns:
        Uyarı metinleri; sorun yoksa boş liste.
    """
    if not plan.groups:
        return []

    warnings: list[str] = []
    distinct = sorted(set(plan.groups.values()))
    if len(distinct) < 2:
        only = distinct[0] if distinct else "—"
        return [
            f"--balance-by '{field_name}' tek bir grup değeri verdi ({only}); dengeleme etkisiz."
        ]
    if len(distinct) == len(plan.groups):
        warnings.append(
            f"--balance-by '{field_name}' her kayda ayrı bir grup verdi "
            f"({len(distinct)} grup / {len(plan.groups)} kayıt); dengeleme anlamsız. "
            "Daha kaba bir alan seçin."
        )

    return warnings


def leakage_warnings(plan: SplitPlan, record_labels: dict[str, str], field_name: str) -> list[str]:
    """Grubun split İÇİNDE etiketi ele verip vermediğini denetler.

    En tehlikeli durum budur: bir split'te her sınıf ayrı gruplara düşerse grup,
    etiketi birebir tahmin eder. Model o kestirmeyi öğrenir; ilişki başka
    split'te değiştiğinde doğruluk şans seviyesinin de altına iner.

    Ayrıca eğitimde hiç görülmeyen gruplarla değerlendirme yapılıp yapılmadığı
    da bildirilir — bu bir hata değil, ama sonuçları okurken bilinmeli.

    Returns:
        Uyarı metinleri; sorun yoksa boş liste.
    """
    if not plan.groups:
        return []

    warnings: list[str] = []
    for name in SPLIT_NAMES:
        records = plan.records_in(name)
        if len(records) < 2:
            continue
        by_label: dict[str, set[str]] = defaultdict(set)
        for record_id in records:
            by_label[record_labels[record_id]].add(plan.groups[record_id])
        if len(by_label) < 2:
            continue
        if not set.intersection(*by_label.values()):
            detail = ", ".join(f"{label}={sorted(gs)}" for label, gs in sorted(by_label.items()))
            warnings.append(
                f"SIZINTI RİSKİ — '{name}' split'inde '{field_name}' değeri sınıfları "
                f"birebir ayırıyor ({detail}). Model etiketi bu alandan okuyabilir; "
                "doğruluk sayıları güvenilmez. Daha fazla kayıt verin veya "
                "--balance-by alanını değiştirin."
            )

    train_groups = {plan.groups[r] for r in plan.records_in("train")}
    for name in ("val", "test"):
        records = plan.records_in(name)
        if not records or not train_groups:
            continue
        unseen = {plan.groups[r] for r in records} - train_groups
        if unseen == {plan.groups[r] for r in records}:
            warnings.append(
                f"'{name}' split'indeki tüm kayıtların '{field_name}' değeri eğitimde hiç "
                f"görülmedi ({sorted(unseen)}). Bu bir dışdeğerleme testidir; doğruluk "
                "beklenenden düşük çıkabilir ve bu bir hata değildir."
            )
    return warnings
