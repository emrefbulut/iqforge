"""Katmanlı, KAYIT BAZLI train/val/test bölmesi (SPEC §5.6).

Bu modülün tek kuralı var: aynı kayıt dosyasından gelen pencereler aynı split'e
gider. Pencere bazlı bölme, komşu pencereleri hem eğitime hem teste düşürdüğü
için test doğruluğunu yapay olarak şişirir.

Kural uygulanamıyorsa sessizce pencere bazlı bölmeye DÜŞÜLMEZ, hata verilir.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from sigkit.io import SigkitError

#: Split adları, manifest'teki sırayla.
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitPlan:
    """Hangi kaydın hangi split'e gittiğini tutar.

    Attributes:
        assignment: Kayıt kimliği -> split adı.
        ratios: Kullanılan oranlar.
        seed: Kullanılan tohum.
    """

    assignment: dict[str, str]
    ratios: tuple[float, float, float]
    seed: int

    def records_in(self, split: str) -> list[str]:
        """Bir split'e düşen kayıt kimliklerini sıralı olarak verir."""
        return sorted(rid for rid, name in self.assignment.items() if name == split)


def parse_ratios(text: str) -> tuple[float, float, float]:
    """`0.7,0.15,0.15` biçimindeki oran dizgisini çözer.

    Raises:
        SigkitError: Biçim bozuksa, negatif değer varsa veya toplam 1 değilse.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise SigkitError(
            f"--split üç değer bekliyor (train,val,test), {len(parts)} verildi: '{text}'. "
            "Örnek: --split 0.7,0.15,0.15"
        )
    try:
        values = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise SigkitError(
            f"--split sayısal olmalı: '{text}'. Örnek: --split 0.7,0.15,0.15"
        ) from exc

    if any(v < 0 for v in values):
        raise SigkitError(f"--split değerleri negatif olamaz: '{text}'.")
    if abs(sum(values) - 1.0) > 1e-6:
        raise SigkitError(f"--split değerlerinin toplamı 1 olmalı, {sum(values):.6g} verildi.")
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


def stratified_record_split(
    record_labels: dict[str, str],
    ratios: tuple[float, float, float],
    seed: int,
) -> SplitPlan:
    """Kayıtları etikete göre katmanlı biçimde split'lere dağıtır.

    Bölme KAYIT bazındadır: bir kaydın tüm pencereleri aynı split'e gider.
    Aynı `seed` ile birebir aynı sonucu verir.

    Args:
        record_labels: Kayıt kimliği -> o kaydın baskın etiketi.
        ratios: `(train, val, test)` oranları.
        seed: Deterministik karıştırma tohumu.

    Returns:
        Kayıt ataması.

    Raises:
        SigkitError: Bir sınıfın kayıt sayısı, oranların gerektirdiği boş olmayan
            split'leri dolduramayacak kadar azsa.
    """
    if not record_labels:
        raise SigkitError("Bölünecek kayıt yok: girdide etiketlenebilir pencere bulunamadı.")

    active = [i for i, r in enumerate(ratios) if r > 0]
    by_label: dict[str, list[str]] = defaultdict(list)
    for record_id, label in record_labels.items():
        by_label[label].append(record_id)

    for label in sorted(by_label):
        available = len(by_label[label])
        if available < len(active):
            needed = ", ".join(f"{SPLIT_NAMES[i]}={ratios[i]:g}" for i in active)
            raise SigkitError(
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
    assignment: dict[str, str] = {}
    for label in sorted(by_label):
        records = sorted(by_label[label])
        order = rng.permutation(len(records))
        shuffled = [records[i] for i in order]

        counts = _allocate(len(shuffled), ratios, active)
        cursor = 0
        for split_index in range(3):
            for record_id in shuffled[cursor : cursor + counts[split_index]]:
                assignment[record_id] = SPLIT_NAMES[split_index]
            cursor += counts[split_index]

    return SplitPlan(assignment=assignment, ratios=ratios, seed=seed)
