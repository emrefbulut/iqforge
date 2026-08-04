"""iqforge.splitting testleri — SPEC §5.6'nın kayıt bazlı bölme kuralı."""

from __future__ import annotations

import pytest

from iqforge.io import IQForgeError
from iqforge.splitting import (
    SPLIT_NAMES,
    balance_warnings,
    leakage_warnings,
    parse_ratios,
    stratified_record_split,
)

DEFAULT = (0.7, 0.15, 0.15)

#: Örnek veri setiyle aynı yapı: iki sınıf, dört taşıyıcı ofset grubu, her
#: sınıf her grubu birer kez kullanıyor.
OFFSET_GROUPS = ("-280", "-180", "+180", "+280")


def _records(counts: dict[str, int]) -> dict[str, str]:
    """`{'bpsk': 4}` -> `{'bpsk_00': 'bpsk', ...}`"""
    return {f"{label}_{i:02d}": label for label, n in counts.items() for i in range(n)}


def _offset_groups(records: dict[str, str]) -> dict[str, str]:
    """Her kayda sırayla bir taşıyıcı ofset grubu atar."""
    groups: dict[str, str] = {}
    per_label: dict[str, int] = {}
    for record_id, label in sorted(records.items()):
        index = per_label.get(label, 0)
        groups[record_id] = OFFSET_GROUPS[index % len(OFFSET_GROUPS)]
        per_label[label] = index + 1
    return groups


def test_parse_ratios_accepts_valid_input() -> None:
    """Geçerli oran dizgisi üç float olarak çözülmeli."""
    assert parse_ratios("0.7,0.15,0.15") == (0.7, 0.15, 0.15)
    assert parse_ratios(" 1.0 , 0 , 0 ") == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("0.7,0.3", "üç değer"),
        ("a,b,c", "sayısal"),
        ("0.5,0.5,0.5", "toplamı 1"),
        ("1.5,-0.5,0", "negatif"),
    ],
)
def test_parse_ratios_rejects_invalid_input(text: str, fragment: str) -> None:
    """Bozuk oran dizgisi ne yapılacağını söyleyen hata vermeli."""
    with pytest.raises(IQForgeError, match=fragment):
        parse_ratios(text)


def test_every_record_lands_in_exactly_one_split() -> None:
    """Bir kayıt tek bir split'e gitmeli — bölmenin temel kuralı."""
    records = _records({"bpsk": 4, "qpsk": 4})
    plan = stratified_record_split(records, DEFAULT, seed=42)

    assert set(plan.assignment) == set(records)
    placed = [r for name in SPLIT_NAMES for r in plan.records_in(name)]
    assert sorted(placed) == sorted(records)
    assert len(placed) == len(set(placed)), "bir kayıt birden fazla split'te"


def test_each_class_is_present_in_every_split() -> None:
    """Katmanlı bölme her sınıfı her split'te temsil etmeli."""
    records = _records({"bpsk": 4, "qpsk": 4})
    plan = stratified_record_split(records, DEFAULT, seed=42)

    for name in SPLIT_NAMES:
        labels = {records[r] for r in plan.records_in(name)}
        assert labels == {"bpsk", "qpsk"}, f"{name} split'inde eksik sınıf: {labels}"


def test_split_sizes_follow_ratios_as_closely_as_possible() -> None:
    """Sınıf başına 4 kayıt, 0.7/0.15/0.15 için 2/1/1 vermeli."""
    plan = stratified_record_split(_records({"bpsk": 4, "qpsk": 4}), DEFAULT, seed=42)

    assert [len(plan.records_in(n)) for n in SPLIT_NAMES] == [4, 2, 2]


def test_same_seed_gives_identical_split() -> None:
    """Aynı seed birebir aynı bölmeyi vermeli (SPEC §5.6 determinizm)."""
    records = _records({"bpsk": 4, "qpsk": 4})

    first = stratified_record_split(records, DEFAULT, seed=42)
    second = stratified_record_split(records, DEFAULT, seed=42)

    assert first.assignment == second.assignment


def test_different_seed_changes_the_split() -> None:
    """Farklı seed farklı bölme vermeli, aksi halde tohum işe yaramıyordur."""
    records = _records({"bpsk": 4, "qpsk": 4})

    assignments = {
        tuple(sorted(stratified_record_split(records, DEFAULT, seed=s).assignment.items()))
        for s in range(8)
    }

    assert len(assignments) > 1


def test_record_order_does_not_affect_the_split() -> None:
    """Girdi sözlüğünün sırası sonucu değiştirmemeli."""
    records = _records({"bpsk": 4, "qpsk": 4})
    reversed_records = dict(reversed(list(records.items())))

    a = stratified_record_split(records, DEFAULT, seed=42)
    b = stratified_record_split(reversed_records, DEFAULT, seed=42)

    assert a.assignment == b.assignment


def test_too_few_records_raises_instead_of_falling_back() -> None:
    """Kayıt sayısı yetmiyorsa HATA verilmeli, pencere bazlı bölmeye düşülmemeli."""
    with pytest.raises(IQForgeError) as exc:
        stratified_record_split(_records({"bpsk": 1, "qpsk": 4}), DEFAULT, seed=42)

    message = str(exc.value)
    assert "'bpsk' sınıfında yalnızca 1 kayıt" in message
    assert "en az 3 gerekli" in message
    assert "--split 1.0,0,0" in message, "hata mesajı çözüm yolunu söylemeli"
    assert "şişirir" in message, "hata mesajı nedeni söylemeli"


def test_two_records_per_class_fails_for_three_way_split() -> None:
    """İki kayıt üç split'i dolduramaz."""
    with pytest.raises(IQForgeError, match="en az 3 gerekli"):
        stratified_record_split(_records({"bpsk": 2, "qpsk": 2}), DEFAULT, seed=42)


def test_two_way_split_needs_only_two_records() -> None:
    """Sıfır oranlı split'ler zorunluluk saymamalı."""
    plan = stratified_record_split(_records({"bpsk": 2, "qpsk": 2}), (0.5, 0.5, 0.0), seed=42)

    assert [len(plan.records_in(n)) for n in SPLIT_NAMES] == [2, 2, 0]


def test_single_record_allowed_with_train_only_split() -> None:
    """--split 1.0,0,0 tek kayıtla açık kaçış yolu olmalı."""
    plan = stratified_record_split({"solo": "bpsk"}, (1.0, 0.0, 0.0), seed=42)

    assert plan.records_in("train") == ["solo"]
    assert plan.records_in("val") == []
    assert plan.records_in("test") == []


def test_empty_input_is_rejected() -> None:
    """Etiketlenebilir kayıt yoksa açık hata verilmeli."""
    with pytest.raises(IQForgeError, match="Bölünecek kayıt yok"):
        stratified_record_split({}, DEFAULT, seed=42)


def test_uneven_class_sizes_are_stratified_independently() -> None:
    """Farklı büyüklükteki sınıflar kendi içlerinde oranlanmalı.

    10 kayıt için 0.7/0.15/0.15 -> 7.0/1.5/1.5, en-büyük-kalan ile 7/2/1.
    4 kayıt için 2.8/0.6/0.6 -> 3/1/0, minimum garantisiyle 2/1/1.
    """
    records = _records({"bpsk": 10, "qpsk": 4})
    plan = stratified_record_split(records, DEFAULT, seed=42)

    bpsk_per_split = {
        name: [records[r] for r in plan.records_in(name)].count("bpsk") for name in SPLIT_NAMES
    }
    qpsk_per_split = {
        name: [records[r] for r in plan.records_in(name)].count("qpsk") for name in SPLIT_NAMES
    }

    assert bpsk_per_split == {"train": 7, "val": 2, "test": 1}
    assert qpsk_per_split == {"train": 2, "val": 1, "test": 1}


@pytest.mark.parametrize("seed", range(8))
def test_group_never_predicts_the_label_within_a_split(seed: int) -> None:
    """Bir split'in İÇİNDE grup, etiketi tahmin edebilir olmamalı.

    Bu dengelemenin asıl amacı. İhlal edilirse felaket olur: gruplar sınıflar
    arasında tamamlayıcı dağıtılırsa (train'de bpsk pozitif ofsetleri, qpsk
    negatifleri alırsa) model kestirmeyi öğrenir, eğitimde %100 yapar ve ilişki
    başka split'te ters döndüğü için testte %0'a düşer — şansın da altına.
    Bu gerileme gerçekten yaşandı; test onu kilitliyor.
    """
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=seed, record_groups=groups)

    for name in SPLIT_NAMES:
        in_split = plan.records_in(name)
        if len(in_split) < 2:
            continue
        by_label: dict[str, set[str]] = {}
        for record_id in in_split:
            by_label.setdefault(records[record_id], set()).add(groups[record_id])
        assert set.intersection(*by_label.values()), (
            f"seed {seed}, '{name}': grup etiketi ele veriyor -> {by_label}"
        )


def test_each_group_stays_in_a_single_split_when_it_has_one_record_per_class() -> None:
    """Grup başına sınıfta tek kayıt varsa grup bölünemez, bütün olarak yerleşir.

    Bu, yukarıdaki bağımsızlık garantisinin bedelidir: train ile test aynı grubu
    paylaşamaz. `leakage_warnings` bunu dışdeğerleme uyarısı olarak bildirir.
    """
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)

    where: dict[str, set[str]] = {}
    for record_id, split in plan.assignment.items():
        where.setdefault(groups[record_id], set()).add(split)
    assert all(len(splits) == 1 for splits in where.values()), where


def test_balancing_preserves_class_stratification() -> None:
    """Dengeleme sınıf başına split kayıt sayılarını değiştirmemeli."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plain = stratified_record_split(records, DEFAULT, seed=42)
    balanced = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)

    for name in SPLIT_NAMES:
        for label in ("bpsk", "qpsk"):
            assert [records[r] for r in plain.records_in(name)].count(label) == [
                records[r] for r in balanced.records_in(name)
            ].count(label)


def test_balancing_is_deterministic() -> None:
    """Aynı seed dengeli bölmede de birebir aynı sonucu vermeli."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    first = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)
    second = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)

    assert first.assignment == second.assignment
    assert first.groups == groups


def test_balancing_keeps_the_record_level_guarantee() -> None:
    """Dengeleme kayıt bazlı bölme kuralını bozmamalı."""
    records = _records({"bpsk": 6, "qpsk": 6})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=3, record_groups=groups)

    placed = [r for name in SPLIT_NAMES for r in plan.records_in(name)]
    assert sorted(placed) == sorted(records)
    assert len(placed) == len(set(placed))


def test_balance_warning_when_every_record_is_its_own_group() -> None:
    """Her kayıt ayrı gruba düşerse dengeleme anlamsızdır, uyarı verilmeli."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = {r: f"grup_{r}" for r in records}

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)
    warnings = balance_warnings(plan, "core:description")

    assert any("her kayda ayrı bir grup" in w for w in warnings)


def test_balance_warning_when_only_one_group_exists() -> None:
    """Tek grup varsa dengeleme etkisizdir, uyarı verilmeli."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = dict.fromkeys(records, "tek")

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)
    warnings = balance_warnings(plan, "core:hw")

    assert any("dengeleme etkisiz" in w for w in warnings)


def test_leakage_warning_when_group_separates_classes_within_a_split() -> None:
    """Grup bir split'te sınıfları ayırıyorsa yüksek sesle uyarılmalı.

    Dengeleme kullanılmadığında (ya da alan sınıfla ilişkiliyse) bu olabilir;
    kullanıcı doğruluk sayılarına güvenmeden önce bilmeli.
    """
    records = _records({"bpsk": 2, "qpsk": 2})
    # Grup etiketle birebir örtüşüyor: en kötü durum.
    groups = {r: ("g_bpsk" if records[r] == "bpsk" else "g_qpsk") for r in records}

    plan = stratified_record_split(records, (0.5, 0.5, 0.0), seed=42, record_groups=groups)
    warnings = leakage_warnings(plan, records, "core:hw")

    assert any("SIZINTI RİSKİ" in w for w in warnings)


def test_leakage_warning_when_evaluation_groups_are_unseen_in_training() -> None:
    """Test grubu eğitimde hiç görülmediyse bu dışdeğerlemedir, bildirilmeli."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)
    warnings = leakage_warnings(plan, records, "core:freq_lower_edge")

    assert any("eğitimde hiç" in w and "test" in w for w in warnings)


def test_no_leakage_warning_without_balancing() -> None:
    """--balance-by kullanılmadıysa sızıntı denetimi sessiz kalmalı."""
    records = _records({"bpsk": 4, "qpsk": 4})
    plan = stratified_record_split(records, DEFAULT, seed=42)

    assert leakage_warnings(plan, records, "core:hw") == []


def test_balance_warnings_are_empty_when_balancing_works() -> None:
    """Dengeleme tuttuğunda uyarı üretilmemeli."""
    records = _records({"bpsk": 4, "qpsk": 4})
    groups = _offset_groups(records)

    plan = stratified_record_split(records, DEFAULT, seed=42, record_groups=groups)

    assert balance_warnings(plan, "core:freq_lower_edge") == []


def test_balance_warnings_empty_without_balancing() -> None:
    """--balance-by kullanılmadıysa uyarı üretilmemeli."""
    plan = stratified_record_split(_records({"bpsk": 4, "qpsk": 4}), DEFAULT, seed=42)

    assert balance_warnings(plan, "core:hw") == []


def test_minimum_guarantee_does_not_inflate_small_splits() -> None:
    """Minimum garantisi büyük split'ten çalmalı, oranları baştan bozmamalı.

    Baştan her split'e birer kayıt ayrılan (hatalı) yaklaşım 10 kayıt için
    6/2/2 verir; doğrusu 7/2/1.
    """
    plan = stratified_record_split(_records({"bpsk": 10}), DEFAULT, seed=42)

    assert [len(plan.records_in(n)) for n in SPLIT_NAMES] == [7, 2, 1]
