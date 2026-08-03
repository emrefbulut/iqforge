"""sigkit.splitting testleri — SPEC §5.6'nın kayıt bazlı bölme kuralı."""

from __future__ import annotations

import pytest

from sigkit.io import SigkitError
from sigkit.splitting import SPLIT_NAMES, parse_ratios, stratified_record_split

DEFAULT = (0.7, 0.15, 0.15)


def _records(counts: dict[str, int]) -> dict[str, str]:
    """`{'bpsk': 4}` -> `{'bpsk_00': 'bpsk', ...}`"""
    return {f"{label}_{i:02d}": label for label, n in counts.items() for i in range(n)}


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
    with pytest.raises(SigkitError, match=fragment):
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
    with pytest.raises(SigkitError) as exc:
        stratified_record_split(_records({"bpsk": 1, "qpsk": 4}), DEFAULT, seed=42)

    message = str(exc.value)
    assert "'bpsk' sınıfında yalnızca 1 kayıt" in message
    assert "en az 3 gerekli" in message
    assert "--split 1.0,0,0" in message, "hata mesajı çözüm yolunu söylemeli"
    assert "şişirir" in message, "hata mesajı nedeni söylemeli"


def test_two_records_per_class_fails_for_three_way_split() -> None:
    """İki kayıt üç split'i dolduramaz."""
    with pytest.raises(SigkitError, match="en az 3 gerekli"):
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
    with pytest.raises(SigkitError, match="Bölünecek kayıt yok"):
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


def test_minimum_guarantee_does_not_inflate_small_splits() -> None:
    """Minimum garantisi büyük split'ten çalmalı, oranları baştan bozmamalı.

    Baştan her split'e birer kayıt ayrılan (hatalı) yaklaşım 10 kayıt için
    6/2/2 verir; doğrusu 7/2/1.
    """
    plan = stratified_record_split(_records({"bpsk": 10}), DEFAULT, seed=42)

    assert [len(plan.records_in(n)) for n in SPLIT_NAMES] == [7, 2, 1]
