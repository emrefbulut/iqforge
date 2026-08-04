"""Kontrol deneyi: test doğruluğu neden %50 civarında kalıyor?

`examples/` her (sınıf, taşıyıcı ofset) çifti için TEK kayıt içerir. Split
içinde ofsetin etiketi ele vermemesi için bir ofsetin tüm kayıtları aynı split'e
gitmek zorundadır — dolayısıyla train ile test hiçbir ofseti paylaşamaz ve model
hiç görmediği taşıyıcı frekanslarında değerlendirilir.

Bu script değişkeni tamamen izole eder: TÜM kayıtlar aynı taşıyıcı ofsetinde
üretilir. Böylece train ile test ofseti zorunlu olarak paylaşır ve geriye tek
soru kalır — boru hattı ve model, modülasyon türünü ayırt edebiliyor mu?

Doğruluk yükselirse boru hattında ve modelde sorun yok demektir; `examples/`
üzerindeki ~%50, taşıyıcı ofsetine genelleme sorunudur.

Kullanım:
    python scripts/control_shared_offsets.py -o artifacts
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_example import (  # noqa: E402
    BASE_SEED,
    RecordPlan,
    build_signal,
    write_record,
)

from iqforge.cli import _run_build  # noqa: E402, PLC2701
from iqforge.training import train_baseline  # noqa: E402

#: Tek ofset: taşıyıcı frekansı değişken olmaktan çıkarılır.
FIXED_OFFSET = 180_000.0
BURST_STARTS = (4_096, 12_288, 8_192, 16_384)
RECORDS_PER_CLASS = 4
TRAIN_SEEDS = (0, 1, 2)


def make_records(out_dir: Path) -> int:
    """Hepsi aynı taşıyıcı ofsetinde, sınıf başına `RECORDS_PER_CLASS` kayıt üretir."""
    index = 0
    for modulation in ("bpsk", "qpsk"):
        for repeat in range(RECORDS_PER_CLASS):
            plan = RecordPlan(
                name=f"{modulation}_{repeat}",
                modulation=modulation,
                carrier_offset=FIXED_OFFSET,
                burst_start=BURST_STARTS[repeat % len(BURST_STARTS)],
                seed=BASE_SEED + 1000 + index,
            )
            write_record(plan, build_signal(plan), out_dir)
            index += 1
    return index


def shared_offsets(dataset_dir: Path) -> tuple[set[float], set[float], set[float]]:
    """Manifest'ten train/val/test ofset kümelerini çıkarır."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return tuple(  # type: ignore[return-value]
        {r["carrier_offset_hz"] for r in manifest["splits"][s]["records"]}
        for s in ("train", "val", "test")
    )


def main() -> None:
    """Kontrol deneyini çalıştırır ve sonucu yazar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="artifacts", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "control_shared_offsets.log"
    log = log_path.open("w", encoding="utf-8")

    def emit(text: str = "") -> None:
        print(text)
        log.write(text + "\n")
        log.flush()

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "records"
        count = make_records(source)
        dataset_dir = Path(tmp) / "ds"
        _run_build(
            input_path=source,
            output=dataset_dir,
            window=1024,
            stride=512,
            source="annotations",
            label_file=None,
            exclude_label=None,
            keep_unlabeled=False,
            split="0.7,0.15,0.15",
            seed=11,
            balance_by="core:freq_lower_edge",
            representation="iq2ch",
            normalize=True,
        )

        train_offsets, val_offsets, test_offsets = shared_offsets(dataset_dir)
        emit(f"kayıt sayısı        : {count} (hepsi {FIXED_OFFSET / 1e3:+.0f} kHz ofsetinde)")
        emit(f"train ofsetleri (kHz): {sorted(o / 1e3 for o in train_offsets)}")
        emit(f"val   ofsetleri (kHz): {sorted(o / 1e3 for o in val_offsets)}")
        emit(f"test  ofsetleri (kHz): {sorted(o / 1e3 for o in test_offsets)}")
        emit(f"train ile test PAYLAŞILAN ofset: {sorted(train_offsets & test_offsets)}")
        emit("")

        accuracies = []
        for seed in TRAIN_SEEDS:
            result = train_baseline(dataset_dir, epochs=args.epochs, seed=seed)
            accuracies.append(result.test_accuracy)
            per_class = "  ".join(f"{k}={v:.2%}" for k, v in result.test_per_class.items())
            emit(
                f"eğitim tohumu {seed}: eğitim {result.final_train_accuracy:.2%}  "
                f"test {result.test_accuracy:.2%}  ({per_class})"
            )

    emit("")
    emit(
        f"test doğruluğu: ortalama {statistics.mean(accuracies):.2%}  "
        f"min {min(accuracies):.2%}  maks {max(accuracies):.2%}"
    )
    emit("")
    if statistics.mean(accuracies) > 0.75:
        emit("SONUÇ: taşıyıcı ofseti sabitlenince doğruluk yükseliyor.")
        emit("       Boru hattı, etiketleme ve model doğru çalışıyor; examples/")
        emit("       üzerindeki ~%50 taşıyıcı ofsetine genelleme sorunudur,")
        emit("       pipeline bug'ı değil.")
    else:
        emit("SONUÇ: ofset sabitken bile doğruluk düşük — boru hattında veya")
        emit("       modelde sorun olabilir, araştırılmalı.")
    log.close()
    print(f"\nyazıldı: {log_path}")


if __name__ == "__main__":
    main()
