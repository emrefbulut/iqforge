"""Faz 4 tohum ızgarası: bölme tohumu x eğitim tohumu.

İki tohum bilerek ayrılmıştır:

  * **bölme tohumu** (`build --seed`) veri setinin İÇERİĞİNİ belirler — hangi
    kayıt hangi split'e gider. Bunun etkisi "başka bir kayıt seçseydim ne
    olurdu" sorusunun cevabıdır.
  * **eğitim tohumu** (`train --seed`) yalnızca ağırlık ilklendirmesini ve batch
    sırasını belirler. Veri seti sabitken bunun etkisi optimizasyon gürültüsüdür.

İkisi karıştırılırsa "modelim ne kadar kararlı" sorusuna yanlış cevap verilir:
bölme tohumu saçılması genelde eğitim tohumu saçılmasından çok daha büyüktür ve
tek bir bölmede ölçülen kararlılık aldatıcıdır.

Kullanım:
    python scripts/run_seed_grid.py -o artifacts
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from iqforge.cli import _run_build  # noqa: PLC2701 — build boru hattı CLI'da yaşıyor
from iqforge.training import train_baseline

SPLIT_SEEDS = (11, 22, 33, 44, 55)
TRAIN_SEEDS = (0, 1, 2)
BALANCE_FIELD = "core:freq_lower_edge"


def build_dataset(source: Path, out_dir: Path, split_seed: int) -> None:
    """Verilen bölme tohumuyla bir veri seti kurar."""
    _run_build(
        input_path=source,
        output=out_dir,
        window=1024,
        stride=512,
        source="annotations",
        label_file=None,
        exclude_label=None,
        keep_unlabeled=False,
        split="0.7,0.15,0.15",
        seed=split_seed,
        balance_by=BALANCE_FIELD,
        representation="iq2ch",
        normalize=True,
    )


def split_layout(dataset_dir: Path) -> dict:
    """Manifest'ten split -> [(kayıt, etiket, ofset)] çıkarır."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        split: [
            (r["id"].replace(".sigmf-meta", ""), r["label"], r["carrier_offset_hz"])
            for r in manifest["splits"][split]["records"]
        ]
        for split in ("train", "val", "test")
    }


def main() -> None:
    """Izgarayı çalıştırır, log ve sonuç tablosunu yazar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="examples", type=Path)
    parser.add_argument("-o", "--output", default="artifacts", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "train_seed_grid.log"
    log = log_path.open("w", encoding="utf-8")

    def emit(text: str = "") -> None:
        print(text)
        log.write(text + "\n")
        log.flush()

    emit(f"kaynak            : {args.source}")
    emit(f"bölme tohumları   : {SPLIT_SEEDS}")
    emit(f"eğitim tohumları  : {TRAIN_SEEDS}")
    emit(f"epoch / batch     : {args.epochs} / {args.batch_size}")
    emit(f"--balance-by      : {BALANCE_FIELD}")
    emit("")

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for split_seed in SPLIT_SEEDS:
            dataset_dir = Path(tmp) / f"ds_{split_seed}"
            build_dataset(args.source, dataset_dir, split_seed)
            layout = split_layout(dataset_dir)

            emit(f"=== bölme tohumu {split_seed} ===")
            for split, records in layout.items():
                shown = ", ".join(f"{n}({o / 1e3:+.0f}k)" for n, _, o in records) or "—"
                emit(f"  {split:<6}: {shown}")

            for train_seed in TRAIN_SEEDS:
                result = train_baseline(
                    dataset_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=train_seed,
                )
                for epoch in result.epochs:
                    val = "—" if epoch.val_accuracy is None else f"{epoch.val_accuracy:.4f}"
                    emit(
                        f"    [split {split_seed} train {train_seed}] epoch {epoch.epoch:>3} "
                        f"kayıp {epoch.train_loss:.4f} eğitim {epoch.train_accuracy:.4f} val {val}"
                    )
                per_class = "  ".join(f"{k}={v:.2%}" for k, v in result.test_per_class.items())
                emit(
                    f"  -> eğitim tohumu {train_seed}: "
                    f"eğitim {result.final_train_accuracy:.2%}  "
                    f"test {result.test_accuracy:.2%}  ({per_class})"
                )
                rows.append(
                    {
                        "split_seed": split_seed,
                        "train_seed": train_seed,
                        "parameters": result.parameters,
                        "train_accuracy": result.final_train_accuracy,
                        "test_accuracy": result.test_accuracy,
                        "test_per_class": result.test_per_class,
                    }
                )
            emit("")

    accuracies = [r["test_accuracy"] for r in rows]

    emit("=" * 68)
    emit("SONUÇ TABLOSU — test doğruluğu")
    emit("=" * 68)
    header = (
        "bölme \\ eğitim | " + " | ".join(f"seed {s:>2}" for s in TRAIN_SEEDS) + " |  ort.  |  std"
    )
    emit(header)
    emit("-" * len(header))
    per_split: dict[int, list[float]] = {}
    for split_seed in SPLIT_SEEDS:
        values = [r["test_accuracy"] for r in rows if r["split_seed"] == split_seed]
        per_split[split_seed] = values
        cells = " | ".join(f"{v:7.2%}" for v in values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        emit(f"    seed {split_seed:>3}    | {cells} | {statistics.mean(values):6.2%} | {std:6.2%}")
    emit("-" * len(header))

    split_means = [statistics.mean(v) for v in per_split.values()]
    within_split_stds = [statistics.stdev(v) for v in per_split.values() if len(v) > 1]

    emit("")
    emit(
        f"tüm koşular          : ortalama {statistics.mean(accuracies):.2%}  "
        f"std {statistics.stdev(accuracies):.2%}  "
        f"min {min(accuracies):.2%}  maks {max(accuracies):.2%}"
    )
    emit(
        f"bölme tohumları arası: ortalama {statistics.mean(split_means):.2%} ± "
        f"{statistics.stdev(split_means):.2%}  (her bölmenin 3 eğitim tohumu ortalaması)"
    )
    mean_within = statistics.mean(within_split_stds)
    emit(
        f"eğitim tohumları arası: sabit bölmede ortalama std {mean_within:.2%}  "
        f"(en büyük {max(within_split_stds):.2%})"
    )
    dominant = (
        "BÖLME tohumu"
        if statistics.stdev(split_means) > statistics.mean(within_split_stds)
        else "EĞİTİM tohumu"
    )
    emit(f"baskın saçılma kaynağı: {dominant}")

    (args.output / "train_seed_grid.json").write_text(
        json.dumps(
            {
                "split_seeds": list(SPLIT_SEEDS),
                "train_seeds": list(TRAIN_SEEDS),
                "epochs": args.epochs,
                "balance_by": BALANCE_FIELD,
                "parameters": rows[0]["parameters"],
                "runs": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    table_path = args.output / "train_seed_grid.md"
    lines = [
        "# Faz 4 — tohum ızgarası sonuçları",
        "",
        f"- kaynak: `{args.source}`  ·  epoch: {args.epochs}  ·  batch: {args.batch_size}",
        f"- `--balance-by {BALANCE_FIELD}`  ·  model: {rows[0]['parameters']} parametre",
        "",
        "## Test doğruluğu",
        "",
        "| bölme tohumu | "
        + " | ".join(f"eğitim {s}" for s in TRAIN_SEEDS)
        + " | ortalama | std |",
        "|---|" + "---|" * (len(TRAIN_SEEDS) + 2),
    ]
    for split_seed in SPLIT_SEEDS:
        values = per_split[split_seed]
        cells = " | ".join(f"{v:.2%}" for v in values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(f"| {split_seed} | {cells} | **{statistics.mean(values):.2%}** | {std:.2%} |")
    lines += [
        "",
        "## Saçılma",
        "",
        "| kaynak | değer |",
        "|---|---|",
        f"| tüm koşular | {statistics.mean(accuracies):.2%} ± {statistics.stdev(accuracies):.2%} "
        f"(min {min(accuracies):.2%}, maks {max(accuracies):.2%}) |",
        f"| bölme tohumları arası | {statistics.mean(split_means):.2%} ± "
        f"{statistics.stdev(split_means):.2%} |",
        f"| eğitim tohumları arası (sabit bölmede) | ortalama std {mean_within:.2%}, "
        f"en büyük {max(within_split_stds):.2%} |",
        f"| baskın kaynak | {dominant} |",
        "",
    ]
    table_path.write_text("\n".join(lines), encoding="utf-8")

    log.close()
    print(f"\nyazıldı: {log_path}")
    print(f"yazıldı: {args.output / 'train_seed_grid.json'}")
    print(f"yazıldı: {table_path}")


if __name__ == "__main__":
    main()
