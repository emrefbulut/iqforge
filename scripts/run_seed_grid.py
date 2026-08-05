"""The Phase 4 seed grid: split seed x training seed.

The two seeds are deliberately separated:

  * the **split seed** (`build --seed`) decides the CONTENT of the dataset -
    which recording goes to which split. Its effect answers "what if I had
    picked different recordings?".
  * the **training seed** (`train --seed`) decides only the weight
    initialisation and the batch order. With the dataset fixed, its effect is
    optimisation noise.

Mixing them answers "how stable is my model?" wrongly: the spread across split
seeds is usually far larger than the spread across training seeds, and stability
measured on a single split is misleading.

Usage:
    python scripts/run_seed_grid.py -o artifacts
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from iqforge.cli import _run_build  # noqa: PLC2701 - the build pipeline lives in the CLI
from iqforge.training import train_baseline

SPLIT_SEEDS = (11, 22, 33, 44, 55)
TRAIN_SEEDS = (0, 1, 2)
BALANCE_FIELD = "core:freq_lower_edge"


def build_dataset(source: Path, out_dir: Path, split_seed: int) -> None:
    """Build a dataset with the given split seed."""
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


def contingency(dataset_dir: Path, split: str) -> tuple[list[str], list[float], dict, float]:
    """Build the class x carrier offset contingency table for one split.

    Returns:
        `(labels, offsets, table, max_deviation)`. `max_deviation` is the largest
        absolute difference between the observed cells and what independence
        would predict (row_total * column_total / n); 0 means offset and label
        are fully independent within that split.
    """
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["splits"][split]["records"]
    labels = sorted(manifest["label_map"])
    offsets = sorted(
        {
            r["carrier_offset_hz"]
            for s in ("train", "val", "test")
            for r in manifest["splits"][s]["records"]
        }
    )

    table = {(label, offset): 0 for label in labels for offset in offsets}
    for record in records:
        table[(record["label"], record["carrier_offset_hz"])] += 1

    n = len(records)
    if n == 0:
        return labels, offsets, table, 0.0
    deviation = 0.0
    for label in labels:
        row = sum(table[(label, o)] for o in offsets)
        for offset in offsets:
            column = sum(table[(lab, offset)] for lab in labels)
            expected = row * column / n
            deviation = max(deviation, abs(table[(label, offset)] - expected))
    return labels, offsets, table, deviation


def format_contingency(dataset_dir: Path, split: str) -> list[str]:
    """Render a contingency table as lines of text."""
    labels, offsets, table, deviation = contingency(dataset_dir, split)
    total = sum(table.values())
    lines = [
        f"  {split} ({total} recordings)",
        "    " + "".join(f"{o / 1e3:>+9.0f}k" for o in offsets) + f"{'total':>10}",
    ]
    for label in labels:
        row = "".join(f"{table[(label, o)]:>10d}" for o in offsets)
        lines.append(f"    {label:<6}{row}{sum(table[(label, o)] for o in offsets):>10d}")
    totals = "".join(f"{sum(table[(lab, o)] for lab in labels):>10d}" for o in offsets)
    lines.append(f"    {'total':<6}{totals}{total:>10d}")
    verdict = "INDEPENDENT" if deviation < 1e-9 else f"DEPENDENT (max dev {deviation:.2f})"
    lines.append(f"    -> offset vs label: {verdict}")
    return lines


def split_layout(dataset_dir: Path) -> dict:
    """Extract split -> [(recording, label, offset)] from the manifest."""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        split: [
            (r["id"].replace(".sigmf-meta", ""), r["label"], r["carrier_offset_hz"])
            for r in manifest["splits"][split]["records"]
        ]
        for split in ("train", "val", "test")
    }


def main() -> None:
    """Run the grid and write the log and result table."""
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

    emit(f"source          : {args.source}")
    emit(f"split seeds     : {SPLIT_SEEDS}")
    emit(f"training seeds  : {TRAIN_SEEDS}")
    emit(f"epochs / batch  : {args.epochs} / {args.batch_size}")
    emit(f"--balance-by    : {BALANCE_FIELD}")
    emit("")

    rows: list[dict] = []
    contingency_rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for split_seed in SPLIT_SEEDS:
            dataset_dir = Path(tmp) / f"ds_{split_seed}"
            build_dataset(args.source, dataset_dir, split_seed)
            layout = split_layout(dataset_dir)

            emit(f"=== split seed {split_seed} ===")
            for split, records in layout.items():
                shown = ", ".join(f"{n}({o / 1e3:+.0f}k)" for n, _, o in records) or "-"
                emit(f"  {split:<6}: {shown}")

            emit("  --- class x offset contingency tables ---")
            deviations = {}
            for split in ("train", "val", "test"):
                for line in format_contingency(dataset_dir, split):
                    emit(line)
                deviations[split] = contingency(dataset_dir, split)[3]

            train_offsets = {o for _, _, o in layout["train"]}
            for split in ("val", "test"):
                offsets = {o for _, _, o in layout[split]}
                shared = sorted(o / 1e3 for o in (offsets & train_offsets))
                emit(f"  offsets shared between train and {split} (kHz): {shared or 'NONE'}")
            contingency_rows.append({"split_seed": split_seed, "deviations": deviations})

            for train_seed in TRAIN_SEEDS:
                result = train_baseline(
                    dataset_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=train_seed,
                )
                for epoch in result.epochs:
                    val = "-" if epoch.val_accuracy is None else f"{epoch.val_accuracy:.4f}"
                    emit(
                        f"    [split {split_seed} train {train_seed}] epoch {epoch.epoch:>3} "
                        f"loss {epoch.train_loss:.4f} train {epoch.train_accuracy:.4f} val {val}"
                    )
                per_class = "  ".join(f"{k}={v:.2%}" for k, v in result.test_per_class.items())
                emit(
                    f"  -> training seed {train_seed}: "
                    f"train {result.final_train_accuracy:.2%}  "
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
    emit("RESULT TABLE - test accuracy")
    emit("=" * 68)
    header = (
        "split \\ train | " + " | ".join(f"seed {s:>2}" for s in TRAIN_SEEDS) + " |  mean  |  std"
    )
    emit(header)
    emit("-" * len(header))
    per_split: dict[int, list[float]] = {}
    for split_seed in SPLIT_SEEDS:
        values = [r["test_accuracy"] for r in rows if r["split_seed"] == split_seed]
        per_split[split_seed] = values
        cells = " | ".join(f"{v:7.2%}" for v in values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        emit(f"   seed {split_seed:>3}    | {cells} | {statistics.mean(values):6.2%} | {std:6.2%}")
    emit("-" * len(header))

    split_means = [statistics.mean(v) for v in per_split.values()]
    within_split_stds = [statistics.stdev(v) for v in per_split.values() if len(v) > 1]

    emit("")
    emit(
        f"all runs            : mean {statistics.mean(accuracies):.2%}  "
        f"std {statistics.stdev(accuracies):.2%}  "
        f"min {min(accuracies):.2%}  max {max(accuracies):.2%}"
    )
    emit(
        f"across split seeds  : mean {statistics.mean(split_means):.2%} ± "
        f"{statistics.stdev(split_means):.2%}  (each split's mean over 3 training seeds)"
    )
    mean_within = statistics.mean(within_split_stds)
    emit(
        f"across training seeds: mean std {mean_within:.2%} at a fixed split  "
        f"(largest {max(within_split_stds):.2%})"
    )
    dominant = (
        "the SPLIT seed"
        if statistics.stdev(split_means) > statistics.mean(within_split_stds)
        else "the TRAINING seed"
    )
    emit(f"dominant source of spread: {dominant}")

    worst = max((d for row in contingency_rows for d in row["deviations"].values()), default=0.0)
    emit(
        f"offset-label independence: max deviation across every split seed and split "
        f"{worst:.3f} ({'fully independent' if worst < 1e-9 else 'DEPENDENT - leakage risk'})"
    )

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
        "# Phase 4 - seed grid results",
        "",
        f"- source: `{args.source}`  ·  epochs: {args.epochs}  ·  batch: {args.batch_size}",
        f"- `--balance-by {BALANCE_FIELD}`  ·  model: {rows[0]['parameters']} parameters",
        "",
        "## Test accuracy",
        "",
        "| split seed | " + " | ".join(f"train {s}" for s in TRAIN_SEEDS) + " | mean | std |",
        "|---|" + "---|" * (len(TRAIN_SEEDS) + 2),
    ]
    for split_seed in SPLIT_SEEDS:
        values = per_split[split_seed]
        cells = " | ".join(f"{v:.2%}" for v in values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        lines.append(f"| {split_seed} | {cells} | **{statistics.mean(values):.2%}** | {std:.2%} |")
    lines += [
        "",
        "## Spread",
        "",
        "| source | value |",
        "|---|---|",
        f"| all runs | {statistics.mean(accuracies):.2%} ± {statistics.stdev(accuracies):.2%} "
        f"(min {min(accuracies):.2%}, max {max(accuracies):.2%}) |",
        f"| across split seeds | {statistics.mean(split_means):.2%} ± "
        f"{statistics.stdev(split_means):.2%} |",
        f"| across training seeds (fixed split) | mean std {mean_within:.2%}, "
        f"largest {max(within_split_stds):.2%} |",
        f"| dominant source | {dominant} |",
        "",
    ]
    table_path.write_text("\n".join(lines), encoding="utf-8")

    log.close()
    print(f"\nwritten: {log_path}")
    print(f"written: {args.output / 'train_seed_grid.json'}")
    print(f"written: {table_path}")


if __name__ == "__main__":
    main()
