"""iqforge komut satırı arayüzü."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

from iqforge import __version__
from iqforge.display import render_inspect
from iqforge.io import META_EXT, IQForgeError, Recording, load
from iqforge.labeling import (
    LABEL_SOURCES,
    LabelingStats,
    annotation_field_value,
    carrier_offset_hz,
    dominant_label,
    label_from_annotations,
    label_from_csv,
    label_from_dirname,
    load_label_csv,
    resolve_exclude_labels,
)
from iqforge.splitting import (
    SPLIT_NAMES,
    SplitPlan,
    balance_warnings,
    parse_ratios,
    stratified_record_split,
)
from iqforge.storage import (
    ShardWriter,
    dataset_size_bytes,
    read_manifest,
    write_manifest,
)
from iqforge.windowing import (
    REPRESENTATIONS,
    iter_window_batches,
    normalize_windows,
    to_representation,
    validate_window_params,
    window_starts,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SigMF kayıtlarını makine öğrenmesine hazır veri setlerine çevirir.",
)
console = Console()
err_console = Console(stderr=True)


def _format_hz(value: float | None) -> str:
    """Hz değerini okunabilir birimle biçimlendirir."""
    if value is None:
        return "—"
    for unit, scale in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.6g} {unit} ({value:.0f} Hz)"
    return f"{value:.0f} Hz"


def _render_overview(rec: Recording) -> Table:
    """Kaydın temel metadata'sını tablo olarak hazırlar."""
    table = Table(title=rec.meta_path.name, title_style="bold", show_header=False)
    table.add_column("Alan", style="cyan", no_wrap=True)
    table.add_column("Değer", style="white")

    g = rec.global_info
    table.add_row("Örnekleme hızı", _format_hz(rec.sample_rate))
    table.add_row("Merkez frekans", _format_hz(rec.center_frequency))
    table.add_row("Veri tipi", rec.datatype)
    table.add_row("Örnek sayısı", f"{rec.num_samples:,}".replace(",", " "))
    table.add_row("Süre", f"{rec.duration_seconds:.6g} s")
    table.add_row("Veri dosyası", f"{rec.data_path.stat().st_size / 1e6:.2f} MB")
    table.add_row("Donanım", str(g.get("core:hw", "—")))
    table.add_row("Yazar", str(g.get("core:author", "—")))
    table.add_row("Kaydeden", str(g.get("core:recorder", "—")))
    table.add_row("SigMF sürümü", str(g.get("core:version", "—")))
    if g.get("core:description"):
        table.add_row("Açıklama", str(g["core:description"]))
    return table


def _render_annotations(rec: Recording) -> Table:
    """Annotation listesini tablo olarak hazırlar."""
    table = Table(title=f"Annotation'lar ({len(rec.annotations)})", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Etiket", style="green")
    table.add_column("Başlangıç", justify="right")
    table.add_column("Örnek", justify="right")
    table.add_column("Zaman (s)", justify="right")
    table.add_column("Frekans aralığı")

    for i, a in enumerate(rec.annotations):
        t0 = a.sample_start / rec.sample_rate
        t1 = a.sample_end / rec.sample_rate
        if a.freq_lower_edge is not None and a.freq_upper_edge is not None:
            lo = (a.freq_lower_edge - (rec.center_frequency or 0.0)) / 1e3
            hi = (a.freq_upper_edge - (rec.center_frequency or 0.0)) / 1e3
            freq = f"merkez {lo:+.1f} … {hi:+.1f} kHz"
        else:
            freq = "—"
        table.add_row(
            str(i),
            a.label or "—",
            f"{a.sample_start:,}".replace(",", " "),
            f"{a.sample_count:,}".replace(",", " "),
            f"{t0:.4f} → {t1:.4f}",
            freq,
        )
    return table


@app.command()
def info(
    path: Annotated[Path, typer.Argument(help="SigMF kaydı (.sigmf-meta veya kayıt adı)")],
) -> None:
    """SigMF kaydının metadata'sını okunabilir tablo olarak yazdırır."""
    try:
        rec = load(path)
    except IQForgeError as exc:
        err_console.print(f"[bold red]Hata:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(_render_overview(rec))
    if rec.annotations:
        console.print(_render_annotations(rec))
    else:
        console.print("[dim]Kayıtta annotation yok.[/]")


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(help="SigMF kaydı (.sigmf-meta veya kayıt adı)")],
    start: Annotated[int, typer.Option("--start", help="Kaçıncı örnekten başlasın")] = 0,
    samples: Annotated[int, typer.Option("--samples", help="Kaç örnek gösterilsin")] = 262_144,
    nfft: Annotated[int, typer.Option("--nfft", help="FFT uzunluğu")] = 1024,
    height: Annotated[
        int | None, typer.Option("--height", help="Spektrogram yüksekliği (karakter satırı)")
    ] = None,
) -> None:
    """Terminalde spektrogram ve zaman ekseninde güç grafiği çizer."""
    try:
        rec = load(path)
        data = rec.read(start=start, count=samples)
        rows = height if height is not None else max(8, min(24, console.size.height - 10))
        panel = render_inspect(rec, data, start, nfft, width=console.size.width, height=rows)
    except IQForgeError as exc:
        err_console.print(f"[bold red]Hata:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        err_console.print(f"[bold red]Hata:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(panel)


@dataclass
class _RecordWork:
    """Tek bir kaydın build sırasında taşınan durumu.

    Attributes:
        record_id: Kaydın veri seti içindeki benzersiz kimliği (göreli yol).
        recording: Açılmış kayıt.
        indices: Etiket alan pencerelerin indisleri.
        labels: `indices` ile aynı sırada etiketler.
        dominant: Kaydın baskın etiketi; katmanlı bölme bunu kullanır.
    """

    record_id: str
    recording: Recording
    indices: np.ndarray
    labels: list[str]
    dominant: str
    offset_hz: float | None = None
    group: str | None = None


def _collect_inputs(path: Path) -> tuple[list[Path], Path]:
    """Girdi yolundan `.sigmf-meta` dosyalarını toplar.

    Returns:
        `(meta_yolları, kök_klasör)` — kök, kayıt kimliklerinin göreli
        hesaplanacağı klasördür.

    Raises:
        IQForgeError: Yol yoksa veya hiç kayıt bulunamazsa.
    """
    if not path.exists():
        raise IQForgeError(f"Girdi bulunamadı: {path}")
    if path.is_dir():
        metas = sorted(path.rglob(f"*{META_EXT}"))
        if not metas:
            raise IQForgeError(
                f"'{path}' içinde (alt klasörler dahil) hiç '*{META_EXT}' dosyası yok."
            )
        return metas, path
    return [path], path.parent


def _label_one(
    rec: Recording,
    starts: np.ndarray,
    *,
    source: str,
    window: int,
    exclude: frozenset[str],
    keep_unlabeled: bool,
    csv_table: dict[str, str] | None,
) -> tuple[list[str | None], LabelingStats]:
    """Seçilen kaynağa göre bir kaydın pencerelerini etiketler."""
    if source == "annotations":
        return label_from_annotations(rec, starts, window, exclude, keep_unlabeled)
    if source == "dirname":
        return label_from_dirname(rec, starts, exclude)
    if source == "csv":
        assert csv_table is not None  # CLI tarafında doğrulanıyor
        return label_from_csv(rec, starts, csv_table, exclude)
    raise IQForgeError(
        f"Bilinmeyen etiket kaynağı '{source}'. Desteklenenler: {', '.join(LABEL_SOURCES)}."
    )


def _group_key(value: Any) -> str:
    """Dengeleme alanının değerini kararlı bir grup anahtarına çevirir.

    Sayısal değerler 12 anlamlı basamağa yazılır: kayan nokta gürültüsünü
    (0.1+0.2 gibi) normalize etmeye yeter ama gerçek farkları silmez. Daha az
    basamak tehlikelidir — 2.45 GHz'lik bir taşıyıcıda 6 basamak yalnızca
    ~10 kHz çözünürlük verir ve yakın ofsetler sessizce aynı gruba düşer.
    """
    if value is None:
        return "(yok)"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.12g}"
    return str(value)


def _format_offset(offset_hz: float | None) -> str:
    """Taşıyıcı ofsetini okunabilir biçimde verir."""
    if offset_hz is None:
        return "—"
    return f"{offset_hz / 1e3:+.1f} kHz"


def _render_split_records(plan: SplitPlan, work: dict[str, _RecordWork]) -> Table:
    """Hangi kaydın hangi split'e düştüğünü isim isim tablolar."""
    table = Table(title="Kayıt bazında bölme (SPEC §5.6)", title_style="bold", show_lines=False)
    table.add_column("Split", style="cyan", no_wrap=True)
    table.add_column("Kayıt", style="white")
    table.add_column("Etiket", style="green")
    table.add_column("Taşıyıcı", justify="right", style="magenta")
    table.add_column("Pencere", justify="right")
    if plan.groups:
        table.add_column("Grup", style="yellow")

    for split in SPLIT_NAMES:
        records = plan.records_in(split)
        if not records:
            table.add_row(split, "[dim]— boş —[/]", "", "", "0", *([""] if plan.groups else []))
            continue
        for i, record_id in enumerate(records):
            item = work[record_id]
            row = [
                split if i == 0 else "",
                record_id,
                item.dominant,
                _format_offset(item.offset_hz),
                str(len(item.labels)),
            ]
            if plan.groups:
                row.append(item.group or "—")
            table.add_row(*row)
    return table


@app.command()
def build(  # noqa: PLR0913 — CLI seçenekleri SPEC §4'te tanımlı
    input_path: Annotated[Path, typer.Argument(help="Tek .sigmf-meta dosyası veya klasör")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Çıktı klasörü")],
    window: Annotated[int, typer.Option("--window", help="Pencere uzunluğu (örnek)")] = 1024,
    stride: Annotated[int, typer.Option("--stride", help="Pencereler arası adım")] = 512,
    labels: Annotated[
        str, typer.Option("--labels", help=f"Etiket kaynağı: {', '.join(LABEL_SOURCES)}")
    ] = "annotations",
    label_file: Annotated[
        Path | None, typer.Option("--label-file", help="--labels csv için CSV yolu")
    ] = None,
    exclude_label: Annotated[
        list[str] | None,
        typer.Option("--exclude-label", help="Etiketlemede yok sayılacak etiket (yinelenebilir)"),
    ] = None,
    keep_unlabeled: Annotated[
        bool, typer.Option("--keep-unlabeled", help="Eşleşmeyen pencereleri atma, 'unlabeled' yap")
    ] = False,
    split: Annotated[
        str, typer.Option("--split", help="train,val,test oranları")
    ] = "0.7,0.15,0.15",
    seed: Annotated[int, typer.Option("--seed", help="Deterministik bölme tohumu")] = 42,
    balance_by: Annotated[
        str | None,
        typer.Option(
            "--balance-by",
            help="Split'lere yayılacak SigMF alanı, ör. core:freq_lower_edge",
        ),
    ] = None,
    representation: Annotated[
        str, typer.Option("--repr", help=f"Temsil: {', '.join(REPRESENTATIONS)}")
    ] = "iq2ch",
    normalize: Annotated[
        bool, typer.Option("--normalize/--no-normalize", help="Pencere başına birim güç")
    ] = True,
) -> None:
    """Kayıtları pencereleyip etiketli, bölünmüş bir veri seti olarak diske yazar."""
    try:
        _run_build(
            input_path=input_path,
            output=output,
            window=window,
            stride=stride,
            source=labels,
            label_file=label_file,
            exclude_label=exclude_label,
            keep_unlabeled=keep_unlabeled,
            split=split,
            seed=seed,
            balance_by=balance_by,
            representation=representation,
            normalize=normalize,
        )
    except IQForgeError as exc:
        err_console.print(f"[bold red]Hata:[/] {exc}")
        raise typer.Exit(code=1) from exc


def _run_build(  # noqa: PLR0913, PLR0915 — tek akışlı boru hattı
    *,
    input_path: Path,
    output: Path,
    window: int,
    stride: int,
    source: str,
    label_file: Path | None,
    exclude_label: list[str] | None,
    keep_unlabeled: bool,
    split: str,
    seed: int,
    balance_by: str | None,
    representation: str,
    normalize: bool,
) -> None:
    """`build` komutunun boru hattı. Hatalar `IQForgeError` olarak yükselir."""
    validate_window_params(window, stride)
    if source not in LABEL_SOURCES:
        raise IQForgeError(
            f"Bilinmeyen etiket kaynağı '{source}'. Desteklenenler: {', '.join(LABEL_SOURCES)}."
        )
    if representation not in REPRESENTATIONS:
        raise IQForgeError(
            f"Bilinmeyen temsil '{representation}'. Desteklenenler: {', '.join(REPRESENTATIONS)}."
        )
    if source == "csv" and label_file is None:
        raise IQForgeError("--labels csv seçildi ama --label-file verilmedi.")

    ratios = parse_ratios(split)
    exclude = resolve_exclude_labels(exclude_label)
    csv_table = load_label_csv(label_file) if source == "csv" else None

    metas, root = _collect_inputs(input_path)
    console.print(f"[dim]{len(metas)} kayıt bulundu:[/] {input_path}")

    work: dict[str, _RecordWork] = {}
    totals = LabelingStats()
    skipped: list[str] = []

    for meta in metas:
        rec = load(meta)
        starts = window_starts(rec.num_samples, window, stride)
        if starts.size == 0:
            skipped.append(f"{meta.name}: {rec.num_samples} örnek, {window} pencereye yetmiyor")
            continue

        window_labels, stats = _label_one(
            rec,
            starts,
            source=source,
            window=window,
            exclude=exclude,
            keep_unlabeled=keep_unlabeled,
            csv_table=csv_table,
        )
        totals.merge(stats)

        kept = [i for i, label in enumerate(window_labels) if label is not None]
        if not kept:
            skipped.append(f"{meta.name}: etiket alan pencere yok")
            continue

        record_id = meta.relative_to(root).as_posix() if meta != root else meta.name
        kept_labels = [window_labels[i] for i in kept]
        assert all(label is not None for label in kept_labels)
        dominant = dominant_label(window_labels)
        assert dominant is not None
        work[record_id] = _RecordWork(
            record_id=record_id,
            recording=rec,
            indices=np.asarray(kept, dtype=np.int64),
            labels=kept_labels,  # type: ignore[arg-type]
            dominant=dominant,
            offset_hz=carrier_offset_hz(rec, dominant, exclude),
        )

    for note in skipped:
        console.print(f"[yellow]atlandı[/] {note}")
    if not work:
        excluded = ", ".join(sorted(exclude)) or "(yok)"
        raise IQForgeError(
            "Hiçbir kayıttan etiketli pencere çıkmadı. "
            f"Etiket kaynağı '{source}', dışlanan etiketler: {excluded}."
        )

    record_groups: dict[str, str] | None = None
    if balance_by is not None:
        missing: list[str] = []
        record_groups = {}
        for record_id, item in work.items():
            value = annotation_field_value(item.recording, balance_by, item.dominant, exclude)
            if value is None:
                missing.append(record_id)
            item.group = _group_key(value)
            record_groups[record_id] = item.group
        if missing:
            console.print(
                f"[yellow]uyarı[/] --balance-by '{balance_by}' şu kayıtlarda bulunamadı, "
                f"'{_group_key(None)}' grubuna alındılar: {', '.join(sorted(missing))}"
            )

    plan = stratified_record_split(
        {k: v.dominant for k, v in work.items()}, ratios, seed, record_groups
    )
    if balance_by is not None:
        for warning in balance_warnings(plan, balance_by):
            console.print(f"[yellow]uyarı[/] {warning}")

    all_labels = sorted({label for item in work.values() for label in item.labels})
    label_map = {label: i for i, label in enumerate(all_labels)}

    output.mkdir(parents=True, exist_ok=True)
    writers = {name: ShardWriter(output, name) for name in SPLIT_NAMES}

    total_windows = sum(len(item.labels) for item in work.values())
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("pencereler yazılıyor", total=total_windows)
        for record_id in sorted(work):
            item = work[record_id]
            writer = writers[plan.assignment[record_id]]
            position = {int(idx): n for n, idx in enumerate(item.indices)}
            for chunk, batch in iter_window_batches(
                item.recording, window, stride, indices=item.indices
            ):
                if normalize:
                    batch = normalize_windows(batch)
                encoded = to_representation(batch, representation)
                writer.add(encoded, [label_map[item.labels[position[int(i)]]] for i in chunk])
                progress.advance(task, chunk.size)

    splits_meta: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        writer = writers[name]
        writer.flush()
        splits_meta[name] = {
            "shards": writer.shards,
            "labels": writer.labels,
            "count": writer.count,
            "records": [
                {
                    "id": record_id,
                    "label": work[record_id].dominant,
                    "windows": len(work[record_id].labels),
                    "carrier_offset_hz": work[record_id].offset_hz,
                    "balance_group": work[record_id].group,
                }
                for record_id in plan.records_in(name)
            ],
        }

    write_manifest(
        output,
        version=__version__,
        config={
            "window": window,
            "stride": stride,
            "repr": representation,
            "normalize": normalize,
            "seed": seed,
            "labels": source,
            "split": list(ratios),
            "exclude_labels": sorted(exclude),
            "keep_unlabeled": keep_unlabeled,
            "balance_by": balance_by,
        },
        label_map=label_map,
        source_files=[work[r].recording.meta_path.as_posix() for r in sorted(work)],
        splits=splits_meta,
    )

    console.print(_render_split_records(plan, work))
    console.print(
        f"[dim]pencere:[/] {totals.total} toplam, {totals.labeled} etiketli, "
        f"{totals.unmatched} eşleşmeyen, {totals.ambiguous} belirsiz (çakışan annotation)"
    )
    if totals.excluded_labels:
        console.print(f"[dim]dışlanan etiketler:[/] {', '.join(sorted(totals.excluded_labels))}")
    console.print(f"[green]yazıldı:[/] {output} ({dataset_size_bytes(output) / 1e6:.2f} MB)")


def _render_offset_summary(manifest: dict[str, Any]) -> Table:
    """Split başına taşıyıcı ofset dağılımını özetler.

    Bir split'teki ofsetlerin tamamı aynı işaretteyse bu, sınıf etiketiyle
    ilgisiz ama split'ler arasında sistematik olarak değişen bir değişken
    demektir — yani dağılım kayması. Tablo bunu gözle görülür kılar.
    """
    table = Table(title="Taşıyıcı ofset dağılımı", title_style="bold")
    table.add_column("Split", style="cyan", no_wrap=True)
    table.add_column("Ofsetler (kHz)", style="magenta")
    table.add_column("Negatif / Pozitif", justify="right")

    for name in SPLIT_NAMES:
        listed = manifest["splits"][name].get("records", [])
        offsets = [e.get("carrier_offset_hz") for e in listed]
        known = [o for o in offsets if o is not None]
        if not known:
            table.add_row(name, "[dim]—[/]", "—")
            continue
        negative = sum(1 for o in known if o < 0)
        positive = len(known) - negative
        skew = "[red]" if (negative == 0 or positive == 0) and len(known) > 1 else ""
        table.add_row(
            name,
            ", ".join(f"{o / 1e3:+.0f}" for o in sorted(known)),
            f"{skew}{negative} / {positive}",
        )
    return table


@app.command()
def stats(
    dataset_dir: Annotated[Path, typer.Argument(help="iqforge build ile üretilmiş klasör")],
) -> None:
    """Kurulmuş veri setinin özetini yazdırır."""
    try:
        manifest = read_manifest(dataset_dir)
    except IQForgeError as exc:
        err_console.print(f"[bold red]Hata:[/] {exc}")
        raise typer.Exit(code=1) from exc

    label_map: dict[str, int] = manifest["label_map"]
    config = manifest["config"]

    overview = Table(title=str(dataset_dir), title_style="bold", show_header=False)
    overview.add_column("Alan", style="cyan", no_wrap=True)
    overview.add_column("Değer")
    overview.add_row("iqforge sürümü", manifest["iqforge_version"])
    overview.add_row("oluşturma", manifest["created"])
    overview.add_row("kaynak kayıt", str(len(manifest["source_files"])))
    overview.add_row("pencere / adım", f"{config['window']} / {config['stride']}")
    overview.add_row("temsil", f"{config['repr']} (normalize={config['normalize']})")
    overview.add_row("etiket kaynağı", str(config.get("labels", "—")))
    overview.add_row("dışlanan etiket", ", ".join(config.get("exclude_labels", [])) or "—")
    overview.add_row("split / seed", f"{config.get('split', '—')} / {config['seed']}")
    overview.add_row("disk", f"{dataset_size_bytes(dataset_dir) / 1e6:.2f} MB")
    console.print(overview)

    distribution = Table(title="Sınıf dağılımı", title_style="bold")
    distribution.add_column("Split", style="cyan")
    distribution.add_column("Kayıt", justify="right")
    distribution.add_column("Pencere", justify="right")
    for label in label_map:
        distribution.add_column(label, justify="right", style="green")
    distribution.add_column("Shard", justify="right")

    for name in SPLIT_NAMES:
        entry = manifest["splits"][name]
        counts = np.bincount(entry["labels"], minlength=len(label_map)) if entry["labels"] else []
        distribution.add_row(
            name,
            str(len(entry.get("records", []))),  # noqa: PD011
            str(entry["count"]),
            *[str(counts[label_map[label]]) if len(counts) else "0" for label in label_map],
            str(len(entry["shards"])),
        )
    console.print(distribution)

    balanced = config.get("balance_by")
    records = Table(title="Split başına kayıt dosyaları", title_style="bold")
    records.add_column("Split", style="cyan", no_wrap=True)
    records.add_column("Kayıt", style="white")
    records.add_column("Etiket", style="green")
    records.add_column("Taşıyıcı", justify="right", style="magenta")
    records.add_column("Pencere", justify="right")
    if balanced:
        records.add_column(f"Grup ({balanced})", style="yellow")

    for name in SPLIT_NAMES:
        listed = manifest["splits"][name].get("records", [])
        if not listed:
            records.add_row(name, "[dim]— boş —[/]", "", "", "0", *([""] if balanced else []))
            continue
        for i, entry in enumerate(listed):
            row = [
                name if i == 0 else "",
                entry["id"],
                entry["label"],
                _format_offset(entry.get("carrier_offset_hz")),
                str(entry["windows"]),
            ]
            if balanced:
                row.append(entry.get("balance_group") or "—")
            records.add_row(*row)
    console.print(records)
    console.print(_render_offset_summary(manifest))


@app.command()
def version() -> None:
    """iqforge sürümünü yazdırır."""
    console.print(f"iqforge {__version__}")


if __name__ == "__main__":
    app()
