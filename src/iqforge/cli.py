"""iqforge command-line interface."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table

from iqforge import TORCH_REQUIRED, __version__
from iqforge.audit import (
    WIDTH,
    AuditReport,
    RecordFeatures,
    audit_dataset,
    audit_recordings,
    collect_meta_paths,
    measure_recording,
    render_json,
    render_text,
)
from iqforge.display import render_inspect
from iqforge.grouping import grouping_warnings, resolve_group_keys
from iqforge.io import META_EXT, IQForgeError, Recording, load
from iqforge.labeling import (
    LABEL_SOURCES,
    AnnotationLabelSurvey,
    LabelingStats,
    annotation_field_value,
    carrier_offset_hz,
    csv_collapse_error,
    csv_declared_labels,
    dirname_at_level,
    dirname_level_warning,
    dominant_label,
    label_from_annotations,
    label_from_csv,
    label_from_dirname,
    load_label_csv,
    resolve_exclude_labels,
)
from iqforge.measurement import (
    DEFAULT_SPLIT,
    DEFAULT_SPLIT_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    MEASUREMENT_SCHEMA,
    BuildSpec,
    GridCell,
    cell_stats,
    guard_artifact_rows,  # noqa: F401 - re-exported for scripts
    run_grid,
    summarise_stride_table,
)
from iqforge.preflight import DecisionStatus, decide
from iqforge.preflight import render_json as render_measure_json
from iqforge.preflight import render_text as render_measure_text
from iqforge.splitting import (
    SPLIT_NAMES,
    SplitPlan,
    balance_warnings,
    leakage_warnings,
    parse_ratios,
    stratified_record_split,
)
from iqforge.storage import (
    MANIFEST_NAME,
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


def _parse_seed_list(text: str, flag: str) -> list[int]:
    """Parse a comma-separated seed list, refusing anything ambiguous."""
    parts = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not parts:
        raise IQForgeError(f"{flag} needs at least one integer, got '{text}'.")
    try:
        seeds = [int(piece) for piece in parts]
    except ValueError as exc:
        raise IQForgeError(f"{flag} must be integers separated by commas, got '{text}'.") from exc
    if len(set(seeds)) != len(seeds):
        raise IQForgeError(f"{flag} repeats a seed: '{text}'. Each pair must be distinct.")
    return seeds


def _run_row(run: Any) -> dict[str, Any]:
    """One `Run` as JSON. Every row carries its own environment."""
    return {
        "stride": run.stride,
        "strategy": run.strategy,
        "split_seed": run.split_seed,
        "train_seed": run.train_seed,
        "test_accuracy": run.test_accuracy,
        "train_accuracy": run.train_accuracy,
        "train_windows": run.train_windows,
        "test_windows": run.test_windows,
        "environment": run.environment,
    }


def _uncertainty(stats: Any) -> str:
    """Interval text, or an explicit refusal to invent one.

    A single pair has no spread to estimate, and printing the standard error
    of one sample as "+/- 0.0" reads as maximum confidence when it means the
    opposite. Say so instead.
    """
    if stats.n < 2:
        return "(uncertainty not estimated, n=1)"
    return f"+/- {stats.stderr * 100.0:.1f} (standard error, n={stats.n})"


def _environment_lines(runs: list[Any]) -> list[str]:
    """The conditions the runs were measured under, wrapped to the block width.

    Wrapped rather than emitted as one string: the report is 78 columns of
    ASCII so it can be pasted unaltered, and five library versions on one line
    overflow that. The width test caught this the first time it was written.
    """
    env = next((run.environment for run in runs if run.environment), {})
    if not env:
        return ["environment: not recorded"]
    keys = ("device", "torch", "numpy", "scipy", "sigmf", "cuda")
    parts = [f"{key} {env[key]}" for key in keys if key in env]
    lines: list[str] = []
    current = "environment:"
    for part in parts:
        candidate = f"{current} {part}" if current.endswith(":") else f"{current} | {part}"
        if len(candidate) > WIDTH:
            lines.append(current)
            current = f"             {part}"
        else:
            current = candidate
    lines.append(current)
    return lines


def _torch_available() -> bool:
    """True when the optional torch extra is importable."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn SigMF recordings into machine-learning-ready datasets.",
)


def _force_utf8_streams() -> None:
    """Make stdout and stderr able to carry the characters we print.

    Every command prints non-ASCII: the annotation table uses `→`, the split
    error carries `§`, the inspector is built out of block-drawing characters.
    When output goes to a console Python encodes as UTF-8 and all of that
    survives. When it goes to a pipe or a file, Python falls back to the locale
    codepage instead — cp1254 on a Turkish Windows install, cp1252 elsewhere —
    and none of those characters exist there, so `iqforge info x > out.txt`
    died with UnicodeEncodeError while the same command on screen was fine.

    Streams replaced by a test harness may not be reconfigurable; those are
    already text-mode objects that accept any string, so skipping them is safe.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # detached or already-closed stream
            pass


_force_utf8_streams()

console = Console()
err_console = Console(stderr=True)


def _fail(exc: Exception) -> typer.Exit:
    """Print an error and build the exit to raise.

    The message is escaped: it carries file names, labels and metadata field
    values, and rich would otherwise read anything in square brackets as a
    style tag and delete it.
    """
    err_console.print(f"[bold red]Error:[/] {escape(str(exc))}")
    return typer.Exit(code=1)


def _format_hz(value: float | None) -> str:
    """Format a value in Hz with a readable unit."""
    if value is None:
        return "-"
    for unit, scale in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.6g} {unit} ({value:.0f} Hz)"
    return f"{value:.0f} Hz"


def _version_cell(rec: Recording) -> str:
    """What to print for `core:version`.

    The recording's own value comes first, because that is the one a user is
    looking for. The sigmf library rewrites the field with the spec version it
    implements, so when the two differ both are shown rather than silently
    presenting the reader's version as the recording's -- real captures
    declaring 1.0.0 were being reported as 1.2.6.
    """
    declared = rec.declared_version
    parsed = rec.global_info.get("core:version")
    if declared is None:
        return "-" if parsed is None else f"{parsed} (from reader; file declares none)"
    if parsed is not None and str(parsed) != declared:
        return f"{declared} (file); {parsed} (reader)"
    return declared


def _render_overview(rec: Recording) -> Table:
    """Build a table of the recording's basic metadata."""
    table = Table(title=escape(rec.meta_path.name), title_style="bold", show_header=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    g = rec.global_info
    table.add_row("Sample rate", _format_hz(rec.sample_rate))
    table.add_row("Centre frequency", _format_hz(rec.center_frequency))
    table.add_row("Datatype", escape(rec.datatype))
    table.add_row("Samples", f"{rec.num_samples:,}".replace(",", " "))
    table.add_row("Duration", f"{rec.duration_seconds:.6g} s")
    table.add_row("Data file", f"{rec.data_path.stat().st_size / 1e6:.2f} MB")
    table.add_row("Hardware", escape(str(g.get("core:hw", "-"))))
    table.add_row("Author", escape(str(g.get("core:author", "-"))))
    table.add_row("Recorder", escape(str(g.get("core:recorder", "-"))))
    table.add_row("SigMF version", escape(_version_cell(rec)))
    if g.get("core:description"):
        table.add_row("Description", escape(str(g["core:description"])))
    return table


def _render_annotations(rec: Recording) -> Table:
    """Build a table of the recording's annotations."""
    table = Table(title=f"Annotations ({len(rec.annotations)})", title_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Label", style="green")
    table.add_column("Start", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Frequency range")

    for i, a in enumerate(rec.annotations):
        t0 = a.sample_start / rec.sample_rate
        t1 = a.sample_end / rec.sample_rate
        if a.freq_lower_edge is not None and a.freq_upper_edge is not None:
            lo = (a.freq_lower_edge - (rec.center_frequency or 0.0)) / 1e3
            hi = (a.freq_upper_edge - (rec.center_frequency or 0.0)) / 1e3
            freq = f"centre {lo:+.1f} … {hi:+.1f} kHz"
        else:
            freq = "-"
        table.add_row(
            str(i),
            escape(a.label or "-"),
            f"{a.sample_start:,}".replace(",", " "),
            f"{a.sample_count:,}".replace(",", " "),
            f"{t0:.4f} → {t1:.4f}",
            freq,
        )
    return table


@app.command()
def info(
    path: Annotated[Path, typer.Argument(help="SigMF recording (.sigmf-meta or recording name)")],
) -> None:
    """Print a SigMF recording's metadata as a readable table."""
    try:
        rec = load(path)
    except IQForgeError as exc:
        raise _fail(exc) from exc

    console.print(_render_overview(rec))
    if rec.annotations:
        console.print(_render_annotations(rec))
    else:
        console.print("[dim]The recording has no annotations.[/]")

    for annotation in rec.annotations_beyond_end:
        err_console.print(
            f"[yellow]warning[/] annotation {annotation.sample_start}..{annotation.sample_end} "
            f"runs past the end of the recording ({rec.num_samples} samples). "
            f"The metadata and the data file disagree."
        )


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(help="SigMF recording (.sigmf-meta or recording name)")],
    start: Annotated[int, typer.Option("--start", help="Sample index to start at")] = 0,
    samples: Annotated[int, typer.Option("--samples", help="How many samples to show")] = 262_144,
    nfft: Annotated[int, typer.Option("--nfft", help="FFT length")] = 1024,
    height: Annotated[
        int | None, typer.Option("--height", help="Spectrogram height in character rows")
    ] = None,
) -> None:
    """Draw a spectrogram and a power-over-time plot in the terminal."""
    try:
        rec = load(path)
        data = rec.read(start=start, count=samples)
        rows = height if height is not None else max(8, min(24, console.size.height - 10))
        panel = render_inspect(rec, data, start, nfft, width=console.size.width, height=rows)
    except IQForgeError as exc:
        raise _fail(exc) from exc
    except ValueError as exc:
        raise _fail(exc) from exc

    console.print(panel)


@dataclass
class _RecordWork:
    """State carried for one recording during a build.

    Attributes:
        record_id: The recording's unique id in the dataset (a relative path).
        recording: The opened recording.
        indices: Indices of the windows that received a label.
        labels: Labels in the same order as `indices`.
        dominant: The recording's dominant label; stratified splitting uses it.
        offset_hz: Carrier offset from the centre frequency, if derivable.
        group: The `--balance-by` group key, if balancing was requested.
    """

    record_id: str
    recording: Recording
    indices: np.ndarray
    labels: list[str]
    dominant: str
    offset_hz: float | None = None
    group: str | None = None


def _collect_inputs(path: Path) -> tuple[list[Path], Path]:
    """Collect the `.sigmf-meta` files under an input path.

    Returns:
        `(meta_paths, root)` where `root` is the directory recording ids are
        made relative to.

    Raises:
        IQForgeError: If the path does not exist or holds no recordings.
    """
    if not path.exists():
        raise IQForgeError(f"Input not found: {path}")
    if path.is_dir():
        metas = sorted(path.rglob(f"*{META_EXT}"))
        if not metas:
            raise IQForgeError(
                f"No '*{META_EXT}' files found under '{path}' (subdirectories included)."
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
    dirname_level: int = 1,
    record_id: str | None = None,
) -> tuple[list[str | None], LabelingStats]:
    """Label one recording's windows using the selected source.

    `record_id` is the recording's path relative to the input directory. A CSV
    table is matched on it first, because a bare file name does not identify a
    recording in a nested layout.
    """
    if source == "annotations":
        return label_from_annotations(rec, starts, window, exclude, keep_unlabeled)
    if source == "dirname":
        return label_from_dirname(rec, starts, exclude, level=dirname_level)
    if source == "csv":
        assert csv_table is not None  # validated on the CLI side
        return label_from_csv(rec, starts, csv_table, exclude, record_id=record_id)
    raise IQForgeError(f"Unknown label source '{source}'. Supported: {', '.join(LABEL_SOURCES)}.")


def _group_key(value: Any) -> str:
    """Turn a balancing field's value into a stable group key.

    Numeric values are written with 12 significant digits: enough to normalise
    floating-point noise (0.1+0.2 and friends) without erasing real differences.
    Fewer digits is dangerous — on a 2.45 GHz carrier, 6 digits gives only about
    10 kHz of resolution and nearby offsets silently collapse into one group.
    """
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.12g}"
    return str(value)


def _format_offset(offset_hz: float | None) -> str:
    """Format a carrier offset for display."""
    if offset_hz is None:
        return "-"
    return f"{offset_hz / 1e3:+.1f} kHz"


def _render_split_records(plan: SplitPlan, work: dict[str, _RecordWork]) -> Table:
    """Tabulate which recording landed in which split, by name."""
    table = Table(title="Recording-level split (SPEC §5.6)", title_style="bold", show_lines=False)
    table.add_column("Split", style="cyan", no_wrap=True)
    table.add_column("Recording", style="white")
    table.add_column("Label", style="green")
    table.add_column("Carrier", justify="right", style="magenta")
    table.add_column("Windows", justify="right")
    if plan.groups:
        table.add_column("Group", style="yellow")

    for split in SPLIT_NAMES:
        records = plan.records_in(split)
        if not records:
            table.add_row(split, "[dim]- empty -[/]", "", "", "0", *([""] if plan.groups else []))
            continue
        for i, record_id in enumerate(records):
            item = work[record_id]
            row = [
                split if i == 0 else "",
                escape(record_id),
                escape(item.dominant),
                _format_offset(item.offset_hz),
                str(len(item.labels)),
            ]
            if plan.groups:
                row.append(escape(item.group or "-"))
            table.add_row(*row)
    return table


@app.command()
def build(  # noqa: PLR0913 — the CLI options are defined in SPEC §4
    input_path: Annotated[Path, typer.Argument(help="A single .sigmf-meta file or a directory")],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output directory")],
    window: Annotated[int, typer.Option("--window", help="Window length in samples")] = 1024,
    stride: Annotated[int, typer.Option("--stride", help="Step between windows")] = 512,
    labels: Annotated[
        str, typer.Option("--labels", help=f"Label source: {', '.join(LABEL_SOURCES)}")
    ] = "annotations",
    label_file: Annotated[
        Path | None, typer.Option("--label-file", help="CSV path for --labels csv")
    ] = None,
    dirname_level: Annotated[
        int,
        typer.Option(
            "--dirname-level",
            help="For --labels dirname: 1 is the recording's own directory, 2 its parent",
        ),
    ] = 1,
    exclude_label: Annotated[
        list[str] | None,
        typer.Option("--exclude-label", help="Label to ignore when labelling (repeatable)"),
    ] = None,
    keep_unlabeled: Annotated[
        bool,
        typer.Option("--keep-unlabeled", help="Keep unmatched windows as 'unlabeled'"),
    ] = False,
    split: Annotated[str, typer.Option("--split", help="train,val,test ratios")] = "0.7,0.15,0.15",
    seed: Annotated[int, typer.Option("--seed", help="Seed for the deterministic split")] = 42,
    balance_by: Annotated[
        str | None,
        typer.Option(
            "--balance-by",
            help="SigMF field to spread across splits, e.g. core:freq_lower_edge",
        ),
    ] = None,
    group_by: Annotated[
        str | None,
        typer.Option(
            "--group-by",
            help="Keep related recordings in one split: path:<regex> or csv:<file>",
        ),
    ] = None,
    representation: Annotated[
        str, typer.Option("--repr", help=f"Representation: {', '.join(REPRESENTATIONS)}")
    ] = "iq2ch",
    normalize: Annotated[
        bool, typer.Option("--normalize/--no-normalize", help="Unit power per window")
    ] = True,
) -> None:
    """Window, label, split and write recordings as a dataset on disk."""
    try:
        _run_build(
            input_path=input_path,
            output=output,
            window=window,
            stride=stride,
            source=labels,
            label_file=label_file,
            dirname_level=dirname_level,
            exclude_label=exclude_label,
            keep_unlabeled=keep_unlabeled,
            split=split,
            seed=seed,
            balance_by=balance_by,
            group_by=group_by,
            representation=representation,
            normalize=normalize,
        )
    except IQForgeError as exc:
        raise _fail(exc) from exc


def _run_build(  # noqa: PLR0913, PLR0915 — one linear pipeline
    *,
    input_path: Path,
    output: Path,
    window: int,
    stride: int,
    source: str,
    label_file: Path | None,
    dirname_level: int,
    exclude_label: list[str] | None,
    keep_unlabeled: bool,
    split: str,
    seed: int,
    balance_by: str | None,
    group_by: str | None,
    representation: str,
    normalize: bool,
) -> None:
    """The `build` pipeline. Failures are raised as `IQForgeError`."""
    validate_window_params(window, stride)
    if source not in LABEL_SOURCES:
        raise IQForgeError(
            f"Unknown label source '{source}'. Supported: {', '.join(LABEL_SOURCES)}."
        )
    if representation not in REPRESENTATIONS:
        raise IQForgeError(
            f"Unknown representation '{representation}'. Supported: {', '.join(REPRESENTATIONS)}."
        )
    if source == "csv" and label_file is None:
        raise IQForgeError("--labels csv was selected but --label-file was not given.")

    ratios = parse_ratios(split)
    exclude = resolve_exclude_labels(exclude_label)
    csv_table = load_label_csv(label_file) if source == "csv" else None

    metas, root = _collect_inputs(input_path)
    console.print(f"[dim]found {len(metas)} recording(s):[/] {escape(str(input_path))}")

    if source == "dirname":
        level_warning = dirname_level_warning(metas, dirname_level)
        if level_warning:
            console.print(f"[yellow]warning[/] {escape(level_warning)}")

    work: dict[str, _RecordWork] = {}
    totals = LabelingStats()
    skipped: list[str] = []
    survey = AnnotationLabelSurvey()

    for meta in metas:
        rec = load(meta)
        survey.observe(rec)
        for annotation in rec.annotations_beyond_end:
            console.print(
                f"[yellow]warning[/] {escape(meta.name)}: annotation "
                f"{annotation.sample_start}..{annotation.sample_end} runs past the end of the "
                f"recording ({rec.num_samples} samples). The metadata and the data file "
                f"disagree; windows in that range cannot be labelled."
            )
        starts = window_starts(rec.num_samples, window, stride)
        if starts.size == 0:
            skipped.append(
                f"{meta.name}: {rec.num_samples} samples, too few for a window of {window}"
            )
            continue

        record_id = meta.relative_to(root).as_posix() if meta != root else meta.name
        window_labels, stats = _label_one(
            rec,
            starts,
            source=source,
            window=window,
            exclude=exclude,
            keep_unlabeled=keep_unlabeled,
            csv_table=csv_table,
            dirname_level=dirname_level,
            record_id=record_id,
        )
        totals.merge(stats)

        kept = [i for i, label in enumerate(window_labels) if label is not None]
        if not kept:
            skipped.append(f"{meta.name}: no window received a label")
            continue

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
        console.print(f"[yellow]skipped[/] {escape(note)}")
    if not work:
        excluded = ", ".join(sorted(exclude)) or "(none)"
        message = (
            "No recording produced a labelled window. "
            f"Label source '{source}', excluded labels: {excluded}."
        )
        if source == "annotations":
            message = f"{message}\n\n{survey.hint()}"
        raise IQForgeError(message)

    assigned_labels = {item.dominant for item in work.values() if item.dominant is not None}
    if source == "csv" and label_file is not None:
        # Compared after exclusion, so --exclude-label and dropped unlabelled
        # windows cannot look like a collapse.
        declared = csv_declared_labels(label_file, sorted(work), exclude)
        collapse = csv_collapse_error(declared, assigned_labels, label_file)
        if collapse is not None:
            raise IQForgeError(collapse)
    if source == "annotations" and len(assigned_labels) == 1:
        only = next(iter(assigned_labels))
        console.print(
            f"[yellow]warning[/] every window was labelled '{escape(only)}'; the dataset "
            f"has one class and nothing to classify. {survey.annotations} annotation(s) "
            f"were scanned. If other classes were expected, check --exclude-label "
            f"({escape(', '.join(sorted(exclude)) or 'none')}) and whether the other "
            f"annotations carry 'core:label'."
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
                f"[yellow]warning[/] --balance-by '{escape(balance_by)}' was not found "
                f"in these recordings, they went into the '{escape(_group_key(None))}' "
                f"group: {escape(', '.join(sorted(missing)))}"
            )

    record_units: dict[str, str] | None = None
    if group_by is not None:
        if group_by == balance_by:
            raise IQForgeError(
                f"--group-by and --balance-by were both given '{group_by}'. Grouping keeps "
                "recordings that share a key together; balancing spreads them apart. One key "
                "cannot do both."
            )
        record_units = resolve_group_keys(
            sorted(work),
            group_by,
            collections={
                rid: item.recording.global_info.get("core:collection") for rid, item in work.items()
            },
        )
        for warning in grouping_warnings(record_units, group_by):
            console.print(f"[yellow]warning[/] {escape(warning)}")

    plan = stratified_record_split(
        {k: v.dominant for k, v in work.items()},
        ratios,
        seed,
        record_groups,
        record_units=record_units,
        group_by=group_by,
    )
    if balance_by is not None:
        for warning in balance_warnings(plan, balance_by):
            console.print(f"[yellow]warning[/] {escape(warning)}")
        for warning in leakage_warnings(plan, {k: v.dominant for k, v in work.items()}, balance_by):
            console.print(f"[bold yellow]warning[/] {escape(warning)}")

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
        task = progress.add_task("writing windows", total=total_windows)
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
                    "group": plan.units.get(record_id),
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
            "group_by": group_by,
        },
        label_map=label_map,
        source_files=[work[r].recording.meta_path.as_posix() for r in sorted(work)],
        splits=splits_meta,
    )

    console.print(_render_split_records(plan, work))
    console.print(
        f"[dim]windows:[/] {totals.total} total, {totals.labeled} labelled, "
        f"{totals.unmatched} unmatched, {totals.ambiguous} ambiguous (overlapping annotations)"
    )
    if totals.excluded_labels:
        console.print(
            f"[dim]excluded labels:[/] {escape(', '.join(sorted(totals.excluded_labels)))}"
        )
    console.print(
        f"[green]written:[/] {escape(str(output))} ({dataset_size_bytes(output) / 1e6:.2f} MB)"
    )


def _render_offset_summary(manifest: dict[str, Any]) -> Table:
    """Summarise the carrier-offset distribution per split.

    If every offset in a split has the same sign, a variable unrelated to the
    class label varies systematically between splits — a distribution shift.
    This table makes that visible.
    """
    table = Table(title="Carrier offset distribution", title_style="bold")
    table.add_column("Split", style="cyan", no_wrap=True)
    table.add_column("Offsets (kHz)", style="magenta")
    table.add_column("Negative / Positive", justify="right")

    for name in SPLIT_NAMES:
        listed = manifest["splits"][name].get("records", [])
        offsets = [e.get("carrier_offset_hz") for e in listed]
        known = [o for o in offsets if o is not None]
        if not known:
            table.add_row(name, "[dim]-[/]", "-")
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
    dataset_dir: Annotated[Path, typer.Argument(help="A directory built by iqforge build")],
) -> None:
    """Print a summary of a built dataset."""
    try:
        manifest = read_manifest(dataset_dir)
    except IQForgeError as exc:
        raise _fail(exc) from exc

    label_map: dict[str, int] = manifest["label_map"]
    config = manifest["config"]

    overview = Table(title=escape(str(dataset_dir)), title_style="bold", show_header=False)
    overview.add_column("Field", style="cyan", no_wrap=True)
    overview.add_column("Value")
    overview.add_row("iqforge version", escape(str(manifest["iqforge_version"])))
    overview.add_row("created", escape(str(manifest["created"])))
    overview.add_row("source recordings", str(len(manifest["source_files"])))
    overview.add_row("window / stride", f"{config['window']} / {config['stride']}")
    overview.add_row("representation", f"{config['repr']} (normalize={config['normalize']})")
    overview.add_row("label source", escape(str(config.get("labels", "-"))))
    overview.add_row("excluded labels", escape(", ".join(config.get("exclude_labels", [])) or "-"))
    overview.add_row("split / seed", f"{config.get('split', '-')} / {config['seed']}")
    overview.add_row("disk", f"{dataset_size_bytes(dataset_dir) / 1e6:.2f} MB")
    console.print(overview)

    distribution = Table(title="Class distribution", title_style="bold")
    distribution.add_column("Split", style="cyan")
    distribution.add_column("Recordings", justify="right")
    distribution.add_column("Windows", justify="right")
    for label in label_map:
        distribution.add_column(escape(label), justify="right", style="green")
    distribution.add_column("Shards", justify="right")

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

    # The chance line, always. A skewed dataset is a normal thing to want, so
    # there is no threshold and no warning here -- but an accuracy figure is
    # unreadable without the number a constant predictor would score, and that
    # number is free to compute.
    window_counts = Counter[str]()
    for name in SPLIT_NAMES:
        entry = manifest["splits"][name]
        if not entry["labels"]:
            continue
        counts = np.bincount(entry["labels"], minlength=len(label_map))
        for label in label_map:
            window_counts[label] += int(counts[label_map[label]])
    total_windows = sum(window_counts.values())
    if total_windows:
        largest, largest_count = window_counts.most_common(1)[0]
        console.print(
            f"[dim]chance:[/] {largest_count / total_windows:.1%} — a constant predictor "
            f"answering '{escape(largest)}' every time. Any accuracy below this is worse "
            f"than guessing."
        )

    balanced = config.get("balance_by")
    grouped = config.get("group_by")
    records = Table(title="Recordings per split", title_style="bold")
    records.add_column("Split", style="cyan", no_wrap=True)
    records.add_column("Recording", style="white")
    records.add_column("Label", style="green")
    records.add_column("Carrier", justify="right", style="magenta")
    records.add_column("Windows", justify="right")
    if balanced:
        records.add_column(f"Group ({escape(str(balanced))})", style="yellow")
    if grouped:
        records.add_column(f"Unit ({escape(str(grouped))})", style="blue")

    extra = ([""] if balanced else []) + ([""] if grouped else [])
    for name in SPLIT_NAMES:
        listed = manifest["splits"][name].get("records", [])
        if not listed:
            records.add_row(name, "[dim]- empty -[/]", "", "", "0", *extra)
            continue
        for i, entry in enumerate(listed):
            row = [
                name if i == 0 else "",
                escape(str(entry["id"])),
                escape(str(entry["label"])),
                _format_offset(entry.get("carrier_offset_hz")),
                str(entry["windows"]),
            ]
            if balanced:
                row.append(escape(str(entry.get("balance_group") or "-")))
            if grouped:
                row.append(escape(str(entry.get("group") or "-")))
            records.add_row(*row)
    console.print(records)
    console.print(_render_offset_summary(manifest))
    if grouped:
        for line in _grouping_summary(manifest, str(grouped), bool(balanced)):
            console.print(line)


def _grouping_summary(manifest: dict[str, Any], group_by: str, balanced: bool) -> list[str]:
    """Lines reporting what `--group-by` did, and whether it held.

    Three things a reader wants confirmed: how much the grouping actually
    collapsed, that no unit ended up straddling a split — the one invariant
    grouping exists to provide, and checkable from the manifest alone — and,
    when balancing is also on, how many units came out `(mixed)`. A run where
    most units are mixed is a run where `--balance-by` is not doing its job,
    and that is invisible without counting.
    """
    where: dict[str, set[str]] = {}
    values: dict[str, set[str]] = {}
    total = 0
    for name in SPLIT_NAMES:
        for entry in manifest["splits"][name].get("records", []):
            unit = entry.get("group")
            if unit is None:
                continue
            total += 1
            where.setdefault(unit, set()).add(name)
            values.setdefault(unit, set()).add(str(entry.get("balance_group")))

    if not where:
        return []
    lines = [f"[dim]grouping:[/] {escape(group_by)} -> {total} recordings in {len(where)} unit(s)"]

    straddling = sorted(u for u, splits in where.items() if len(splits) > 1)
    if straddling:
        lines.append(
            f"[bold red]warning[/] {len(straddling)} unit(s) span more than one split: "
            f"{escape(', '.join(straddling[:3]))} - grouping did not hold"
        )
    else:
        lines.append("[dim]         no unit spans more than one split[/]")

    if balanced:
        # A unit is mixed when its members disagree on the balancing field, so
        # it carries no single value into its split. Derived here rather than
        # stored: the manifest already holds the per-recording values, and a
        # second copy could drift from them.
        mixed = sum(1 for members in values.values() if len(members) > 1)
        note = " - --balance-by has little left to work with" if mixed * 2 >= len(where) else ""
        lines.append(
            f"[dim]         {mixed} of {len(where)} unit(s) mixed for --balance-by{note}[/]"
        )
    return lines


@app.command()
def train(
    dataset_dir: Annotated[Path, typer.Argument(help="A directory built by iqforge build")],
    epochs: Annotated[int, typer.Option("--epochs", help="Number of epochs")] = 10,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size")] = 64,
    seed: Annotated[
        int, typer.Option("--seed", help="TRAINING seed (weight init + batch order)")
    ] = 0,
    learning_rate: Annotated[float, typer.Option("--lr", help="Adam learning rate")] = 1e-3,
    device: Annotated[
        str, typer.Option("--device", help="auto, cpu or cuda. Default cpu, deliberately")
    ] = "cpu",
) -> None:
    """Train a small baseline CNN.

    The point is not a record accuracy but proof that the dataset is actually
    trainable. `--seed` affects training only; how the dataset was split is
    fixed by `build --seed` and cannot be changed here.

    `--device` defaults to CPU on purpose: the same seed gives the same bytes
    there, and it does not on a GPU.
    """
    try:
        # torch is optional: import only when `train` runs, so that
        # info/inspect/build/stats keep working without it.
        from iqforge.models import MAX_PARAMETERS
        from iqforge.training import train_baseline
    except ImportError as exc:  # pragma: no cover - only when torch is absent
        err_console.print(
            f"[bold red]Error:[/] {escape(TORCH_REQUIRED.format(what='`iqforge train`'))}"
        )
        raise typer.Exit(code=1) from exc

    def _report(epoch_result: Any) -> None:
        line = (
            f"epoch {epoch_result.epoch:>3}  loss {epoch_result.train_loss:.4f}  "
            f"train {epoch_result.train_accuracy:6.2%}"
        )
        if epoch_result.val_accuracy is not None:
            line += f"  val {epoch_result.val_accuracy:6.2%}"
        console.print(line)

    try:
        result = train_baseline(
            dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            learning_rate=learning_rate,
            on_epoch=_report,
            device_choice=device,
        )
    except IQForgeError as exc:
        raise _fail(exc) from exc

    console.print(
        f"[dim]model:[/] {result.parameters} trainable parameters "
        f"(budget {MAX_PARAMETERS})  [dim]training seed:[/] {seed}"
    )
    if result.test_accuracy is None:
        console.print("[yellow]the test split is empty - no test accuracy computed[/]")
        return

    if result.environment.get("device") == "cuda":
        err_console.print(
            "[yellow]warning[/] trained on CUDA. cuDNN selects kernels by heuristic, so "
            "these numbers are NOT bit-comparable with CPU runs or with each other across "
            "machines. Do not mix devices inside one paired experiment - see "
            "docs/methodology.md §4."
        )
    console.print(f"[bold]test accuracy: {result.test_accuracy:.2%}[/]")
    for name, accuracy in result.test_per_class.items():
        console.print(f"[dim]  {escape(name)}:[/] {accuracy:.2%}")


@app.command()
def audit(
    path: Annotated[
        Path, typer.Argument(help="A dataset built by iqforge, or a folder of recordings")
    ],
    window: Annotated[int, typer.Option("--window", help="Window length, folder mode only")] = 1024,
    stride: Annotated[
        int, typer.Option("--stride", help="Step between windows, folder mode")
    ] = 512,
    labels: Annotated[
        str, typer.Option("--labels", help=f"Label source, folder mode: {', '.join(LABEL_SOURCES)}")
    ] = "dirname",
    label_file: Annotated[
        Path | None, typer.Option("--label-file", help="CSV path for --labels csv")
    ] = None,
    dirname_level: Annotated[
        int, typer.Option("--dirname-level", help="Ancestor directory to read as the class")
    ] = 1,
    output_format: Annotated[str, typer.Option("--format", help="text or json")] = "text",
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit non-zero on RISK as well as LEAK")
    ] = False,
) -> None:
    """Report leakage risk and whether a leakage measurement is even possible.

    Prints what it checked, what it found, and -- always -- what it could not
    check. It never certifies a dataset as clean; see `docs/methodology.md` §6
    for why that distinction is the whole point.
    """
    if output_format not in ("text", "json"):
        raise _fail(IQForgeError(f"--format must be text or json, got '{output_format}'."))

    try:
        if (path / MANIFEST_NAME).exists():
            report = audit_dataset(path, read_manifest(path), __version__)
        else:
            report = _audit_folder(path, window, stride, labels, dirname_level, label_file)
    except IQForgeError as exc:
        raise _fail(exc) from exc

    print(render_json(report) if output_format == "json" else render_text(report))
    summary = report.summary
    if summary["leaks"] or (strict and summary["risk"]):
        raise typer.Exit(code=1)


@app.command("measure-leakage")
def measure_leakage(  # noqa: PLR0913 — flags match `audit` plus --force / --group-by
    path: Annotated[
        Path, typer.Argument(help="A dataset built by iqforge, or a folder of recordings")
    ],
    window: Annotated[int, typer.Option("--window", help="Window length, folder mode only")] = 1024,
    stride: Annotated[
        int, typer.Option("--stride", help="Step between windows, folder mode")
    ] = 512,
    labels: Annotated[
        str, typer.Option("--labels", help=f"Label source, folder mode: {', '.join(LABEL_SOURCES)}")
    ] = "dirname",
    label_file: Annotated[
        Path | None, typer.Option("--label-file", help="CSV path for --labels csv")
    ] = None,
    dirname_level: Annotated[
        int, typer.Option("--dirname-level", help="Ancestor directory to read as the class")
    ] = 1,
    group_by: Annotated[
        str | None,
        typer.Option(
            "--group-by", help="Hold related recordings together: path:<re> or csv:<file>"
        ),
    ] = None,
    balance_by: Annotated[
        str | None,
        typer.Option(
            "--balance-by",
            help="Spread nuisance keys across splits during build, e.g. core:freq_lower_edge",
        ),
    ] = None,
    split: Annotated[
        str, typer.Option("--split", help="train,val,test ratios a later measurement would use")
    ] = DEFAULT_SPLIT,
    force: Annotated[
        bool,
        typer.Option("--force", help="Measure anyway; the overridden category stays in the header"),
    ] = False,
    output_format: Annotated[str, typer.Option("--format", help="text or json")] = "text",
    sweep: Annotated[
        str | None,
        typer.Option("--sweep", help="Optional sweep mode. Only 'stride' is supported."),
    ] = None,
    split_seeds: Annotated[
        str, typer.Option("--split-seeds", help="Comma-separated split seeds")
    ] = DEFAULT_SPLIT_SEEDS,
    train_seeds: Annotated[
        str, typer.Option("--train-seeds", help="Comma-separated training seeds")
    ] = DEFAULT_TRAIN_SEEDS,
) -> None:
    """Measure leakage inflation after running the refuse-path gate.

    Runs `audit`, classifies the result into the six categories in
    `docs/methodology.md` §6, then:
    - `REFUSED` exits non-zero
    - `WOULD MEASURE` trains both arms (recording-level vs window-level)

    Default mode runs one operating point at the selected stride, over every
    seed pair. `--sweep stride` runs the fixed overlap ladder at the same pairs.

    The seed lists default to the five split seeds and three training seeds the
    published tables used -- 15 pairs. They are flags rather than constants so a
    run can be made cheaper deliberately and visibly, and the count is printed
    with the result: a measurement whose sample size is not on the page cannot
    be read.
    """
    if output_format not in ("text", "json"):
        raise _fail(IQForgeError(f"--format must be text or json, got '{output_format}'."))
    if sweep not in (None, "stride"):
        raise _fail(IQForgeError(f"--sweep must be 'stride' when set, got '{sweep}'."))
    try:
        split_seed_list = _parse_seed_list(split_seeds, "--split-seeds")
        train_seed_list = _parse_seed_list(train_seeds, "--train-seeds")
    except IQForgeError as exc:
        raise _fail(exc) from exc

    unreadable_error: str | None = None
    report: AuditReport | None
    try:
        if (path / MANIFEST_NAME).exists():
            report = audit_dataset(path, read_manifest(path), __version__)
        else:
            report = _audit_folder(path, window, stride, labels, dirname_level, label_file)
    except IQForgeError as exc:
        message = str(exc)
        if "could not be read" in message or "Unsupported datatype" in message:
            report = None
            unreadable_error = message
        else:
            raise _fail(exc) from exc

    group_keys: dict[str, str] | None = None
    if group_by is not None and report is not None and report.features:
        try:
            group_keys = resolve_group_keys(
                [f.record_id for f in report.features],
                group_by,
                collections={f.record_id: f.collection for f in report.features},
            )
        except IQForgeError as exc:
            raise _fail(exc) from exc

    decision = decide(
        report,
        force=force,
        split=split,
        window=window,
        stride=stride,
        group_keys=group_keys,
        unreadable_error=unreadable_error,
    )
    printed = (
        render_measure_json(decision) if output_format == "json" else render_measure_text(decision)
    )
    should_print_preflight_now = output_format == "text" or (
        output_format == "json" and decision.status is not DecisionStatus.WOULD_MEASURE
    )
    if should_print_preflight_now:
        print(printed, flush=True)
    if decision.status is not DecisionStatus.WOULD_MEASURE:
        raise typer.Exit(code=1)

    if not _torch_available():
        if sweep is not None:
            raise _fail(IQForgeError(TORCH_REQUIRED.format(what="iqforge measure-leakage --sweep")))
        return

    if not path.is_dir():
        raise _fail(
            IQForgeError(
                "measurement mode needs a recording folder path. A built dataset is "
                "supported for refuse-path classification only."
            )
        )

    spec = BuildSpec(
        window=window,
        stride=stride,
        split=split,
        labels=labels,
        label_file=label_file,
        dirname_level=dirname_level if labels == "dirname" else None,
        group_by=group_by,
        balance_by=balance_by,
    )

    if sweep == "stride":
        strides = (1024, 768, 512, 256, 128)
        cells = [
            GridCell(
                records=path,
                spec=BuildSpec(
                    window=window,
                    stride=s,
                    split=split,
                    labels=labels,
                    label_file=label_file,
                    dirname_level=dirname_level if labels == "dirname" else None,
                    group_by=group_by,
                    balance_by=balance_by,
                ),
                stride=s,
                label=f"stride={s}",
            )
            for s in strides
        ]
        runs = run_grid(cells, split_seed_list, train_seed_list, verbose=False)
        pairs = len(split_seed_list) * len(train_seed_list)
        caption = (
            f"Stride sweep over {pairs} seed pair(s) "
            f"(split {','.join(map(str, split_seed_list))} x "
            f"train {','.join(map(str, train_seed_list))}). "
            "Inflation is the mean paired difference, window-level minus "
            "recording-level."
        )
        table = summarise_stride_table(runs, window=window, caption=caption)
        if output_format == "json":
            payload = {
                "preflight": json.loads(render_measure_json(decision)),
                "measurement": {
                    "mode": "sweep_stride",
                    "split_seeds": split_seed_list,
                    "train_seeds": train_seed_list,
                    "seed_pairs": pairs,
                    "strides": list(strides),
                    "table": table,
                    "rows": [
                        {
                            "stride": run.stride,
                            "strategy": run.strategy,
                            "split_seed": run.split_seed,
                            "train_seed": run.train_seed,
                            "test_accuracy": run.test_accuracy,
                            "train_accuracy": run.train_accuracy,
                            "train_windows": run.train_windows,
                            "test_windows": run.test_windows,
                            "environment": run.environment,
                        }
                        for run in runs
                    ],
                },
            }
            print(json.dumps(payload, indent=2, ensure_ascii=True))
        else:
            print()
            print(table)
        return

    cell = GridCell(records=path, spec=spec, stride=stride, label=f"stride={stride}")
    runs = run_grid([cell], split_seed_list, train_seed_list, verbose=False)
    pairs = len(split_seed_list) * len(train_seed_list)
    stats = cell_stats(runs)
    if stats is None:
        raise _fail(IQForgeError("no seed pair produced both arms; nothing to report."))
    inflation_pp = stats.mean_diff * 100.0
    forced_prefix = "FORCED " if decision.forced else ""

    if output_format == "json":
        payload = {
            "preflight": json.loads(render_measure_json(decision)),
            "measurement": {
                "measurement_schema": MEASUREMENT_SCHEMA,
                "mode": "single",
                "forced": decision.forced,
                "split_seeds": split_seed_list,
                "train_seeds": train_seed_list,
                "seed_pairs": pairs,
                "stride": stride,
                "inflation_pp": inflation_pp,
                "stderr_pp": stats.stderr * 100.0 if stats.n > 1 else None,
                "n": stats.n,
                "rows": [_run_row(run) for run in runs],
            },
        }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print()
        console.print(f"[bold]{forced_prefix}MEASUREMENT[/]")
        console.print(
            f"stride={stride}  seed pairs={stats.n}  "
            f"split={','.join(map(str, split_seed_list))}  "
            f"train={','.join(map(str, train_seed_list))}"
        )
        console.print(f"inflation={inflation_pp:+.1f} pp {_uncertainty(stats)}")
        console.print(
            f"recording-level: test {stats.rec_mean:.2%}, window-level: test {stats.win_mean:.2%}"
        )
        for line in _environment_lines(runs):
            console.print(line)


def _audit_folder(
    path: Path,
    window: int,
    stride: int,
    labels: str,
    dirname_level: int,
    label_file: Path | None = None,
) -> AuditReport:
    """Measure a folder of recordings for `audit`."""
    meta_paths = collect_meta_paths(path) if path.is_dir() else [path]
    if not meta_paths:
        raise IQForgeError(f"No {META_EXT} files found under '{path}'.")

    if labels not in LABEL_SOURCES:
        raise IQForgeError(
            f"--labels must be one of {', '.join(LABEL_SOURCES)} for audit, got '{labels}'."
        )
    csv_table: dict[str, str] | None = None
    if labels == "csv":
        if label_file is None:
            raise IQForgeError("--labels csv needs --label-file.")
        csv_table = load_label_csv(label_file)

    exclude = resolve_exclude_labels(None)
    features: list[RecordFeatures] = []
    unreadable: list[tuple[str, str]] = []
    unlisted: list[str] = []
    for meta in meta_paths:
        # One unreadable recording used to abort the whole run, so a single
        # cf16_le file in a 330-file set produced no report at all. A dataset
        # this tool cannot fully read is exactly the case worth reporting on,
        # and the dataset-mode path already skipped per file -- this makes the
        # two agree.
        try:
            rec = load(meta)
            record_id = _record_id(meta, path)
            if labels == "dirname":
                label: str | None = dirname_at_level(meta, dirname_level)
            elif labels == "csv":
                assert csv_table is not None
                try:
                    window_labels, _ = label_from_csv(
                        rec, np.array([0]), csv_table, exclude, record_id=record_id
                    )
                    label = dominant_label(window_labels)
                except IQForgeError:
                    # A recording the table does not cover is unlabelled here,
                    # not fatal: audit reports what it found, and a partial
                    # table is itself worth reporting rather than refusing.
                    label, _ = None, unlisted.append(record_id)
            else:
                starts = np.array([0], dtype=np.int64)
                window_labels, _ = label_from_annotations(rec, starts, window, exclude, False)
                label = dominant_label(window_labels)
            features.append(measure_recording(rec, record_id, label))
        except (IQForgeError, OSError, ValueError) as exc:
            unreadable.append((_record_id(meta, path), str(exc)))

    if not features:
        raise IQForgeError(
            f"None of the {len(meta_paths)} recording(s) under '{path}' could be read. "
            f"First error: {unreadable[0][1] if unreadable else 'unknown'}"
        )

    label_source: dict[str, object] | None = None
    if labels == "csv" and label_file is not None:
        assigned = {f.label for f in features if f.label is not None}
        label_source = {
            "path": label_file,
            "declared": csv_declared_labels(label_file, [f.record_id for f in features], exclude),
            "assigned": assigned,
            "unlisted": unlisted,
        }

    return audit_recordings(
        path,
        features,
        window,
        stride,
        __version__,
        meta_paths,
        unreadable=unreadable,
        label_source=label_source,
    )


def _record_id(meta: Path, root: Path) -> str:
    """Identify a recording by its path relative to the audited root.

    The bare file name is ambiguous: a LoRa capture set holds a `3.sigmf-meta`
    under every session and receiver, so a finding that named two of them read
    `3.sigmf-meta / 3.sigmf-meta` and pointed at nothing.
    """
    try:
        return meta.relative_to(root).as_posix()
    except ValueError:
        return meta.name


@app.command()
def version() -> None:
    """Print the iqforge version."""
    console.print(f"iqforge {__version__}")


if __name__ == "__main__":
    app()
