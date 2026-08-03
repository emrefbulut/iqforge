"""sigkit komut satırı arayüzü."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from sigkit import __version__
from sigkit.io import Recording, SigkitError, load

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
    except SigkitError as exc:
        err_console.print(f"[bold red]Hata:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(_render_overview(rec))
    if rec.annotations:
        console.print(_render_annotations(rec))
    else:
        console.print("[dim]Kayıtta annotation yok.[/]")


@app.command()
def version() -> None:
    """sigkit sürümünü yazdırır."""
    console.print(f"sigkit {__version__}")


if __name__ == "__main__":
    app()
