"""Generate LinkedIn banner for iqforge (1200x627).

Outputs:
  linkedin.svg  — vector source (edit this)
  linkedin.png  — raster for LinkedIn upload

Usage:
    uv run python docs/linkedin/make_linkedin.py
"""

from __future__ import annotations

from pathlib import Path

W, H = 1200, 627
ROOT = Path(__file__).resolve().parent

# Palette (matches repo banner / viridis terminal spectrogram)
BG = "#0a0d12"
PANEL = "#121820"
BORDER = "#1e2836"
INK = "#e8edf4"
MUTED = "#8b97a8"
FAINT = "#4a5568"
ACCENT = "#35b779"
ACCENT2 = "#31688e"
WARN = "#fde725"

MONO = "Consolas, 'Courier New', monospace"


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def pipeline_box(x: int, y: int, w: int, h: int, title: str, lines: list[str]) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{PANEL}" '
        f'stroke="{BORDER}" stroke-width="1.5"/>',
        f'<text x="{x + 16}" y="{y + 28}" font-family="{MONO}" font-size="15" '
        f'font-weight="700" fill="{ACCENT}">{svg_escape(title)}</text>',
    ]
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x + 16}" y="{y + 52 + i * 22}" font-family="{MONO}" '
            f'font-size="13" fill="{MUTED}">{svg_escape(line)}</text>'
        )
    return "\n".join(parts)


def arrow(x1: int, y: int, x2: int) -> str:
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2 - 10}" y2="{y}" stroke="{FAINT}" '
        f'stroke-width="2" stroke-linecap="round"/>'
        f'<polygon points="{x2 - 10},{y - 5} {x2},{y} {x2 - 10},{y + 5}" fill="{FAINT}"/>'
    )


def build_svg() -> str:
    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="iqforge: SigMF to PyTorch datasets">'
    )
    lines.append("<title>iqforge</title>")
    lines.append(
        "<desc>CLI tool that converts SDR SigMF recordings into leak-safe "
        "PyTorch datasets with recording-level splits.</desc>"
    )

    # Background
    lines.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    lines.append(
        f'<rect x="0" y="0" width="{W}" height="4" fill="url(#grad)"/>'
    )
    lines.append(
        "<defs>"
        '<linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%" stop-color="#440154"/>'
        f'<stop offset="50%" stop-color="#35b779"/>'
        f'<stop offset="100%" stop-color="{WARN}"/>'
        "</linearGradient></defs>"
    )

    # Header
    lines.append(
        f'<text x="48" y="58" font-family="{MONO}" font-size="42" font-weight="700" '
        f'fill="{INK}" letter-spacing="2">iqforge</text>'
    )
    lines.append(
        f'<text x="48" y="88" font-family="{MONO}" font-size="17" fill="{MUTED}">'
        "Turn SDR captures into PyTorch datasets &#8212; without silent data leakage"
        "</text>"
    )

    # Purpose panel (left)
    lines.append(
        f'<rect x="48" y="110" width="340" height="200" rx="12" fill="{PANEL}" '
        f'stroke="{BORDER}" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="68" y="142" font-family="{MONO}" font-size="14" font-weight="700" '
        f'fill="{WARN}">THE GAP</text>'
    )
    purpose = [
        "TorchSig generates synthetic RF data.",
        "IQEngine inspects recordings in-browser.",
        "Nothing reliably bridges real SigMF",
        "captures to trainable PyTorch datasets.",
        "",
        "iqforge fills that step with explicit,",
        "tested pipeline decisions.",
    ]
    for i, row in enumerate(purpose):
        color = FAINT if not row else MUTED
        lines.append(
            f'<text x="68" y="{168 + i * 22}" font-family="{MONO}" font-size="13" '
            f'fill="{color}">{svg_escape(row) if row else " "}</text>'
        )

    # Pipeline (center)
    py = 330
    boxes = [
        (48, "SigMF input", [".sigmf-meta + .sigmf-data", "cf32_le / ci16_le / ci8", "memory-mapped I/O"]),
        (280, "iqforge CLI", ["info / inspect / build", "window + label + split", "shard + manifest.json"]),
        (512, "PyTorch output", ["IQForgeDataset", "iq2ch (2 x N) tensors", "baseline CNN train"]),
    ]
    bw, bh = 210, 100
    for i, (bx, title, blines) in enumerate(boxes):
        lines.append(pipeline_box(bx, py, bw, bh, title, blines))
        if i < len(boxes) - 1:
            lines.append(arrow(bx + bw + 6, py + bh // 2, boxes[i + 1][0] - 6))

    # Key guarantee callout
    lines.append(
        f'<rect x="48" y="450" width="674" height="72" rx="10" fill="#0f1a14" '
        f'stroke="{ACCENT}" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="68" y="478" font-family="{MONO}" font-size="13" font-weight="700" '
        f'fill="{ACCENT}">Recording-level split guarantee</text>'
    )
    lines.append(
        f'<text x="68" y="502" font-family="{MONO}" font-size="12" fill="{MUTED}">'
        "Windows from the same recording always share one split. "
        "Impossible splits error out &#8212; never fall back to window-level leakage."
        "</text>"
    )

    # Tech stack (right column)
    lines.append(
        f'<rect x="760" y="110" width="392" height="200" rx="12" fill="{PANEL}" '
        f'stroke="{BORDER}" stroke-width="1.5"/>'
    )
    lines.append(
        f'<text x="780" y="142" font-family="{MONO}" font-size="14" font-weight="700" '
        f'fill="{ACCENT2}">TECH STACK</text>'
    )
    stack = [
        ("Language", "Python 3.11+"),
        ("Format", "SigMF (sigmf-python)"),
        ("Numerics", "NumPy, SciPy (STFT)"),
        ("CLI / UI", "Typer, Rich (terminal spectrogram)"),
        ("ML", "PyTorch optional [torch]"),
        ("Quality", "pytest, ruff, GitHub Actions CI"),
    ]
    for i, (label, value) in enumerate(stack):
        y = 168 + i * 26
        lines.append(
            f'<text x="780" y="{y}" font-family="{MONO}" font-size="12" '
            f'fill="{FAINT}">{svg_escape(label + ":")}</text>'
        )
        lines.append(
            f'<text x="880" y="{y}" font-family="{MONO}" font-size="12" '
            f'fill="{MUTED}">{svg_escape(value)}</text>'
        )

    # Footer
    lines.append(
        f'<rect x="48" y="548" width="1104" height="52" rx="8" fill="{PANEL}" '
        f'stroke="{BORDER}" stroke-width="1"/>'
    )
    lines.append(
        f'<text x="68" y="580" font-family="{MONO}" font-size="14" fill="{INK}">'
        "github.com/emrefbulut/iqforge"
        "</text>"
    )
    lines.append(
        f'<text x="420" y="580" font-family="{MONO}" font-size="13" fill="{MUTED}">'
        "MIT  &#183;  Open Source  &#183;  v0.1.0 alpha"
        "</text>"
    )
    lines.append(
        f'<text x="900" y="580" font-family="{MONO}" font-size="13" fill="{ACCENT}">'
        "pip install coming soon  &#183;  clone + uv sync today"
        "</text>"
    )

    lines.append("</svg>")
    return "\n".join(lines)


def build_png(svg_path: Path, png_path: Path) -> bool:
    """Rasterize via Pillow (simple fallback, no cairo needed)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed; PNG skipped (SVG is ready)")
        return False

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    def font(size: int, bold: bool = False):
        candidates = [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    # Simplified raster version mirroring SVG layout
    draw.rectangle([0, 0, W, 4], fill="#35b779")
    draw.text((48, 24), "iqforge", fill=INK, font=font(36, True))
    draw.text(
        (48, 68),
        "Turn SDR captures into PyTorch datasets - without silent data leakage",
        fill=MUTED,
        font=font(14),
    )

    def panel(x, y, w, h, title, body_lines, title_color):
        draw.rounded_rectangle([x, y, x + w, y + h], radius=12, fill=PANEL, outline=BORDER)
        draw.text((x + 20, y + 16), title, fill=title_color, font=font(13, True))
        for i, line in enumerate(body_lines):
            if line:
                draw.text((x + 20, y + 44 + i * 20), line, fill=MUTED, font=font(12))

    panel(
        48, 110, 340, 200, "THE GAP",
        [
            "TorchSig: synthetic RF data",
            "IQEngine: browser inspection",
            "iqforge: real capture -> Dataset",
            "Explicit pipeline, no silent leaks",
        ],
        WARN,
    )
    panel(
        760, 110, 392, 200, "TECH STACK",
        [
            "Python 3.11+  |  SigMF",
            "NumPy, SciPy, Typer, Rich",
            "PyTorch [torch] optional",
            "pytest + ruff + CI",
        ],
        ACCENT2,
    )

    for i, (bx, title, subs) in enumerate(
        [
            (48, "SigMF input", [".sigmf-meta + .sigmf-data", "cf32_le / ci16_le / ci8"]),
            (280, "iqforge CLI", ["info / inspect / build / stats"]),
            (512, "PyTorch output", ["IQForgeDataset + baseline train"]),
        ]
    ):
        panel(bx, 330, 210, 100, title, subs, ACCENT)
        if i < 2:
            ax = bx + 215
            draw.line([(ax, 380), (ax + 28, 380)], fill=FAINT, width=2)

    draw.rounded_rectangle([48, 450, 722, 522], radius=10, fill="#0f1a14", outline=ACCENT)
    draw.text((68, 462), "Recording-level split guarantee", fill=ACCENT, font=font(12, True))
    draw.text(
        (68, 486),
        "Same recording -> one split only. Errors instead of window-level leakage.",
        fill=MUTED,
        font=font(11),
    )

    draw.rounded_rectangle([48, 548, 1152, 600], radius=8, fill=PANEL, outline=BORDER)
    draw.text((68, 562), "github.com/emrefbulut/iqforge", fill=INK, font=font(13))
    draw.text((420, 562), "MIT | Open Source | v0.1.0 alpha", fill=MUTED, font=font(12))

    img.save(png_path, "PNG", optimize=True)
    return True


def main() -> None:
    svg = build_svg()
    svg_path = ROOT / "linkedin.svg"
    png_path = ROOT / "linkedin.png"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"Wrote {svg_path}")
    if build_png(svg_path, png_path):
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
