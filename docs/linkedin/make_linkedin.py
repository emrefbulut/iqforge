"""Generate LinkedIn banner for iqforge (1200x627).

Visual-first: the continuous SigMF spectrogram fracturing into labelled
dataset windows — same language as docs/banner.svg / the demo GIF.
Text is minimal; the LinkedIn caption carries the explanation.

Usage:
    uv run python docs/linkedin/make_linkedin.py
"""

from __future__ import annotations

import random
from pathlib import Path

W, H = 1200, 627
ROOT = Path(__file__).resolve().parent

BG = "#0a0d12"
INK = "#e8edf4"
MUTED = "#8b97a8"
FAINT = "#4a5568"

ANCHORS = [
    (0.000, (68, 1, 84)),
    (0.125, (72, 40, 120)),
    (0.250, (62, 73, 137)),
    (0.375, (49, 104, 142)),
    (0.500, (38, 130, 142)),
    (0.625, (31, 158, 137)),
    (0.750, (53, 183, 121)),
    (0.875, (110, 206, 88)),
    (1.000, (253, 231, 37)),
]

MONO = "Consolas, 'Courier New', monospace"


def viridis(t: float) -> str:
    t = max(0.0, min(1.0, t))
    for i in range(len(ANCHORS) - 1):
        t0, c0 = ANCHORS[i]
        t1, c1 = ANCHORS[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            r, g, b = (round(c0[j] + f * (c1[j] - c0[j])) for j in range(3))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#fde725"


def build_svg() -> str:
    rng = random.Random(20260804)

    # Spectrogram band fills most of the frame
    NWIN = 18
    CELLS = 8
    ROWS = 28
    BAND_Y, BAND_H = 72, 380
    RH = BAND_H / ROWS
    X0, X1 = 48, 1152
    SPAN = X1 - X0

    gaps: list[float] = []
    for k in range(NWIN):
        if k == 0 or k < 8:
            gaps.append(0.0)
        else:
            gaps.append(min((k - 7) * 3.2, 18.0))
    total_gap = sum(gaps)
    WIN_W = (SPAN - total_gap) / NWIN

    REF_ROW = 10
    BPSK_WINS, BPSK_ROWS = range(2, 6), range(18, 25)
    QPSK_WINS, QPSK_ROWS = range(13, 17), range(3, 9)

    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="iqforge — SigMF captures into PyTorch datasets">'
    )
    out.append("<title>iqforge</title>")
    out.append(
        "<desc>A viridis spectrogram of an SDR capture, cut into labelled "
        "dataset windows toward the right.</desc>"
    )
    out.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

    # Thin viridis accent line at top
    out.append(
        "<defs>"
        '<linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">'
        '<stop offset="0%" stop-color="#440154"/>'
        '<stop offset="50%" stop-color="#35b779"/>'
        '<stop offset="100%" stop-color="#fde725"/>'
        "</linearGradient></defs>"
    )
    out.append(f'<rect x="0" y="0" width="{W}" height="3" fill="url(#grad)"/>')

    # Spectrogram cells
    out.append('<g shape-rendering="crispEdges">')
    x = float(X0)
    win_x: list[float] = []
    for k in range(NWIN):
        x += gaps[k]
        win_x.append(x)
        cw = WIN_W / CELLS
        for r in range(ROWS):
            for c in range(CELLS):
                if r == REF_ROW:
                    lvl = 0.92 + rng.random() * 0.08
                elif (k in BPSK_WINS and r in BPSK_ROWS) or (k in QPSK_WINS and r in QPSK_ROWS):
                    lvl = 0.62 + rng.random() * 0.20
                else:
                    lvl = 0.08 + rng.random() * 0.15
                out.append(
                    f'<rect x="{x + c * cw:.2f}" y="{BAND_Y + r * RH:.2f}" '
                    f'width="{cw + 0.4:.2f}" height="{RH + 0.4:.2f}" '
                    f'fill="{viridis(lvl)}"/>'
                )
        x += WIN_W
    out.append("</g>")

    # Cut line where windowing becomes visible
    cut_x = win_x[8] - gaps[8] / 2
    out.append(
        f'<line x1="{cut_x:.1f}" y1="{BAND_Y - 16}" x2="{cut_x:.1f}" '
        f'y2="{BAND_Y + BAND_H + 16}" stroke="{FAINT}" stroke-width="1.5" '
        f'stroke-dasharray="4 5"/>'
    )

    # Class labels under fully separated windows (qpsk burst lives here)
    for k in range(13, NWIN):
        cx = win_x[k] + WIN_W / 2
        out.append(
            f'<text x="{cx:.1f}" y="{BAND_Y + BAND_H + 28:.0f}" '
            f'text-anchor="middle" font-family="{MONO}" font-size="13" '
            f'letter-spacing="1" fill="{MUTED}">qpsk</text>'
        )

    # Brand block — bottom left, sparse
    out.append(
        f'<text x="48" y="560" font-family="{MONO}" font-size="48" '
        f'font-weight="700" letter-spacing="3" fill="{INK}">iqforge</text>'
    )
    out.append(
        f'<text x="48" y="592" font-family="{MONO}" font-size="16" '
        f'letter-spacing="1" fill="{MUTED}">'
        "SDR capture &#8594; leak-safe PyTorch dataset"
        "</text>"
    )
    out.append(
        f'<text x="1152" y="592" text-anchor="end" font-family="{MONO}" '
        f'font-size="14" letter-spacing="1.5" fill="{FAINT}">'
        "SigMF &#183; recording-level split"
        "</text>"
    )

    out.append("</svg>")
    return "\n".join(out)


def build_png(png_path: Path) -> bool:
    """Rasterize the same visual with Pillow (no cairo)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed; PNG skipped (SVG is ready)")
        return False

    rng = random.Random(20260804)
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    def font(size: int, bold: bool = False):
        names = [
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/consola.ttf",
        ]
        for path in names:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def viridis_rgb(t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        for i in range(len(ANCHORS) - 1):
            t0, c0 = ANCHORS[i]
            t1, c1 = ANCHORS[i + 1]
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0)
                return tuple(round(c0[j] + f * (c1[j] - c0[j])) for j in range(3))  # type: ignore[return-value]
        return (253, 231, 37)

    NWIN = 18
    CELLS = 8
    ROWS = 28
    BAND_Y, BAND_H = 72, 380
    RH = BAND_H / ROWS
    X0, X1 = 48, 1152
    SPAN = X1 - X0
    gaps = [0.0 if k < 8 else min((k - 7) * 3.2, 18.0) for k in range(NWIN)]
    gaps[0] = 0.0
    WIN_W = (SPAN - sum(gaps)) / NWIN
    REF_ROW = 10
    BPSK_WINS, BPSK_ROWS = range(2, 6), range(18, 25)
    QPSK_WINS, QPSK_ROWS = range(13, 17), range(3, 9)

    # Top accent
    for i in range(W):
        t = i / (W - 1)
        draw.point((i, 0), fill=viridis_rgb(t))
        draw.point((i, 1), fill=viridis_rgb(t))
        draw.point((i, 2), fill=viridis_rgb(t))

    x = float(X0)
    win_x: list[float] = []
    for k in range(NWIN):
        x += gaps[k]
        win_x.append(x)
        cw = WIN_W / CELLS
        for r in range(ROWS):
            for c in range(CELLS):
                if r == REF_ROW:
                    lvl = 0.92 + rng.random() * 0.08
                elif (k in BPSK_WINS and r in BPSK_ROWS) or (k in QPSK_WINS and r in QPSK_ROWS):
                    lvl = 0.62 + rng.random() * 0.20
                else:
                    lvl = 0.08 + rng.random() * 0.15
                x0 = x + c * cw
                y0 = BAND_Y + r * RH
                draw.rectangle(
                    [x0, y0, x0 + cw + 0.4, y0 + RH + 0.4],
                    fill=viridis_rgb(lvl),
                )
        x += WIN_W

    cut_x = win_x[8] - gaps[8] / 2
    # dashed cut line
    y = BAND_Y - 16
    while y < BAND_Y + BAND_H + 16:
        draw.line([(cut_x, y), (cut_x, min(y + 4, BAND_Y + BAND_H + 16))], fill=FAINT, width=1)
        y += 9

    for k in range(13, NWIN):
        cx = win_x[k] + WIN_W / 2
        draw.text((cx - 16, BAND_Y + BAND_H + 12), "qpsk", fill=MUTED, font=font(12))

    draw.text((48, 520), "iqforge", fill=INK, font=font(44, True))
    draw.text(
        (48, 575),
        "SDR capture  ->  leak-safe PyTorch dataset",
        fill=MUTED,
        font=font(15),
    )
    # right footer approx
    draw.text(
        (820, 575),
        "SigMF  ·  recording-level split",
        fill=FAINT,
        font=font(13),
    )

    img.save(png_path, "PNG", optimize=True)
    return True


def main() -> None:
    svg = build_svg()
    svg_path = ROOT / "linkedin.svg"
    png_path = ROOT / "linkedin.png"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"Wrote {svg_path}")
    if build_png(png_path):
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
