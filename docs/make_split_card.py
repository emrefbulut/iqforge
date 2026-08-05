"""Generate the LinkedIn card that argues the split (1200x1200).

The existing `make_linkedin.py` shows a spectrogram. This one shows the *claim*:
the same recording, split two ways. Random window splitting scatters
neighbouring windows across train/val/test; recording-level splitting keeps each
recording whole. The contrast between a noisy stripe and clean blocks carries the
argument without the reader parsing a caption.

Square rather than 1200x627: LinkedIn shows a square image at roughly double the
height in a mobile feed, and the feed is mostly mobile.

The scene is described once and rendered to both SVG and PNG, so the two cannot
drift apart.

Usage:
    uv run python docs/linkedin/make_split_card.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

W = H = 1200
ROOT = Path(__file__).resolve().parent

BG = "#0a0d12"
INK = "#e8edf4"
MUTED = "#8b97a8"
FAINT = "#3d4757"

# Split accents. Chosen to stay legible against viridis, which owns the
# blue-green-yellow range: the bars must not read as more spectrogram.
TRAIN = "#5b8cff"
VAL = "#ffb020"
TEST = "#ff5c8a"

MONO = "Consolas, 'DejaVu Sans Mono', 'Courier New', monospace"

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


def viridis_rgb(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    for i in range(len(ANCHORS) - 1):
        t0, c0 = ANCHORS[i]
        t1, c1 = ANCHORS[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(round(c0[j] + f * (c1[j] - c0[j])) for j in range(3))  # type: ignore[return-value]
    return (253, 231, 37)


def viridis(t: float) -> str:
    r, g, b = viridis_rgb(t)
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------
# A scene is a list of primitives. Both renderers walk the same list, which is
# how the SVG and the PNG stay identical.
# --------------------------------------------------------------------------


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    fill: str


@dataclass
class Text:
    x: float
    y: float  # baseline
    s: str
    size: int
    fill: str
    bold: bool = False
    anchor: str = "start"  # "start" | "middle"
    tracking: float = 0.0


@dataclass
class Scene:
    ops: list = field(default_factory=list)

    def rect(self, x: float, y: float, w: float, h: float, fill: str) -> None:
        self.ops.append(Rect(x, y, w, h, fill))

    def text(self, x: float, y: float, s: str, size: int, fill: str, **kw) -> None:
        self.ops.append(Text(x, y, s, size, fill, **kw))


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

M = 72  # page margin
CONTENT = W - 2 * M

NREC = 8  # 4 bpsk + 4 qpsk
WINS_PER_REC = 7  # windows cut from each recording
ROWS = 22  # spectrogram frequency bins
REF_ROW = 7  # the +100 kHz reference tone, present in every recording

# Class layout mirrors examples/: bpsk sits low, qpsk sits high, ref tone flat.
CLASSES = ["bpsk", "qpsk", "bpsk", "qpsk", "bpsk", "qpsk", "bpsk", "qpsk"]
BPSK_ROWS = range(14, 19)
QPSK_ROWS = range(2, 6)

# 0.5 / 0.25 / 0.25, stratified: train takes two of each class, val and test one.
RECORD_SPLIT = [TRAIN, TRAIN, TRAIN, TRAIN, VAL, VAL, TEST, TEST]

BAND_H = 176
BAR_H = 22

PANEL_A_LABEL_Y = 238
PANEL_A_BAND_Y = 272
PANEL_A_BAR_Y = PANEL_A_BAND_Y + BAND_H + 14
PANEL_A_NOTE_Y = PANEL_A_BAR_Y + BAR_H + 40

DIVIDER_Y = 576

PANEL_B_LABEL_Y = 640
PANEL_B_BAND_Y = 674
PANEL_B_BAR_Y = PANEL_B_BAND_Y + BAND_H + 14
PANEL_B_TICK_Y = PANEL_B_BAR_Y + BAR_H + 30
PANEL_B_NOTE_Y = PANEL_B_TICK_Y + 74

LEGEND_Y = 1074
FOOTER_Y = 1146


def cell_level(rng: random.Random, rec: int, row: int) -> float:
    """Power in one spectrogram cell, in 0..1."""
    if row == REF_ROW:
        return 0.90 + rng.random() * 0.10
    rows = BPSK_ROWS if CLASSES[rec] == "bpsk" else QPSK_ROWS
    if row in rows:
        return 0.58 + rng.random() * 0.22
    return 0.06 + rng.random() * 0.14


def verdict(scene: Scene, y: float, s: str, colour: str) -> None:
    """A one-line takeaway, marked with a swatch rather than a glyph.

    Check and cross marks (U+2713 / U+2717) are missing from Consolas and render
    as tofu — verified by rendering, not assumed. A filled square is font-proof
    and echoes the legend, where the same colours already mean something.
    """
    scene.rect(M, y - 14, 15, 15, colour)
    scene.text(M + 30, y, s, 19, colour)


def draw_band(scene: Scene, rng: random.Random, y0: float, gap: float) -> list[float]:
    """Draw the spectrogram. Returns the left x of each recording."""
    total_gap = gap * (NREC - 1)
    rec_w = (CONTENT - total_gap) / NREC
    win_w = rec_w / WINS_PER_REC
    rh = BAND_H / ROWS

    lefts: list[float] = []
    x = float(M)
    for rec in range(NREC):
        lefts.append(x)
        for w in range(WINS_PER_REC):
            for row in range(ROWS):
                lvl = cell_level(rng, rec, row)
                scene.rect(
                    x + w * win_w,
                    y0 + row * rh,
                    win_w + 0.5,
                    rh + 0.5,
                    viridis(lvl),
                )
        x += rec_w + gap
    return lefts


def build_scene() -> Scene:
    scene = Scene()
    scene.rect(0, 0, W, H, BG)

    # Top accent: the viridis ramp, so the card is recognisably the same family
    # as the README banner and the demo GIF.
    for i in range(W):
        scene.rect(i, 0, 1.4, 6, viridis(i / (W - 1)))

    scene.text(M, 118, "iqforge", 58, INK, bold=True)
    scene.text(M, 158, "SDR capture  \u2192  leak-safe PyTorch dataset", 20, MUTED)

    # ---------------- Panel A: the failure mode ----------------
    scene.text(M, PANEL_A_LABEL_Y, "RANDOM WINDOW SPLIT", 17, MUTED, bold=True, tracking=2.2)
    scene.text(
        M + 268,
        PANEL_A_LABEL_Y,
        "\u2014  what most pipelines do by default",
        17,
        FAINT,
    )

    rng = random.Random(20260805)
    draw_band(scene, rng, PANEL_A_BAND_Y, gap=0.0)

    # One bar segment per window, assigned at random: neighbouring windows end
    # up on opposite sides of the split.
    nwin = NREC * WINS_PER_REC
    seg_w = CONTENT / nwin
    pick = random.Random(7)
    for i in range(nwin):
        colour = pick.choices([TRAIN, VAL, TEST], weights=[2, 1, 1])[0]
        scene.rect(M + i * seg_w, PANEL_A_BAR_Y, seg_w + 0.5, BAR_H, colour)

    verdict(
        scene,
        PANEL_A_NOTE_Y,
        "adjacent windows land in train and test. accuracy goes up, the model doesn't.",
        TEST,
    )

    scene.rect(M, DIVIDER_Y, CONTENT, 1, FAINT)

    # ---------------- Panel B: what iqforge does ----------------
    scene.text(M, PANEL_B_LABEL_Y, "RECORDING-LEVEL SPLIT", 17, INK, bold=True, tracking=2.2)
    scene.text(M + 292, PANEL_B_LABEL_Y, "\u2014  iqforge, by default", 17, MUTED)

    rng = random.Random(20260805)
    gap = 9.0
    lefts = draw_band(scene, rng, PANEL_B_BAND_Y, gap=gap)
    rec_w = (CONTENT - gap * (NREC - 1)) / NREC

    for rec in range(NREC):
        scene.rect(lefts[rec], PANEL_B_BAR_Y, rec_w, BAR_H, RECORD_SPLIT[rec])
        cx = lefts[rec] + rec_w / 2
        scene.text(cx, PANEL_B_TICK_Y, CLASSES[rec], 15, MUTED, anchor="middle")
        scene.text(cx, PANEL_B_TICK_Y + 21, f"rec_{rec + 1:02d}", 13, FAINT, anchor="middle")

    verdict(
        scene,
        PANEL_B_NOTE_Y,
        "every window from one recording stays on one side. classes stay stratified.",
        "#4ade80",
    )

    # ---------------- Legend + footer ----------------
    x = float(M)
    for colour, name in ((TRAIN, "train"), (VAL, "val"), (TEST, "test")):
        scene.rect(x, LEGEND_Y - 13, 26, 13, colour)
        scene.text(x + 36, LEGEND_Y, name, 17, MUTED)
        x += 130

    scene.text(
        W - M,
        LEGEND_Y,
        "when it can't, it errors \u2014 it never falls back",
        17,
        FAINT,
        anchor="end",
    )

    scene.text(M, FOOTER_Y, "SigMF  \u00b7  cf32_le / ci16_le / ci8  \u00b7  MIT", 16, FAINT)
    scene.text(W - M, FOOTER_Y, "github.com/emrefbulut/iqforge", 16, MUTED, anchor="end")

    return scene


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------


def to_svg(scene: Scene) -> str:
    esc = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="iqforge: random window splitting versus recording-level splitting">',
        "<desc>Two spectrogram strips of the same SDR recording. In the first, a "
        "colour bar assigns individual windows to train, validation and test at "
        "random. In the second, every window of a recording carries a single "
        "colour, so no recording is divided between splits.</desc>",
    ]
    for op in scene.ops:
        if isinstance(op, Rect):
            out.append(
                f'<rect x="{op.x:.2f}" y="{op.y:.2f}" width="{op.w:.2f}" '
                f'height="{op.h:.2f}" fill="{op.fill}"/>'
            )
        else:
            s = "".join(esc.get(c, c) for c in op.s)
            anchor = {"start": "start", "middle": "middle", "end": "end"}[op.anchor]
            weight = ' font-weight="700"' if op.bold else ""
            track = f' letter-spacing="{op.tracking}"' if op.tracking else ""
            out.append(
                f'<text x="{op.x:.2f}" y="{op.y:.2f}" font-family="{MONO}" '
                f'font-size="{op.size}"{weight}{track} fill="{op.fill}" '
                f'text-anchor="{anchor}">{s}</text>'
            )
    out.append("</svg>")
    return "\n".join(out)


def to_png(scene: Scene, path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed; PNG skipped (SVG is ready)")
        return False

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    def font(size: int, bold: bool):
        for p in (
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
        return ImageFont.load_default()

    for op in scene.ops:
        if isinstance(op, Rect):
            draw.rectangle([op.x, op.y, op.x + op.w, op.y + op.h], fill=op.fill)
        else:
            f = font(op.size, op.bold)
            if op.tracking:
                # Pillow has no letter-spacing; step glyph by glyph.
                x = op.x
                for ch in op.s:
                    draw.text((x, op.y), ch, fill=op.fill, font=f, anchor="ls")
                    x += draw.textlength(ch, font=f) + op.tracking
            else:
                anchor = {"start": "ls", "middle": "ms", "end": "rs"}[op.anchor]
                draw.text((op.x, op.y), op.s, fill=op.fill, font=f, anchor=anchor)

    img.save(path, "PNG", optimize=True)
    return True


def main() -> None:
    scene = build_scene()

    svg_path = ROOT / "split_card.svg"
    svg_path.write_text(to_svg(scene), encoding="utf-8")
    print(f"wrote {svg_path}  ({svg_path.stat().st_size / 1024:.0f} KB)")

    png_path = ROOT / "split_card.png"
    if to_png(scene, png_path):
        print(f"wrote {png_path}  ({png_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
