"""Banner generator for iqforge — GitHub social preview.

Thesis: one object, transforming. A continuous spectrogram enters at the left and
progressively fractures into discrete, labelled dataset windows toward the right.
It is not two pictures joined by an arrow — it is the same signal, being cut.

Sized 1280x640, which is what GitHub's social preview slot expects. The previous
version was 1280x340: a letterbox strip dropped into a 2:1 frame, which the card
renderer scales to fit, so everything came out oversized. Aspect ratio is the
whole reason this file was rewritten — keep it at 2:1.

Type is set large on purpose. The card is usually seen shrunk to a few hundred
pixels wide in a link preview, so anything sized for full-resolution viewing
becomes unreadable exactly where it is meant to work.

Content is honest to the project's own sample capture: a continuous reference
tone, and bursts of EQUAL duration and bandwidth at different times and
frequencies. Both classes appear in the cut region, so the labels show what the
tool actually produces rather than a single class name.

Palette is viridis because that is the colormap the tool renders. Class labels
are set in type, not colour, so nothing competes with viridis's own meaning.

Usage:
    uv run python docs/make_banner.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scene import Scene, to_png, to_svg, viridis  # noqa: E402

ROOT = Path(__file__).resolve().parent

#: GitHub social preview. Do not change without re-reading the module docstring.
W, H = 1280, 640

GROUND = "#0a0d12"
INK = "#e8edf4"
MUTED = "#8b97a8"
FAINT = "#4a5568"

MARGIN = 72

# --- spectrogram geometry -------------------------------------------------
NWIN = 16
CELLS_PER_WIN = 8
ROWS = 26
BAND_Y, BAND_H = 286, 236
RH = BAND_H / ROWS

X0, X1 = MARGIN, W - MARGIN
SPAN = X1 - X0

#: The reference tone runs through every window, as it does in examples/.
REF_ROW = 9

#: Bursts. The left pair sits in the continuous region; the right pair sits in
#: the separated windows, one per class, so both labels have something to name.
LEFT_BURST_WINS, LEFT_BURST_ROWS = range(1, 5), range(17, 22)
BPSK_WINS, BPSK_ROWS = range(10, 13), range(17, 22)
QPSK_WINS, QPSK_ROWS = range(13, 16), range(3, 8)

#: Windows are flush until here, then the gap between them opens: the cut
#: becoming visible rather than announced.
FIRST_CUT = 8


def _gaps() -> list[float]:
    return [0.0 if k < FIRST_CUT else min((k - FIRST_CUT + 1) * 3.4, 20.0) for k in range(NWIN)]


def build() -> Scene:
    rng = random.Random(20260804)
    scene = Scene()
    scene.rect(0, 0, W, H, GROUND)

    # Top accent: the viridis ramp, tying the card to the tool's own output.
    for i in range(W):
        scene.rect(i, 0, 1.4, 7, viridis(i / (W - 1)))

    scene.text(MARGIN, 150, "iqforge", 96, INK, bold=True, tracking=4)
    scene.text(
        MARGIN,
        199,
        "Turn SDR captures into PyTorch datasets.",
        27,
        MUTED,
        tracking=0.6,
    )

    gaps = _gaps()
    win_w = (SPAN - sum(gaps)) / NWIN
    cw = win_w / CELLS_PER_WIN

    x = float(X0)
    win_x: list[float] = []
    for k in range(NWIN):
        x += gaps[k]
        win_x.append(x)
        for r in range(ROWS):
            for c in range(CELLS_PER_WIN):
                if r == REF_ROW:
                    level = 0.92 + rng.random() * 0.08
                elif (
                    (k in LEFT_BURST_WINS and r in LEFT_BURST_ROWS)
                    or (k in BPSK_WINS and r in BPSK_ROWS)
                    or (k in QPSK_WINS and r in QPSK_ROWS)
                ):
                    level = 0.62 + rng.random() * 0.20
                else:
                    level = 0.08 + rng.random() * 0.15
                scene.rect(x + c * cw, BAND_Y + r * RH, cw + 0.5, RH + 0.5, viridis(level))
        x += win_w

    # Hairline where windowing begins.
    cut_x = win_x[FIRST_CUT] - gaps[FIRST_CUT] / 2
    y = BAND_Y - 16
    while y < BAND_Y + BAND_H + 16:
        scene.rect(cut_x, y, 1, 5, FAINT)
        y += 11
    scene.text(cut_x + 12, BAND_Y - 22, "window / label / split", 19, FAINT, tracking=1.2)

    # Class labels, only under windows the cut has fully separated.
    for wins, name in ((BPSK_WINS, "bpsk"), (QPSK_WINS, "qpsk")):
        centre = (win_x[wins[0]] + win_x[wins[-1]] + win_w) / 2
        scene.text(centre, BAND_Y + BAND_H + 34, name, 21, MUTED, anchor="middle", tracking=1)

    scene.text(MARGIN, H - 36, "SigMF  ·  recording-level splits  ·  MIT", 20, FAINT, tracking=1)
    scene.text(
        W - MARGIN,
        H - 36,
        "github.com/emrefbulut/iqforge",
        20,
        MUTED,
        anchor="end",
        tracking=0.5,
    )
    return scene


def main() -> None:
    scene = build()

    svg_path = ROOT / "banner.svg"
    svg_path.write_text(
        to_svg(
            scene,
            width=W,
            height=H,
            aria="iqforge - turn SDR captures into PyTorch datasets",
            desc=(
                "A spectrogram of an SDR capture. On the left it is continuous; toward "
                "the right it separates into discrete dataset windows, the last of which "
                "are labelled bpsk and qpsk. A reference tone runs through the whole "
                "recording."
            ),
        ),
        encoding="utf-8",
    )
    print(f"wrote {svg_path.name}  {W}x{H}  ({svg_path.stat().st_size / 1024:.0f} KB)")

    png_path = ROOT / "banner.png"
    if to_png(scene, png_path, width=W, height=H, background=GROUND):
        print(f"wrote {png_path.name}  {W}x{H}  ({png_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
