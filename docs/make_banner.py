"""Banner generator for iqforge.

Thesis: one object, transforming. A continuous spectrogram enters at the left and
progressively fractures into discrete, labelled dataset windows toward the right.
It is not two pictures joined by an arrow - it is the same signal, being cut.

Content is honest to the project's own sample capture: a continuous reference tone,
and two bursts of EQUAL duration and bandwidth at different times and frequencies.

Palette is viridis because that is the colormap the tool renders. Class labels are
set in type, not colour, so nothing competes with viridis's own meaning (intensity).
"""

import random

W, H = 1280, 340
GROUND = "#0a0d12"
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


rng = random.Random(20260804)

NWIN = 16
CELLS_PER_WIN = 8  # noise columns inside one window
ROWS = 22
BAND_Y, BAND_H = 40, 176
RH = BAND_H / ROWS

X0, X1 = 64, 1216
SPAN = X1 - X0

# gap before window k grows toward the right: the cut becoming visible
gaps = []
for k in range(NWIN):
    if k == 0:
        gaps.append(0.0)
    elif k < 7:
        gaps.append(0.0)
    else:
        gaps.append(min((k - 6) * 2.6, 15.0))
total_gap = sum(gaps)
WIN_W = (SPAN - total_gap) / NWIN

# signal content, honest to the sample capture
REF_ROW = 8
BPSK_WINS, BPSK_ROWS = range(2, 6), range(15, 20)
QPSK_WINS, QPSK_ROWS = range(12, 16), range(2, 7)

out = []
out.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}" role="img" '
    f'aria-label="iqforge - turn SDR captures into PyTorch datasets">'
)
out.append("<title>iqforge</title>")
out.append(
    "<desc>A spectrogram showing a continuous reference tone and two signal bursts, "
    "progressively cut into discrete labelled dataset windows toward the right.</desc>"
)
out.append(f'<rect width="{W}" height="{H}" fill="{GROUND}"/>')
out.append('<g shape-rendering="crispEdges">')

x = X0
win_x = []
for k in range(NWIN):
    x += gaps[k]
    win_x.append(x)
    cw = WIN_W / CELLS_PER_WIN
    for r in range(ROWS):
        for c in range(CELLS_PER_WIN):
            if r == REF_ROW:
                lvl = 0.92 + rng.random() * 0.08
            elif (k in BPSK_WINS and r in BPSK_ROWS) or (k in QPSK_WINS and r in QPSK_ROWS):
                lvl = 0.62 + rng.random() * 0.20
            else:
                lvl = 0.08 + rng.random() * 0.15
            out.append(
                f'<rect x="{x + c * cw:.2f}" y="{BAND_Y + r * RH:.2f}" '
                f'width="{cw + 0.4:.2f}" height="{RH + 0.4:.2f}" fill="{viridis(lvl)}"/>'
            )
    x += WIN_W
out.append("</g>")

MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

# labels appear only where the cut has fully opened
for k in range(12, NWIN):
    cx = win_x[k] + WIN_W / 2
    out.append(
        f'<text x="{cx:.1f}" y="{BAND_Y + BAND_H + 18:.0f}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="12" letter-spacing="0.5" fill="{MUTED}">qpsk</text>'
    )

# hairline marking where windowing begins
cut_x = win_x[7] - gaps[7] / 2
out.append(
    f'<line x1="{cut_x:.1f}" y1="{BAND_Y - 10}" x2="{cut_x:.1f}" y2="{BAND_Y + BAND_H + 10}" '
    f'stroke="{FAINT}" stroke-width="1" stroke-dasharray="3 4"/>'
)
out.append(
    f'<text x="{cut_x + 8:.1f}" y="{BAND_Y - 14}" font-family="{MONO}" font-size="12" '
    f'letter-spacing="1" fill="{FAINT}">window / label / split</text>'
)

out.append(
    f'<text x="64" y="292" font-family="{MONO}" font-size="44" font-weight="600" '
    f'letter-spacing="3" fill="{INK}">iqforge</text>'
)
out.append(
    f'<text x="64" y="318" font-family="{MONO}" font-size="15" letter-spacing="1.2" '
    f'fill="{MUTED}">Turn SDR captures into PyTorch datasets.</text>'
)
out.append(
    f'<text x="1216" y="318" text-anchor="end" font-family="{MONO}" font-size="13" '
    f'letter-spacing="2" fill="{FAINT}">SigMF &#8594; torch.Dataset</text>'
)
out.append("</svg>")

with open("banner.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("banner.svg written")
