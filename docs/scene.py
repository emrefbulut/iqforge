"""A tiny scene description rendered to both SVG and PNG.

Shared by the image generators in this directory. A scene is a flat list of
primitives; each renderer walks the same list, so the two formats cannot drift
apart -- which is what happens when a generator keeps one copy of its drawing
code per output format.

Not part of the installed package; these are development tools (SPEC §2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

#: Viridis control points. The tool renders spectrograms in viridis, so the
#: images that advertise it use the same colours.
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
    """Viridis colour for `t` in 0..1."""
    t = max(0.0, min(1.0, t))
    for i in range(len(ANCHORS) - 1):
        t0, c0 = ANCHORS[i]
        t1, c1 = ANCHORS[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0)
            return tuple(round(c0[j] + f * (c1[j] - c0[j])) for j in range(3))  # type: ignore[return-value]
    return (253, 231, 37)


def viridis(t: float) -> str:
    """Viridis colour for `t` as a hex string."""
    r, g, b = viridis_rgb(t)
    return f"#{r:02x}{g:02x}{b:02x}"


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
    anchor: str = "start"  # start | middle | end
    tracking: float = 0.0


@dataclass
class Scene:
    """An ordered list of primitives, painted back to front."""

    ops: list = field(default_factory=list)

    def rect(self, x: float, y: float, w: float, h: float, fill: str) -> None:
        self.ops.append(Rect(x, y, w, h, fill))

    def text(self, x: float, y: float, s: str, size: int, fill: str, **kw) -> None:
        self.ops.append(Text(x, y, s, size, fill, **kw))


def to_svg(scene: Scene, *, width: int, height: int, aria: str, desc: str) -> str:
    """Render the scene as an SVG document."""
    escapes = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{aria}">',
        f"<desc>{desc}</desc>",
        '<g shape-rendering="crispEdges">',
    ]
    in_group = True
    for op in scene.ops:
        if isinstance(op, Rect):
            if not in_group:
                out.append('<g shape-rendering="crispEdges">')
                in_group = True
            out.append(
                f'<rect x="{op.x:.2f}" y="{op.y:.2f}" width="{op.w:.2f}" '
                f'height="{op.h:.2f}" fill="{op.fill}"/>'
            )
        else:
            if in_group:
                out.append("</g>")
                in_group = False
            s = "".join(escapes.get(c, c) for c in op.s)
            weight = ' font-weight="700"' if op.bold else ""
            track = f' letter-spacing="{op.tracking}"' if op.tracking else ""
            out.append(
                f'<text x="{op.x:.2f}" y="{op.y:.2f}" font-family="{MONO}" '
                f'font-size="{op.size}"{weight}{track} fill="{op.fill}" '
                f'text-anchor="{op.anchor}">{s}</text>'
            )
    if in_group:
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


def _font(size: int, bold: bool):
    from PIL import ImageFont

    candidates = (
        "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_png(scene: Scene, path, *, width: int, height: int, background: str) -> bool:
    """Rasterise the scene with Pillow. Returns False if Pillow is missing."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed; PNG skipped (SVG is ready)")
        return False

    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)
    for op in scene.ops:
        if isinstance(op, Rect):
            draw.rectangle([op.x, op.y, op.x + op.w, op.y + op.h], fill=op.fill)
        else:
            font = _font(op.size, op.bold)
            if op.tracking:
                # Pillow has no letter-spacing, so glyphs are stepped manually.
                # The advance has to be measured first: drawing from op.x and
                # letting Pillow's own anchor handle alignment silently ignores
                # the anchor, which left-aligned every tracked label -- the
                # footer URL ran off the right edge and the class labels sat
                # beside their windows instead of under them.
                advance = sum(draw.textlength(ch, font=font) for ch in op.s)
                advance += op.tracking * max(len(op.s) - 1, 0)
                x = op.x - {"start": 0.0, "middle": advance / 2, "end": advance}[op.anchor]
                for ch in op.s:
                    draw.text((x, op.y), ch, fill=op.fill, font=font, anchor="ls")
                    x += draw.textlength(ch, font=font) + op.tracking
            else:
                anchor = {"start": "ls", "middle": "ms", "end": "rs"}[op.anchor]
                draw.text((op.x, op.y), op.s, fill=op.fill, font=font, anchor=anchor)

    img.save(path, "PNG", optimize=True)
    return True
