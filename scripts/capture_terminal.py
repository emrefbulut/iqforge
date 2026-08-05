"""Save `iqforge inspect` output, colours included, under artifacts/.

The terminal spectrogram carries all of its information in colour; when the
output is redirected to a file `rich` turns colour off and all that remains is a
uniform block of `▀`. This script builds the console with `force_terminal=True`
and writes two formats:

  * `.ansi.txt` - with ANSI escape sequences; view it with `cat` in a terminal.
  * `.svg`      - a vector image with the colours embedded, viewable in a browser.

Usage:
    python scripts/capture_terminal.py --samples 32768 -o artifacts/inspect_bpsk_01
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from iqforge.display import render_inspect
from iqforge.io import load


def main() -> None:
    """Render the requested window and save it as ANSI text and SVG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="examples/bpsk_01.sigmf-meta")
    parser.add_argument("-o", "--output", required=True, help="output path without extension")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=262_144)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--height", type=int, default=24)
    args = parser.parse_args()

    rec = load(args.path)
    data = rec.read(start=args.start, count=args.samples)

    console = Console(width=args.width, force_terminal=True, record=True, color_system="truecolor")
    console.print(
        render_inspect(rec, data, args.start, args.nfft, width=args.width, height=args.height)
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # clear=False is required: the default empties the record buffer and the
    # save_svg that follows would produce an empty SVG.
    console.save_text(str(out.with_suffix(".ansi.txt")), styles=True, clear=False)
    console.save_svg(str(out.with_suffix(".svg")), title=f"iqforge inspect - {rec.meta_path.name}")
    print(f"written: {out.with_suffix('.ansi.txt')}")
    print(f"written: {out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
