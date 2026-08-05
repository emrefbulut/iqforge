"""Export docs/linkedin/linkedin.svg to PNG for LinkedIn (1200x627).

Requires: pip install cairosvg  (or run on a machine with Inkscape/rsvg-convert)

Usage:
    uv run python docs/linkedin/export_png.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SVG = ROOT / "linkedin.svg"
PNG = ROOT / "linkedin.png"


def main() -> None:
    try:
        import cairosvg
    except ImportError as exc:
        raise SystemExit(
            "Install cairosvg: uv pip install cairosvg\n"
            "Or export manually: inkscape linkedin.svg -o linkedin.png -w 1200 -h 627"
        ) from exc

    cairosvg.svg2png(url=str(SVG), write_to=str(PNG), output_width=1200, output_height=627)
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
