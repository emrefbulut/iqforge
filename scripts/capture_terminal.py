"""`sigkit inspect` çıktısını renkleriyle birlikte artifacts/ altına kaydeder.

Terminal spektrogramı bilgisini tamamen renkte taşır; çıktı bir dosyaya
yönlendirildiğinde `rich` renkleri kapatır ve geriye tekdüze bir `▀` bloğu
kalır. Bu script konsolu `force_terminal=True` ile kurar ve iki biçimde yazar:

  * `.ansi.txt` — ANSI kaçış dizileriyle; `cat` ile terminalde görüntülenir.
  * `.svg`      — tarayıcıda açılabilen, renkleri gömülü vektör görüntü.

Kullanım:
    python scripts/capture_terminal.py --samples 262144 -o artifacts/inspect_default
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from sigkit.display import render_inspect
from sigkit.io import load


def main() -> None:
    """Belirtilen pencereyi render edip ANSI metin ve SVG olarak kaydeder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="examples/sample.sigmf-meta")
    parser.add_argument("-o", "--output", required=True, help="uzantısız çıktı yolu")
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
    # clear=False şart: varsayılan davranış kayıt tamponunu boşaltır ve ardından
    # gelen save_svg boş bir SVG üretir.
    console.save_text(str(out.with_suffix(".ansi.txt")), styles=True, clear=False)
    console.save_svg(str(out.with_suffix(".svg")), title=f"sigkit inspect — {rec.meta_path.name}")
    print(f"yazıldı: {out.with_suffix('.ansi.txt')}")
    print(f"yazıldı: {out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
