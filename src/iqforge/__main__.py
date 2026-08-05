"""Entry point for `python -m iqforge`.

Equivalent to the `iqforge` console script, and works when that script is not
on PATH — after a plain `pip install --user`, or inside a virtualenv that has
not been activated.
"""

from __future__ import annotations

from iqforge.cli import app

if __name__ == "__main__":
    app()
