"""Entry point for ``python -m mechabrain``.

Equivalent to the ``mechabrain`` console script; useful when the package is
importable but its script is not on PATH.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
