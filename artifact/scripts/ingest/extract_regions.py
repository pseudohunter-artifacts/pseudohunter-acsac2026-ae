"""Thin wrapper around :mod:`android_packer.cli.extract_regions`.

Prefer the ``android-packer-extract-regions`` console script installed by
``pip install -e .``. Running this file directly still works when the package
is on ``PYTHONPATH`` (for example ``$env:PYTHONPATH='src'``).
"""

from __future__ import annotations

from android_packer.cli.extract_regions import main


if __name__ == "__main__":
    raise SystemExit(main())
