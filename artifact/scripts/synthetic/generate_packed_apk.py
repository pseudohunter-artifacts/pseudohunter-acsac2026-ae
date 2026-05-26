"""Thin wrapper around :mod:`android_packer.cli.generate_packed_apk`."""

from __future__ import annotations

from android_packer.cli.generate_packed_apk import main


if __name__ == "__main__":
    raise SystemExit(main())
