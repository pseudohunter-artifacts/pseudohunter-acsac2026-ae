"""Thin wrapper around :mod:`android_packer.cli.build_training_labels`."""

from __future__ import annotations

from android_packer.cli.build_training_labels import main


if __name__ == "__main__":
    raise SystemExit(main())
