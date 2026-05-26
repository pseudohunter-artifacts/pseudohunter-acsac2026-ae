"""Project path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

_DEFAULT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git")


def find_project_root(
    start: Optional[Path] = None,
    markers: Iterable[str] = _DEFAULT_MARKERS,
) -> Path:
    """Locate the repository root by walking up from ``start``.

    The root is the first ancestor directory that contains any of ``markers``
    (by default ``pyproject.toml`` or ``.git``). If none is found, the current
    working directory is returned as a best-effort fallback.

    ``start`` defaults to this module's location so that callers inside the
    installed package still find the original source checkout when running in
    editable mode. For sdist/wheel installs without a ``pyproject.toml`` next
    to the code (typical production deployment), the fallback still gives a
    predictable value rather than raising.
    """

    marker_tuple = tuple(markers)
    origin = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (origin, *origin.parents):
        if any((candidate / marker).exists() for marker in marker_tuple):
            return candidate
    return Path.cwd()
