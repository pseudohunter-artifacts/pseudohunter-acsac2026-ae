"""Shared helpers for the byte-level baseline CLIs.

Both ``train_ngram_baseline`` and ``run_ngram_baseline`` need to map
``apk_id`` -> APK file path. The mapping comes from either:

- an explicit JSONL with ``{apk_id, apk_path}`` rows, or
- a synthetic manifest (rows carry ``generated_apk_id`` and
  ``generated_apk_path``).

We auto-detect on a per-row basis so a single file can mix both shapes
(useful when concatenating manifests from multiple runs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

from android_packer.utils.jsonl import read_jsonl


def build_apk_index(path: Path) -> Dict[str, Path]:
    """Build a deterministic ``apk_id -> apk_path`` mapping.

    Later rows for the same ``apk_id`` override earlier ones, which
    lets users refresh a stale path by appending to the file.
    """

    index: Dict[str, Path] = {}
    for row in read_jsonl(path):
        apk_id, apk_path = _extract_apk_pair(row)
        if apk_id and apk_path:
            index[apk_id] = Path(apk_path)
    return index


def _extract_apk_pair(row: Mapping) -> tuple[str, str]:
    if "generated_apk_path" in row:
        apk_id = str(row.get("generated_apk_id") or row.get("apk_id", ""))
        apk_path = str(row["generated_apk_path"])
        return apk_id, apk_path
    apk_id = str(row.get("apk_id", ""))
    apk_path = str(row.get("apk_path", ""))
    return apk_id, apk_path


__all__ = ["build_apk_index"]
