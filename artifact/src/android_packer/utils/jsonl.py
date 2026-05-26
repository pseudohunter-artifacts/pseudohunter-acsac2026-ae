"""JSON Lines helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield one decoded object per non-empty line.

    Malformed lines raise :class:`json.JSONDecodeError` enriched with the file
    path and 1-based line number so callers can quickly locate corrupt rows in
    multi-million-line JSONL datasets.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"{exc.msg} (file={path}, line={lineno})",
                    exc.doc,
                    exc.pos,
                ) from exc
