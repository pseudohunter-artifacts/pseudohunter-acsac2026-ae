"""Optional JSON Schema validation helpers.

Schemas live in ``configs/data/schemas/*.schema.json`` and define the data
contracts for the JSONL / JSON artefacts produced by the pipeline. Validation
is *optional* — the core baselines do not run it on every record to keep the
pipeline fast — but unit tests and ad-hoc triage scripts use it to catch
schema drift early.

The ``jsonschema`` package is installed via the ``[dev]`` extra. If it is not
available the validators raise :class:`ImportError` with an actionable hint
instead of silently passing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from android_packer.utils.jsonl import read_jsonl
from android_packer.utils.paths import find_project_root


class SchemaValidationError(ValueError):
    """Raised when a record or JSONL file fails schema validation."""


def schema_path(schema_name: str) -> Path:
    """Return the absolute path to ``configs/data/schemas/<schema_name>``.

    ``schema_name`` may be bare (``"region_metadata"``) or with the
    ``.schema.json`` suffix; both forms are accepted.
    """

    if not schema_name.endswith(".schema.json"):
        schema_name = f"{schema_name}.schema.json"
    root = find_project_root()
    candidate = root / "configs" / "data" / "schemas" / schema_name
    if not candidate.exists():
        raise FileNotFoundError(f"schema not found: {candidate}")
    return candidate


@lru_cache(maxsize=None)
def load_schema(schema_name: str) -> dict:
    """Read and JSON-parse a schema, cached by name."""

    return json.loads(schema_path(schema_name).read_text(encoding="utf-8"))


def _get_validator(schema_name: str):
    """Return a ``jsonschema`` Draft 2020-12 validator for ``schema_name``.

    Defer the ``jsonschema`` import so production installs without the
    ``[dev]`` extra don't pay the import cost.
    """

    try:
        from jsonschema import Draft202012Validator  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on dev extras.
        raise ImportError(
            "jsonschema is required for validation; install the project with "
            "'pip install -e \".[dev]\"' or add jsonschema>=4.18 explicitly."
        ) from exc
    return Draft202012Validator(load_schema(schema_name))


def validate_record(record: Mapping, schema_name: str) -> None:
    """Raise :class:`SchemaValidationError` if ``record`` does not match the schema."""

    validator = _get_validator(schema_name)
    errors = sorted(validator.iter_errors(record), key=lambda err: err.path)
    if errors:
        raise SchemaValidationError(_format_errors(schema_name, None, errors))


def validate_records(records: Iterable[Mapping], schema_name: str) -> int:
    """Validate an iterable of records; return the count processed.

    Raises the first :class:`SchemaValidationError` found; callers that want
    to collect *all* errors should iterate :func:`iter_validation_errors`
    manually.
    """

    count = 0
    for record in records:
        validate_record(record, schema_name)
        count += 1
    return count


def validate_jsonl(path: Path, schema_name: str) -> int:
    """Validate every JSON object in a JSONL file; return the record count."""

    return validate_records(read_jsonl(path), schema_name)


def iter_validation_errors(
    records: Iterable[Mapping], schema_name: str
) -> Iterator[tuple[int, str]]:
    """Yield ``(index, formatted_error)`` for every failing record.

    ``index`` is 0-based over the input iterable. Useful when a JSONL has
    many rows and you want to inspect all schema violations at once.
    """

    validator = _get_validator(schema_name)
    for index, record in enumerate(records):
        errors = sorted(validator.iter_errors(record), key=lambda err: err.path)
        if errors:
            yield index, _format_errors(schema_name, index, errors)


def _format_errors(schema_name: str, index: int | None, errors) -> str:
    header = (
        f"schema '{schema_name}' validation failed"
        + (f" at record #{index}" if index is not None else "")
        + ":"
    )
    details = []
    for err in errors:
        path = "/".join(str(p) for p in err.absolute_path) or "<root>"
        details.append(f"  - {path}: {err.message}")
    return header + "\n" + "\n".join(details)


__all__ = [
    "SchemaValidationError",
    "iter_validation_errors",
    "load_schema",
    "schema_path",
    "validate_jsonl",
    "validate_record",
    "validate_records",
]
