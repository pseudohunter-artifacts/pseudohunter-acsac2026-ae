"""Adapter that turns packer source-injection logs into ``SyntheticLabel``.

Track B Path A: we patch each open-source packer so that it writes an
``inject_labels.jsonl`` file recording byte-exact payload offsets when it
produces ``packed.apk``. Commercial packers (no source access) instead run
through ``commercial_rule_engine.py`` which emits the same JSONL schema.

This adapter:

* reads such a JSONL file
* validates each record against the schema (raises on unknown required keys)
* converts records into :class:`SyntheticLabel` instances that flow into the
  existing ``build_training_labels()`` pipeline

The goal is **schema parity with Track A**: Track B labels go through the
exact same region / object / apk aggregation code path. No downstream
baseline or evaluation change is required.

See ``docs/workstreams/track_b/labeling_injection_spec.md`` sections 3-4
for per-packer injection points and the full JSONL schema.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence

from android_packer.labeling.synthetic import (
    HIDDEN_EXECUTABLE_PAYLOAD,
    SyntheticLabel,
)


# ---------------------------------------------------------------------------
# Payload kind vocabulary (kept small & explicit; extend via code review only).
# ---------------------------------------------------------------------------

# Produced by open-source packers (S1..S5) and rule engines alike.
_PAYLOAD_KINDS = frozenset(
    {
        "encrypted_dex",  # whole classes.dex -> ZIP entry under assets/
        "extracted_method_body",  # Gen3: method bodies carved out of DEX
        "metadata_table",  # offset / key tables written alongside payload
        "compressed_payload",  # Deflate/LZMA wrapped payload
    }
)

# The non-payload classes we also want to log (for negative regions).
#
# ``benign_other`` (L42 fix, 2026-05-07): catch-all for APK objects that
# are **not** packer-introduced -- e.g. native ``res/drawable/icon.png``,
# native ``kotlin/internal/module.kotlin_module``, etc.  Added so the
# Typed-Instance MIL encoder has a dedicated "not a loader / not a
# payload" type id for the ~1500 benign instances per bag, rather than
# mis-labelling them as ``shim`` (the legacy default fallback before the
# fix).  See ``docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md``
# §L42 for context.
_LOADER_KINDS = frozenset(
    {
        "shim",  # benign loader replacing classes.dex
        "native_stub",  # packer's libXXX.so that calls into JNI decrypt
        "benign_other",  # L42: non-packer-introduced APK object
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InjectLabelSchemaError(ValueError):
    """Raised when an ``inject_labels.jsonl`` record is malformed."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectedEntryRecord:
    """One entry inside an inject_labels.jsonl record.

    This mirrors what the injected packer (or commercial rule engine) writes
    per ZIP entry in the packed APK. Field semantics align with
    :class:`SyntheticLabel`:

    - ``offset_start`` / ``offset_end`` are object-local in ``packed.apk``
    - ``source_offset_start`` / ``source_offset_end`` are object-local in
      ``benign.apk`` (only meaningful for sub-range transforms)
    """

    object_path: str
    offset_start: int
    offset_end: int
    label: str
    payload_kind: str
    transform_family: str
    source_object_path: Optional[str] = None
    source_offset_start: Optional[int] = None
    source_offset_end: Optional[int] = None
    payload_sha256: Optional[str] = None
    injection_point: Optional[str] = None  # free-form diagnostic

    def to_synthetic_label(self, *, apk_id: str, source_apk_id: Optional[str]) -> Optional[SyntheticLabel]:
        """Convert to :class:`SyntheticLabel`; returns None for loader regions.

        The **label** (not ``payload_kind``) decides positivity: anything
        tagged ``hidden_executable_payload`` is emitted as a positive
        SyntheticLabel regardless of whether its container is an
        encrypted DEX blob, a native stub, or a compressed archive.
        ``payload_kind`` remains in the JSONL for provenance / diagnostics
        but does not alter the training signal.

        Anything else (``benign_loader`` etc.) is **recorded for provenance**
        but NOT emitted, because Track A convention is: SyntheticLabel only
        represents positive ``payload`` intervals. Everything unreferenced
        becomes ``benign`` automatically via ``build_training_labels``.
        """
        if self.label != HIDDEN_EXECUTABLE_PAYLOAD:
            return None
        if self.payload_sha256 is None:
            raise InjectLabelSchemaError(
                f"payload record missing payload_sha256: entry={self.object_path}"
            )
        return SyntheticLabel(
            apk_id=apk_id,
            object_path=self.object_path,
            offset_start=self.offset_start,
            offset_end=self.offset_end,
            label=self.label,
            transform_family=self.transform_family,
            payload_sha256=self.payload_sha256,
            source_apk_id=source_apk_id,
            source_object_path=self.source_object_path,
            source_offset_start=self.source_offset_start,
            source_offset_end=self.source_offset_end,
        )


@dataclass(frozen=True)
class InjectLabelRecord:
    """Top-level record in an ``inject_labels.jsonl`` line.

    One line per (benign, packed) pair. The packer / rule engine emits exactly
    one line when packing completes; ``flush()`` may overwrite the file on
    retries.
    """

    apk_id: str
    source_apk_id: str
    packer_name: str
    packer_commit: Optional[str]
    label_source: str  # "source_injected" | "rule_based"
    timestamp_utc: Optional[str]
    entries: Sequence[InjectedEntryRecord] = field(default_factory=tuple)

    def to_synthetic_labels(self) -> List[SyntheticLabel]:
        out: List[SyntheticLabel] = []
        for entry in self.entries:
            label = entry.to_synthetic_label(
                apk_id=self.apk_id, source_apk_id=self.source_apk_id
            )
            if label is not None:
                out.append(label)
        return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


_REQUIRED_TOP_FIELDS = ("apk_id", "source_apk_id", "packer_name", "entries")
_REQUIRED_ENTRY_FIELDS = (
    "object_path",
    "offset_start",
    "offset_end",
    "label",
    "payload_kind",
    "transform_family",
)
_ALLOWED_LABEL_SOURCES = frozenset({"source_injected", "rule_based"})


def _validate_entry(payload: Mapping[str, Any], *, record_index: int, entry_index: int) -> InjectedEntryRecord:
    missing = [key for key in _REQUIRED_ENTRY_FIELDS if key not in payload]
    if missing:
        raise InjectLabelSchemaError(
            f"record[{record_index}].entries[{entry_index}] missing required fields: {missing}"
        )

    kind = str(payload["payload_kind"])
    if kind not in _PAYLOAD_KINDS and kind not in _LOADER_KINDS:
        raise InjectLabelSchemaError(
            f"record[{record_index}].entries[{entry_index}] has unknown payload_kind={kind!r}; "
            f"expected one of payload={sorted(_PAYLOAD_KINDS)} or loader={sorted(_LOADER_KINDS)}"
        )

    start = int(payload["offset_start"])
    end = int(payload["offset_end"])
    if end < start:
        raise InjectLabelSchemaError(
            f"record[{record_index}].entries[{entry_index}]: offset_end({end}) < offset_start({start})"
        )

    transform_family = str(payload["transform_family"])
    if not transform_family.startswith("packer_"):
        raise InjectLabelSchemaError(
            f"record[{record_index}].entries[{entry_index}]: transform_family must start with 'packer_' "
            f"(got {transform_family!r})"
        )

    return InjectedEntryRecord(
        object_path=str(payload["object_path"]),
        offset_start=start,
        offset_end=end,
        label=str(payload["label"]),
        payload_kind=kind,
        transform_family=transform_family,
        source_object_path=(
            str(payload["source_object_path"]) if payload.get("source_object_path") is not None else None
        ),
        source_offset_start=(
            int(payload["source_offset_start"]) if payload.get("source_offset_start") is not None else None
        ),
        source_offset_end=(
            int(payload["source_offset_end"]) if payload.get("source_offset_end") is not None else None
        ),
        payload_sha256=(
            str(payload["payload_sha256"]) if payload.get("payload_sha256") is not None else None
        ),
        injection_point=(
            str(payload["injection_point"]) if payload.get("injection_point") is not None else None
        ),
    )


def _validate_record(payload: Mapping[str, Any], *, record_index: int) -> InjectLabelRecord:
    missing = [key for key in _REQUIRED_TOP_FIELDS if key not in payload]
    if missing:
        raise InjectLabelSchemaError(
            f"record[{record_index}] missing required fields: {missing}"
        )

    label_source = str(payload.get("label_source", "source_injected"))
    if label_source not in _ALLOWED_LABEL_SOURCES:
        raise InjectLabelSchemaError(
            f"record[{record_index}].label_source must be one of {sorted(_ALLOWED_LABEL_SOURCES)}; "
            f"got {label_source!r}"
        )

    entries_raw = payload["entries"]
    if not isinstance(entries_raw, list):
        raise InjectLabelSchemaError(
            f"record[{record_index}].entries must be a list; got {type(entries_raw).__name__}"
        )

    entries = tuple(
        _validate_entry(entry, record_index=record_index, entry_index=i)
        for i, entry in enumerate(entries_raw)
    )

    return InjectLabelRecord(
        apk_id=str(payload["apk_id"]),
        source_apk_id=str(payload["source_apk_id"]),
        packer_name=str(payload["packer_name"]),
        packer_commit=(
            str(payload["packer_commit"]) if payload.get("packer_commit") is not None else None
        ),
        label_source=label_source,
        timestamp_utc=(
            str(payload["timestamp_utc"]) if payload.get("timestamp_utc") is not None else None
        ),
        entries=entries,
    )


def _read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise InjectLabelSchemaError(
                    f"{path}:{line_no}: invalid JSON ({exc.msg})"
                ) from exc


def parse_inject_labels(jsonl_path: Path) -> List[InjectLabelRecord]:
    """Read ``inject_labels.jsonl`` and return validated records.

    Raises :class:`InjectLabelSchemaError` when any record fails validation.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"inject_labels.jsonl not found: {path}")
    records = []
    for record_index, payload in enumerate(_read_jsonl(path)):
        records.append(_validate_record(payload, record_index=record_index))
    return records


def to_synthetic_labels(records: Iterable[InjectLabelRecord]) -> List[SyntheticLabel]:
    """Flatten records to a list of SyntheticLabel (loader entries dropped)."""
    out: List[SyntheticLabel] = []
    for record in records:
        out.extend(record.to_synthetic_labels())
    return out


# ---------------------------------------------------------------------------
# Convenience helper used by the orchestrator (B-c-1 / B-c-2)
# ---------------------------------------------------------------------------


def load_synthetic_labels(jsonl_path: Path) -> List[SyntheticLabel]:
    """One-shot helper: parse + flatten so callers don't need both functions.

    Typical usage::

        labels = load_synthetic_labels(task_dir / "inject_labels.jsonl")
        training = build_training_labels(regions, labels, ...)
    """
    return to_synthetic_labels(parse_inject_labels(jsonl_path))


# ---------------------------------------------------------------------------
# Cross-validation with Path B (diff-based alignment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossValidationResult:
    """Outcome of comparing Path A (source-injected) vs Path B (diff-based).

    See ``labeling_injection_spec.md`` section 6.2 for the verdict table.
    """

    iou: float
    verdict: str  # "solid" | "partial_mismatch" | "low_confidence"
    per_object_iou: Mapping[str, float]

    @property
    def is_solid(self) -> bool:
        return self.verdict == "solid"

    @property
    def needs_manual_review(self) -> bool:
        return self.verdict == "low_confidence"


def _coverage_bitmask(
    labels: Iterable[SyntheticLabel], object_path: str
) -> List[tuple[int, int]]:
    ranges = [
        (label.offset_start, label.offset_end)
        for label in labels
        if label.object_path == object_path and label.offset_end > label.offset_start
    ]
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _range_intersection_length(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]) -> int:
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _range_union_length(ranges: Sequence[tuple[int, int]]) -> int:
    return sum(end - start for start, end in ranges)


def cross_validate(
    path_a_labels: Iterable[SyntheticLabel],
    path_b_labels: Iterable[SyntheticLabel],
    *,
    solid_threshold: float = 0.9,
    review_threshold: float = 0.5,
) -> CrossValidationResult:
    """Compute IoU between Path A and Path B labels, per object_path.

    Overall IoU is the ``sum(intersections) / sum(unions)`` across objects,
    which is the standard "mean IoU over token set" formulation -- it
    penalises large objects with large disagreement more than small ones.
    """
    if solid_threshold <= review_threshold:
        raise ValueError("solid_threshold must be > review_threshold")

    path_a = list(path_a_labels)
    path_b = list(path_b_labels)
    objects = sorted({lbl.object_path for lbl in path_a} | {lbl.object_path for lbl in path_b})

    per_object: dict[str, float] = {}
    total_inter = 0
    total_union = 0
    for object_path in objects:
        a_ranges = _coverage_bitmask(path_a, object_path)
        b_ranges = _coverage_bitmask(path_b, object_path)
        inter = _range_intersection_length(a_ranges, b_ranges)
        union = _range_union_length(a_ranges) + _range_union_length(b_ranges) - inter
        per_object[object_path] = (inter / union) if union > 0 else 1.0
        total_inter += inter
        total_union += union

    overall = (total_inter / total_union) if total_union > 0 else 1.0

    if overall >= solid_threshold:
        verdict = "solid"
    elif overall >= review_threshold:
        verdict = "partial_mismatch"
    else:
        verdict = "low_confidence"

    return CrossValidationResult(
        iou=round(overall, 6),
        verdict=verdict,
        per_object_iou={path: round(v, 6) for path, v in per_object.items()},
    )


# ---------------------------------------------------------------------------
# Writer (used by patched open-source packers + commercial rule engine)
# ---------------------------------------------------------------------------


def compute_payload_sha256(data: bytes) -> str:
    """SHA-256 of payload bytes; callers pass the *encrypted* bytes we record."""
    return hashlib.sha256(data).hexdigest()


def write_inject_labels(
    jsonl_path: Path,
    records: Sequence[InjectLabelRecord],
) -> None:
    """Atomic write: dump records to a temp file then rename.

    Packer patches usually call this once at the end of packing via a
    ``PackerLabelEmitter.flush()`` bridge. Rule engine does the same.
    """
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            row: dict[str, Any] = {
                "apk_id": record.apk_id,
                "source_apk_id": record.source_apk_id,
                "packer_name": record.packer_name,
                "label_source": record.label_source,
                "entries": [
                    {
                        key: value
                        for key, value in {
                            "object_path": entry.object_path,
                            "offset_start": entry.offset_start,
                            "offset_end": entry.offset_end,
                            "label": entry.label,
                            "payload_kind": entry.payload_kind,
                            "transform_family": entry.transform_family,
                            "source_object_path": entry.source_object_path,
                            "source_offset_start": entry.source_offset_start,
                            "source_offset_end": entry.source_offset_end,
                            "payload_sha256": entry.payload_sha256,
                            "injection_point": entry.injection_point,
                        }.items()
                        if value is not None
                    }
                    for entry in record.entries
                ],
            }
            if record.packer_commit is not None:
                row["packer_commit"] = record.packer_commit
            if record.timestamp_utc is not None:
                row["timestamp_utc"] = record.timestamp_utc
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    tmp.replace(jsonl_path)


__all__ = [
    "CrossValidationResult",
    "InjectLabelRecord",
    "InjectLabelSchemaError",
    "InjectedEntryRecord",
    "compute_payload_sha256",
    "cross_validate",
    "load_synthetic_labels",
    "parse_inject_labels",
    "to_synthetic_labels",
    "write_inject_labels",
]
