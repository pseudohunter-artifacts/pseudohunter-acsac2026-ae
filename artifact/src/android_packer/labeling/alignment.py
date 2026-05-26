"""Align synthetic payload labels to extracted byte-window regions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping, Sequence

from android_packer.labeling.synthetic import HIDDEN_EXECUTABLE_PAYLOAD


BENIGN = "benign"


@dataclass(frozen=True)
class RegionTrainingLabel:
    apk_id: str
    object_id: str
    region_id: str
    object_path: str
    object_type: str
    offset_start: int
    offset_end: int
    size: int
    sha256: str
    entropy: float
    printable_ratio: float
    label: str
    label_id: int
    overlap_bytes: int
    overlap_ratio: float
    max_iou: float
    matched_label_count: int
    transform_families: List[str]
    payload_sha256s: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObjectTrainingLabel:
    apk_id: str
    object_id: str
    object_path: str
    object_type: str
    label: str
    label_id: int
    region_count: int
    positive_region_count: int
    max_region_iou: float
    transform_families: List[str]
    payload_sha256s: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ApkTrainingLabel:
    apk_id: str
    label: str
    label_id: int
    object_count: int
    positive_object_count: int
    region_count: int
    positive_region_count: int
    transform_families: List[str]
    payload_sha256s: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingLabels:
    region_labels: List[RegionTrainingLabel]
    object_labels: List[ObjectTrainingLabel]
    apk_labels: List[ApkTrainingLabel]


@dataclass(frozen=True)
class _PayloadInterval:
    apk_id: str
    object_path: str
    offset_start: int
    offset_end: int
    transform_family: str
    payload_sha256: str


def build_training_labels(
    regions: Iterable[Mapping],
    synthetic_labels: Iterable[Mapping],
    *,
    min_overlap_bytes: int = 1,
    min_overlap_ratio: float = 0.0,
) -> TrainingLabels:
    if min_overlap_bytes < 1:
        raise ValueError("min_overlap_bytes must be at least 1")
    if not 0.0 <= min_overlap_ratio <= 1.0:
        raise ValueError("min_overlap_ratio must be between 0 and 1")

    payloads = _payload_intervals(synthetic_labels)
    payloads_by_object: dict[tuple[str, str], list[_PayloadInterval]] = {}
    payloads_by_apk: dict[str, list[_PayloadInterval]] = {}
    for payload in payloads:
        payloads_by_object.setdefault((payload.apk_id, payload.object_path), []).append(payload)
        payloads_by_apk.setdefault(payload.apk_id, []).append(payload)

    region_rows = list(regions)
    region_labels = [
        _label_region(
            region,
            payloads_by_object.get((str(region["apk_id"]), str(region["object_path"])), []),
            min_overlap_bytes=min_overlap_bytes,
            min_overlap_ratio=min_overlap_ratio,
        )
        for region in region_rows
    ]
    object_labels = _aggregate_objects(region_labels, payloads_by_object)
    apk_labels = _aggregate_apks(region_labels, object_labels, payloads_by_apk)
    return TrainingLabels(
        region_labels=region_labels,
        object_labels=object_labels,
        apk_labels=apk_labels,
    )


def interval_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def interval_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    overlap = interval_overlap(start_a, end_a, start_b, end_b)
    union = max(end_a, end_b) - min(start_a, start_b)
    if overlap <= 0 or union <= 0:
        return 0.0
    return overlap / union


def _payload_intervals(synthetic_labels: Iterable[Mapping]) -> list[_PayloadInterval]:
    intervals = []
    for row in synthetic_labels:
        if row.get("label") != HIDDEN_EXECUTABLE_PAYLOAD:
            continue
        start = int(row["offset_start"])
        end = int(row["offset_end"])
        if end <= start:
            continue
        intervals.append(
            _PayloadInterval(
                apk_id=str(row["apk_id"]),
                object_path=str(row["object_path"]),
                offset_start=start,
                offset_end=end,
                transform_family=str(row["transform_family"]),
                payload_sha256=str(row["payload_sha256"]),
            )
        )
    return intervals


def _label_region(
    region: Mapping,
    payloads: Sequence[_PayloadInterval],
    *,
    min_overlap_bytes: int,
    min_overlap_ratio: float,
) -> RegionTrainingLabel:
    start = int(region["offset_start"])
    end = int(region["offset_end"])
    region_size = max(0, end - start)
    matches: list[_PayloadInterval] = []
    max_iou = 0.0
    clipped_ranges: list[tuple[int, int]] = []

    for payload in payloads:
        overlap = interval_overlap(start, end, payload.offset_start, payload.offset_end)
        if overlap <= 0:
            continue
        ratio = overlap / region_size if region_size else 0.0
        max_iou = max(max_iou, interval_iou(start, end, payload.offset_start, payload.offset_end))
        clipped_ranges.append(
            (max(start, payload.offset_start), min(end, payload.offset_end))
        )
        if overlap >= min_overlap_bytes and ratio >= min_overlap_ratio:
            matches.append(payload)

    # The region can be covered by several payload intervals that may overlap
    # (e.g. decoy + real payload, or partially nested ranges). Merge them into a
    # disjoint union before computing ``overlap_bytes`` so the figure stays
    # bounded by ``region_size`` without double counting.
    total_overlap = _union_length(clipped_ranges)
    label = HIDDEN_EXECUTABLE_PAYLOAD if matches else BENIGN
    return RegionTrainingLabel(
        apk_id=str(region["apk_id"]),
        object_id=str(region["object_id"]),
        region_id=str(region["region_id"]),
        object_path=str(region["object_path"]),
        object_type=str(region["object_type"]),
        offset_start=start,
        offset_end=end,
        size=int(region["size"]),
        sha256=str(region["sha256"]),
        entropy=float(region["entropy"]),
        printable_ratio=float(region["printable_ratio"]),
        label=label,
        label_id=1 if matches else 0,
        overlap_bytes=total_overlap,
        overlap_ratio=round(total_overlap / region_size, 6) if region_size else 0.0,
        max_iou=round(max_iou, 6),
        matched_label_count=len(matches),
        transform_families=_unique_sorted(payload.transform_family for payload in matches),
        payload_sha256s=_unique_sorted(payload.payload_sha256 for payload in matches),
    )


def _union_length(ranges: Sequence[tuple[int, int]]) -> int:
    """Return the total length covered by a set of half-open intervals."""

    if not ranges:
        return 0
    sorted_ranges = sorted(ranges)
    total = 0
    current_start, current_end = sorted_ranges[0]
    for start, end in sorted_ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += max(0, current_end - current_start)
            current_start, current_end = start, end
    total += max(0, current_end - current_start)
    return total


def _aggregate_objects(
    region_labels: Sequence[RegionTrainingLabel],
    payloads_by_object: Mapping[tuple[str, str], Sequence[_PayloadInterval]],
) -> list[ObjectTrainingLabel]:
    groups: dict[tuple[str, str], list[RegionTrainingLabel]] = {}
    for row in region_labels:
        groups.setdefault((row.apk_id, row.object_id), []).append(row)

    object_labels = []
    for (_apk_id, _object_id), rows in sorted(groups.items()):
        first = rows[0]
        payloads = list(payloads_by_object.get((first.apk_id, first.object_path), []))
        positive = bool(payloads)
        object_labels.append(
            ObjectTrainingLabel(
                apk_id=first.apk_id,
                object_id=first.object_id,
                object_path=first.object_path,
                object_type=first.object_type,
                label=HIDDEN_EXECUTABLE_PAYLOAD if positive else BENIGN,
                label_id=1 if positive else 0,
                region_count=len(rows),
                positive_region_count=sum(row.label_id for row in rows),
                max_region_iou=round(max((row.max_iou for row in rows), default=0.0), 6),
                transform_families=_unique_sorted(payload.transform_family for payload in payloads),
                payload_sha256s=_unique_sorted(payload.payload_sha256 for payload in payloads),
            )
        )
    return object_labels


def _aggregate_apks(
    region_labels: Sequence[RegionTrainingLabel],
    object_labels: Sequence[ObjectTrainingLabel],
    payloads_by_apk: Mapping[str, Sequence[_PayloadInterval]],
) -> list[ApkTrainingLabel]:
    region_counts: dict[str, int] = {}
    positive_region_counts: dict[str, int] = {}
    object_counts: dict[str, int] = {}
    positive_object_counts: dict[str, int] = {}

    for row in region_labels:
        region_counts[row.apk_id] = region_counts.get(row.apk_id, 0) + 1
        positive_region_counts[row.apk_id] = positive_region_counts.get(row.apk_id, 0) + row.label_id
    for row in object_labels:
        object_counts[row.apk_id] = object_counts.get(row.apk_id, 0) + 1
        positive_object_counts[row.apk_id] = (
            positive_object_counts.get(row.apk_id, 0) + row.label_id
        )

    apk_ids = sorted(set(region_counts) | set(payloads_by_apk))
    apk_labels = []
    for apk_id in apk_ids:
        payloads = list(payloads_by_apk.get(apk_id, []))
        positive = bool(payloads)
        apk_labels.append(
            ApkTrainingLabel(
                apk_id=apk_id,
                label=HIDDEN_EXECUTABLE_PAYLOAD if positive else BENIGN,
                label_id=1 if positive else 0,
                object_count=object_counts.get(apk_id, 0),
                positive_object_count=positive_object_counts.get(apk_id, 0),
                region_count=region_counts.get(apk_id, 0),
                positive_region_count=positive_region_counts.get(apk_id, 0),
                transform_families=_unique_sorted(payload.transform_family for payload in payloads),
                payload_sha256s=_unique_sorted(payload.payload_sha256 for payload in payloads),
            )
        )
    return apk_labels


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})
