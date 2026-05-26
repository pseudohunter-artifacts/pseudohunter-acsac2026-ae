"""Commercial packer rule engine -- Track B Path A for closed-source packers.

See ``docs/workstreams/track_b/labeling_injection_spec.md`` section 9 for the
full rationale and rule schema.

Motivation: open-source packers (S1..S5) expose their source so we can patch
``PackerLabelEmitter`` calls into them directly (Path A, primary). Commercial
packers (CS1..CS5: 360 Jiagu, ijiami, Bangcle, Tencent Legu, DexProtector)
are closed-source and offer no hook points. We instead encode each packer's
published landing conventions -- derived from PackerGrind (TSE 2022),
DroidUnpack (NDSS 2018), Wermke (CCS 2018), and other surveys -- into a
YAML rule file. This engine matches those rules against ZIP entries of a
packed APK and emits the same ``inject_labels.jsonl`` schema that
``injected_packer_adapter.py`` already consumes.

Limitations (must be acknowledged in the paper section 5.4):

* Rules are **literature-derived approximations**, not packer-source ground
  truth. Confidence is "medium": when a commercial packer upgrades its
  landing layout, the rule goes stale until refreshed.
* Track B main table must therefore report CS rows separately from S rows.
* Each rule file must carry ``references`` (at minimum 1 peer-reviewed
  paper) and ``packer_version``.
"""

from __future__ import annotations

import dataclasses
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from android_packer.labeling.injected_packer_adapter import (
    InjectLabelRecord,
    InjectedEntryRecord,
    compute_payload_sha256,
    write_inject_labels,
)
from android_packer.labeling.synthetic import HIDDEN_EXECUTABLE_PAYLOAD


# ---------------------------------------------------------------------------
# Rule file loading + validation
# ---------------------------------------------------------------------------


class CommercialRuleSchemaError(ValueError):
    """Raised when a commercial rule file fails validation."""


@dataclasses.dataclass(frozen=True)
class EmitSpec:
    label: str
    payload_kind: str
    transform_family: str
    offset_start: Any  # int or "__file_size__"
    offset_end: Any


@dataclasses.dataclass(frozen=True)
class MatchSpec:
    object_path_regex: re.Pattern


@dataclasses.dataclass(frozen=True)
class Rule:
    rule_id: str
    match: MatchSpec
    emit: EmitSpec


@dataclasses.dataclass(frozen=True)
class CommercialPackerSpec:
    packer_id: str
    packer_version: str
    gen_level: str
    references: Sequence[Mapping[str, Any]]
    rules: Sequence[Rule]
    limitations: Sequence[str]


_ALLOWED_EMIT_LABELS = frozenset({HIDDEN_EXECUTABLE_PAYLOAD, "benign_loader"})
_ALLOWED_PAYLOAD_KINDS = frozenset(
    {
        "encrypted_dex",
        "extracted_method_body",
        "metadata_table",
        "compressed_payload",
        "shim",
        "native_stub",
    }
)
_FILE_SIZE_SENTINEL = "__file_size__"


def load_rule_file(path: Path) -> CommercialPackerSpec:
    """Parse and validate a commercial packer rule YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise CommercialRuleSchemaError(f"{path}: top-level must be a mapping")

    missing = [key for key in ("packer_id", "packer_version", "gen_level", "references", "rules") if key not in raw]
    if missing:
        raise CommercialRuleSchemaError(f"{path}: missing required keys: {missing}")

    references = raw.get("references") or []
    if not isinstance(references, list) or len(references) < 1:
        raise CommercialRuleSchemaError(
            f"{path}: 'references' must be a non-empty list (at least 1 citation)"
        )

    rules_raw = raw.get("rules") or []
    if not isinstance(rules_raw, list) or not rules_raw:
        raise CommercialRuleSchemaError(f"{path}: 'rules' must be a non-empty list")

    rules: List[Rule] = []
    seen_ids: set[str] = set()
    for i, rule_raw in enumerate(rules_raw):
        rules.append(_parse_rule(rule_raw, path=path, index=i, seen_ids=seen_ids))

    return CommercialPackerSpec(
        packer_id=str(raw["packer_id"]),
        packer_version=str(raw["packer_version"]),
        gen_level=str(raw["gen_level"]),
        references=tuple(references),
        rules=tuple(rules),
        limitations=tuple(raw.get("limitations") or []),
    )


def _parse_rule(
    rule_raw: Mapping[str, Any], *, path: Path, index: int, seen_ids: set[str]
) -> Rule:
    for key in ("rule_id", "match", "emit"):
        if key not in rule_raw:
            raise CommercialRuleSchemaError(
                f"{path}: rules[{index}] missing required key {key!r}"
            )
    rule_id = str(rule_raw["rule_id"])
    if rule_id in seen_ids:
        raise CommercialRuleSchemaError(f"{path}: duplicate rule_id {rule_id!r}")
    seen_ids.add(rule_id)

    match_raw = rule_raw["match"]
    if not isinstance(match_raw, dict) or "object_path_regex" not in match_raw:
        raise CommercialRuleSchemaError(
            f"{path}: rules[{index}].match must include object_path_regex"
        )
    try:
        regex = re.compile(match_raw["object_path_regex"])
    except re.error as exc:
        raise CommercialRuleSchemaError(
            f"{path}: rules[{index}] invalid regex: {exc}"
        ) from exc

    emit_raw = rule_raw["emit"]
    if not isinstance(emit_raw, dict):
        raise CommercialRuleSchemaError(f"{path}: rules[{index}].emit must be a mapping")
    for key in ("label", "payload_kind", "transform_family", "offset_start", "offset_end"):
        if key not in emit_raw:
            raise CommercialRuleSchemaError(
                f"{path}: rules[{index}].emit missing {key!r}"
            )

    label = str(emit_raw["label"])
    if label not in _ALLOWED_EMIT_LABELS:
        raise CommercialRuleSchemaError(
            f"{path}: rules[{index}].emit.label must be one of {sorted(_ALLOWED_EMIT_LABELS)}"
        )

    kind = str(emit_raw["payload_kind"])
    if kind not in _ALLOWED_PAYLOAD_KINDS:
        raise CommercialRuleSchemaError(
            f"{path}: rules[{index}].emit.payload_kind must be one of {sorted(_ALLOWED_PAYLOAD_KINDS)}"
        )

    transform_family = str(emit_raw["transform_family"])
    if not transform_family.startswith("packer_"):
        raise CommercialRuleSchemaError(
            f"{path}: rules[{index}].emit.transform_family must start with 'packer_'"
        )

    emit = EmitSpec(
        label=label,
        payload_kind=kind,
        transform_family=transform_family,
        offset_start=_coerce_offset(emit_raw["offset_start"], path=path, index=index, field="offset_start"),
        offset_end=_coerce_offset(emit_raw["offset_end"], path=path, index=index, field="offset_end"),
    )
    return Rule(rule_id=rule_id, match=MatchSpec(object_path_regex=regex), emit=emit)


def _coerce_offset(value: Any, *, path: Path, index: int, field: str) -> Any:
    if value == _FILE_SIZE_SENTINEL:
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CommercialRuleSchemaError(
            f"{path}: rules[{index}].emit.{field} must be int or {_FILE_SIZE_SENTINEL!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ApplyResult:
    """What ``apply_rules_to_apk`` returned, pre-write."""

    record: InjectLabelRecord
    unmatched_entries: List[str]
    matched_entries: List[str]


def apply_rules_to_apk(
    packed_apk_path: Path,
    spec: CommercialPackerSpec,
    *,
    apk_id: str,
    source_apk_id: str,
    compute_sha256: bool = True,
) -> ApplyResult:
    """Run rules over every ZIP entry in ``packed_apk_path``.

    For each rule whose regex matches an entry path, emit one
    :class:`InjectedEntryRecord`. Rules are evaluated in order; a single
    entry can match multiple rules (rare, but supported -- useful when one
    rule tags the loader and another tags a payload inside the same object).

    Parameters
    ----------
    compute_sha256:
        When True (default) we read each matched entry and hash its bytes
        for ``payload_sha256``. Disable on huge APKs where you only need
        the offset ranges (e.g. during rule-file development).
    """
    entries: List[InjectedEntryRecord] = []
    matched: List[str] = []
    unmatched: List[str] = []

    with zipfile.ZipFile(packed_apk_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            any_hit = False
            for rule in spec.rules:
                if rule.match.object_path_regex.search(name):
                    any_hit = True
                    matched.append(f"{rule.rule_id}:{name}")
                    entry = _rule_to_entry_record(
                        rule, info, zf=zf, compute_sha256=compute_sha256,
                    )
                    entries.append(entry)
            if not any_hit:
                unmatched.append(name)

    record = InjectLabelRecord(
        apk_id=apk_id,
        source_apk_id=source_apk_id,
        packer_name=spec.packer_id,
        packer_commit=None,  # no source -> no commit
        label_source="rule_based",
        timestamp_utc=None,
        entries=tuple(entries),
    )
    return ApplyResult(record=record, unmatched_entries=unmatched, matched_entries=matched)


def _rule_to_entry_record(
    rule: Rule,
    info: zipfile.ZipInfo,
    *,
    zf: zipfile.ZipFile,
    compute_sha256: bool,
) -> InjectedEntryRecord:
    start = 0 if rule.emit.offset_start == _FILE_SIZE_SENTINEL else int(rule.emit.offset_start)
    if rule.emit.offset_end == _FILE_SIZE_SENTINEL:
        end = info.file_size
    else:
        end = int(rule.emit.offset_end)
        if end < 0:
            end = info.file_size + end  # allow negative offsets relative to file end

    payload_sha256: Optional[str] = None
    if compute_sha256 and rule.emit.label == HIDDEN_EXECUTABLE_PAYLOAD:
        with zf.open(info) as fh:
            payload_sha256 = compute_payload_sha256(fh.read())

    return InjectedEntryRecord(
        object_path=info.filename,
        offset_start=start,
        offset_end=end,
        label=rule.emit.label,
        payload_kind=rule.emit.payload_kind,
        transform_family=rule.emit.transform_family,
        payload_sha256=payload_sha256,
        injection_point=f"rule:{rule.rule_id}",
    )


def run_commercial_rule_engine(
    packed_apk_path: Path,
    rule_file: Path,
    output_jsonl: Path,
    *,
    apk_id: str,
    source_apk_id: str,
    compute_sha256: bool = True,
) -> Dict[str, Any]:
    """Top-level entrypoint suitable for CLI: load rules, apply, write JSONL.

    Returns a JSON-safe summary so callers can log coverage statistics.
    """
    spec = load_rule_file(rule_file)
    result = apply_rules_to_apk(
        Path(packed_apk_path),
        spec,
        apk_id=apk_id,
        source_apk_id=source_apk_id,
        compute_sha256=compute_sha256,
    )
    write_inject_labels(Path(output_jsonl), [result.record])
    return {
        "packer_id": spec.packer_id,
        "packer_version": spec.packer_version,
        "rule_file": str(rule_file),
        "packed_apk": str(packed_apk_path),
        "output_jsonl": str(output_jsonl),
        "entries_emitted": len(result.record.entries),
        "matched_entries": len(result.matched_entries),
        "unmatched_entries": len(result.unmatched_entries),
    }


__all__ = [
    "ApplyResult",
    "CommercialPackerSpec",
    "CommercialRuleSchemaError",
    "EmitSpec",
    "MatchSpec",
    "Rule",
    "apply_rules_to_apk",
    "load_rule_file",
    "run_commercial_rule_engine",
]
