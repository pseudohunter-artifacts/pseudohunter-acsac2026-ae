"""Adapter: convert a Track B Path B :class:`DiffReport` into SyntheticLabel.

This is the **B-b-3** deliverable. It closes the loop between Path B
(byte-diff alignment) and the unified label pipeline:

    Path B:   align(benign, packed) -> DiffReport                    [B-b-1]
    Adapter:  diff_report_to_synthetic_labels(report, ...)           [B-b-3]
    Downstream: build_training_labels(regions, synthetic_labels, ...)  [existing]

Because the adapter runs **after** Path B's diff, it needs an additional
signal to decide whether a new-in-packed entry (no benign counterpart at all)
is a payload container or a benign loader. That decision is encapsulated in
a :class:`NewEntryPolicy` abstraction:

* :class:`AllNewEntriesArePayload` -- permissive default for open-source
  packers where the packer source + S5 patch already emitted Path A labels
  and Path B is the cross-check.
* :class:`RuleBasedNewEntryPolicy` -- consults a list of regex rules
  (matching the ``commercial_rule_engine`` schema) to classify each new
  entry as ``payload_container`` or ``benign_loader``. This is the primary
  codepath for CS1-CS5 commercial packers under the 2026-04-30 corrected
  stance: Path B diff produces byte-level labels; Path A-rule provides the
  semantic classification for new_in_packed entries.

See ``docs/workstreams/track_b/diff_alignment_spec.md`` and
``docs/workstreams/track_b/labeling_injection_spec.md`` section 9 for the
full design context.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from android_packer.labeling.diff_alignment import (
    BRANCH_BYTE_MODIFIED,
    BRANCH_NEW_IN_PACKED,
    BRANCH_REMOVED,
    BRANCH_RENAMED,
    BRANCH_UNCHANGED,
    DiffRange,
    DiffReport,
    EntryMapping,
)
from android_packer.labeling.synthetic import (
    HIDDEN_EXECUTABLE_PAYLOAD,
    SyntheticLabel,
)

# ---------------------------------------------------------------------------
# New-entry classification policies
# ---------------------------------------------------------------------------


# Decision outcomes returned by NewEntryPolicy.classify().
NEW_ENTRY_PAYLOAD = "payload_container"
NEW_ENTRY_LOADER = "benign_loader"
NEW_ENTRY_UNKNOWN = "unknown"


class NewEntryPolicy:
    """Classify a ``new_in_packed`` entry as payload / loader / unknown.

    Subclasses override :meth:`classify`; unknown lets the adapter skip the
    entry (labelled as neither payload nor loader, which degenerates to
    benign via ``build_training_labels``).
    """

    def classify(self, entry: EntryMapping) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class AllNewEntriesArePayload(NewEntryPolicy):
    """Treat every new-in-packed entry as a payload container.

    This is the permissive default for open-source packers whose Path A
    source-injection labels are already authoritative; Path B is only a
    cross-check so the false-positive risk is bounded by the IoU
    comparison against Path A. Not recommended as the sole labeling path.
    """

    def classify(self, entry: EntryMapping) -> str:
        return NEW_ENTRY_PAYLOAD


class NoNewEntriesArePayload(NewEntryPolicy):
    """Treat every new-in-packed entry as a benign loader.

    Useful as a pessimistic baseline to reason about the lower bound of
    Path B's recall: under this policy the adapter emits a
    ``SyntheticLabel`` only for byte-modified intervals.
    """

    def classify(self, entry: EntryMapping) -> str:
        return NEW_ENTRY_LOADER


@dataclass(frozen=True)
class NewEntryRule:
    """Minimal rule subset accepted by :class:`RuleBasedNewEntryPolicy`.

    Intentionally smaller than ``commercial_rule_engine.CommercialPackerSpec``
    to avoid circular imports; callers who already have a rule engine loaded
    should adapt its rules into this form with ``new_entry_rules_from_spec``.
    """

    object_path_regex: str
    classification: str  # NEW_ENTRY_PAYLOAD or NEW_ENTRY_LOADER

    def __post_init__(self) -> None:
        if self.classification not in {NEW_ENTRY_PAYLOAD, NEW_ENTRY_LOADER}:
            raise ValueError(
                f"NewEntryRule.classification must be one of "
                f"{{{NEW_ENTRY_PAYLOAD!r}, {NEW_ENTRY_LOADER!r}}}; "
                f"got {self.classification!r}"
            )


class RuleBasedNewEntryPolicy(NewEntryPolicy):
    """Apply a sequence of :class:`NewEntryRule` in order.

    The first rule whose ``object_path_regex`` matches
    ``entry.packed_entry`` determines the classification. If no rule
    matches, returns ``NEW_ENTRY_UNKNOWN`` (the adapter will then skip the
    entry, avoiding false positives from uncovered packer variants).
    """

    def __init__(self, rules: Sequence[NewEntryRule]) -> None:
        self._rules = list(rules)
        self._compiled = [re.compile(r.object_path_regex) for r in self._rules]

    def classify(self, entry: EntryMapping) -> str:
        if entry.packed_entry is None:
            return NEW_ENTRY_UNKNOWN
        for pattern, rule in zip(self._compiled, self._rules):
            if pattern.search(entry.packed_entry):
                return rule.classification
        return NEW_ENTRY_UNKNOWN


def new_entry_rules_from_spec(spec: Any) -> List[NewEntryRule]:
    """Convert ``commercial_rule_engine.CommercialPackerSpec`` -> rules list.

    ``CommercialPackerSpec.rules`` is a sequence of frozen
    :class:`commercial_rule_engine.Rule` dataclasses whose ``match`` is a
    :class:`MatchSpec` (with a pre-compiled ``object_path_regex``) and
    whose ``emit`` is an :class:`EmitSpec`. We pull the regex pattern
    string back out (so the policy can re-compile it under its own
    namespace) and map ``emit.payload_kind`` to our classification:

    * ``encrypted_dex`` / ``extracted_method_body`` / ``compressed_payload``
      / ``embedded_asset`` / ``native_stub`` -> ``payload_container``
    * ``shim`` / anything labelled ``benign_loader`` -> ``benign_loader``
    """
    mapped: List[NewEntryRule] = []
    for rule in getattr(spec, "rules", []):
        match = getattr(rule, "match", None)
        emit = getattr(rule, "emit", None)
        if match is None or emit is None:
            continue
        raw_regex = getattr(match, "object_path_regex", None)
        if raw_regex is None:
            continue
        # MatchSpec stores a compiled pattern; recover its source string.
        regex = raw_regex.pattern if hasattr(raw_regex, "pattern") else str(raw_regex)
        label = getattr(emit, "label", None)
        payload_kind = getattr(emit, "payload_kind", None)
        if label == HIDDEN_EXECUTABLE_PAYLOAD and payload_kind not in {None, "shim"}:
            classification = NEW_ENTRY_PAYLOAD
        else:
            classification = NEW_ENTRY_LOADER
        mapped.append(NewEntryRule(object_path_regex=regex, classification=classification))
    return mapped


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


_BRANCHES_REQUIRING_PACKED_BYTES = {BRANCH_BYTE_MODIFIED, BRANCH_NEW_IN_PACKED}


def diff_report_to_synthetic_labels(
    report: DiffReport,
    *,
    packer_name: str,
    apk_id: str,
    source_apk_id: Optional[str] = None,
    new_entry_policy: Optional[NewEntryPolicy] = None,
    packed_apk_path: Optional[Path] = None,
    transform_family: Optional[str] = None,
    include_when_degenerate: bool = False,
) -> List[SyntheticLabel]:
    """Convert a :class:`DiffReport` into SyntheticLabel positives.

    :param report: Output of :func:`diff_alignment.align`.
    :param packer_name: Short packer id (e.g. ``"s5_timscriptov"``,
        ``"cs1_360_jiagu"``). Used to derive ``transform_family`` when the
        caller does not override it.
    :param apk_id: Packed APK identifier (stored on each emitted label).
    :param source_apk_id: Benign APK identifier; passed through so
        downstream cross-track joins can pair Track A vs Track B labels
        that originated from the same benign seed.
    :param new_entry_policy: Decision helper for new-in-packed entries.
        Defaults to :class:`AllNewEntriesArePayload`; commercial CS*
        packers should pass a :class:`RuleBasedNewEntryPolicy`.
    :param packed_apk_path: Path to the packed APK so the adapter can
        compute ``payload_sha256`` over each exact byte range. If omitted,
        ``payload_sha256`` falls back to the entry-level sha from
        ``EntryMapping.packed_sha256`` (acceptable for whole-entry
        payloads but less precise for partial byte_modified intervals).
    :param transform_family: Override for the label's ``transform_family``
        field. Defaults to ``"packer_<packer_name>"`` to match Track A's
        ``xor``-style family naming and keep downstream unique-value
        counts honest.
    :param include_when_degenerate: If False (default), return ``[]`` for
        a degenerate report (``payload_ratio > 0.95``) so the caller can
        route the task to ``needs_manual_review`` rather than polluting
        training data. Set True only from the cross-validation driver
        when it has already routed the task to manual review.
    """
    if packed_apk_path is not None:
        _packed_apk_path = Path(packed_apk_path)
        if not _packed_apk_path.exists():
            raise FileNotFoundError(
                f"packed_apk_path does not exist: {_packed_apk_path}"
            )
    else:
        _packed_apk_path = None

    if report.alignment_failed:
        return []
    if report.degenerate_flag and not include_when_degenerate:
        return []

    policy = new_entry_policy or AllNewEntriesArePayload()
    family = transform_family or f"packer_{packer_name}"

    # Lazy load packed bytes only if needed (skipping for degenerate/early
    # return paths above).
    packed_entry_bytes = _load_packed_entry_bytes(
        _packed_apk_path, report.entries
    )

    out: List[SyntheticLabel] = []
    for entry in report.entries:
        if entry.branch == BRANCH_BYTE_MODIFIED:
            out.extend(
                _emit_byte_modified_labels(
                    entry,
                    family=family,
                    apk_id=apk_id,
                    source_apk_id=source_apk_id,
                    packed_entry_bytes=packed_entry_bytes,
                )
            )
        elif entry.branch == BRANCH_NEW_IN_PACKED:
            classification = policy.classify(entry)
            if classification != NEW_ENTRY_PAYLOAD:
                continue  # skip loader or unknown
            lbl = _emit_new_entry_label(
                entry,
                family=family,
                apk_id=apk_id,
                source_apk_id=source_apk_id,
                packed_entry_bytes=packed_entry_bytes,
            )
            if lbl is not None:
                out.append(lbl)
        # BRANCH_UNCHANGED / BRANCH_RENAMED / BRANCH_REMOVED produce no
        # positive labels: they are either benign bytes (unchanged /
        # renamed copies of benign entries) or absent from the packed
        # APK (removed).
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_packed_entry_bytes(
    packed_apk_path: Optional[Path],
    entries: Sequence[EntryMapping],
) -> Optional[Mapping[str, bytes]]:
    """Return ``{entry_name: bytes}`` or None if path not supplied.

    Only loads entries that the adapter will need (``byte_modified`` or
    ``new_in_packed``). For degenerate / removed / renamed branches we
    skip the zip read entirely.
    """
    if packed_apk_path is None:
        return None
    path = Path(packed_apk_path)
    if not path.exists():
        raise FileNotFoundError(f"packed_apk_path does not exist: {path}")
    needed = {
        e.packed_entry
        for e in entries
        if e.branch in _BRANCHES_REQUIRING_PACKED_BYTES and e.packed_entry
    }
    if not needed:
        return {}
    bytes_by_name: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.filename in needed:
                with zf.open(info) as fh:
                    bytes_by_name[info.filename] = fh.read()
    return bytes_by_name


def _emit_byte_modified_labels(
    entry: EntryMapping,
    *,
    family: str,
    apk_id: str,
    source_apk_id: Optional[str],
    packed_entry_bytes: Optional[Mapping[str, bytes]],
) -> List[SyntheticLabel]:
    labels: List[SyntheticLabel] = []
    entry_bytes = None
    if packed_entry_bytes is not None and entry.packed_entry is not None:
        entry_bytes = packed_entry_bytes.get(entry.packed_entry)
    for rng in entry.diff_ranges:
        payload_sha = _payload_sha_for_range(entry, rng, entry_bytes)
        labels.append(
            SyntheticLabel(
                apk_id=apk_id,
                object_path=entry.packed_entry or "",
                offset_start=rng.start,
                offset_end=rng.end,
                label=HIDDEN_EXECUTABLE_PAYLOAD,
                transform_family=family,
                payload_sha256=payload_sha,
                source_apk_id=source_apk_id,
                source_object_path=entry.benign_entry,
            )
        )
    return labels


def _emit_new_entry_label(
    entry: EntryMapping,
    *,
    family: str,
    apk_id: str,
    source_apk_id: Optional[str],
    packed_entry_bytes: Optional[Mapping[str, bytes]],
) -> Optional[SyntheticLabel]:
    if entry.packed_entry is None or entry.packed_size == 0:
        return None
    # New entries have a single [0, packed_size) range (or none when empty);
    # the diff alignment implementation guarantees at most one range.
    if entry.diff_ranges:
        rng = entry.diff_ranges[0]
    else:
        rng = DiffRange(start=0, end=entry.packed_size)
    entry_bytes = (
        packed_entry_bytes.get(entry.packed_entry)
        if packed_entry_bytes is not None
        else None
    )
    payload_sha = _payload_sha_for_range(entry, rng, entry_bytes)
    return SyntheticLabel(
        apk_id=apk_id,
        object_path=entry.packed_entry,
        offset_start=rng.start,
        offset_end=rng.end,
        label=HIDDEN_EXECUTABLE_PAYLOAD,
        transform_family=family,
        payload_sha256=payload_sha,
        source_apk_id=source_apk_id,
        # New entries have no benign source -- leave as None.
        source_object_path=None,
    )


def _payload_sha_for_range(
    entry: EntryMapping,
    rng: DiffRange,
    entry_bytes: Optional[bytes],
) -> str:
    """Return a sha256 hex for ``entry_bytes[rng.start:rng.end]``.

    Falls back to ``entry.packed_sha256`` when the caller did not supply
    the bytes (so the label still has *some* provenance hash). The fallback
    is acceptable for whole-entry payloads (branch=new_in_packed,
    rng covering the whole entry) but loses specificity for partial
    byte_modified intervals; downstream spotcheck code should prefer
    entries whose labels were produced with ``packed_apk_path`` set.
    """
    if entry_bytes is None:
        if entry.packed_sha256:
            return entry.packed_sha256
        # Last-resort: sha of empty string would lie, so raise.
        raise ValueError(
            f"Cannot compute payload_sha256: no bytes and no packed_sha256 "
            f"for entry {entry.packed_entry!r}"
        )
    slice_bytes = entry_bytes[rng.start : rng.end]
    return hashlib.sha256(slice_bytes).hexdigest()


__all__ = [
    "AllNewEntriesArePayload",
    "NEW_ENTRY_LOADER",
    "NEW_ENTRY_PAYLOAD",
    "NEW_ENTRY_UNKNOWN",
    "NewEntryPolicy",
    "NewEntryRule",
    "NoNewEntriesArePayload",
    "RuleBasedNewEntryPolicy",
    "diff_report_to_synthetic_labels",
    "new_entry_rules_from_spec",
]
