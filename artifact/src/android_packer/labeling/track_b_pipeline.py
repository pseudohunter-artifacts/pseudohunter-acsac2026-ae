"""Track B labeling pipeline: packed APKs -> full SyntheticLabel set.

This is the **B-c-1** / **B-c-2** production module. It takes the raw
output of ``scripts/build_track_b_corpus.py`` (one packed APK plus, for
open-source packers, one ``inject_labels.jsonl``) and produces the
complete labeling bundle expected by the downstream Track A evaluation
pipeline:

    <out_dir>/<packer_id>/<benign_stem>/
        path_a_labels.jsonl      # from inject_labels.jsonl (if present)
        path_b_labels.jsonl      # from diff_alignment + diff_adapter
        apkid_report.json        # from apkid_cross_check (if apkid available)
        merged_labels.jsonl      # final SyntheticLabel set (training-ready)
        summary.json             # per-apk outcome

For **open-source packers** (S1..S6) we prefer Path A labels when they
exist (byte-exact ground truth). Path B (diff) is still computed and
recorded as a sanity check; the IoU between the two paths becomes a
data-quality metric in the final summary.

For **commercial packers** (CS1..CS5) there is no Path A source
injection; instead we invoke :func:`apply_rules_to_apk` which gives a
literature-derived best-guess labeling. We then cross-validate against
Path B via :func:`cross_validate_commercial_packer`; the decision
matrix (SOLID / PARTIAL_MISMATCH / LOW_CONFIDENCE / ...) chooses which
source wins.

For **all** packers (open + commercial + even benign sanity runs) we
additionally run APKiD as an *independent* third-party check and record
the agreement.

The module is intentionally I/O-heavy but dependency-light: every
step-level function is isolated and can be mocked in tests.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from android_packer.labeling.apkid_cross_check import (
    AGREEMENT_APKID_FAILED,
    ApkidCrossCheckReport,
    ApkidFamilyMap,
    cross_check_apk as _cross_check_apkid,
)
from android_packer.labeling.commercial_rule_engine import (
    CommercialPackerSpec,
    load_rule_file,
)
from android_packer.labeling.cs_cross_validate import (
    CsCrossValidationReport,
    cross_validate_commercial_packer,
)
from android_packer.labeling.diff_adapter import (
    AllNewEntriesArePayload,
    NewEntryPolicy,
    RuleBasedNewEntryPolicy,
    diff_report_to_synthetic_labels,
    new_entry_rules_from_spec,
)
from android_packer.labeling.diff_alignment import align
from android_packer.labeling.injected_packer_adapter import (
    load_synthetic_labels,
)
from android_packer.labeling.synthetic import SyntheticLabel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_SOURCE_PATH_A = "path_a_injected"
LABEL_SOURCE_PATH_A_RULE = "path_a_rule"
LABEL_SOURCE_PATH_B = "path_b_diff"
LABEL_SOURCE_NONE = "none"

#: Reasons why ``final_labels`` may be empty. Surface into summary.
REASON_NO_PACKED_APK = "no_packed_apk"
REASON_NO_BENIGN_APK = "no_benign_apk"
REASON_DIFF_DEGENERATE = "diff_degenerate"
REASON_DIFF_ALIGNMENT_FAILED = "diff_alignment_failed"
REASON_RULE_FILE_MISSING = "rule_file_missing"
REASON_LOW_CONFIDENCE = "low_confidence"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TrackBLabelingError(RuntimeError):
    """Unrecoverable per-pair failure (distinct from per-pair skips)."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackerIdent:
    """Lightweight projection of a ``track_b_packers.yaml`` entry.

    We don't want the rest of the pipeline to know the full YAML schema;
    this structure captures everything ``process_pair`` needs.
    """

    packer_id: str
    #: Group tag: ``open_source`` (S1..S6), ``commercial`` (CS1..CS5), or
    #: ``registered_not_patched`` (e.g. S7 · Bangcle-OSS).
    group: str
    transform_family: str
    path_a_enabled: bool
    path_b_enabled: bool  # (True / False / "partial" normalised to True)
    rule_file: Optional[Path]  # only for CS* group

    @classmethod
    def from_registry_entry(
        cls,
        packer_id: str,
        spec: Mapping[str, Any],
        *,
        rules_dir: Optional[Path] = None,
    ) -> "PackerIdent":
        label = spec.get("label", {}) or {}
        transform_family = label.get("transform_family")
        if not transform_family:
            raise ValueError(
                f"{packer_id!r}: missing label.transform_family in registry entry"
            )
        license_str = str(spec.get("license", "")).strip().upper()
        status = str(spec.get("status", "")).strip()
        path_a_raw = label.get("path_a", False)
        path_b_raw = label.get("path_b", False)

        # Group heuristic:
        # - "candidate" + license in {COMMERCIAL_*} -> commercial
        # - status starts with registered_not_patched -> registered_not_patched
        # - explicit "license: NONE" on an open-source entry stays open_source
        # - otherwise: open_source
        if status == "registered_not_patched":
            group = "registered_not_patched"
        elif license_str.startswith("COMMERCIAL") or packer_id.startswith("cs"):
            group = "commercial"
        else:
            group = "open_source"

        rule_file: Optional[Path] = None
        if group == "commercial":
            rf = label.get("rule_file")
            if rf:
                # Three fallback strategies, tried in order:
                # 1. ``rf`` is already a usable path (absolute or relative
                #    to the current working directory / repo root).
                # 2. Otherwise join with ``rules_dir`` using the filename only.
                # 3. Else leave ``rule_file = None``; the downstream pipeline
                #    surfaces this as ``REASON_RULE_FILE_MISSING`` in the
                #    per-pair summary.
                candidate_direct = Path(rf)
                if candidate_direct.exists():
                    rule_file = candidate_direct
                elif rules_dir is not None:
                    candidate_joined = rules_dir / Path(rf).name
                    if candidate_joined.exists():
                        rule_file = candidate_joined

        return cls(
            packer_id=packer_id,
            group=group,
            transform_family=transform_family,
            path_a_enabled=bool(path_a_raw),
            # "partial" -> True; rely on diff_alignment's own degenerate flag.
            # For commercial packers the spec file may say ``path_b: false``
            # because Path B is not the *primary* source, but we still need
            # to run the diff in order to execute ``cs_cross_validate``. So
            # Path B is force-enabled for the commercial group here; the
            # actual per-call gating happens in ``cs_cross_validate_commercial_packer``.
            path_b_enabled=(
                bool(path_b_raw)
                or path_b_raw == "partial"
                or group == "commercial"
            ),
            rule_file=rule_file,
        )


@dataclass(frozen=True)
class PairInputs:
    """Locations of the input files for one (packer, benign) pair."""

    packer: PackerIdent
    benign_apk: Path
    packed_apk: Optional[Path]
    inject_labels_jsonl: Optional[Path]
    apk_id: str
    source_apk_id: str


@dataclass
class PairOutcome:
    """Result of processing one pair. Safe to serialise to JSON."""

    packer_id: str
    apk_id: str
    source_apk_id: str
    transform_family: str
    group: str
    chosen_source: str  # one of LABEL_SOURCE_*
    path_a_label_count: int = 0
    path_b_label_count: int = 0
    final_label_count: int = 0
    needs_manual_review: bool = False
    path_b_payload_ratio: Optional[float] = None
    diff_degenerate: bool = False
    diff_alignment_failed: bool = False
    cs_decision: Optional[str] = None  # only for commercial packers
    cs_iou: Optional[float] = None
    apkid_agreement: Optional[str] = None
    apkid_has_packer_hit: Optional[bool] = None
    reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: Paths of persisted artifacts (all relative to out_dir for portability).
    artifacts: Dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_slug(s: str) -> str:
    """Make a filename-safe slug from an APK stem."""
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def _write_jsonl(labels: Sequence[SyntheticLabel], out: Path) -> None:
    """Write ``SyntheticLabel``s as JSONL (one per line)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for lab in labels:
            fh.write(
                json.dumps(lab.to_dict(), ensure_ascii=False, sort_keys=True)
            )
            fh.write("\n")


def _safe_name(path: Path) -> str:
    return _safe_slug(path.stem)


# ---------------------------------------------------------------------------
# Step-level functions (each returns data; no I/O side effects except
# the JSONL / JSON writes indicated by ``artifacts`` in PairOutcome)
# ---------------------------------------------------------------------------


def _do_path_a(
    inputs: PairInputs,
    pair_out_dir: Path,
) -> Tuple[List[SyntheticLabel], Optional[str]]:
    """Load Path A labels from inject_labels.jsonl if available.

    Returns ``(labels, error_reason_or_None)``. The error reason is a
    short constant (not a full message) or None on success; it is
    intended to be appended to ``PairOutcome.reasons``.
    """
    if not inputs.packer.path_a_enabled:
        return [], None  # silently skipped
    jsonl = inputs.inject_labels_jsonl
    if jsonl is None or not jsonl.exists():
        return [], "path_a_jsonl_missing"
    try:
        labels = load_synthetic_labels(jsonl)
    except Exception as e:  # InjectLabelSchemaError or OSError
        return [], f"path_a_parse_error:{type(e).__name__}"

    out_path = pair_out_dir / "path_a_labels.jsonl"
    _write_jsonl(labels, out_path)
    return list(labels), None


def _do_path_b(
    inputs: PairInputs,
    pair_out_dir: Path,
    *,
    new_entry_policy: Optional[NewEntryPolicy] = None,
) -> Tuple[List[SyntheticLabel], Optional[str], Dict[str, Any]]:
    """Align benign vs packed and emit Path B SyntheticLabels.

    ``meta`` contains ``{degenerate, alignment_failed, payload_ratio}``
    even when the list is empty, so the caller can populate summary.
    """
    meta: Dict[str, Any] = {
        "degenerate": False,
        "alignment_failed": False,
        "payload_ratio": None,
    }
    if not inputs.packer.path_b_enabled:
        return [], None, meta
    if inputs.packed_apk is None or not inputs.packed_apk.exists():
        return [], REASON_NO_PACKED_APK, meta
    if not inputs.benign_apk.exists():
        return [], REASON_NO_BENIGN_APK, meta

    report = align(inputs.benign_apk, inputs.packed_apk)
    meta["degenerate"] = report.degenerate_flag
    meta["alignment_failed"] = report.alignment_failed
    meta["payload_ratio"] = report.payload_ratio

    policy = new_entry_policy or AllNewEntriesArePayload()
    labels = diff_report_to_synthetic_labels(
        report,
        packer_name=inputs.packer.packer_id,
        apk_id=inputs.apk_id,
        source_apk_id=inputs.source_apk_id,
        new_entry_policy=policy,
        packed_apk_path=inputs.packed_apk,
        transform_family=inputs.packer.transform_family,
        include_when_degenerate=False,
    )

    out_path = pair_out_dir / "path_b_labels.jsonl"
    _write_jsonl(labels, out_path)

    reason: Optional[str] = None
    if report.alignment_failed:
        reason = REASON_DIFF_ALIGNMENT_FAILED
    elif report.degenerate_flag:
        reason = REASON_DIFF_DEGENERATE
    return list(labels), reason, meta


def _do_apkid(
    inputs: PairInputs,
    pair_out_dir: Path,
    *,
    family_map: Optional[ApkidFamilyMap],
    apkid_cmd: str,
    timeout: float,
) -> Optional[ApkidCrossCheckReport]:
    """Run APKiD on the packed APK (if present) for independent check."""
    if inputs.packed_apk is None or not inputs.packed_apk.exists():
        return None
    report = _cross_check_apkid(
        inputs.packed_apk,
        apk_id=inputs.apk_id,
        expected_family=inputs.packer.transform_family,
        family_map=family_map,
        apkid_cmd=apkid_cmd,
        timeout=timeout,
        graceful=True,
    )
    out = pair_out_dir / "apkid_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return report


def _do_cs_cross_validate(
    inputs: PairInputs,
    pair_out_dir: Path,
    *,
    solid_threshold: float,
    review_threshold: float,
) -> Tuple[Optional[CsCrossValidationReport], Optional[str]]:
    """Run CS* rule+Path-B cross-validation; only for commercial group.

    Returns ``(report, error_reason_or_None)``.
    """
    if inputs.packer.group != "commercial":
        return None, None
    rf = inputs.packer.rule_file
    if rf is None or not rf.exists():
        return None, REASON_RULE_FILE_MISSING
    if inputs.packed_apk is None or not inputs.packed_apk.exists():
        return None, REASON_NO_PACKED_APK
    if not inputs.benign_apk.exists():
        return None, REASON_NO_BENIGN_APK

    spec = load_rule_file(rf)
    report = cross_validate_commercial_packer(
        benign_apk=inputs.benign_apk,
        packed_apk=inputs.packed_apk,
        rule_spec=spec,
        apk_id=inputs.apk_id,
        source_apk_id=inputs.source_apk_id,
        solid_threshold=solid_threshold,
        review_threshold=review_threshold,
    )
    out = pair_out_dir / "cs_cross_validate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return report, None


# ---------------------------------------------------------------------------
# Main per-pair processor
# ---------------------------------------------------------------------------


def process_pair(
    inputs: PairInputs,
    out_dir: Path,
    *,
    apkid_family_map: Optional[ApkidFamilyMap] = None,
    apkid_cmd: str = "apkid",
    apkid_timeout: float = 60.0,
    run_apkid: bool = True,
    solid_threshold: float = 0.8,
    review_threshold: float = 0.5,
) -> PairOutcome:
    """Process one (packer, benign_apk) pair end-to-end.

    This function is the pipeline's core unit: given a :class:`PairInputs`
    locator, it produces all per-pair artifacts (Path A / Path B JSONL,
    merged JSONL, APKiD / CS* reports, summary) under
    ``out_dir/<packer_id>/<benign_stem>/``.
    """
    t0 = time.perf_counter()
    pair_dir = out_dir / inputs.packer.packer_id / _safe_name(inputs.benign_apk)

    outcome = PairOutcome(
        packer_id=inputs.packer.packer_id,
        apk_id=inputs.apk_id,
        source_apk_id=inputs.source_apk_id,
        transform_family=inputs.packer.transform_family,
        group=inputs.packer.group,
        chosen_source=LABEL_SOURCE_NONE,
    )

    # ------------------------------------------------------------------
    # Path A
    # ------------------------------------------------------------------
    path_a_labels, path_a_reason = _do_path_a(inputs, pair_dir)
    outcome.path_a_label_count = len(path_a_labels)
    if path_a_reason:
        outcome.reasons.append(path_a_reason)
    if path_a_labels:
        outcome.artifacts["path_a_labels"] = str(
            (pair_dir / "path_a_labels.jsonl").as_posix()
        )

    # ------------------------------------------------------------------
    # Path B (policy depends on group)
    # ------------------------------------------------------------------
    new_entry_policy: Optional[NewEntryPolicy] = None
    if inputs.packer.group == "commercial" and inputs.packer.rule_file is not None:
        try:
            spec = load_rule_file(inputs.packer.rule_file)
            rules = new_entry_rules_from_spec(spec)
            new_entry_policy = RuleBasedNewEntryPolicy(rules)
        except Exception as e:
            outcome.notes.append(
                f"rule-based new_entry_policy unavailable ({type(e).__name__}): "
                "falling back to AllNewEntriesArePayload"
            )
            new_entry_policy = None
    path_b_labels, path_b_reason, diff_meta = _do_path_b(
        inputs, pair_dir, new_entry_policy=new_entry_policy
    )
    outcome.path_b_label_count = len(path_b_labels)
    outcome.diff_degenerate = bool(diff_meta.get("degenerate"))
    outcome.diff_alignment_failed = bool(diff_meta.get("alignment_failed"))
    outcome.path_b_payload_ratio = diff_meta.get("payload_ratio")
    if path_b_reason:
        outcome.reasons.append(path_b_reason)
    if path_b_labels:
        outcome.artifacts["path_b_labels"] = str(
            (pair_dir / "path_b_labels.jsonl").as_posix()
        )

    # ------------------------------------------------------------------
    # APKiD independent cross-check (group-agnostic)
    # ------------------------------------------------------------------
    apkid_report: Optional[ApkidCrossCheckReport] = None
    if run_apkid:
        apkid_report = _do_apkid(
            inputs,
            pair_dir,
            family_map=apkid_family_map,
            apkid_cmd=apkid_cmd,
            timeout=apkid_timeout,
        )
        if apkid_report is not None:
            outcome.apkid_agreement = apkid_report.agreement
            outcome.apkid_has_packer_hit = apkid_report.has_packer_hit or apkid_report.has_protector_hit
            if apkid_report.needs_manual_review:
                outcome.needs_manual_review = True
            outcome.artifacts["apkid_report"] = str(
                (pair_dir / "apkid_report.json").as_posix()
            )

    # ------------------------------------------------------------------
    # Decide which labels to adopt
    # ------------------------------------------------------------------
    final_labels: List[SyntheticLabel] = []
    if inputs.packer.group == "commercial":
        cs_report, cs_reason = _do_cs_cross_validate(
            inputs,
            pair_dir,
            solid_threshold=solid_threshold,
            review_threshold=review_threshold,
        )
        if cs_reason:
            outcome.reasons.append(cs_reason)
        if cs_report is not None:
            outcome.cs_decision = cs_report.decision
            outcome.cs_iou = cs_report.iou
            if cs_report.needs_manual_review:
                outcome.needs_manual_review = True
            final_labels = list(cs_report.final_labels)
            outcome.chosen_source = _pick_cs_label_source(cs_report)
            outcome.artifacts["cs_cross_validate_report"] = str(
                (pair_dir / "cs_cross_validate_report.json").as_posix()
            )
        else:
            # No rules to cross-validate against: fall back to Path B alone.
            if path_b_labels:
                final_labels = list(path_b_labels)
                outcome.chosen_source = LABEL_SOURCE_PATH_B
                outcome.needs_manual_review = True
                outcome.notes.append(
                    "commercial packer with no rule file; using Path B only "
                    "(needs_manual_review=true)"
                )
    else:
        # Open-source: prefer Path A if we have it, else Path B
        if path_a_labels:
            final_labels = path_a_labels
            outcome.chosen_source = LABEL_SOURCE_PATH_A
        elif path_b_labels:
            final_labels = path_b_labels
            outcome.chosen_source = LABEL_SOURCE_PATH_B
            outcome.notes.append(
                "open-source packer fell back to Path B (inject_labels.jsonl missing)"
            )
        elif outcome.diff_degenerate or outcome.diff_alignment_failed:
            outcome.chosen_source = LABEL_SOURCE_NONE
            outcome.needs_manual_review = True

    outcome.final_label_count = len(final_labels)
    if final_labels:
        _write_jsonl(final_labels, pair_dir / "merged_labels.jsonl")
        outcome.artifacts["merged_labels"] = str(
            (pair_dir / "merged_labels.jsonl").as_posix()
        )

    # ------------------------------------------------------------------
    # Persist per-pair summary
    # ------------------------------------------------------------------
    outcome.duration_s = time.perf_counter() - t0
    summary_path = pair_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(outcome.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    outcome.artifacts["summary"] = str(summary_path.as_posix())
    return outcome


def _pick_cs_label_source(report: CsCrossValidationReport) -> str:
    """Map a CS cross-validate decision to a label-source tag."""
    if not report.final_labels:
        return LABEL_SOURCE_NONE
    # cs_cross_validate prefers Path B whenever it has labels;
    # falls back to rule only when Path B fails (alignment / degenerate).
    if report.path_b_label_count > 0 and report.final_label_count == report.path_b_label_count:
        return LABEL_SOURCE_PATH_B
    return LABEL_SOURCE_PATH_A_RULE


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------


@dataclass
class BatchSummary:
    """Aggregate across all (packer, benign) pairs in one run."""

    total_pairs: int = 0
    ok_pairs: int = 0
    degenerate_pairs: int = 0
    alignment_failed_pairs: int = 0
    needs_manual_review_pairs: int = 0
    label_source_histogram: Dict[str, int] = field(default_factory=dict)
    apkid_agreement_histogram: Dict[str, int] = field(default_factory=dict)
    group_histogram: Dict[str, int] = field(default_factory=dict)
    per_packer_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, o: PairOutcome) -> None:
        self.total_pairs += 1
        if o.final_label_count > 0:
            self.ok_pairs += 1
        if o.diff_degenerate:
            self.degenerate_pairs += 1
        if o.diff_alignment_failed:
            self.alignment_failed_pairs += 1
        if o.needs_manual_review:
            self.needs_manual_review_pairs += 1
        self.label_source_histogram[o.chosen_source] = (
            self.label_source_histogram.get(o.chosen_source, 0) + 1
        )
        if o.apkid_agreement is not None:
            self.apkid_agreement_histogram[o.apkid_agreement] = (
                self.apkid_agreement_histogram.get(o.apkid_agreement, 0) + 1
            )
        self.group_histogram[o.group] = self.group_histogram.get(o.group, 0) + 1
        p = self.per_packer_stats.setdefault(
            o.packer_id,
            {"total": 0, "with_labels": 0, "needs_review": 0},
        )
        p["total"] += 1
        if o.final_label_count > 0:
            p["with_labels"] += 1
        if o.needs_manual_review:
            p["needs_review"] += 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def process_batch(
    pair_inputs: Sequence[PairInputs],
    out_dir: Path,
    *,
    apkid_family_map: Optional[ApkidFamilyMap] = None,
    apkid_cmd: str = "apkid",
    apkid_timeout: float = 60.0,
    run_apkid: bool = True,
    solid_threshold: float = 0.8,
    review_threshold: float = 0.5,
) -> Tuple[List[PairOutcome], BatchSummary]:
    """Process many pairs. Writes aggregate summary files at the end."""
    outcomes: List[PairOutcome] = []
    summary = BatchSummary()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-pair summary.jsonl is appended incrementally so long runs are
    # recoverable even if the process crashes halfway.
    summary_jsonl = out_dir / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as fh:
        for inputs in pair_inputs:
            outcome = process_pair(
                inputs,
                out_dir,
                apkid_family_map=apkid_family_map,
                apkid_cmd=apkid_cmd,
                apkid_timeout=apkid_timeout,
                run_apkid=run_apkid,
                solid_threshold=solid_threshold,
                review_threshold=review_threshold,
            )
            outcomes.append(outcome)
            summary.record(outcome)
            fh.write(json.dumps(outcome.to_dict(), ensure_ascii=False, sort_keys=True))
            fh.write("\n")

    aggregate_path = out_dir / "summary.json"
    aggregate_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return outcomes, summary


# ---------------------------------------------------------------------------
# Discovery helpers for scripts/run_track_b_labeling.py
# ---------------------------------------------------------------------------


def discover_pair_inputs(
    registry: Mapping[str, Mapping[str, Any]],
    *,
    packed_dir: Path,
    benign_dir: Path,
    rules_dir: Optional[Path] = None,
    packer_allowlist: Optional[Sequence[str]] = None,
    include_registered_not_patched: bool = False,
) -> List[PairInputs]:
    """Walk ``packed_dir`` and match up packed.apk files with benign APKs.

    Expected layout::

        packed_dir/<packer_id>/<benign_stem>/packed.apk
        packed_dir/<packer_id>/<benign_stem>/inject_labels.jsonl (optional)
        benign_dir/<benign_stem>.apk

    Missing packed.apk files are silently skipped (they are usually
    pending packer runs, not errors). Callers should use the returned
    list length against the registry x benign product to see how many
    pairs are still missing.
    """
    pairs: List[PairInputs] = []
    allow = set(packer_allowlist) if packer_allowlist else None
    for packer_id, spec in registry.items():
        if allow is not None and packer_id not in allow:
            continue
        ident = PackerIdent.from_registry_entry(
            packer_id, spec, rules_dir=rules_dir
        )
        if (
            ident.group == "registered_not_patched"
            and not include_registered_not_patched
        ):
            continue
        packer_packed_dir = packed_dir / packer_id
        if not packer_packed_dir.is_dir():
            continue
        for apk_dir in sorted(packer_packed_dir.iterdir()):
            if not apk_dir.is_dir():
                continue
            packed_apk = apk_dir / "packed.apk"
            if not packed_apk.exists():
                continue
            # Match a benign apk whose stem equals apk_dir.name
            benign_apk = _find_benign(benign_dir, apk_dir.name)
            if benign_apk is None:
                continue
            inject_jsonl = apk_dir / "inject_labels.jsonl"
            apk_id = f"track_b:{packer_id}:{apk_dir.name}"
            source_apk_id = f"benign:{apk_dir.name}"
            pairs.append(
                PairInputs(
                    packer=ident,
                    benign_apk=benign_apk,
                    packed_apk=packed_apk,
                    inject_labels_jsonl=inject_jsonl if inject_jsonl.exists() else None,
                    apk_id=apk_id,
                    source_apk_id=source_apk_id,
                )
            )
    return pairs


def _find_benign(benign_dir: Path, stem: str) -> Optional[Path]:
    if not benign_dir.is_dir():
        return None
    candidate = benign_dir / f"{stem}.apk"
    if candidate.exists():
        return candidate
    # Fallback: case-insensitive / slug-equivalent match
    target = _safe_slug(stem).lower()
    for p in benign_dir.glob("*.apk"):
        if _safe_slug(p.stem).lower() == target:
            return p
    return None
