"""Cross-validate Path B (byte diff) against Path A-rule (commercial rules).

This is the **B-g-3** deliverable. Under the 2026-04-30 corrected stance
for CS1-CS5 commercial packers, Path B is the primary labelling source
and Path A-rule serves as an independent check. This module wires them
together:

    (benign_apk, packed_apk, rule_spec)
        |
        +--- diff_alignment.align(...)          -> DiffReport
        |    diff_adapter.diff_report_to_*(...) -> Path B SyntheticLabel
        |
        +--- commercial_rule_engine.apply_rules_to_apk(...) -> InjectLabelRecord
             injected_packer_adapter.to_synthetic_labels(...) -> Path A-rule SyntheticLabel
        |
        +--- injected_packer_adapter.cross_validate(...) -> CrossValidationResult
        |
        +--- decide which side to trust -> CsCrossValidationReport

Decision matrix (see ``labeling_injection_spec.md`` sections 9.3/9.4):

+-------------------------+----------------+-------+-------------------------+-----------------+--------------+
| Path B status           | Rule output    | IoU   | decision                | final_labels    | manual_review|
+=========================+================+=======+=========================+=================+==============+
| alignment_failed        | any            | --    | rule_only_alignment_failed | rule          | True         |
| degenerate              | any            | --    | rule_only_degenerate       | rule          | True         |
| normal                  | empty          | --    | path_b_only_no_rule_match  | path_b        | True         |
| normal                  | non-empty      | >=0.8 | solid                      | path_b         | False        |
| normal                  | non-empty      | 0.5-0.8| partial_mismatch          | path_b         | True         |
| normal                  | non-empty      | <0.5  | low_confidence             | []            | True         |
+-------------------------+----------------+-------+-------------------------+-----------------+--------------+

The 0.8 solid threshold is looser than the open-source S1-S5 threshold of
0.9 because Path A-rule is a literature-derived estimator, not ground
truth; see ``labeling_injection_spec.md`` section 9.3 row 3.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from android_packer.labeling.commercial_rule_engine import (
    ApplyResult,
    CommercialPackerSpec,
    apply_rules_to_apk,
)
from android_packer.labeling.diff_adapter import (
    NewEntryPolicy,
    RuleBasedNewEntryPolicy,
    diff_report_to_synthetic_labels,
    new_entry_rules_from_spec,
)
from android_packer.labeling.diff_alignment import DiffReport, align
from android_packer.labeling.injected_packer_adapter import (
    CrossValidationResult,
    cross_validate,
    to_synthetic_labels,
)
from android_packer.labeling.synthetic import SyntheticLabel


DECISION_SOLID = "solid"
DECISION_PARTIAL_MISMATCH = "partial_mismatch"
DECISION_LOW_CONFIDENCE = "low_confidence"
DECISION_RULE_ONLY_DEGENERATE = "rule_only_degenerate"
DECISION_RULE_ONLY_ALIGNMENT_FAILED = "rule_only_alignment_failed"
DECISION_PATH_B_ONLY_NO_RULE_MATCH = "path_b_only_no_rule_match"
DECISION_NO_SIGNAL = "no_signal"


@dataclass(frozen=True)
class CsCrossValidationReport:
    """Per-APK outcome of CS* Path B x Path A-rule cross-validation."""

    packer_id: str
    apk_id: str
    source_apk_id: str
    decision: str
    needs_manual_review: bool
    iou: Optional[float]
    per_object_iou: Optional[dict]
    path_b_label_count: int
    path_a_rule_label_count: int
    final_label_count: int
    diff_report_degenerate: bool
    diff_report_alignment_failed: bool
    path_b_payload_ratio: float
    rule_matched_entries: Sequence[str]
    rule_unmatched_entries: Sequence[str]
    path_b_labels: Sequence[SyntheticLabel]
    path_a_rule_labels: Sequence[SyntheticLabel]
    final_labels: Sequence[SyntheticLabel]
    notes: Sequence[str]

    def to_dict(self) -> dict:
        return {
            "packer_id": self.packer_id,
            "apk_id": self.apk_id,
            "source_apk_id": self.source_apk_id,
            "decision": self.decision,
            "needs_manual_review": self.needs_manual_review,
            "iou": self.iou,
            "per_object_iou": dict(self.per_object_iou or {}),
            "path_b_label_count": self.path_b_label_count,
            "path_a_rule_label_count": self.path_a_rule_label_count,
            "final_label_count": self.final_label_count,
            "diff_report_degenerate": self.diff_report_degenerate,
            "diff_report_alignment_failed": self.diff_report_alignment_failed,
            "path_b_payload_ratio": self.path_b_payload_ratio,
            "rule_matched_entries": list(self.rule_matched_entries),
            "rule_unmatched_entries": list(self.rule_unmatched_entries),
            "path_b_labels": [lbl.to_dict() for lbl in self.path_b_labels],
            "path_a_rule_labels": [lbl.to_dict() for lbl in self.path_a_rule_labels],
            "final_labels": [lbl.to_dict() for lbl in self.final_labels],
            "notes": list(self.notes),
        }


def cross_validate_commercial_packer(
    benign_apk: Path,
    packed_apk: Path,
    rule_spec: CommercialPackerSpec,
    *,
    apk_id: str,
    source_apk_id: str,
    solid_threshold: float = 0.8,
    review_threshold: float = 0.5,
    new_entry_policy: Optional[NewEntryPolicy] = None,
) -> CsCrossValidationReport:
    """Run Path B and Path A-rule against the same packed APK and reconcile.

    Returns a :class:`CsCrossValidationReport` summarising both paths and
    the final verdict. ``final_labels`` is the sequence that downstream
    training / evaluation should consume -- possibly empty if both paths
    disagreed strongly.
    """
    if solid_threshold <= review_threshold:
        raise ValueError("solid_threshold must be > review_threshold")

    benign_path = Path(benign_apk)
    packed_path = Path(packed_apk)
    if not packed_path.exists():
        raise FileNotFoundError(f"packed_apk does not exist: {packed_path}")
    if not benign_path.exists():
        raise FileNotFoundError(f"benign_apk does not exist: {benign_path}")

    notes: List[str] = []

    # --- Path B -----------------------------------------------------------
    diff_report: DiffReport = align(benign_path, packed_path)
    if new_entry_policy is None:
        # Use the CommercialPackerSpec's rules to decide which new-in-packed
        # entries are payloads vs loaders.
        new_entry_policy = RuleBasedNewEntryPolicy(new_entry_rules_from_spec(rule_spec))

    # Normal case: only emit Path B labels when diff is non-degenerate and
    # alignment succeeded. Degenerate/failed cases go to rule-only below.
    path_b_labels = diff_report_to_synthetic_labels(
        diff_report,
        packer_name=rule_spec.packer_id,
        apk_id=apk_id,
        source_apk_id=source_apk_id,
        new_entry_policy=new_entry_policy,
        packed_apk_path=packed_path,
        include_when_degenerate=False,
    )

    # --- Path A-rule ------------------------------------------------------
    apply_result: ApplyResult = apply_rules_to_apk(
        packed_path,
        rule_spec,
        apk_id=apk_id,
        source_apk_id=source_apk_id,
        compute_sha256=True,
    )
    path_a_rule_labels = to_synthetic_labels(iter([apply_result.record]))

    # --- Decide -----------------------------------------------------------
    decision, needs_manual_review, final_labels, iou_res = _decide(
        diff_report=diff_report,
        path_b_labels=path_b_labels,
        path_a_rule_labels=path_a_rule_labels,
        solid_threshold=solid_threshold,
        review_threshold=review_threshold,
        notes=notes,
    )

    return CsCrossValidationReport(
        packer_id=rule_spec.packer_id,
        apk_id=apk_id,
        source_apk_id=source_apk_id,
        decision=decision,
        needs_manual_review=needs_manual_review,
        iou=iou_res.iou if iou_res is not None else None,
        per_object_iou=(
            dict(iou_res.per_object_iou) if iou_res is not None else None
        ),
        path_b_label_count=len(path_b_labels),
        path_a_rule_label_count=len(path_a_rule_labels),
        final_label_count=len(final_labels),
        diff_report_degenerate=diff_report.degenerate_flag,
        diff_report_alignment_failed=diff_report.alignment_failed,
        path_b_payload_ratio=diff_report.payload_ratio,
        rule_matched_entries=tuple(apply_result.matched_entries),
        rule_unmatched_entries=tuple(apply_result.unmatched_entries),
        path_b_labels=tuple(path_b_labels),
        path_a_rule_labels=tuple(path_a_rule_labels),
        final_labels=tuple(final_labels),
        notes=tuple(notes),
    )


def write_cs_reports_jsonl(
    reports: Iterable[CsCrossValidationReport], path: Path
) -> None:
    """Serialize a batch of reports as one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for report in reports:
            fh.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _decide(
    *,
    diff_report: DiffReport,
    path_b_labels: List[SyntheticLabel],
    path_a_rule_labels: List[SyntheticLabel],
    solid_threshold: float,
    review_threshold: float,
    notes: List[str],
) -> tuple[str, bool, List[SyntheticLabel], Optional[CrossValidationResult]]:
    """Apply the decision matrix; mutates ``notes`` with reasoning."""

    # 1. Path B alignment failed -> rule-only, manual review.
    if diff_report.alignment_failed:
        notes.append(
            "Path B alignment_failed (no entry overlap at all); falling back to rule-only labels."
        )
        return (
            DECISION_RULE_ONLY_ALIGNMENT_FAILED,
            True,
            list(path_a_rule_labels),
            None,
        )

    # 2. Path B degenerate -> rule-only, manual review.
    if diff_report.degenerate_flag:
        notes.append(
            f"Path B degenerate (payload_ratio={diff_report.payload_ratio:.4f} > 0.95); "
            "falling back to rule-only labels."
        )
        return (
            DECISION_RULE_ONLY_DEGENERATE,
            True,
            list(path_a_rule_labels),
            None,
        )

    # Normal case -- Path B has byte-level labels.
    # 3. Rule produced nothing -> Path B only, flag for review (rule
    #    corpus may be incomplete for this packer variant).
    if not path_a_rule_labels:
        if not path_b_labels:
            notes.append(
                "Neither Path B nor Path A-rule produced any labels; no signal."
            )
            return (DECISION_NO_SIGNAL, True, [], None)
        notes.append(
            "Rule engine matched no entries; trusting Path B but flagging for review "
            "(rule corpus may be stale or packer variant not yet catalogued)."
        )
        return (
            DECISION_PATH_B_ONLY_NO_RULE_MATCH,
            True,
            list(path_b_labels),
            None,
        )

    # 4. Both paths produced labels -> compare IoU.
    iou_res = cross_validate(
        path_a_rule_labels,  # A side = rule (reference) -- matches the
        path_b_labels,       # B side = diff (primary) -- matches the
        solid_threshold=solid_threshold,
        review_threshold=review_threshold,
    )

    if iou_res.iou >= solid_threshold:
        notes.append(
            f"IoU={iou_res.iou:.4f} >= solid_threshold={solid_threshold}; "
            "both paths agree -- Path B labels accepted as solid."
        )
        return (DECISION_SOLID, False, list(path_b_labels), iou_res)

    if iou_res.iou >= review_threshold:
        notes.append(
            f"IoU={iou_res.iou:.4f} in [{review_threshold}, {solid_threshold}); "
            "partial mismatch -- emitting Path B labels but flagging for manual review."
        )
        return (
            DECISION_PARTIAL_MISMATCH,
            True,
            list(path_b_labels),
            iou_res,
        )

    notes.append(
        f"IoU={iou_res.iou:.4f} < review_threshold={review_threshold}; "
        "low confidence -- dropping all labels and flagging for manual review."
    )
    return (DECISION_LOW_CONFIDENCE, True, [], iou_res)


__all__ = [
    "CsCrossValidationReport",
    "DECISION_LOW_CONFIDENCE",
    "DECISION_NO_SIGNAL",
    "DECISION_PARTIAL_MISMATCH",
    "DECISION_PATH_B_ONLY_NO_RULE_MATCH",
    "DECISION_RULE_ONLY_ALIGNMENT_FAILED",
    "DECISION_RULE_ONLY_DEGENERATE",
    "DECISION_SOLID",
    "cross_validate_commercial_packer",
    "write_cs_reports_jsonl",
]
