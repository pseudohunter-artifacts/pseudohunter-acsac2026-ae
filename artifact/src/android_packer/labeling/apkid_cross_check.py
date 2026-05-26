"""Cross-check our Path A / Path B labels against APKiD (rednaga/APKiD).

This is the **B-g-4** deliverable. APKiD is a third-party YARA-based
packer/protector identifier; we use it as an *independent* check against
our own ``transform_family`` values:

- For Track B packed APKs we expect APKiD to identify the packer family
  (``packer`` or ``protector`` rule category) that matches the one we
  injected (Path A) or diffed (Path B).
- For Track A benign APKs we expect APKiD to *not* report any packer
  (any ``packer`` / ``protector`` hit on a benign APK is surfaced as an
  APKID_FALSE_POSITIVE finding for manual review).

APKiD is GPL-licensed, so we *never* import it as a library; we always
invoke it as an external CLI via ``subprocess.run(...)`` and parse its
``-j / --json`` output. This keeps APKiD in the "external tool" category
alongside ``binwalk`` / ``nmap`` and avoids polluting our distribution
license.

The APKiD JSON schema (verified on v3.1.0, 2026-04-30) is::

    {
        "apkid_version": "3.1.0",
        "files": [
            {
                "filename": "<apk_path>" or "<apk_path>!<inner_entry>",
                "matches": {
                    "<category>": ["<hit1>", "<hit2>", ...]
                }
            },
            ...
        ],
        "rules_sha256": "<hex>"
    }

Where ``<category>`` is one of
``packer / protector / obfuscator / manipulator / anti_vm / anti_debug /
anti_disassembly / abnormal / compiler / device_type``. For
cross-checking we only look at ``packer`` and ``protector`` hits --
everything else is metadata.

A separate YAML mapping file (``configs/data/apkid_family_map.yaml``)
translates APKiD hit strings (e.g. ``"Bangcle"`` / ``"Tencent's Legu"``)
to our canonical ``transform_family`` slugs (e.g. ``packer_cs3_bangcle``
/ ``packer_cs4_tencent_legu``). The mapping is intentionally decoupled
from the code: when APKiD adds new rules, we only need to append a line
to the YAML, not re-release code.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Public constants -- APKiD JSON schema
# ---------------------------------------------------------------------------

#: APKiD rule categories we care about for cross-checking.
APKID_CATEGORY_PACKER = "packer"
APKID_CATEGORY_PROTECTOR = "protector"

#: Full set of APKiD categories (informational; we keep but do not
#: cross-check non-packer categories).
APKID_CATEGORIES_ALL: Tuple[str, ...] = (
    "packer",
    "protector",
    "obfuscator",
    "manipulator",
    "anti_vm",
    "anti_debug",
    "anti_disassembly",
    "abnormal",
    "compiler",
    "device_type",
)

#: Categories that, if hit, indicate a packer/protector was present.
APKID_PACKER_LIKE_CATEGORIES: Tuple[str, ...] = (
    APKID_CATEGORY_PACKER,
    APKID_CATEGORY_PROTECTOR,
)


# ---------------------------------------------------------------------------
# Agreement values (what the cross-check concluded)
# ---------------------------------------------------------------------------

AGREEMENT_SOLID = "solid"
"""APKiD identified exactly the packer family we expected (or compatible)."""

AGREEMENT_MISMATCH = "mismatch"
"""APKiD identified a packer, but a DIFFERENT family than we expected."""

AGREEMENT_NO_APKID_DETECTION = "no_apkid_detection"
"""We expected a packer, but APKiD found no packer/protector hits.

This is common for packers not covered by APKiD's rule set yet (e.g.,
new Gen3 open-source tools) and is surfaced as ``needs_manual_review``
but not an outright failure.
"""

AGREEMENT_NO_EXPECTATION = "no_expectation"
"""No expected family (benign APK); APKiD also found no packer hits."""

AGREEMENT_APKID_FALSE_POSITIVE = "apkid_false_positive"
"""Benign APK (no expected family) but APKiD hit a packer rule anyway."""

AGREEMENT_APKID_FAILED = "apkid_failed"
"""APKiD invocation failed (CLI missing, timeout, malformed JSON)."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ApkidError(RuntimeError):
    """Raised when we cannot produce an ``ApkidResult`` at all.

    Typically means the CLI binary is missing *and* the caller didn't
    ask for graceful degradation.
    """


class ApkidFamilyMapError(ValueError):
    """Raised for malformed ``apkid_family_map.yaml`` content."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApkidMatch:
    """One hit from an APKiD scan."""

    #: Filename as APKiD reports it (``<apk>`` or ``<apk>!<entry>``).
    filename: str
    #: Rule category (``packer``, ``protector``, ``compiler``, etc.).
    category: str
    #: Rule hit string (e.g. ``"Bangcle"``, ``"r8"``).
    hit: str

    def is_packer_like(self) -> bool:
        return self.category in APKID_PACKER_LIKE_CATEGORIES


@dataclass(frozen=True)
class ApkidResult:
    """Parsed and normalised output of a single ``apkid -j <apk>`` run."""

    apkid_version: str
    rules_sha256: str
    matches: Tuple[ApkidMatch, ...]
    #: Raw JSON text, kept for provenance and debugging. Never parsed
    #: downstream except via :func:`parse_apkid_json`.
    raw_json: str = ""
    #: Set when the run was not fully successful; caller decides whether
    #: to treat this as fatal.
    error: Optional[str] = None

    def packer_like_matches(self) -> Tuple[ApkidMatch, ...]:
        return tuple(m for m in self.matches if m.is_packer_like())

    def to_dict(self) -> dict:
        return {
            "apkid_version": self.apkid_version,
            "rules_sha256": self.rules_sha256,
            "matches": [
                {"filename": m.filename, "category": m.category, "hit": m.hit}
                for m in self.matches
            ],
            "error": self.error,
        }


@dataclass(frozen=True)
class ApkidCrossCheckReport:
    """Outcome of ``expected_family`` vs APKiD for a single APK."""

    apk_id: str
    apk_path: str
    expected_family: Optional[str]
    detected_families: Tuple[str, ...]
    agreement: str
    needs_manual_review: bool
    has_packer_hit: bool
    has_protector_hit: bool
    apkid_result: ApkidResult
    notes: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "apk_id": self.apk_id,
            "apk_path": self.apk_path,
            "expected_family": self.expected_family,
            "detected_families": list(self.detected_families),
            "agreement": self.agreement,
            "needs_manual_review": self.needs_manual_review,
            "has_packer_hit": self.has_packer_hit,
            "has_protector_hit": self.has_protector_hit,
            "apkid_result": self.apkid_result.to_dict(),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_apkid_json(json_text: str) -> ApkidResult:
    """Parse an ``apkid -j`` stdout blob into an ``ApkidResult``.

    Raises ``ApkidError`` on malformed input (not a dict, missing keys,
    unexpected value types). Empty / whitespace-only input is treated
    as an error as well.
    """
    text = json_text.strip()
    if not text:
        raise ApkidError("apkid JSON output is empty")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ApkidError(f"apkid JSON output is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ApkidError(f"apkid JSON top-level is not an object: {type(payload).__name__}")

    version = str(payload.get("apkid_version", "")).strip()
    rules_sha = str(payload.get("rules_sha256", "")).strip()
    files = payload.get("files", [])

    if not isinstance(files, list):
        raise ApkidError(f"apkid JSON 'files' is not a list: {type(files).__name__}")

    matches: List[ApkidMatch] = []
    for i, f in enumerate(files):
        if not isinstance(f, dict):
            raise ApkidError(f"apkid JSON files[{i}] is not an object")
        filename = str(f.get("filename", ""))
        if not filename:
            raise ApkidError(f"apkid JSON files[{i}] has empty filename")
        raw_matches = f.get("matches", {}) or {}
        if not isinstance(raw_matches, dict):
            raise ApkidError(
                f"apkid JSON files[{i}].matches is not an object: "
                f"{type(raw_matches).__name__}"
            )
        for category, hits in raw_matches.items():
            category = str(category)
            if not isinstance(hits, list):
                raise ApkidError(
                    f"apkid JSON files[{i}].matches[{category}] is not a list"
                )
            for hit in hits:
                matches.append(
                    ApkidMatch(
                        filename=filename,
                        category=category,
                        hit=str(hit),
                    )
                )

    return ApkidResult(
        apkid_version=version,
        rules_sha256=rules_sha,
        matches=tuple(matches),
        raw_json=text,
    )


# ---------------------------------------------------------------------------
# Subprocess wrapper
# ---------------------------------------------------------------------------


def run_apkid(
    apk_path: Path | str,
    *,
    apkid_cmd: str = "apkid",
    timeout: float = 60.0,
    graceful: bool = True,
) -> ApkidResult:
    """Invoke the ``apkid`` CLI on one APK and return the parsed result.

    Parameters
    ----------
    apk_path:
        Path to the APK to scan. Passed as-is to ``apkid``; APKiD also
        recursively scans nested dex files by default.
    apkid_cmd:
        Override the CLI binary (useful for tests and for environments
        where ``apkid`` is installed under a different name).
    timeout:
        Seconds. APKiD's internal default YARA timeout is 30s; we add a
        small safety margin on top.
    graceful:
        If True (default), transport-level failures (CLI missing,
        timeout, non-zero exit) are captured into ``ApkidResult.error``
        rather than raised. This lets the caller still record a
        ``needs_manual_review`` report. If False, raises ``ApkidError``
        on any failure.
    """
    apk = Path(apk_path)
    if not apk.exists():
        msg = f"APK does not exist: {apk}"
        if graceful:
            return ApkidResult(
                apkid_version="",
                rules_sha256="",
                matches=(),
                raw_json="",
                error=msg,
            )
        raise ApkidError(msg)

    try:
        completed = subprocess.run(
            [apkid_cmd, "-j", str(apk)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        msg = f"apkid CLI not found: {exc}"
        if graceful:
            return ApkidResult(
                apkid_version="",
                rules_sha256="",
                matches=(),
                raw_json="",
                error=msg,
            )
        raise ApkidError(msg) from exc
    except subprocess.TimeoutExpired as exc:
        msg = f"apkid timed out after {timeout}s on {apk}"
        if graceful:
            return ApkidResult(
                apkid_version="",
                rules_sha256="",
                matches=(),
                raw_json="",
                error=msg,
            )
        raise ApkidError(msg) from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode != 0 and not stdout.strip():
        msg = (
            f"apkid exit={completed.returncode}; stderr={stderr.strip()[:200]}"
        )
        if graceful:
            return ApkidResult(
                apkid_version="",
                rules_sha256="",
                matches=(),
                raw_json=stdout,
                error=msg,
            )
        raise ApkidError(msg)

    try:
        result = parse_apkid_json(stdout)
    except ApkidError as exc:
        if graceful:
            return ApkidResult(
                apkid_version="",
                rules_sha256="",
                matches=(),
                raw_json=stdout,
                error=str(exc),
            )
        raise

    # Attach any non-fatal stderr as a note on the result.
    if completed.returncode != 0 and stderr.strip():
        return dataclasses.replace(
            result,
            error=(
                f"apkid exit={completed.returncode} (non-fatal); "
                f"stderr={stderr.strip()[:200]}"
            ),
        )
    return result


# ---------------------------------------------------------------------------
# Family mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApkidFamilyMap:
    """Case-insensitive mapping from APKiD hit string to transform_family.

    Example YAML::

        schema_version: 1
        mappings:
          # key = APKiD hit, value = our transform_family slug
          "Bangcle": packer_cs3_bangcle
          "Bangcle (SecShell)": packer_cs3_bangcle
          "Ijiami": packer_cs2_ijiami
          "Qihoo 360 Packer": packer_cs1_360_jiagu
          "Tencent's Legu": packer_cs4_tencent_legu
          "DexProtector": packer_cs5_dexprotector
    """

    mappings: Mapping[str, str]

    def lookup(self, apkid_hit: str) -> Optional[str]:
        """Case-insensitive exact match first, then lowercase substring."""
        if not apkid_hit:
            return None
        key = apkid_hit.strip().lower()
        if not self._ci_cache:
            # build once on first call (dataclass is frozen -> object.__setattr__)
            object.__setattr__(
                self,
                "_ci_cache",
                {k.strip().lower(): v for k, v in self.mappings.items()},
            )
        cache = self._ci_cache  # type: ignore[attr-defined]
        if key in cache:
            return cache[key]
        # Substring fallback for APKiD hits like "Bangcle v2 (new)".
        for mk, mv in cache.items():
            if mk and mk in key:
                return mv
        return None

    def __post_init__(self) -> None:
        # Initialize empty cache; populated lazily.
        object.__setattr__(self, "_ci_cache", {})


def load_apkid_family_map(path: Path | str) -> ApkidFamilyMap:
    """Load ``apkid_family_map.yaml`` from disk.

    Defers the ``yaml`` import so that importing this module doesn't
    require PyYAML if the caller never uses the map file.
    """
    import yaml  # local import -- optional dep at module scope

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"apkid family map not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return apkid_family_map_from_dict(data, source=str(p))


def apkid_family_map_from_dict(
    data: object, *, source: str = "<dict>"
) -> ApkidFamilyMap:
    """Normalise a loaded dict into an ``ApkidFamilyMap``; strict schema."""
    if not isinstance(data, dict):
        raise ApkidFamilyMapError(
            f"{source}: top-level must be a mapping, got {type(data).__name__}"
        )
    schema = data.get("schema_version", 1)
    if schema != 1:
        raise ApkidFamilyMapError(
            f"{source}: unsupported schema_version={schema!r}, expected 1"
        )
    mappings = data.get("mappings", {})
    if not isinstance(mappings, dict):
        raise ApkidFamilyMapError(
            f"{source}: 'mappings' must be a dict, got {type(mappings).__name__}"
        )
    out: Dict[str, str] = {}
    for k, v in mappings.items():
        if not isinstance(k, str) or not k.strip():
            raise ApkidFamilyMapError(f"{source}: map key must be non-empty str, got {k!r}")
        if not isinstance(v, str) or not v.strip():
            raise ApkidFamilyMapError(
                f"{source}: map value for {k!r} must be non-empty str, got {v!r}"
            )
        out[k] = v
    return ApkidFamilyMap(mappings=out)


# ---------------------------------------------------------------------------
# Cross-check entry point
# ---------------------------------------------------------------------------


def _extract_detected_families(
    result: ApkidResult, family_map: Optional[ApkidFamilyMap]
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return ``(mapped_families, unmapped_hits)``.

    Mapped families are deduplicated and order-stable (insertion order).
    Unmapped hits are APKiD hit strings with no entry in ``family_map``.
    """
    seen: Dict[str, None] = {}
    unmapped: List[str] = []
    for m in result.packer_like_matches():
        fam: Optional[str] = None
        if family_map is not None:
            fam = family_map.lookup(m.hit)
        if fam is None:
            if m.hit not in unmapped:
                unmapped.append(m.hit)
        else:
            seen.setdefault(fam, None)
    return tuple(seen.keys()), tuple(unmapped)


def cross_check(
    apkid_result: ApkidResult,
    *,
    apk_id: str,
    apk_path: Path | str,
    expected_family: Optional[str],
    family_map: Optional[ApkidFamilyMap] = None,
) -> ApkidCrossCheckReport:
    """Compute the cross-check decision for one APK."""
    if not apk_id:
        raise ValueError("apk_id must be non-empty")
    notes: List[str] = []

    if apkid_result.error:
        notes.append(f"apkid error: {apkid_result.error}")

    packer_hits = tuple(
        m for m in apkid_result.packer_like_matches()
        if m.category == APKID_CATEGORY_PACKER
    )
    protector_hits = tuple(
        m for m in apkid_result.packer_like_matches()
        if m.category == APKID_CATEGORY_PROTECTOR
    )
    detected_families, unmapped_hits = _extract_detected_families(
        apkid_result, family_map
    )

    has_packer_hit = bool(packer_hits)
    has_protector_hit = bool(protector_hits)
    any_hit = has_packer_hit or has_protector_hit

    expected = (expected_family or "").strip() or None

    if apkid_result.error and not any_hit:
        agreement = AGREEMENT_APKID_FAILED
        needs_review = True
        if expected is not None:
            notes.append(
                f"expected packer family {expected!r} but apkid did not run; "
                "retry or install apkid"
            )
    elif expected is None and not any_hit:
        agreement = AGREEMENT_NO_EXPECTATION
        needs_review = False
    elif expected is None and any_hit:
        agreement = AGREEMENT_APKID_FALSE_POSITIVE
        needs_review = True
        notes.append(
            "benign apk (no expected family) but apkid identified packer/protector: "
            + ", ".join(sorted({m.hit for m in packer_hits + protector_hits}))
        )
    elif expected is not None and not any_hit:
        agreement = AGREEMENT_NO_APKID_DETECTION
        needs_review = True
        notes.append(
            f"expected packer family {expected!r} but apkid rule set did not match; "
            "either apkid lacks a rule for this packer or the injection is too subtle"
        )
    else:
        # expected is not None and we have at least one packer-like hit
        if expected in detected_families:
            agreement = AGREEMENT_SOLID
            needs_review = False
            if unmapped_hits:
                notes.append(
                    "additional unmapped apkid hits (informational): "
                    + ", ".join(unmapped_hits)
                )
        else:
            agreement = AGREEMENT_MISMATCH
            needs_review = True
            if detected_families:
                notes.append(
                    f"expected {expected!r} but apkid mapped to "
                    + ", ".join(detected_families)
                )
            if unmapped_hits:
                notes.append(
                    f"expected {expected!r} but apkid produced unmapped hits: "
                    + ", ".join(unmapped_hits)
                    + " (update apkid_family_map.yaml if appropriate)"
                )

    return ApkidCrossCheckReport(
        apk_id=apk_id,
        apk_path=str(apk_path),
        expected_family=expected,
        detected_families=detected_families,
        agreement=agreement,
        needs_manual_review=needs_review,
        has_packer_hit=has_packer_hit,
        has_protector_hit=has_protector_hit,
        apkid_result=apkid_result,
        notes=tuple(notes),
    )


def cross_check_apk(
    apk_path: Path | str,
    *,
    apk_id: str,
    expected_family: Optional[str],
    family_map: Optional[ApkidFamilyMap] = None,
    apkid_cmd: str = "apkid",
    timeout: float = 60.0,
    graceful: bool = True,
) -> ApkidCrossCheckReport:
    """Convenience: run APKiD then ``cross_check`` in one call."""
    result = run_apkid(
        apk_path, apkid_cmd=apkid_cmd, timeout=timeout, graceful=graceful
    )
    return cross_check(
        result,
        apk_id=apk_id,
        apk_path=apk_path,
        expected_family=expected_family,
        family_map=family_map,
    )


# ---------------------------------------------------------------------------
# JSONL writer (matches cs_cross_validate's write_cs_reports_jsonl style)
# ---------------------------------------------------------------------------


def write_apkid_reports_jsonl(
    reports: Iterable[ApkidCrossCheckReport], output_path: Path | str
) -> Path:
    """Write reports as one JSON object per line; atomic replace on success."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in reports:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    tmp.replace(out)
    return out
