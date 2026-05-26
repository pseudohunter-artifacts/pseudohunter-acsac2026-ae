"""APKiD reference baseline for Android packer detection.

APKiD (https://github.com/rednaga/APKiD) is the community-recognised,
YARA-backed rule engine for identifying Android packers, protectors,
obfuscators and compilers. It is the *reported* rule baseline for this
project: the rules are curated by the RedNaga security community rather
than by us, which makes it a methodologically honest comparison point
for the paper.

Scope limitations (by design, not a bug):

- APKiD detects at the *APK granularity*. It does not produce object- or
  region-level localization. The report will therefore populate only
  APK-level metrics, with ``localization_granularity = "apk_only"``.
- APKiD relies on fingerprints of known packer families. On our
  synthetic APKs, which carry no real packer magic, its recall will be
  low by construction. This is a feature of the experiment, not a flaw:
  it demonstrates that existing rule-based tools cannot generalise to
  unseen packer variants, which is precisely the motivation for
  learning-based object/region-level localization.

The module never imports :mod:`apkid` at module load time; the heavy
dependency is lazily imported on first invocation so that the rest of
:mod:`android_packer.baselines` stays importable without the optional
extra installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional, Sequence

from android_packer.evaluation import binary_classification_metrics


# Categories inside the ``matches`` dict that APKiD emits for each scanned
# file. The default config treats the first four as "positive" evidence
# of packing / hardening; ``compiler`` and ``anti_vm`` are informational
# by default but can be promoted via config.
_PACKER_CATEGORIES = (
    "packer",
    "protector",
    "obfuscator",
    "manipulator",
)
_AUX_CATEGORIES = (
    "anti_vm",
    "anti_disassembly",
    "anti_debug",
)


class ApkidNotInstalledError(RuntimeError):
    """Raised when the optional ``apkid`` package is not importable.

    The baseline ships as an optional extra so that users who only want
    the entropy / sanity-rules baselines do not need to pull in yara.
    """


@dataclass(frozen=True)
class ApkidBaselineConfig:
    """Configuration for the APKiD-backed APK-level baseline.

    Attributes:
        include_aux_categories: If True, matches in ``anti_vm`` /
            ``anti_disassembly`` / ``anti_debug`` also count towards the
            positive score. Defaults to False so that only strong
            "packer family" evidence drives the decision, matching how
            APKiD is typically interpreted in the literature.
        min_hits: An APK is predicted positive iff at least this many
            distinct (category, family) matches are present. Defaults to
            1: any identified packer family flips the decision.
        timeout_seconds: Per-APK upper bound for the APKiD scan. A run
            that exceeds it is treated as a scan failure, producing a
            negative prediction and a diagnostic note in the report.
    """

    include_aux_categories: bool = False
    min_hits: int = 1
    timeout_seconds: float = 120.0

    def positive_categories(self) -> tuple[str, ...]:
        if self.include_aux_categories:
            return _PACKER_CATEGORIES + _AUX_CATEGORIES
        return _PACKER_CATEGORIES


@dataclass(frozen=True)
class ApkidMatch:
    """A single (category, family) match attributed to a file inside the APK."""

    filename: str
    category: str
    family: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ApkidApkPrediction:
    """APK-level prediction produced by the APKiD baseline."""

    apk_id: str
    apk_path: str
    score: int
    threshold: int
    predicted_label_id: int
    true_label_id: int
    matches: List[ApkidMatch] = field(default_factory=list)
    detected_families: List[str] = field(default_factory=list)
    scan_ok: bool = True
    scan_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "apk_id": self.apk_id,
            "apk_path": self.apk_path,
            "score": self.score,
            "threshold": self.threshold,
            "predicted_label_id": self.predicted_label_id,
            "true_label_id": self.true_label_id,
            "matches": [m.to_dict() for m in self.matches],
            "detected_families": list(self.detected_families),
            "scan_ok": self.scan_ok,
            "scan_error": self.scan_error,
        }


@dataclass(frozen=True)
class ApkidBaselineResult:
    apk_predictions: List[ApkidApkPrediction]
    report: dict


# Type alias for an injected scan function. Tests pass a fake here so
# that the baseline can be exercised without installing APKiD at all.
ApkidScanFn = Callable[[Path, float], Mapping]


def run_apkid_baseline(
    apk_entries: Iterable[Mapping],
    config: Optional[ApkidBaselineConfig] = None,
    scan_fn: Optional[ApkidScanFn] = None,
) -> ApkidBaselineResult:
    """Run the APKiD baseline over a list of APK entries.

    Args:
        apk_entries: Iterable of mappings with at least ``apk_id``,
            ``apk_path`` and ``true_label_id`` keys. ``apk_path`` must
            point to an existing APK file on disk.
        config: Baseline configuration. Uses defaults when omitted.
        scan_fn: Optional injection point for the scanning backend. When
            provided it replaces the bundled APKiD caller entirely,
            which keeps the tests completely decoupled from the real
            APKiD runtime. The callable must accept ``(apk_path,
            timeout_seconds)`` and return an APKiD-shaped JSON mapping
            (``{"files": [{"filename": ..., "matches": {...}}, ...]}``).
    """

    config = config or ApkidBaselineConfig()
    _validate_config(config)

    scan = scan_fn if scan_fn is not None else _default_scan_fn()
    positive_cats = config.positive_categories()

    predictions: list[ApkidApkPrediction] = []
    scan_failures = 0

    for entry in apk_entries:
        apk_id = str(entry["apk_id"])
        apk_path = str(entry["apk_path"])
        true_label_id = int(entry.get("true_label_id", 0))

        scan_ok = True
        scan_error: Optional[str] = None
        matches: list[ApkidMatch] = []
        try:
            payload = scan(Path(apk_path), config.timeout_seconds)
            matches = _flatten_matches(payload, positive_cats)
        except Exception as exc:  # noqa: BLE001 - any backend error is reported
            scan_ok = False
            scan_error = f"{type(exc).__name__}: {exc}"
            scan_failures += 1

        families = sorted({m.family for m in matches})
        score = len(matches)
        predicted = 1 if scan_ok and score >= config.min_hits else 0

        predictions.append(
            ApkidApkPrediction(
                apk_id=apk_id,
                apk_path=apk_path,
                score=score,
                threshold=config.min_hits,
                predicted_label_id=predicted,
                true_label_id=true_label_id,
                matches=matches,
                detected_families=families,
                scan_ok=scan_ok,
                scan_error=scan_error,
            )
        )

    report = _build_report(predictions, config, scan_failures)
    return ApkidBaselineResult(apk_predictions=predictions, report=report)


def _validate_config(config: ApkidBaselineConfig) -> None:
    if config.min_hits < 1:
        raise ValueError(f"min_hits must be >= 1, got {config.min_hits}")
    if config.timeout_seconds <= 0:
        raise ValueError(
            f"timeout_seconds must be positive, got {config.timeout_seconds}"
        )


def _flatten_matches(
    payload: Mapping, positive_categories: Sequence[str]
) -> List[ApkidMatch]:
    """Flatten APKiD's nested JSON into a flat list of hits."""

    files = payload.get("files") or []
    flat: list[ApkidMatch] = []
    for file_entry in files:
        filename = str(file_entry.get("filename", ""))
        raw_matches = file_entry.get("matches") or {}
        for category, families in raw_matches.items():
            if category not in positive_categories:
                continue
            for family in families or []:
                flat.append(
                    ApkidMatch(
                        filename=filename,
                        category=str(category),
                        family=str(family),
                    )
                )
    return flat


def _build_report(
    predictions: Sequence[ApkidApkPrediction],
    config: ApkidBaselineConfig,
    scan_failures: int,
) -> dict:
    truth = [row.true_label_id for row in predictions]
    hard = [row.predicted_label_id for row in predictions]
    # APKiD's "score" is a non-negative integer match count. That is a
    # sensible ordering for AUROC even though it's not a probability.
    scores = [float(row.score) for row in predictions]

    family_histogram: dict[str, int] = {}
    for row in predictions:
        for family in row.detected_families:
            family_histogram[family] = family_histogram.get(family, 0) + 1

    return {
        "baseline": "apkid",
        # Critically signal to consumers that this baseline only reports
        # at the APK level; do NOT synthesise object/region-level
        # predictions from it.
        "localization_granularity": "apk_only",
        "config": asdict(config),
        "counts": {
            "apks": len(predictions),
            "scan_failures": scan_failures,
        },
        "metrics": {
            "apk": binary_classification_metrics(
                truth=truth,
                predictions=hard,
                scores=scores,
            ).to_dict(),
        },
        "detected_family_histogram": dict(sorted(family_histogram.items())),
    }


def _default_scan_fn() -> ApkidScanFn:
    """Build the real APKiD-backed scan function.

    Supports both APKiD 2.x (``Scanner(rules, output_formatter, timeout)``
    with ``scan()`` returning nothing and writing through the formatter)
    and APKiD 3.x (``Scanner(rules, options)`` where ``scan_file(path)``
    directly returns a ``Dict[filepath, List[yara.Match]]``). The 3.x
    path is preferred when available because it avoids scraping the
    formatter's stdout.

    Raises :class:`ApkidNotInstalledError` the first time it is invoked
    on a machine that does not have the ``apkid`` package available,
    with an actionable hint for the user.
    """

    try:
        from apkid.apkid import Scanner  # type: ignore
        from apkid.rules import RulesManager  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by a dedicated test
        message = (
            "The 'apkid' package is not installed. Install it with "
            "'pip install android-packer[apkid]' (or 'pip install apkid "
            "yara-python'). APKiD 3.x can scan directly after installation; "
            "older 2.x environments may require 'apkid --prepare' to compile "
            "the YARA rule set. On Windows, if the trampoline exe fails to "
            "install, run 'python -c \"from apkid.rules import RulesManager; "
            "RulesManager().compile()\"' once to build the rule cache."
        )
        raise ApkidNotInstalledError(message) from exc

    # Try the 3.x Options API first; fall back to the 2.x
    # OutputFormatter path when Options is absent.
    try:
        from apkid.apkid import Options  # type: ignore
    except ImportError:
        Options = None  # type: ignore

    try:
        rules = RulesManager().load()
    except Exception:
        # Rules not yet compiled: try an in-process compile (works
        # even when `apkid --prepare` cannot be invoked, e.g. Windows
        # pip installs where the trampoline exe was rejected).
        rules = RulesManager().compile()

    if Options is not None:
        # --- APKiD 3.x path ---
        # Scanner(rules, options) + scan_file(path) returns
        # Dict[str, List[yara.Match]] directly.
        def scan(apk_path: Path, timeout_seconds: float) -> Mapping:
            options = Options(
                timeout=max(1, int(timeout_seconds)),
                json=False,           # We process matches directly, no stdout.
                output_dir=None,
                typing="magic",
                scan_depth=2,
                recursive=False,
                include_types=False,
            )
            scanner = Scanner(rules=rules, options=options)
            raw = scanner.scan_file(str(apk_path))
            # Convert the 3.x dict into the 2.x ``{files: [...]}``
            # structure expected by _normalise_apkid_result so the
            # downstream code remains stable across APKiD versions.
            files_payload = []
            for filename, matches in (raw or {}).items():
                files_payload.append({
                    "filename": filename,
                    "matches": [
                        {
                            "rule": getattr(m, "rule", str(m)),
                            "tags": list(getattr(m, "tags", ()) or ()),
                        }
                        for m in matches
                    ],
                })
            shaped = {"files": files_payload}
            return _normalise_apkid_result(shaped, apk_path)

        return scan

    # --- APKiD 2.x fallback ---
    from apkid.output import OutputFormatter  # type: ignore

    formatter = OutputFormatter(
        json_output=True,
        output_dir=None,
        rules_manager=RulesManager(),
        include_types=False,
    )
    scanner = Scanner(rules=rules, output_formatter=formatter, timeout=0)

    def scan(apk_path: Path, timeout_seconds: float) -> Mapping:
        # APKiD 2.x's Scanner.scan returns nothing directly; it writes
        # JSON to the formatter. Build a minimal wrapper that collects
        # the JSON for the requested APK.
        scanner.timeout = int(timeout_seconds)
        results = scanner.scan(str(apk_path))
        return _normalise_apkid_result(results, apk_path)

    return scan


def _normalise_apkid_result(results, apk_path: Path) -> Mapping:
    """Coerce APKiD's output into the ``{files: [...]}`` shape we expect.

    Different APKiD versions emit slightly different structures; we
    depend only on the ``files`` -> ``filename`` / ``matches`` pattern,
    which has been stable since 2.x.
    """

    if isinstance(results, Mapping):
        return results  # already in the expected shape
    # Fallback: best-effort construction from an iterable of (filename,
    # matches) pairs. If this ever fails, surface a scan error so the
    # report records the failure rather than masking it.
    try:
        files = []
        for item in results:
            filename = item.get("filename") if isinstance(item, Mapping) else None
            matches = item.get("matches") if isinstance(item, Mapping) else None
            files.append({"filename": filename or str(apk_path), "matches": matches or {}})
        return {"files": files}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Unexpected APKiD output shape for {apk_path}: {type(results).__name__}"
        ) from exc


__all__ = [
    "ApkidApkPrediction",
    "ApkidBaselineConfig",
    "ApkidBaselineResult",
    "ApkidMatch",
    "ApkidNotInstalledError",
    "ApkidScanFn",
    "run_apkid_baseline",
]
