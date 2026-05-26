"""Fetch F-Droid origin APKs for PackerGrind's 49-app benchmark.

PackerGrind (Xue et al., ICSE'17 / TSE'22) evaluated 6 commercial Android
packers (Ali, Baidu, Bangcle, Ijiami, Qihoo360, Tencent) on a set of 49
FOSS apps drawn from F-Droid. Their upstream repo only ships the
post-pack static signatures, not the origin APKs. This script reconstructs
the origin side by resolving each PackerGrind app name to an F-Droid
``packageName`` and downloading a reasonable APK version for it.

Pipeline
--------
  1. Load ``data/real_world/packergrind/app_list.json`` (produced by
     ``scripts/data/build_packergrind_index.py``). This gives us the
     canonical 49-app list and the API levels (15, 16) they were tested on.

  2. Download F-Droid ``index-v1.jar`` (cached).

  3. For each PackerGrind app name, resolve it to an F-Droid
     ``packageName`` by scanning ``name``, ``localized.*.name``,
     ``packageName`` (suffix match on last label), and ``autoName``.
     Unresolved apps are recorded but do not fail the batch.

  4. For each resolved app, pick the best APK version: prefer the oldest
     APK whose ``minSdkVersion <= 16`` (closest in spirit to what
     PackerGrind packed); fall back to F-Droid's ``suggestedVersionCode``.

  5. Download via ``curl`` (direct + Tsinghua mirror fallback, matches
     the convention of ``download_fdroid_seed_apks.py``), verify SHA-256
     and size, and emit a manifest at
     ``data/real_world/packergrind/origins_manifest.json``.

Third-party APK bytes never enter git. The manifest is the reproducible
artifact.

Usage::

    # Plan only, using existing cached index.
    python scripts/data/fetch_fdroid_origins.py

    # Actually download.
    python scripts/data/fetch_fdroid_origins.py --execute

    # Custom app list / dirs.
    python scripts/data/fetch_fdroid_origins.py \
        --app-list data/real_world/packergrind/app_list.json \
        --origins-dir data/real_world/packergrind/origins \
        --manifest-out data/real_world/packergrind/origins_manifest.json \
        --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_APP_LIST = REPO_ROOT / "data" / "real_world" / "packergrind" / "app_list.json"
DEFAULT_ORIGINS_DIR = REPO_ROOT / "data" / "real_world" / "packergrind" / "origins"
DEFAULT_MANIFEST_OUT = REPO_ROOT / "data" / "real_world" / "packergrind" / "origins_manifest.json"
DEFAULT_OVERRIDES = REPO_ROOT / "data" / "real_world" / "packergrind" / "resolver_overrides.json"
DEFAULT_INDEX_URL = "https://mirrors.tuna.tsinghua.edu.cn/fdroid/repo/index-v1.jar"
DEFAULT_ARCHIVE_INDEX_URL = "https://mirrors.tuna.tsinghua.edu.cn/fdroid/archive/index-v1.jar"
DEFAULT_REPO_URL = "https://mirrors.tuna.tsinghua.edu.cn/fdroid/repo"
DEFAULT_ARCHIVE_REPO_URL = "https://mirrors.tuna.tsinghua.edu.cn/fdroid/archive"
DEFAULT_FALLBACK_REPO_URL = "https://f-droid.org/repo"
DEFAULT_FALLBACK_ARCHIVE_REPO_URL = "https://f-droid.org/archive"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "real_world" / "packergrind" / ".cache"


# Lowercase normalization: drop non-alphanum so "aRelevation" ~ "arelevation".
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _NORMALIZE_RE.sub("", s.lower())


# Hand-curated resolver overrides. Keys are PackerGrind canonical app names;
# values are ``{"package_name": <fdroid-pkg>, "source_repo": "main"|"archive"}``.
#
# This dict is the SEED set: it was derived by scanning the union of the
# F-Droid main + archive indexes against the 49 PackerGrind app names and
# taking every unambiguous exact / suffix / high-confidence substring hit.
# It is overridden at runtime by ``data/real_world/packergrind/resolver_overrides.json``
# when that file exists, so reviewers can pin ambiguous entries (``coin``,
# ``gestures``, ``n``, etc.) without editing this module.
HAND_OVERRIDES: Dict[str, Dict[str, str]] = {
    "2048": {"package_name": "com.uberspot.a2048", "source_repo": "archive"},
    "PersianCalendar": {"package_name": "com.byagowi.persiancalendar", "source_repo": "main"},
    "Shopt": {"package_name": "eu.domob.shopt2", "source_repo": "main"},
    "Shorty": {"package_name": "org.billthefarmer.shorty", "source_repo": "main"},
    "SignalGenerator": {"package_name": "org.billthefarmer.siggen", "source_repo": "main"},
    "SimpleC25K": {"package_name": "nl.ttys0.simplec25k", "source_repo": "archive"},
    "aRevelation": {"package_name": "com.github.marmalade.aRevelation", "source_repo": "archive"},
    "adsdroid": {"package_name": "hu.vsza.adsdroid", "source_repo": "main"},
    "amplayer": {"package_name": "com.orphan.amplayer", "source_repo": "archive"},
    "androidcpg": {"package_name": "com.isanexusdev.androidcpg", "source_repo": "main"},
    "androidrun": {"package_name": "fr.asterope", "source_repo": "archive"},
    "andstatus": {"package_name": "org.andstatus.app", "source_repo": "main"},
    "birthdroid": {"package_name": "com.rigid.birthdroid", "source_repo": "main"},
    "bjtrainer": {"package_name": "eu.domob.bjtrainer", "source_repo": "archive"},
    "blockinger": {"package_name": "org.blockinger.game", "source_repo": "archive"},
    "calc": {"package_name": "home.jmstudios.calc", "source_repo": "archive"},
    "calculator": {"package_name": "com.simplemobiletools.calculator", "source_repo": "archive"},
    "coin_flip": {"package_name": "com.banasiak.coinflip", "source_repo": "archive"},
    "colorpicker": {"package_name": "com.nauj27.android.colorpicker", "source_repo": "archive"},
    "diary": {"package_name": "jpf.android.diary", "source_repo": "archive"},
    "dudo": {"package_name": "it.ecosw.dudo", "source_repo": "main"},
    "epub3reader": {"package_name": "it.angrydroids.epub3reader", "source_repo": "archive"},
    "fileexplorer": {"package_name": "net.micode.fileexplorer", "source_repo": "archive"},
    "filemanager": {"package_name": "com.cyanogenmod.filemanager.ics", "source_repo": "archive"},
    "h2droid": {"package_name": "com.frankcalise.h2droid", "source_repo": "archive"},
    "holocounter": {"package_name": "com.omegavesko.holocounter", "source_repo": "archive"},
    "holoken": {"package_name": "com.tortuca.holoken", "source_repo": "archive"},
    "hydromemo": {"package_name": "de.boesling.hydromemo", "source_repo": "archive"},
    "igo": {"package_name": "com.idunnololz.igo", "source_repo": "archive"},
    "number_guesser": {"package_name": "com.numguesser.tonio_rpchp.numberguesser", "source_repo": "main"},
    "opentimer": {"package_name": "edu.killerud.kitchentimer", "source_repo": "archive"},
    "passandroid": {"package_name": "org.ligi.passandroid", "source_repo": "main"},
    "passcard": {"package_name": "com.passcard", "source_repo": "main"},
    "persiancalendar": {"package_name": "com.byagowi.persiancalendar", "source_repo": "main"},
    "politedroid": {"package_name": "com.politedroid", "source_repo": "main"},
    "pomodoro": {"package_name": "com.hlidskialf.android.pomodoro", "source_repo": "archive"},
    "postcode": {"package_name": "net.tevp.postcode", "source_repo": "main"},
    "pwdhash": {"package_name": "com.uploadedlobster.PwdHash", "source_repo": "archive"},
    "timetracker": {"package_name": "de.live.gdev.timetracker", "source_repo": "main"},
    # Entries below had ambiguous or no-confidence matches at scan time;
    # flashlight/clip/j/number were high-substring hits but names are generic,
    # so we keep them for now but expect reviewer to replace via resolver_overrides.json.
    "flashlight": {"package_name": "com.abitsinc.andr", "source_repo": "archive", "note": "generic name, review"},
    "clip": {"package_name": "me.lucky.clipeus", "source_repo": "main", "note": "substring match, review"},
    "j": {"package_name": "im.r_c.android.jigsaw", "source_repo": "main", "note": "single-char name, review"},
    "number": {"package_name": "org.uaraven.e", "source_repo": "main", "note": "substring match, review"},
}

# Apps known to be unresolved against the current merged F-Droid index.
# We record them explicitly so downstream tooling (and the CLI summary) can
# surface them as "expected misses" rather than re-scanning each run.
KNOWN_UNRESOLVED = {
    "aRelevation": "likely typo for aRevelation; PackerGrind's own feature dir had both forms",
    "browseropen": "no F-Droid app with this name; vendor demo app",
    "coin": "ambiguous; pick coin_flip explicitly or override via resolver_overrides.json",
    "gestures": "ambiguous; several gesture apps exist, override if needed",
    "n": "single-char name; skip or override",
    "timertracker": "likely typo for timetracker; kept separate because PackerGrind tests both",
}


@dataclass
class ResolutionResult:
    app: str
    package_name: Optional[str]
    match_type: str  # exact | localized_name | suffix | autoname | override | unresolved
    candidate_score: float = 0.0
    note: str = ""


@dataclass
class FetchResult:
    app: str
    package_name: Optional[str]
    resolution: str
    status: str  # planned | downloaded | skipped_already_ok
                  # | sha256_mismatch | http_error | unresolved | no_apk_found
    local_path: Optional[str] = None
    apk_name: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    version_name: Optional[str] = None
    version_code: Optional[int] = None
    min_sdk_version: Optional[int] = None
    target_sdk_version: Optional[int] = None
    downloaded_url: Optional[str] = None
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Index handling                                                               #
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_of(path: Path, *, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _curl_download(
    url: str,
    dst: Path,
    *,
    curl_bin: str,
    proxy: Optional[str],
    connect_timeout: int,
    retry: int,
    retry_delay: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        curl_bin,
        "-L",
        "--fail",
        "--retry",
        str(retry),
        "--retry-delay",
        str(retry_delay),
        "--connect-timeout",
        str(connect_timeout),
        "-o",
        str(dst),
        url,
    ]
    if proxy:
        cmd[1:1] = ["-x", proxy]
    subprocess.run(cmd, check=True)


def _download_with_fallback(
    apk_name: str,
    destination: Path,
    *,
    repo_url: str,
    fallback_repo_url: Optional[str],
    curl_bin: str,
    proxy: Optional[str],
    connect_timeout: int,
    retry: int,
    retry_delay: int,
) -> str:
    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    primary = f"{repo_url.rstrip('/')}/{apk_name}"
    fallback = f"{fallback_repo_url.rstrip('/')}/{apk_name}" if fallback_repo_url else None
    try:
        _curl_download(
            primary,
            tmp,
            curl_bin=curl_bin,
            proxy=proxy,
            connect_timeout=connect_timeout,
            retry=retry,
            retry_delay=retry_delay,
        )
        url_used = primary
    except subprocess.CalledProcessError:
        if not fallback:
            raise
        if tmp.exists():
            tmp.unlink()
        _curl_download(
            fallback,
            tmp,
            curl_bin=curl_bin,
            proxy=proxy,
            connect_timeout=connect_timeout,
            retry=retry,
            retry_delay=retry_delay,
        )
        url_used = fallback
    tmp.replace(destination)
    return url_used


def _load_fdroid_index(jar_path: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(jar_path) as zf:
        return json.loads(zf.read("index-v1.json"))


# --------------------------------------------------------------------------- #
# Resolver                                                                     #
# --------------------------------------------------------------------------- #


def _localized_names(app: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    loc = app.get("localized") or {}
    for lang in ("en-US", "en", "zh-CN", "zh", "de"):
        n = (loc.get(lang) or {}).get("name")
        if n:
            out.append(str(n))
    return out


def resolve_app_name(
    app_name: str,
    apps_by_package: Dict[str, Dict[str, Any]],
    *,
    prefer_override: bool = True,
    external_overrides: Optional[Dict[str, Dict[str, str]]] = None,
) -> ResolutionResult:
    """Resolve a PackerGrind app name to an F-Droid ``packageName``.

    Lookup order:
      1. ``external_overrides`` (from ``resolver_overrides.json``) - reviewer-pinned.
      2. ``HAND_OVERRIDES`` - auto-scanned seed set, shipped in this module.
      3. ``KNOWN_UNRESOLVED`` - short-circuits apps known to have no F-Droid
         counterpart, avoiding an expensive scan.
      4. Fuzzy fallback: exact name / package suffix / substring.
    """
    # Helper: treat either string (legacy) or dict (new) override values.
    def _coerce(entry: Any, match_type: str, note: str) -> ResolutionResult:
        if isinstance(entry, str):
            pkg = entry
            source_repo = None
        elif isinstance(entry, dict):
            pkg = entry.get("package_name")
            source_repo = entry.get("source_repo")
            if entry.get("note"):
                note = f"{note}; {entry['note']}"
        else:
            return ResolutionResult(
                app=app_name,
                package_name=None,
                match_type="unresolved",
                note=f"malformed override entry: {entry!r}",
            )
        if pkg is None:
            return ResolutionResult(
                app=app_name,
                package_name=None,
                match_type="unresolved",
                note="override missing package_name",
            )
        if pkg not in apps_by_package:
            return ResolutionResult(
                app=app_name,
                package_name=None,
                match_type="unresolved",
                note=f"override {pkg!r} not in F-Droid index ({note})",
            )
        result = ResolutionResult(
            app=app_name,
            package_name=pkg,
            match_type=match_type,
            candidate_score=1.0,
            note=note,
        )
        if source_repo:
            # Piggyback via note - FetchResult.extra captures this downstream.
            result.note = f"{note} [source_repo={source_repo}]"
        return result

    if external_overrides and app_name in external_overrides:
        return _coerce(external_overrides[app_name], "override_external", "external override")
    if prefer_override and app_name in HAND_OVERRIDES:
        return _coerce(HAND_OVERRIDES[app_name], "override", "hand-scanned override")
    if app_name in KNOWN_UNRESOLVED:
        return ResolutionResult(
            app=app_name,
            package_name=None,
            match_type="unresolved",
            note=f"known-unresolved: {KNOWN_UNRESOLVED[app_name]}",
        )

    needle = _norm(app_name)
    if not needle:
        return ResolutionResult(app=app_name, package_name=None, match_type="unresolved", note="empty name")

    # Pass 1: exact match against app name / localized name.
    for pkg, app in apps_by_package.items():
        candidates = []
        if app.get("name"):
            candidates.append(str(app["name"]))
        if app.get("autoName"):
            candidates.append(str(app["autoName"]))
        candidates.extend(_localized_names(app))
        for cand in candidates:
            if _norm(cand) == needle:
                return ResolutionResult(
                    app=app_name,
                    package_name=pkg,
                    match_type="exact",
                    candidate_score=1.0,
                    note=f"matched name {cand!r}",
                )

    # Pass 2: package last-label match, e.g. "a2048" in "com.uberspot.a2048".
    for pkg, app in apps_by_package.items():
        last = pkg.split(".")[-1]
        if _norm(last) == needle:
            return ResolutionResult(
                app=app_name,
                package_name=pkg,
                match_type="suffix",
                candidate_score=0.8,
                note=f"matched pkg suffix {last!r}",
            )

    # Pass 3: substring containment on localized name.
    candidates_sub: List[Tuple[float, str, str]] = []
    for pkg, app in apps_by_package.items():
        names = [str(app.get("name") or ""), *_localized_names(app)]
        for n in names:
            nn = _norm(n)
            if not nn:
                continue
            if needle and needle in nn:
                score = len(needle) / len(nn)
                candidates_sub.append((score, pkg, n))
                break
    if candidates_sub:
        candidates_sub.sort(key=lambda x: -x[0])
        s, pkg, n = candidates_sub[0]
        return ResolutionResult(
            app=app_name,
            package_name=pkg,
            match_type="localized_name",
            candidate_score=float(s),
            note=f"substring match on {n!r} score={s:.2f}",
        )

    return ResolutionResult(
        app=app_name,
        package_name=None,
        match_type="unresolved",
        note="no index entry matched",
    )


# --------------------------------------------------------------------------- #
# Version selection                                                            #
# --------------------------------------------------------------------------- #


def select_apk_version(
    versions: List[Dict[str, Any]],
    app: Dict[str, Any],
    *,
    max_min_sdk: int = 16,
) -> Optional[Dict[str, Any]]:
    """Pick an APK version aligned with PackerGrind's evaluation era (Android 4.x-4.4).

    Strategy:
      1. Prefer versions with ``minSdkVersion <= max_min_sdk`` (so it can run on
         the API 15/16 emulator that PackerGrind used).
      2. Among those, prefer the *oldest* ``added`` timestamp to match the
         Xue'17 evaluation window.
      3. Fall back to the F-Droid ``suggestedVersionCode`` entry.
      4. Final fallback: the newest available APK.
    """
    if not versions:
        return None

    low_sdk = [v for v in versions if int(v.get("minSdkVersion") or 999) <= max_min_sdk]
    if low_sdk:
        low_sdk.sort(key=lambda v: int(v.get("added") or 0))
        return low_sdk[0]

    suggested = app.get("suggestedVersionCode")
    if suggested is not None:
        for v in versions:
            if int(v.get("versionCode") or -1) == int(suggested):
                return v

    # Newest by added timestamp.
    sorted_versions = sorted(versions, key=lambda v: -int(v.get("added") or 0))
    return sorted_versions[0] if sorted_versions else None


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-list", type=Path, default=DEFAULT_APP_LIST)
    parser.add_argument("--origins-dir", type=Path, default=DEFAULT_ORIGINS_DIR)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES,
                        help="Optional resolver_overrides.json written by reviewer.")
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL,
                        help="F-Droid MAIN index-v1.jar URL (Tsinghua mirror by default).")
    parser.add_argument("--archive-index-url", default=DEFAULT_ARCHIVE_INDEX_URL,
                        help="F-Droid ARCHIVE index-v1.jar URL (holds delisted PackerGrind-era apps).")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL,
                        help="F-Droid MAIN repo URL for APK downloads.")
    parser.add_argument("--archive-repo-url", default=DEFAULT_ARCHIVE_REPO_URL,
                        help="F-Droid ARCHIVE repo URL for APK downloads.")
    parser.add_argument("--fallback-repo-url", default=DEFAULT_FALLBACK_REPO_URL,
                        help="Fallback MAIN repo URL (default: f-droid.org direct).")
    parser.add_argument("--fallback-archive-repo-url", default=DEFAULT_FALLBACK_ARCHIVE_REPO_URL,
                        help="Fallback ARCHIVE repo URL (default: f-droid.org direct).")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--index-path", type=Path, default=None,
                        help="Override path to the MAIN index-v1.jar cache.")
    parser.add_argument("--archive-index-path", type=Path, default=None,
                        help="Override path to the ARCHIVE index-v1.jar cache.")
    parser.add_argument("--skip-index-download", action="store_true")
    parser.add_argument("--no-archive", action="store_true",
                        help="Skip the F-Droid archive index entirely (main repo only).")
    parser.add_argument("--curl-bin", default="curl.exe" if os.name == "nt" else "curl")
    parser.add_argument("--proxy", default=os.environ.get("FDROID_PROXY"))
    parser.add_argument("--connect-timeout", type=int, default=20)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=2)
    parser.add_argument("--max-min-sdk", type=int, default=16, help="Prefer APKs with minSdkVersion <= this.")
    parser.add_argument("--only", nargs="+", default=None, help="Restrict to these PackerGrind app names.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    dry_run = not args.execute

    if not args.app_list.exists():
        print(f"[ABORT] app list not found: {args.app_list}", file=sys.stderr)
        return 2

    app_list = json.loads(args.app_list.read_text(encoding="utf-8"))
    app_names = sorted(app_list.get("by_app", {}).keys())
    if args.only:
        only = set(args.only)
        app_names = [a for a in app_names if a in only]
    if not app_names:
        print("[ABORT] no apps to process (after --only filter)", file=sys.stderr)
        return 2

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    main_index_path = args.index_path or (args.cache_dir / "index-v1.jar")
    arch_index_path = args.archive_index_path or (args.cache_dir / "index-v1-archive.jar")

    # Decide which indexes we actually need.
    need_main = True
    need_arch = not args.no_archive

    # Download indexes if missing + we're in execute mode.
    if not dry_run and not args.skip_index_download:
        if need_main:
            print(f"[STEP] downloading F-Droid MAIN index -> {main_index_path}")
            _curl_download(
                args.index_url, main_index_path,
                curl_bin=args.curl_bin, proxy=args.proxy,
                connect_timeout=args.connect_timeout,
                retry=args.retry, retry_delay=args.retry_delay,
            )
        if need_arch:
            try:
                print(f"[STEP] downloading F-Droid ARCHIVE index -> {arch_index_path}")
                _curl_download(
                    args.archive_index_url, arch_index_path,
                    curl_bin=args.curl_bin, proxy=args.proxy,
                    connect_timeout=args.connect_timeout,
                    retry=args.retry, retry_delay=args.retry_delay,
                )
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] archive index download failed ({exc}); "
                      "continuing with MAIN only", file=sys.stderr)
                need_arch = False

    # In dry-run mode, refuse to proceed if the main index isn't cached yet.
    if dry_run and not main_index_path.exists() and not args.skip_index_download:
        print(
            f"[DRY-RUN] MAIN index not cached at {main_index_path}; pass --execute "
            f"(or pre-cache and use --skip-index-download) to proceed."
        )
        return 0

    if not main_index_path.exists():
        print(f"[ABORT] MAIN index not available at {main_index_path}", file=sys.stderr)
        return 2

    # Load MAIN + ARCHIVE and merge. MAIN wins when a package appears in both.
    main_index = _load_fdroid_index(main_index_path)
    main_apps = {app["packageName"]: app for app in main_index["apps"]}
    main_packages = main_index["packages"]
    pkg_sources: Dict[str, str] = {pkg: "main" for pkg in main_apps}
    merged_apps: Dict[str, Dict[str, Any]] = dict(main_apps)
    merged_packages: Dict[str, List[Dict[str, Any]]] = dict(main_packages)

    arch_index: Optional[Dict[str, Any]] = None
    arch_sha256: Optional[str] = None
    if need_arch and arch_index_path.exists():
        try:
            arch_index = _load_fdroid_index(arch_index_path)
            arch_sha256 = _sha256_of(arch_index_path)
            arch_apps = {app["packageName"]: app for app in arch_index["apps"]}
            for pkg, app in arch_apps.items():
                if pkg not in merged_apps:
                    merged_apps[pkg] = app
                    pkg_sources[pkg] = "archive"
                    merged_packages[pkg] = arch_index["packages"].get(pkg) or []
        except Exception as exc:
            print(f"[WARN] failed to parse archive index {arch_index_path}: {exc}",
                  file=sys.stderr)

    main_sha256 = _sha256_of(main_index_path)
    print(
        f"[OK] merged F-Droid index: apps={len(merged_apps)} "
        f"(main={len(main_apps)} archive={len(merged_apps) - len(main_apps) if arch_index else 0})"
    )

    # Optional external overrides (reviewer-pinned).
    external_overrides: Optional[Dict[str, Dict[str, str]]] = None
    if args.overrides and args.overrides.exists():
        try:
            external_overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
            print(f"[OK] loaded external overrides from {args.overrides} ({len(external_overrides)} entries)")
        except Exception as exc:
            print(f"[WARN] failed to load overrides {args.overrides}: {exc}",
                  file=sys.stderr)

    args.origins_dir.mkdir(parents=True, exist_ok=True)

    results: List[FetchResult] = []
    generated_at = _utc_now()

    for name in app_names:
        res = resolve_app_name(
            name, merged_apps,
            external_overrides=external_overrides,
        )
        if res.package_name is None:
            results.append(FetchResult(
                app=name, package_name=None, resolution=res.match_type,
                status="unresolved", message=res.note,
            ))
            continue

        versions = merged_packages.get(res.package_name) or []
        app_meta = merged_apps.get(res.package_name, {})
        picked = select_apk_version(versions, app_meta, max_min_sdk=args.max_min_sdk)
        if picked is None:
            results.append(FetchResult(
                app=name, package_name=res.package_name, resolution=res.match_type,
                status="no_apk_found",
                message=f"packages[{res.package_name}] returned no versions",
            ))
            continue

        apk_name = picked["apkName"]
        expected_sha = str(picked["hash"]).lower()
        local = args.origins_dir / apk_name

        # Select repo + fallback based on which index the package came from.
        source_repo = pkg_sources.get(res.package_name, "main")
        if source_repo == "archive":
            repo_url = args.archive_repo_url
            fallback_url = args.fallback_archive_repo_url
        else:
            repo_url = args.repo_url
            fallback_url = args.fallback_repo_url

        common_extra = {
            "source_repo": source_repo,
            "resolution_note": res.note,
        }

        if local.exists() and local.stat().st_size > 0:
            actual = _sha256_of(local).lower()
            if actual == expected_sha:
                results.append(FetchResult(
                    app=name, package_name=res.package_name, resolution=res.match_type,
                    status="skipped_already_ok",
                    local_path=str(local), apk_name=apk_name,
                    sha256=expected_sha, size_bytes=local.stat().st_size,
                    version_name=picked.get("versionName"),
                    version_code=picked.get("versionCode"),
                    min_sdk_version=picked.get("minSdkVersion"),
                    target_sdk_version=picked.get("targetSdkVersion"),
                    downloaded_url=f"{repo_url.rstrip('/')}/{apk_name}",
                    message="already on disk, hash matches",
                    extra=common_extra,
                ))
                continue

        if dry_run:
            results.append(FetchResult(
                app=name, package_name=res.package_name, resolution=res.match_type,
                status="planned",
                local_path=str(local), apk_name=apk_name,
                sha256=expected_sha, size_bytes=int(picked.get("size") or 0),
                version_name=picked.get("versionName"),
                version_code=picked.get("versionCode"),
                min_sdk_version=picked.get("minSdkVersion"),
                target_sdk_version=picked.get("targetSdkVersion"),
                downloaded_url=f"{repo_url.rstrip('/')}/{apk_name}",
                message=f"would GET {apk_name} from {source_repo}",
                extra=common_extra,
            ))
            continue

        try:
            url_used = _download_with_fallback(
                apk_name, local,
                repo_url=repo_url, fallback_repo_url=fallback_url,
                curl_bin=args.curl_bin, proxy=args.proxy,
                connect_timeout=args.connect_timeout,
                retry=args.retry, retry_delay=args.retry_delay,
            )
        except subprocess.CalledProcessError as exc:
            results.append(FetchResult(
                app=name, package_name=res.package_name, resolution=res.match_type,
                status="http_error", apk_name=apk_name,
                message=f"curl failed: exit={exc.returncode} (source_repo={source_repo})",
                extra=common_extra,
            ))
            continue

        actual = _sha256_of(local).lower()
        if actual != expected_sha:
            results.append(FetchResult(
                app=name, package_name=res.package_name, resolution=res.match_type,
                status="sha256_mismatch", apk_name=apk_name,
                sha256=actual, local_path=str(local),
                message=f"expected {expected_sha}, got {actual}",
                extra=common_extra,
            ))
            continue

        results.append(FetchResult(
            app=name, package_name=res.package_name, resolution=res.match_type,
            status="downloaded",
            local_path=str(local), apk_name=apk_name,
            sha256=expected_sha, size_bytes=local.stat().st_size,
            version_name=picked.get("versionName"),
            version_code=picked.get("versionCode"),
            min_sdk_version=picked.get("minSdkVersion"),
            target_sdk_version=picked.get("targetSdkVersion"),
            downloaded_url=url_used,
            message="fetched and verified",
            extra=common_extra,
        ))

    # Summarize.
    ok_statuses = {"downloaded", "skipped_already_ok", "planned"}
    manifest = {
        "schema_version": 2,
        "source": "packergrind+fdroid",
        "generated_at": generated_at,
        "app_list": str(args.app_list),
        "origins_dir": str(args.origins_dir),
        "index": {
            "main_url": args.index_url,
            "main_path": str(main_index_path),
            "main_sha256": main_sha256,
            "main_timestamp": main_index.get("repo", {}).get("timestamp"),
            "archive_url": args.archive_index_url if arch_index else None,
            "archive_path": str(arch_index_path) if arch_index else None,
            "archive_sha256": arch_sha256,
            "archive_timestamp": (arch_index or {}).get("repo", {}).get("timestamp"),
        },
        "counts": {
            "total": len(results),
            "ok": sum(1 for r in results if r.status in ok_statuses),
            "downloaded": sum(1 for r in results if r.status == "downloaded"),
            "planned": sum(1 for r in results if r.status == "planned"),
            "skipped": sum(1 for r in results if r.status == "skipped_already_ok"),
            "unresolved": sum(1 for r in results if r.status == "unresolved"),
            "no_apk_found": sum(1 for r in results if r.status == "no_apk_found"),
            "sha256_mismatch": sum(1 for r in results if r.status == "sha256_mismatch"),
            "http_error": sum(1 for r in results if r.status == "http_error"),
        },
        "results": [asdict(r) for r in results],
    }

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mode_tag = "DRY-RUN" if dry_run else "EXECUTE"
    c = manifest["counts"]
    print(f"[{mode_tag}] manifest={args.manifest_out}")
    print(
        f"  total={c['total']} ok={c['ok']} downloaded={c['downloaded']} "
        f"planned={c['planned']} skipped={c['skipped']} "
        f"unresolved={c['unresolved']} no_apk_found={c['no_apk_found']} "
        f"sha256_mismatch={c['sha256_mismatch']} http_error={c['http_error']}"
    )
    for r in results:
        short_app = r.app[:22].ljust(22)
        pkg = (r.package_name or "?").ljust(40)[:40]
        src = f"[{r.extra.get('source_repo','-'):7s}]" if r.extra else "[-      ]"
        print(f"  [{r.status:20}] {short_app} {pkg} {src} {r.message}")

    # Treat unresolved / no_apk_found as non-fatal; hard failures are sha256_mismatch / http_error.
    hard_failures = c["sha256_mismatch"] + c["http_error"]
    return 0 if hard_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
