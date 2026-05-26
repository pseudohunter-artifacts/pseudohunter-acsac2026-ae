"""Download and verify the F-Droid seed APK set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_URL = "https://f-droid.org/repo/index-v1.jar"
DEFAULT_REPO_URL = "https://f-droid.org/repo"
DEFAULT_FALLBACK_REPO_URL = "https://mirrors.tuna.tsinghua.edu.cn/fdroid/repo"

TARGETS: Tuple[Dict[str, Any], ...] = (
    {
        "package_name": "org.fdroid.fdroid",
        "app_name": "F-Droid",
        "category": "App Store & Updater",
    },
    {
        "package_name": "org.schabi.newpipe",
        "app_name": "NewPipe",
        "category": "Online Media Player",
    },
    {
        "package_name": "de.danoeh.antennapod",
        "app_name": "AntennaPod",
        "category": "Podcast",
    },
    {
        "package_name": "org.videolan.vlc",
        "app_name": "VLC",
        "category": "Local Media Player",
        "required_native_code": "arm64-v8a",
    },
    {
        "package_name": "com.fsck.k9",
        "app_name": "K-9 Mail",
        "category": "Email",
    },
    {
        "package_name": "org.tasks",
        "app_name": "Tasks.org",
        "category": "Task",
    },
    {
        "package_name": "com.kunzisoft.keepass.libre",
        "app_name": "KeePassDX",
        "category": "Password & 2FA",
    },
    {
        "package_name": "de.dennisguse.opentracks",
        "app_name": "OpenTracks",
        "category": "Sports & Health",
    },
    {
        "package_name": "com.termux",
        "app_name": "Termux",
        "category": "Development",
    },
)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download representative F-Droid seed APKs and write a manifest."
    )
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--fallback-repo-url", default=DEFAULT_FALLBACK_REPO_URL)
    parser.add_argument("--proxy", default=os.environ.get("FDROID_PROXY"))
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic" / "seed_apks",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=REPO_ROOT / "data" / "synthetic" / "seed_apks_manifest.json",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Cached F-Droid index JAR path. Defaults under seed-dir/.cache.",
    )
    parser.add_argument(
        "--skip-index-download",
        action="store_true",
        help="Use the cached index-path without downloading a fresh index.",
    )
    parser.add_argument(
        "--curl-bin",
        default="curl.exe" if os.name == "nt" else "curl",
        help="curl executable to use for downloads.",
    )
    parser.add_argument("--connect-timeout", type=int, default=20)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--retry-delay", type=int, default=2)
    return parser.parse_args(argv)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def relpath(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curl_download(url: str, out_path: Path, args: argparse.Namespace) -> None:
    command = [
        args.curl_bin,
        "-L",
        "--fail",
        "--retry",
        str(args.retry),
        "--retry-delay",
        str(args.retry_delay),
        "--connect-timeout",
        str(args.connect_timeout),
        "-o",
        str(out_path),
        url,
    ]
    if args.proxy:
        command[1:1] = ["-x", args.proxy]
    subprocess.run(command, check=True)


def download_with_fallback(
    apk_name: str,
    destination: Path,
    args: argparse.Namespace,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    primary_url = f"{args.repo_url.rstrip('/')}/{apk_name}"
    fallback_url = (
        f"{args.fallback_repo_url.rstrip('/')}/{apk_name}"
        if args.fallback_repo_url
        else None
    )
    try:
        curl_download(primary_url, tmp_path, args)
        downloaded_url = primary_url
    except subprocess.CalledProcessError:
        if not fallback_url:
            raise
        if tmp_path.exists():
            tmp_path.unlink()
        curl_download(fallback_url, tmp_path, args)
        downloaded_url = fallback_url

    tmp_path.replace(destination)
    return downloaded_url


def load_fdroid_index(index_path: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(index_path) as archive:
        return json.loads(archive.read("index-v1.json"))


def localized_app_name(app: Dict[str, Any], fallback: str) -> str:
    for locale in ("en-US", "en", "zh-CN"):
        localized = app.get("localized", {}).get(locale, {})
        name = localized.get("name")
        if name:
            return str(name)
    return str(app.get("name") or app.get("autoName") or fallback)


def by_version_order(entry: Dict[str, Any]) -> Tuple[int, int]:
    return int(entry.get("added") or 0), int(entry.get("versionCode") or 0)


def select_package(
    target: Dict[str, Any],
    app: Dict[str, Any],
    versions: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    candidates = list(versions)
    required_native_code = target.get("required_native_code")
    if required_native_code:
        candidates = [
            item
            for item in candidates
            if required_native_code in (item.get("nativecode") or [])
        ]
        suggested_name = app.get("suggestedVersionName")
        name_matches = [
            item for item in candidates if item.get("versionName") == suggested_name
        ]
        if name_matches:
            candidates = name_matches
        if candidates:
            return max(candidates, key=by_version_order)
        raise ValueError(
            f"No {target['package_name']} APK for native code {required_native_code}"
        )

    suggested_code = app.get("suggestedVersionCode")
    if suggested_code is not None:
        matches = [
            item
            for item in candidates
            if int(item.get("versionCode") or -1) == int(suggested_code)
        ]
        if matches:
            return matches[0]

    if not candidates:
        raise ValueError(f"No APK versions found for {target['package_name']}")
    return candidates[0]


def apk_zip_stats(path: Path) -> Dict[str, int]:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"{path} has corrupt ZIP member {bad_member}")
        members = [info.filename for info in archive.infolist() if not info.is_dir()]

    lower_members = [member.lower() for member in members]
    return {
        "zip_member_count": len(members),
        "dex_count": sum(1 for member in lower_members if member.endswith(".dex")),
        "native_lib_count": sum(
            1 for member in lower_members if member.endswith(".so")
        ),
        "asset_count": sum(1 for member in lower_members if member.startswith("assets/")),
    }


def ensure_apk(
    selected: Dict[str, Any],
    destination: Path,
    args: argparse.Namespace,
) -> Tuple[str, bool]:
    expected_sha256 = selected["hash"].lower()
    if destination.exists() and destination.stat().st_size > 0:
        if sha256_file(destination).lower() == expected_sha256:
            return f"{args.repo_url.rstrip('/')}/{selected['apkName']}", False

    downloaded_url = download_with_fallback(selected["apkName"], destination, args)
    actual_sha256 = sha256_file(destination).lower()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{destination} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return downloaded_url, True


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    seed_dir = args.seed_dir.resolve()
    seed_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index_path or seed_dir / ".cache" / "index-v1.jar"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    fetched_at = utc_now()
    if not args.skip_index_download:
        curl_download(args.index_url, index_path, args)

    index_sha256 = sha256_file(index_path)
    index = load_fdroid_index(index_path)
    apps_by_package = {app["packageName"]: app for app in index["apps"]}
    packages = index["packages"]

    downloaded_at = utc_now()
    entries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    downloaded_count = 0

    for target in TARGETS:
        package_name = target["package_name"]
        try:
            app = apps_by_package[package_name]
            selected = select_package(target, app, packages[package_name])
            apk_name = selected["apkName"]
            local_path = seed_dir / apk_name
            downloaded_url, downloaded = ensure_apk(selected, local_path, args)
            if downloaded:
                downloaded_count += 1

            actual_size = local_path.stat().st_size
            expected_size = int(selected["size"])
            if actual_size != expected_size:
                raise ValueError(
                    f"{local_path} size mismatch: expected {expected_size}, "
                    f"got {actual_size}"
                )

            stats = apk_zip_stats(local_path)
            categories = app.get("categories") or []
            entries.append(
                {
                    "source": "f-droid",
                    "package_name": package_name,
                    "app_name": target.get("app_name")
                    or localized_app_name(app, package_name),
                    "category": target.get("category")
                    or (categories[0] if categories else None),
                    "categories": categories,
                    "version_name": selected.get("versionName"),
                    "version_code": selected.get("versionCode"),
                    "apk_name": apk_name,
                    "download_url": f"{args.repo_url.rstrip('/')}/{apk_name}",
                    "downloaded_url": downloaded_url,
                    "local_path": relpath(local_path),
                    "size_bytes": actual_size,
                    "sha256": selected["hash"].lower(),
                    "hash_type": selected.get("hashType"),
                    "downloaded_at": downloaded_at,
                    "native_code": selected.get("nativecode") or [],
                    "min_sdk_version": selected.get("minSdkVersion"),
                    "target_sdk_version": selected.get("targetSdkVersion"),
                    **stats,
                }
            )
            print(
                f"ok {package_name} {selected.get('versionName')} "
                f"{actual_size} bytes"
            )
        except Exception as exc:  # noqa: BLE001 - collect all target failures.
            failures.append({"package_name": package_name, "error": str(exc)})
            print(f"failed {package_name}: {exc}", file=sys.stderr)

    manifest = {
        "schema_version": 1,
        "source": "f-droid",
        "seed_dir": relpath(seed_dir),
        "generated_at": downloaded_at,
        "downloaded_count": downloaded_count,
        "entry_count": len(entries),
        "repo": {
            "index_url": args.index_url,
            "repo_url": args.repo_url,
            "fallback_repo_url": args.fallback_repo_url,
            "index_path": relpath(index_path),
            "index_sha256": index_sha256,
            "index_fetched_at": fetched_at,
            "index_timestamp": index.get("repo", {}).get("timestamp"),
        },
        "entries": entries,
        "failures": failures,
    }
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    manifest = build_manifest(args)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={relpath(args.manifest_out)} entries={manifest['entry_count']}")
    return 1 if manifest["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
