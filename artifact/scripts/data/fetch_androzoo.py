"""Fetch APKs from AndroZoo by SHA256.

AndroZoo provides a single HTTP endpoint parameterized by SHA256 and an
API key. See https://androzoo.uni.lu/api_doc for details. This helper is a
batch wrapper around that endpoint with dry-run, idempotency, rate-limiting,
and a JSON summary, mirroring the style of ``scripts/fetch_track_b_benign.py``.

Third-party APK bytes never enter git. The script writes APKs into
``data/androzoo/apks/<prefix>/<sha256>.apk`` (sharded by sha256 prefix to
avoid million-entry directories), and emits a JSON summary that is safe to
commit.

INPUT FORMATS
    --sha256-file    Plain text, one SHA256 per line (``#`` comments OK).
    --manifest       JSON/YAML list of objects with at least ``sha256``;
                     other fields (``package_name``, ``label``, ``packer``)
                     are passed through to the summary.

Usage::

    # Plan only (default):
    python scripts/data/fetch_androzoo.py --sha256-file my_hashes.txt

    # Actually fetch (reads API key from $ANDROZOO_API_KEY):
    python scripts/data/fetch_androzoo.py --sha256-file my_hashes.txt --execute

    # Explicit key + custom output dir + throttle:
    python scripts/data/fetch_androzoo.py \
        --manifest targets.json --execute \
        --api-key $env:ANDROZOO_API_KEY \
        --out-dir data/androzoo/apks \
        --sleep-sec 0.5 \
        --json-summary reports/androzoo_fetch.json

Exit code is 0 iff every requested SHA256 ended up in state
``downloaded`` or ``skipped_already_ok``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# AndroZoo's documented download endpoint.
ANDROZOO_DOWNLOAD_URL = "https://androzoo.uni.lu/api/download"

DEFAULT_OUT_DIR = Path("data/androzoo/apks")
DEFAULT_TIMEOUT_SEC = 180
DEFAULT_SLEEP_SEC = 0.25  # be polite to uni.lu's mirror
DEFAULT_USER_AGENT = "android-packer-androzoo-fetcher/1.0 (+research)"

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass
class FetchOutcome:
    sha256: str
    status: str  # one of: planned | downloaded | skipped_already_ok
                  #        | sha256_mismatch | http_error | network_error
                  #        | invalid_sha256 | filtered_out
    local_path: Optional[Path]
    http_status: Optional[int]
    extra: Dict[str, Any]
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "status": self.status,
            "local_path": str(self.local_path) if self.local_path else None,
            "http_status": self.http_status,
            "extra": self.extra,
            "message": self.message,
        }


def _sha256_of(path: Path, *, buf_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _sharded_path(out_dir: Path, sha256: str) -> Path:
    """``<out_dir>/<prefix-2>/<sha256>.apk``.

    Sharding keeps any single directory well under 65k entries even when we
    scale to 100k+ APKs.
    """
    prefix = sha256[:2].lower()
    return out_dir / prefix / f"{sha256.lower()}.apk"


def _load_targets_from_sha256_file(path: Path) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    # ``utf-8-sig`` transparently strips an optional BOM that Windows tooling
    # (Notepad, PowerShell's ``Out-File -Encoding utf8``) likes to prepend.
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Allow trailing ``# comment``.
        tok = line.split("#", 1)[0].strip()
        if not tok:
            continue
        targets.append({"sha256": tok})
    return targets


def _load_targets_from_manifest(path: Path) -> List[Dict[str, Any]]:
    # YAML is accepted if PyYAML is available, otherwise fall back to JSON.
    text = path.read_text(encoding="utf-8")
    data: Any
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: WPS433

            data = yaml.safe_load(text)
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(f"PyYAML required to read {path}: {exc}") from exc
    else:
        data = json.loads(text)

    if isinstance(data, dict) and "apks" in data:
        data = data["apks"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list (or dict with 'apks'), got {type(data).__name__}")

    targets: List[Dict[str, Any]] = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "sha256" not in entry:
            raise ValueError(f"{path}: entry [{i}] missing 'sha256' key")
        targets.append(dict(entry))
    return targets


def _fetch_one(
    target: Dict[str, Any],
    *,
    out_dir: Path,
    api_key: str,
    dry_run: bool,
    timeout_sec: int,
    user_agent: str,
) -> FetchOutcome:
    sha = str(target["sha256"]).strip().lower()
    extra = {k: v for k, v in target.items() if k != "sha256"}

    if not SHA256_RE.match(sha):
        return FetchOutcome(
            sha256=sha,
            status="invalid_sha256",
            local_path=None,
            http_status=None,
            extra=extra,
            message="not a 64-char hex string",
        )

    local_path = _sharded_path(out_dir, sha)

    # Idempotent fast-path: already-downloaded and sha256 still matches.
    if local_path.exists():
        actual = _sha256_of(local_path)
        if actual == sha:
            return FetchOutcome(
                sha256=sha,
                status="skipped_already_ok",
                local_path=local_path,
                http_status=None,
                extra=extra,
                message="already on disk; hash matches",
            )
        return FetchOutcome(
            sha256=sha,
            status="sha256_mismatch",
            local_path=local_path,
            http_status=None,
            extra=extra,
            message=(
                f"local file hash={actual} != requested {sha}; "
                "delete it manually if you trust the request"
            ),
        )

    if dry_run:
        return FetchOutcome(
            sha256=sha,
            status="planned",
            local_path=local_path,
            http_status=None,
            extra=extra,
            message=f"would GET {ANDROZOO_DOWNLOAD_URL}?sha256={sha} -> {local_path}",
        )

    # Actually fetch.
    query = urllib.parse.urlencode({"apikey": api_key, "sha256": sha})
    url = f"{ANDROZOO_DOWNLOAD_URL}?{query}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".part")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp, tmp.open("wb") as out:
            http_status = resp.status
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        return FetchOutcome(
            sha256=sha,
            status="http_error",
            local_path=None,
            http_status=exc.code,
            extra=extra,
            message=f"HTTP {exc.code}: {exc.reason}",
        )
    except (urllib.error.URLError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        return FetchOutcome(
            sha256=sha,
            status="network_error",
            local_path=None,
            http_status=None,
            extra=extra,
            message=f"{type(exc).__name__}: {exc}",
        )

    # Verify downloaded bytes.
    actual = _sha256_of(tmp)
    if actual != sha:
        tmp.unlink(missing_ok=True)
        return FetchOutcome(
            sha256=sha,
            status="sha256_mismatch",
            local_path=None,
            http_status=http_status,
            extra=extra,
            message=f"server returned bytes hashing to {actual}; discarded",
        )

    tmp.replace(local_path)
    return FetchOutcome(
        sha256=sha,
        status="downloaded",
        local_path=local_path,
        http_status=http_status,
        extra=extra,
        message="fetched and verified",
    )


def _resolve_api_key(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value.strip()
    for env_var in ("ANDROZOO_API_KEY", "ANDROZOO_KEY"):
        env_val = os.environ.get(env_var, "").strip()
        if env_val:
            return env_val
    raise SystemExit(
        "AndroZoo API key not provided. Pass --api-key or set "
        "$ANDROZOO_API_KEY / $ANDROZOO_KEY in the environment."
    )


def _iter_limited(targets: Iterable[Dict[str, Any]], limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    if limit is None or limit <= 0:
        yield from targets
        return
    for i, t in enumerate(targets):
        if i >= limit:
            return
        yield t


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sha256-file", type=Path, help="Plain text, one SHA256 per line.")
    src.add_argument("--manifest", type=Path, help="JSON/YAML list or dict with 'apks'.")

    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Sharded output directory.")
    parser.add_argument("--api-key", type=str, default=None, help="AndroZoo API key (default: $ANDROZOO_API_KEY).")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SEC, help="Per-APK timeout in seconds.")
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC, help="Sleep between requests.")
    parser.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT, help="HTTP User-Agent.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N targets (for smoke tests).")
    parser.add_argument("--only", nargs="+", default=None, help="Restrict to these SHA256 values.")
    parser.add_argument(
        "--json-summary",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of outcomes.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="Plan only (default).")
    mode.add_argument("--execute", action="store_true", help="Actually download.")
    args = parser.parse_args(argv)
    dry_run = not args.execute

    # Load target list.
    if args.sha256_file is not None:
        targets = _load_targets_from_sha256_file(args.sha256_file.resolve())
        source = str(args.sha256_file.resolve())
    else:
        targets = _load_targets_from_manifest(args.manifest.resolve())
        source = str(args.manifest.resolve())

    if args.only:
        only = {s.lower() for s in args.only}
        targets = [t for t in targets if str(t.get("sha256", "")).lower() in only]

    targets = list(_iter_limited(targets, args.limit))
    if not targets:
        print(f"[ABORT] no targets from source={source}")
        return 2

    # API key only required in execute mode.
    api_key = "" if dry_run else _resolve_api_key(args.api_key)
    out_dir = args.out_dir.resolve()

    outcomes: List[FetchOutcome] = []
    for i, tgt in enumerate(targets):
        outcomes.append(
            _fetch_one(
                tgt,
                out_dir=out_dir,
                api_key=api_key,
                dry_run=dry_run,
                timeout_sec=args.timeout,
                user_agent=args.user_agent,
            )
        )
        # Throttle only between real requests that actually hit the network.
        if (
            not dry_run
            and outcomes[-1].status in {"downloaded", "http_error", "network_error"}
            and i < len(targets) - 1
            and args.sleep_sec > 0
        ):
            time.sleep(args.sleep_sec)

    ok_states = {"downloaded", "skipped_already_ok", "planned"}
    summary = {
        "dry_run": dry_run,
        "source": source,
        "out_dir": str(out_dir),
        "total": len(outcomes),
        "ok": sum(1 for o in outcomes if o.status in ok_states),
        "downloaded": sum(1 for o in outcomes if o.status == "downloaded"),
        "skipped_already_ok": sum(1 for o in outcomes if o.status == "skipped_already_ok"),
        "failed": sum(1 for o in outcomes if o.status in {"sha256_mismatch", "http_error", "network_error", "invalid_sha256"}),
        "outcomes": [o.to_dict() for o in outcomes],
    }

    if args.json_summary:
        args.json_summary.parent.mkdir(parents=True, exist_ok=True)
        args.json_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mode_tag = "DRY-RUN" if dry_run else "EXECUTE"
    print(f"[{mode_tag}] source={source}")
    print(f"  out_dir={out_dir}")
    print(
        f"  total={summary['total']} ok={summary['ok']} "
        f"downloaded={summary['downloaded']} skipped={summary['skipped_already_ok']} "
        f"failed={summary['failed']}"
    )
    # Compact per-outcome log (truncated hash for readability).
    for o in outcomes:
        tag = f"[{o.status:20}]"
        short = o.sha256[:12] + "..." if len(o.sha256) == 64 else o.sha256
        extra = ""
        if "package_name" in o.extra:
            extra = f" pkg={o.extra['package_name']}"
        if "packer" in o.extra:
            extra += f" packer={o.extra['packer']}"
        print(f"  {tag} {short}{extra}  {o.message}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
