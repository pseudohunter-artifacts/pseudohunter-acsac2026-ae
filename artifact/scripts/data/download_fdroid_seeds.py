"""Download additional F-Droid APKs to expand Track B seed pool.

Downloads 20 diverse F-Droid apps (different from existing 9) for:
- Expanding Track B with more packed samples (APKProtector, DPT Shell)
- Creating fully app-disjoint test splits
- Strengthening data scale for ACSAC submission

Usage:
    python scripts/data/download_fdroid_seeds.py [--count 20] [--out-dir data/real_world/track_b_v2/benign]
"""

import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

# Existing 9 apps we already have (exclude from download)
EXISTING_PACKAGES = {
    "com.fsck.k9",
    "com.kunzisoft.keepass.libre",
    "com.termux",
    "de.danoeh.antennapod",
    "de.dennisguse.opentracks",
    "org.fdroid.fdroid",
    "org.schabi.newpipe",
    "org.tasks",
    "org.videolan.vlc",
}

# 30 diverse F-Droid apps (different categories, sizes, architectures)
# These are well-known open-source apps with stable APK downloads
FDROID_APPS = [
    # Communication
    ("org.thoughtcrime.securesms", "Signal-FOSS"),
    ("im.vector.app", "Element"),
    ("org.telegram.messenger", "Telegram-FOSS"),
    ("com.simplemobiletools.smsmessenger", "Simple SMS"),
    # Productivity
    ("com.nextcloud.client", "Nextcloud"),
    ("at.bitfire.davdroid", "DAVx5"),
    ("org.mozilla.fenix", "Firefox"),
    ("com.orgzly", "Orgzly"),
    # Media
    ("org.fossasia.phimpme", "Phimpme"),
    ("com.simplemobiletools.gallery.pro", "Simple Gallery"),
    ("org.jellyfin.androidtv", "Jellyfin"),
    ("app.grapheneos.camera", "GrapheneOS Camera"),
    # Security / Privacy
    ("org.torproject.torbrowser", "Tor Browser"),
    ("net.bierbaumer.otp_authenticator", "Aegis Authenticator"),
    ("com.wireguard.android", "WireGuard"),
    ("org.kde.kdeconnect_tp", "KDE Connect"),
    # System / Tools
    ("com.termoneplus", "TermOneePlus"),
    ("eu.faircode.email", "FairEmail"),
    ("com.amaze.filemanager", "Amaze File Manager"),
    ("com.github.axet.bookreader", "Book Reader"),
    # Games / Other
    ("org.supertuxkart.stk", "SuperTuxKart"),
    ("org.moire.opensudoku", "Open Sudoku"),
    ("io.github.nfdz.cryptool", "Cryptool"),
    ("com.simplemobiletools.calculator", "Simple Calculator"),
    # Development
    ("com.arachnoid.sshelper", "SSHelper"),
    ("org.connectbot", "ConnectBot"),
    ("com.foxdebug.acode", "Acode"),
    ("com.termux.api", "Termux API"),
    # Navigation
    ("net.osmand.plus", "OsmAnd"),
    ("de.westnordost.streetcomplete", "StreetComplete"),
]


def download_from_fdroid(package: str, out_dir: Path, timeout: int = 60) -> Path | None:
    """Try to download APK from F-Droid repo."""
    # F-Droid repo URL format
    base_url = f"https://f-droid.org/repo/{package}"

    # Try the index API first to get latest version
    index_url = f"https://f-droid.org/api/v1/packages/{package}"

    try:
        # Simple approach: try common version patterns
        req = urllib.request.Request(index_url)
        req.add_header("User-Agent", "Mozilla/5.0 (research)")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json
            data = json.loads(resp.read())
            if "suggestedVersionCode" in data:
                vcode = data["suggestedVersionCode"]
                apk_url = f"https://f-droid.org/repo/{package}_{vcode}.apk"
            elif "packages" in data and data["packages"]:
                vcode = data["packages"][0]["versionCode"]
                apk_url = f"https://f-droid.org/repo/{package}_{vcode}.apk"
            else:
                return None
    except Exception:
        return None

    out_path = out_dir / f"{package}_{vcode}.apk"
    if out_path.exists():
        print(f"  Already exists: {out_path.name}")
        return out_path

    try:
        print(f"  Downloading: {apk_url}")
        req = urllib.request.Request(apk_url)
        req.add_header("User-Agent", "Mozilla/5.0 (research)")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
            if len(data) < 10000:  # Too small, likely error page
                return None
            out_path.write_bytes(data)
            print(f"  Saved: {out_path.name} ({len(data)/1e6:.1f} MB)")
            return out_path
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20,
                        help="Target number of new APKs to download")
    parser.add_argument("--out-dir", type=str,
                        default="data/real_world/track_b_v2/benign")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading up to {args.count} new F-Droid APKs to {out_dir}")
    print(f"Excluding {len(EXISTING_PACKAGES)} existing packages\n")

    downloaded = 0
    for package, name in FDROID_APPS:
        if package in EXISTING_PACKAGES:
            continue
        if downloaded >= args.count:
            break

        print(f"[{downloaded+1}/{args.count}] {name} ({package})")
        result = download_from_fdroid(package, out_dir)
        if result:
            downloaded += 1
        else:
            print(f"  Skipped (unavailable)")

        time.sleep(1)  # Be polite to F-Droid servers

    print(f"\n=== Downloaded {downloaded} new APKs ===")
    existing = list(out_dir.glob("*.apk"))
    print(f"Total in {out_dir}: {len(existing)} APKs")


if __name__ == "__main__":
    main()
