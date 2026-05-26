"""APK object extraction utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Iterator, Optional, Tuple
import posixpath
import zipfile


ARCHIVE_SUFFIXES = {".apk", ".jar", ".zip"}


class ApkReadError(RuntimeError):
    """Raised when an APK cannot be parsed as a ZIP archive."""


@dataclass(frozen=True)
class ApkObject:
    apk_id: str
    object_id: str
    object_path: str
    object_type: str
    size: int
    sha256: str
    depth: int
    container_path: str
    compression: str
    compressed_size: int

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_object(object_path: str, data: bytes) -> str:
    normalized = object_path.replace("\\", "/")
    lower_path = normalized.lower()
    name = posixpath.basename(lower_path)
    suffix = Path(name).suffix.lower()

    if _is_classes_dex(name) or suffix == ".dex":
        return "dex"
    if suffix == ".so" or lower_path.startswith("lib/"):
        return "native_lib"
    if lower_path.startswith("meta-inf/"):
        return "signature"
    if _looks_like_zip(data) or suffix in ARCHIVE_SUFFIXES:
        return "embedded_archive"
    if lower_path.startswith("assets/"):
        return "asset_blob"
    if lower_path.startswith("res/") or lower_path == "resources.arsc":
        return "resource"
    return "unknown_blob"


def iter_apk_objects(
    apk_path: Path,
    *,
    max_depth: int = 1,
    max_member_bytes: Optional[int] = None,
) -> Iterator[Tuple[ApkObject, bytes]]:
    apk_path = Path(apk_path)
    apk_id = file_sha256(apk_path)

    try:
        with zipfile.ZipFile(apk_path) as archive:
            counter = _Counter()
            yield from _iter_zip_members(
                archive=archive,
                apk_id=apk_id,
                counter=counter,
                container_path=apk_path.name,
                path_prefix="",
                depth=0,
                max_depth=max_depth,
                max_member_bytes=max_member_bytes,
            )
    except zipfile.BadZipFile as exc:
        raise ApkReadError(f"Not a valid APK/ZIP archive: {apk_path}") from exc


class _Counter:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> int:
        current = self.value
        self.value += 1
        return current


def _iter_zip_members(
    *,
    archive: zipfile.ZipFile,
    apk_id: str,
    counter: _Counter,
    container_path: str,
    path_prefix: str,
    depth: int,
    max_depth: int,
    max_member_bytes: Optional[int],
) -> Iterator[Tuple[ApkObject, bytes]]:
    for info in archive.infolist():
        if info.is_dir():
            continue
        if max_member_bytes is not None and info.file_size > max_member_bytes:
            continue

        data = archive.read(info)
        relative_path = info.filename.replace("\\", "/")
        object_path = f"{path_prefix}!{relative_path}" if path_prefix else relative_path
        object_type = classify_object(object_path, data)
        index = counter.next()
        metadata = ApkObject(
            apk_id=apk_id,
            object_id=f"{apk_id[:12]}:{index:06d}",
            object_path=object_path,
            object_type=object_type,
            size=len(data),
            sha256=sha256(data).hexdigest(),
            depth=depth,
            container_path=container_path,
            compression=_compression_name(info.compress_type),
            compressed_size=info.compress_size,
        )
        yield metadata, data

        if object_type == "embedded_archive" and depth < max_depth:
            try:
                with zipfile.ZipFile(BytesIO(data)) as nested:
                    yield from _iter_zip_members(
                        archive=nested,
                        apk_id=apk_id,
                        counter=counter,
                        container_path=object_path,
                        path_prefix=object_path,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_member_bytes=max_member_bytes,
                    )
            except zipfile.BadZipFile:
                continue


def _is_classes_dex(name: str) -> bool:
    if not name.endswith(".dex"):
        return False
    stem = name[:-4]
    if stem == "classes":
        return True
    return stem.startswith("classes") and stem[7:].isdigit()


def _looks_like_zip(data: bytes) -> bool:
    return data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06")


def _compression_name(compress_type: int) -> str:
    if compress_type == zipfile.ZIP_STORED:
        return "stored"
    if compress_type == zipfile.ZIP_DEFLATED:
        return "deflated"
    if compress_type == zipfile.ZIP_BZIP2:
        return "bzip2"
    if compress_type == zipfile.ZIP_LZMA:
        return "lzma"
    return f"unknown:{compress_type}"
