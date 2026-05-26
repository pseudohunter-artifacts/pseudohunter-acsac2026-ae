"""DEX-aware structural features for fusion head.

These features are designed to capture Android-specific structural patterns
that byte-level models might miss. All features are pure Python stdlib-only
and compute in <1 ms on 4 KiB regions.

Key insight: DEX files have rigid header structure that survives packing
transforms (xor, base64, splitting) but gets scrambled by signature stripping.
These features help the fusion head distinguish between "DEX-like structure
that's been transformed" vs "random bytes that happen to look DEX-ish".

Status (F2a, demoted to ablation-only input)
--------------------------------------------
As of batch F2, the 8 scalar features exported here are **no longer a core
selling point of Ours** and are **disabled by default** in the Ours baseline.
Rationale: their novelty is insufficient to support a top-tier venue claim
(see ``docs/research_framing.md`` §4.3 / §5.3). They are retained purely as
an optional ablation input for the ``ours_with_scalar_struct`` configuration
in ``configs/eval/ablation/``, which exists only to *demonstrate* that
switching them on delivers far smaller Δ than the grammar-aware item-type
auxiliary supervision introduced in F2b + F5 (see
``docs/method/ours_method_spec.md`` §3.2.1b / §5.1 / §6.3).

The corresponding defaults enforced elsewhere:

- ``FusionHeadConfig.structural_feature_dim = 0`` (F4).
- ``OursBaselineConfig.use_structural = False`` (F6).
- ``GatedFusionHead`` bypasses the structural path when ``structural`` is
  ``None`` or ``structural_feature_dim == 0``.

Do not import these features from ``training/`` or ``baselines/ours.py`` in
the default code path; only the ablation entrypoint is allowed to wire them
in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DexStructuralFeatureConfig:
    """Configuration for DEX structural feature extraction.
    
    Attributes:
        include_header_magic: Whether to check for DEX magic signatures
        include_map_list_hints: Whether to check map_list offset plausibility
        include_alignment_hints: Whether to check alignment constraints
        include_string_ids_sanity: Whether to check string_ids offset sanity
    """
    include_header_magic: bool = True
    include_map_list_hints: bool = True
    include_alignment_hints: bool = True
    include_string_ids_sanity: bool = True


def extract_region_structural_features(
    raw_bytes: bytes,
    object_path: str,
    offset_start: int,
    offset_end: int,
    config: DexStructuralFeatureConfig = DexStructuralFeatureConfig(),
) -> Dict[str, float]:
    """Extract DEX-aware structural features from a region.
    
    Args:
        raw_bytes: The raw bytes of the region
        object_path: Path of the object within the APK
        offset_start: Start offset within the object
        offset_end: End offset within the object
        config: Feature configuration
        
    Returns:
        Dictionary mapping feature names to float values (0.0 or 1.0)
    """
    features: Dict[str, float] = {}
    
    # Feature 1: DEX magic presence
    if config.include_header_magic:
        features["dex_magic_present"] = _check_dex_magic_present(raw_bytes)
    
    # Feature 2: DEX header file size plausibility
    if config.include_header_magic:
        features["dex_header_plausible_file_size"] = _check_file_size_plausible(raw_bytes)
    
    # Feature 3: Map list offset plausibility
    if config.include_map_list_hints:
        features["map_list_offset_plausible"] = _check_map_list_offset_plausible(raw_bytes)
    
    # Feature 4: String IDs offset alignment
    if config.include_alignment_hints:
        features["string_ids_offset_aligned"] = _check_string_ids_aligned(raw_bytes)
    
    # Feature 5: Nearby ZIP local header
    features["nearby_zip_local_header"] = _check_nearby_zip_header(raw_bytes)
    
    # Feature 6: Object path DEX-like pattern
    features["object_path_is_dex_like"] = _check_object_path_dex_like(object_path)
    
    # Feature 7: Object path in assets/
    features["object_path_is_asset"] = _check_object_path_asset(object_path)
    
    # Feature 8: Object path is lib/*/*.so (hard negative)
    features["object_path_is_lib_so"] = _check_object_path_lib_so(object_path)
    
    return features


def _check_dex_magic_present(data: bytes) -> float:
    """Check if data contains DEX magic signature."""
    if len(data) < 8:
        return 0.0
    
    # DEX magic patterns for versions 035-039
    dex_magics = [
        bytes([100, 101, 120, 10, 48, 51, 53, 0]),  # dex\n035\x00
        bytes([100, 101, 120, 10, 48, 51, 54, 0]),  # dex\n036\x00
        bytes([100, 101, 120, 10, 48, 51, 55, 0]),  # dex\n037\x00
        bytes([100, 101, 120, 10, 48, 51, 56, 0]),  # dex\n038\x00
        bytes([100, 101, 120, 10, 48, 51, 57, 0]),  # dex\n039\x00
    ]
    
    for magic in dex_magics:
        if data.startswith(magic):
            return 1.0
    
    return 0.0


def _check_file_size_plausible(data: bytes) -> float:
    """Check if DEX header file_size field is plausible."""
    if len(data) < 112:  # Need at least up to file_size field at offset 0x20
        return 0.0
    
    # DEX file_size is at offset 0x20 (32 bytes)
    if len(data) < 36:  # Need 4 bytes for file_size
        return 0.0
    
    try:
        file_size = int.from_bytes(data[32:36], byteorder='little', signed=False)
        
        # Plausible file size: 1KB to 100MB
        if 1024 <= file_size <= 100 * 1024 * 1024:
            # Also check that file_size <= actual data length
            if file_size <= len(data):
                return 1.0
    except (ValueError, IndexError):
        pass
    
    return 0.0


def _check_map_list_offset_plausible(data: bytes) -> float:
    """Check if map_list offset is plausible."""
    if len(data) < 116:  # Need at least up to map_off field at offset 0x34
        return 0.0
    
    # map_off is at offset 0x34 (52 bytes)
    if len(data) < 56:  # Need 4 bytes for map_off
        return 0.0
    
    try:
        map_off = int.from_bytes(data[52:56], byteorder='little', signed=False)
        
        # Map list should be after header (>=112 bytes) and within file bounds
        if 112 <= map_off < len(data):
            # Check 4-byte alignment
            if map_off % 4 == 0:
                return 1.0
    except (ValueError, IndexError):
        pass
    
    return 0.0


def _check_string_ids_aligned(data: bytes) -> float:
    """Check if string_ids offset is 4-byte aligned."""
    if len(data) < 108:  # Need at least up to string_ids_off field at offset 0x3C
        return 0.0
    
    # string_ids_off is at offset 0x3C (60 bytes)
    if len(data) < 64:  # Need 4 bytes for string_ids_off
        return 0.0
    
    try:
        string_ids_off = int.from_bytes(data[60:64], byteorder='little', signed=False)
        
        # Check 4-byte alignment
        if string_ids_off % 4 == 0:
            return 1.0
    except (ValueError, IndexError):
        pass
    
    return 0.0


def _check_nearby_zip_header(data: bytes) -> float:
    """Check for ZIP local header nearby."""
    if len(data) < 4:
        return 0.0
    
    # ZIP local header signature: PK\x03\x04
    zip_magic = b"PK\x03\x04"
    
    # Check within first 512 bytes
    search_window = data[:min(512, len(data))]
    if zip_magic in search_window:
        return 1.0
    
    # Check last 512 bytes
    if len(data) > 512:
        search_window = data[-512:]
        if zip_magic in search_window:
            return 1.0
    
    return 0.0


def _check_object_path_dex_like(object_path: str) -> float:
    """Check if object path suggests DEX content."""
    import re
    
    # Patterns that suggest DEX content
    patterns = [
        r'\.dex$',  # Ends with .dex
        r'classes\d*\.dex',  # classes.dex, classes2.dex, etc.
    ]
    
    for pattern in patterns:
        if re.search(pattern, object_path, re.IGNORECASE):
            return 1.0
    
    return 0.0


def _check_object_path_asset(object_path: str) -> float:
    """Check if object path is in assets/ directory."""
    return 1.0 if object_path.lower().startswith('assets/') else 0.0


def _check_object_path_lib_so(object_path: str) -> float:
    """Check if object path is a native library (hard negative)."""
    import re
    
    # Pattern for native libraries
    pattern = r'^lib/[^/]+/.*\.so$'
    return 1.0 if re.match(pattern, object_path) else 0.0


__all__ = [
    "DexStructuralFeatureConfig",
    "extract_region_structural_features",
]