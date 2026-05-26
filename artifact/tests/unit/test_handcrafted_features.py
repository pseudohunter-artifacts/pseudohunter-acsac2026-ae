"""Unit tests for the handcrafted feature assembly.

Covers Pass-2a (15 dims, paper-cell):
* feature vocabulary (``handcrafted_feature_names``) is stable and
  matches the config toggles
* Group A (entropy raw + deltas) flows through unchanged from the
  entropy_delta module
* Group B (byte distribution) gives hand-verifiable values on small
  crafted byte sequences
* Group G (region position) normalises correctly across objects /
  APKs
* Byte loader is optional and defaults to zeros

Also covers the optional Group C (compression / magic) and Group F
(ZIP / object context) extensions. These two groups are the building
blocks of Pass-2b (22 dims). Pass-2b as a FEATURE PACK is retracted
(L1 ablation 2026-05-03 showed it regresses vs Pass-2a on 4 Gen3
folds; see docs/progress/sessions/2026-05-02_overnight_results_report.md
section 1.5), but the two underlying TOGGLES remain first-class API
features -- other pipelines (e.g. DEX-only probes for Stage B smoke
runs) can still opt into Group C / F independently. These unit tests
therefore pin the TOGGLE behaviour, not the deprecated Pass-2b
pack.

Uses pure stdlib only; no pytorch, numpy, or sklearn.
"""

from __future__ import annotations

import pytest

from android_packer.features import (
    EntropyDeltaConfig,
    HandcraftedFeatureConfig,
    extract_handcrafted_features,
    handcrafted_feature_names,
)


def _mk_row(apk, obj, obj_path, offset, entropy):
    return {
        "apk_id": apk,
        "object_id": obj,
        "object_path": obj_path,
        "offset_start": offset,
        "offset_end": offset + 16,
        "entropy": entropy,
    }


# ---------------------------------------------------------------------------
# Feature-vocabulary shape tests
# ---------------------------------------------------------------------------


def test_default_config_gives_15_features():
    names = handcrafted_feature_names()
    # 1 (entropy_raw) + 3 (deltas) + 8 (byte dist) + 3 (region pos) = 15
    assert len(names) == 15
    # Check the head + tail anchors (order is part of the contract).
    assert names[0] == "entropy_raw"
    assert names[1:4] == [
        "entropy_delta_neighbor",
        "entropy_delta_entry",
        "entropy_delta_apk",
    ]
    assert "byte_printable_ratio" in names
    assert names[-1] == "object_index_in_apk_norm"


def test_toggle_entropy_deltas_off():
    cfg = HandcraftedFeatureConfig(include_entropy_deltas=False)
    names = handcrafted_feature_names(cfg)
    assert "entropy_delta_neighbor" not in names
    assert "entropy_raw" in names


def test_reserved_groups_d_e_do_not_appear_when_flags_flipped():
    # Group C (compression_signature) and Group F (zip_context) are now
    # implemented (landed 2026-05-02) and ADD dims when enabled; Group
    # D (dex_structural) and Group E (bigram_top_k) are still reserved
    # placeholders -- the implementation ignores their flags so toggling
    # them has zero effect on the feature vocabulary.
    cfg = HandcraftedFeatureConfig(
        include_compression_signature=False,  # off -> no Group C
        include_zip_context=False,            # off -> no Group F
        include_dex_structural=True,          # reserved -- no-op
        include_bigram_top_k=True,            # reserved -- no-op
    )
    names = handcrafted_feature_names(cfg)
    # Pass-2a baseline (15) because the two reserved flags are ignored
    # and the two implemented flags are off.
    assert len(names) == 15
    # None of the reserved Group-D / Group-E names may leak in.
    for unexpected in (
        "dex_section_ratio",
        "dex_magic_at_offset_0",
        "bigram_top1_ratio",
    ):
        assert unexpected not in names


# ---------------------------------------------------------------------------
# Group A: entropy raw + deltas
# ---------------------------------------------------------------------------


def test_group_a_entropy_raw_and_deltas_populated():
    rows = [
        _mk_row("apk", "obj1", "assets/a", 0, 2.0),
        _mk_row("apk", "obj1", "assets/a", 16, 4.0),
        _mk_row("apk", "obj2", "assets/b", 0, 6.0),
    ]
    extract_handcrafted_features(rows, byte_loader=None)
    for r in rows:
        assert "entropy_raw" in r
        assert "entropy_delta_neighbor" in r
        assert "entropy_delta_entry" in r
        assert "entropy_delta_apk" in r
    # entropy_raw matches the input entropy
    assert rows[0]["entropy_raw"] == pytest.approx(2.0)
    assert rows[2]["entropy_raw"] == pytest.approx(6.0)
    # entry-mean for assets/a = 3.0 -> deltas [-1, +1]
    assert rows[0]["entropy_delta_entry"] == pytest.approx(-1.0)
    assert rows[1]["entropy_delta_entry"] == pytest.approx(1.0)
    # apk-mean = 4 -> deltas [-2, 0, +2]
    assert rows[0]["entropy_delta_apk"] == pytest.approx(-2.0)
    assert rows[1]["entropy_delta_apk"] == pytest.approx(0.0)
    assert rows[2]["entropy_delta_apk"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Group B: byte distribution
# ---------------------------------------------------------------------------


def test_group_b_zero_bytes_all_ratios_extreme():
    rows = [_mk_row("apk", "o", "p", 0, 0.0)]
    loader = lambda _row: b"\x00" * 256
    extract_handcrafted_features(rows, byte_loader=loader)
    r = rows[0]
    assert r["byte_zero_ratio"] == pytest.approx(1.0)
    assert r["byte_printable_ratio"] == pytest.approx(0.0)
    assert r["byte_high_bit_ratio"] == pytest.approx(0.0)
    assert r["byte_ascii_ratio"] == pytest.approx(0.0)
    # Only one unique byte value -> log2(1) = 0
    assert r["byte_unique_count_log2"] == pytest.approx(0.0)
    # range = max - min = 0
    assert r["byte_range_span"] == pytest.approx(0.0)
    # max run = 256 -> log2_floor(256) = 8
    assert r["byte_max_run_len_log2"] == pytest.approx(8.0)


def test_group_b_uniform_random_like_high_unique_count():
    # All 256 byte values once each: uniform distribution.
    rows = [_mk_row("apk", "o", "p", 0, 7.5)]
    loader = lambda _row: bytes(range(256))
    extract_handcrafted_features(rows, byte_loader=loader)
    r = rows[0]
    # Unique values = 256 -> log2(256) = 8
    assert r["byte_unique_count_log2"] == pytest.approx(8.0)
    # Range spans full [0, 255] -> 1.0
    assert r["byte_range_span"] == pytest.approx(1.0)
    # Chi-square distance to uniform should be ~0 (perfectly uniform).
    assert abs(r["byte_chi2_uniform"]) < 1e-6
    # Max run = 1 -> log2_floor(1) = 0
    assert r["byte_max_run_len_log2"] == pytest.approx(0.0)


def test_group_b_ascii_text_printable_ratio_one():
    rows = [_mk_row("apk", "o", "p", 0, 5.0)]
    text = b"Hello, World!\n" * 20  # all printable chars
    loader = lambda _row: text
    extract_handcrafted_features(rows, byte_loader=loader)
    assert rows[0]["byte_printable_ratio"] == pytest.approx(1.0)
    assert rows[0]["byte_zero_ratio"] == pytest.approx(0.0)
    assert rows[0]["byte_high_bit_ratio"] == pytest.approx(0.0)


def test_group_b_byte_loader_none_fills_zeros():
    rows = [_mk_row("apk", "o", "p", 0, 3.0)]
    extract_handcrafted_features(rows, byte_loader=None)
    r = rows[0]
    # All Group B fields must exist and be 0.0 (shape-preserving).
    assert r["byte_printable_ratio"] == pytest.approx(0.0)
    assert r["byte_chi2_uniform"] == pytest.approx(0.0)
    assert r["byte_unique_count_log2"] == pytest.approx(0.0)


def test_group_b_empty_region_yields_zeros_without_crashing():
    rows = [_mk_row("apk", "o", "p", 0, 0.0)]
    extract_handcrafted_features(rows, byte_loader=lambda _row: b"")
    r = rows[0]
    for name in (
        "byte_printable_ratio",
        "byte_zero_ratio",
        "byte_high_bit_ratio",
        "byte_ascii_ratio",
        "byte_chi2_uniform",
        "byte_unique_count_log2",
        "byte_range_span",
        "byte_max_run_len_log2",
    ):
        assert r[name] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Group G: region position
# ---------------------------------------------------------------------------


def test_group_g_region_position_normalised():
    # Single APK, two objects, 3 regions in obj1 and 2 in obj2.
    rows = [
        _mk_row("apk", "obj1", "a", 0, 1.0),
        _mk_row("apk", "obj1", "a", 100, 1.0),
        _mk_row("apk", "obj1", "a", 200, 1.0),
        _mk_row("apk", "obj2", "b", 0, 1.0),
        _mk_row("apk", "obj2", "b", 50, 1.0),
    ]
    extract_handcrafted_features(rows, byte_loader=None)

    # Object-level: obj1 has regions at offsets [0, 100, 200]; the
    # middle region (offset=100) is at rank 1 / (3-1) = 0.5.
    assert rows[1]["region_index_in_object_norm"] == pytest.approx(0.5)
    # First region in obj1 is at offset 0 / 200 = 0.0.
    assert rows[0]["region_offset_in_object_norm"] == pytest.approx(0.0)
    # Last region is at offset 200 / 200 = 1.0.
    assert rows[2]["region_offset_in_object_norm"] == pytest.approx(1.0)

    # APK-level: 2 objects in the APK, so object_index_in_apk_norm is
    # 0/1 = 0 for obj1 and 1/1 = 1 for obj2.
    assert rows[0]["object_index_in_apk_norm"] == pytest.approx(0.0)
    assert rows[3]["object_index_in_apk_norm"] == pytest.approx(1.0)


def test_group_g_handles_single_object_apk_without_div_by_zero():
    # If an APK has only one object, the denominator
    # (object_count - 1) would be 0; the implementation must clamp.
    rows = [
        _mk_row("apk", "only", "a", 0, 1.0),
        _mk_row("apk", "only", "a", 100, 1.0),
    ]
    extract_handcrafted_features(rows, byte_loader=None)
    # Both regions in the single object get object_index_in_apk_norm = 0.
    assert rows[0]["object_index_in_apk_norm"] == pytest.approx(0.0)
    assert rows[1]["object_index_in_apk_norm"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Multi-APK isolation
# ---------------------------------------------------------------------------


def test_cross_apk_isolation_in_position_features():
    # Two APKs, each with its own object list. APK B's object_index
    # numbering must restart at 0 (not pick up from APK A).
    rows = [
        _mk_row("A", "objA1", "a", 0, 1.0),
        _mk_row("A", "objA2", "b", 0, 1.0),
        _mk_row("B", "objB1", "c", 0, 1.0),
        _mk_row("B", "objB2", "d", 0, 1.0),
    ]
    extract_handcrafted_features(rows, byte_loader=None)
    # Within APK A: objA1 -> 0, objA2 -> 1 / (2-1) = 1.
    assert rows[0]["object_index_in_apk_norm"] == pytest.approx(0.0)
    assert rows[1]["object_index_in_apk_norm"] == pytest.approx(1.0)
    # Within APK B: objB1 -> 0, objB2 -> 1.
    assert rows[2]["object_index_in_apk_norm"] == pytest.approx(0.0)
    assert rows[3]["object_index_in_apk_norm"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end: every returned row has all 15 named feature keys
# ---------------------------------------------------------------------------


def test_extract_writes_every_feature_name_on_every_row():
    rows = [
        _mk_row("apk", "obj", "p", 0, 3.0),
        _mk_row("apk", "obj", "p", 16, 5.0),
    ]
    extract_handcrafted_features(rows, byte_loader=lambda _r: b"\x00\x01\x02" * 16)
    expected = set(handcrafted_feature_names())
    for r in rows:
        for name in expected:
            assert name in r, f"row missing feature key {name!r}"


def test_empty_input_is_noop():
    # Must not raise. No changes to make.
    extract_handcrafted_features([], byte_loader=None)


def test_custom_entropy_delta_neighbor_half_window():
    # Pass a narrower neighbor window via the entropy_delta_config
    # nested dataclass; verify the produced delta differs.
    rows_default = [
        _mk_row("apk", "o", "p", i * 16, float(i + 1)) for i in range(5)
    ]
    rows_narrow = [dict(r) for r in rows_default]

    extract_handcrafted_features(rows_default, byte_loader=None)
    narrow_cfg = HandcraftedFeatureConfig(
        entropy_delta_config=EntropyDeltaConfig(neighbor_half_window=1)
    )
    extract_handcrafted_features(rows_narrow, byte_loader=None, config=narrow_cfg)

    # At index 2 with window=2 neighbours are [1, 2, 4, 5], mean 3, delta 0.
    # With window=1 neighbours are [2, 4], mean 3, delta 0. Both happen
    # to be 0, so compare a non-central index instead.
    # At index 0: window=2 -> neighbours [2, 3] mean 2.5, delta = 1-2.5 = -1.5
    #             window=1 -> neighbours [2] mean 2.0, delta = 1-2.0 = -1.0
    assert rows_default[0]["entropy_delta_neighbor"] == pytest.approx(-1.5)
    assert rows_narrow[0]["entropy_delta_neighbor"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Group C (compression/magic) + Group F (ZIP/object context) toggle tests
# ---------------------------------------------------------------------------
# NOTE 2026-05-03: the combination Group A+B+G+C+F was historically called
# "Pass-2b" (22 dims). Pass-2b as a feature pack is RETRACTED (L1
# ablation regressed vs Pass-2a on Gen3 4-fold). The individual
# Group-C and Group-F toggles remain live API features; these tests
# pin the toggle behaviour, not the retracted pack.


def _mk_row_with_type(apk, obj, obj_path, offset, entropy, object_type):
    """Variant of _mk_row carrying an ``object_type`` (needed for Group F)."""
    return {
        "apk_id": apk,
        "object_id": obj,
        "object_path": obj_path,
        "object_type": object_type,
        "offset_start": offset,
        "offset_end": offset + 16,
        "entropy": entropy,
    }


def test_group_c_compression_signature_adds_four_dims():
    """Enabling ``include_compression_signature`` appends exactly the
    four Group-C names in the documented order, right after Group G."""
    cfg = HandcraftedFeatureConfig(include_compression_signature=True)
    names = handcrafted_feature_names(cfg)
    # Pass-2a = 15, + 4 Group-C = 19.
    assert len(names) == 19
    # Tail anchors: Group C must come after object_index_in_apk_norm
    # (last Group-G entry) and in the C-specific order.
    assert names[-5:] == [
        "object_index_in_apk_norm",
        "has_dex_magic",
        "has_elf_magic",
        "has_png_magic",
        "has_zip_local_header",
    ]


def test_group_f_zip_context_adds_three_dims():
    """Enabling ``include_zip_context`` appends exactly the three
    Group-F names."""
    cfg = HandcraftedFeatureConfig(include_zip_context=True)
    names = handcrafted_feature_names(cfg)
    assert len(names) == 18  # 15 + 3
    assert names[-3:] == [
        "object_size_log2",
        "object_type_is_embedded_archive",
        "object_type_is_asset_blob",
    ]


def test_groups_c_and_f_together_yield_22_dims():
    """Group C + Group F together on top of Pass-2a give 15 + 4 + 3 = 22.

    This combination historically shipped as "Pass-2b"; the pack is
    RETRACTED (see module docstring) but the dimension arithmetic on
    the toggles is still a correctness contract. Group D / E are still
    reserved, so the final 34-dim target is not reached until a
    subsequent pass lands them.
    """
    cfg = HandcraftedFeatureConfig(
        include_compression_signature=True,
        include_zip_context=True,
    )
    names = handcrafted_feature_names(cfg)
    assert len(names) == 22


def test_group_c_dex_magic_on_raw_dex():
    """A region whose first bytes are ``dex\\n035\\0`` fires has_dex_magic=1."""
    rows = [_mk_row_with_type("apk", "o", "classes.dex", 0, 7.0, "dex")]
    # Fake byte loader returns raw DEX magic followed by padding.
    fake_dex = b"dex\n035\x00" + b"\x00" * 200

    def loader(row):
        return fake_dex

    cfg = HandcraftedFeatureConfig(
        include_compression_signature=True,
        include_byte_distribution=False,  # keep vector focused on Group C
    )
    extract_handcrafted_features(rows, byte_loader=loader, config=cfg)
    assert rows[0]["has_dex_magic"] == pytest.approx(1.0)
    assert rows[0]["has_elf_magic"] == pytest.approx(0.0)
    assert rows[0]["has_png_magic"] == pytest.approx(0.0)
    assert rows[0]["has_zip_local_header"] == pytest.approx(0.0)


def test_group_c_detects_shifted_dex():
    """DEX magic at offset 50 (within the 256-byte scan window) still
    fires the flag -- this matches the 'shifted DEX' layout used by
    some open-source packers."""
    rows = [_mk_row_with_type("apk", "o", "assets/a.dat", 0, 7.0, "asset_blob")]
    # 50 random bytes, then DEX magic, then padding.
    fake_blob = b"\x01" * 50 + b"dex\n037\x00" + b"\x00" * 200

    def loader(row):
        return fake_blob

    cfg = HandcraftedFeatureConfig(
        include_compression_signature=True,
        include_byte_distribution=False,
    )
    extract_handcrafted_features(rows, byte_loader=loader, config=cfg)
    assert rows[0]["has_dex_magic"] == pytest.approx(1.0)


def test_group_c_zip_local_header():
    """``PK\\x03\\x04`` (ZIP local-file header) fires has_zip_local_header
    -- the key feature for catching ZIP-in-ZIP (``embedded_archive``)."""
    rows = [_mk_row_with_type("apk", "o", "assets/hide.png", 0, 7.9, "embedded_archive")]
    fake_zip = b"PK\x03\x04" + b"\x00" * 200

    def loader(row):
        return fake_zip

    cfg = HandcraftedFeatureConfig(
        include_compression_signature=True,
        include_byte_distribution=False,
    )
    extract_handcrafted_features(rows, byte_loader=loader, config=cfg)
    assert rows[0]["has_zip_local_header"] == pytest.approx(1.0)
    assert rows[0]["has_dex_magic"] == pytest.approx(0.0)


def test_group_c_elf_and_png():
    rows = [
        _mk_row_with_type("a1", "o1", "lib/x86/libfoo.so", 0, 6.8, "native_lib"),
        _mk_row_with_type("a2", "o2", "res/drawable/icon.png", 0, 7.5, "resource"),
    ]

    def loader(row):
        if row["object_id"] == "o1":
            return b"\x7fELF" + b"\x00" * 200
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 200

    cfg = HandcraftedFeatureConfig(
        include_compression_signature=True,
        include_byte_distribution=False,
    )
    extract_handcrafted_features(rows, byte_loader=loader, config=cfg)
    assert rows[0]["has_elf_magic"] == pytest.approx(1.0)
    assert rows[0]["has_png_magic"] == pytest.approx(0.0)
    assert rows[1]["has_png_magic"] == pytest.approx(1.0)
    assert rows[1]["has_elf_magic"] == pytest.approx(0.0)


def test_group_c_without_loader_yields_zeros():
    """When no byte_loader is provided, Group C fills zeros so the
    feature-vector shape stays fixed (mirrors Group B behaviour)."""
    rows = [_mk_row_with_type("apk", "o", "assets/a.dat", 0, 7.0, "asset_blob")]
    cfg = HandcraftedFeatureConfig(
        include_compression_signature=True,
        include_byte_distribution=False,
    )
    extract_handcrafted_features(rows, byte_loader=None, config=cfg)
    assert rows[0]["has_dex_magic"] == pytest.approx(0.0)
    assert rows[0]["has_elf_magic"] == pytest.approx(0.0)
    assert rows[0]["has_png_magic"] == pytest.approx(0.0)
    assert rows[0]["has_zip_local_header"] == pytest.approx(0.0)


def test_group_f_size_log2_and_type_flags():
    """Group F writes object_size_log2 (via max offset_end) and two
    one-hot type flags (embedded_archive vs asset_blob)."""
    # Object o1 is an embedded_archive with regions ending at 4096 and
    # 8192 (object size = 8192 -> log2 = 13).
    rows = [
        _mk_row_with_type("apk", "o1", "assets/hide.png", 0, 7.9, "embedded_archive"),
        _mk_row_with_type("apk", "o1", "assets/hide.png", 4096, 7.8, "embedded_archive"),
        # Object o2 is an asset_blob with a single region ending at 2048
        # (-> log2 = 11).
        _mk_row_with_type("apk", "o2", "assets/config.dat", 0, 5.0, "asset_blob"),
    ]
    # offset_end is offset_start + 16 per _mk_row_with_type, so
    # manually bump them to the intended sizes:
    rows[0]["offset_end"] = 4096
    rows[1]["offset_end"] = 8192
    rows[2]["offset_end"] = 2048

    cfg = HandcraftedFeatureConfig(
        include_zip_context=True,
        include_byte_distribution=False,
    )
    extract_handcrafted_features(rows, byte_loader=None, config=cfg)

    # Object size = max(offset_end) per object.
    assert rows[0]["object_size_log2"] == pytest.approx(13.0)  # log2(8192)
    assert rows[1]["object_size_log2"] == pytest.approx(13.0)
    assert rows[2]["object_size_log2"] == pytest.approx(11.0)  # log2(2048)

    # Type flags.
    assert rows[0]["object_type_is_embedded_archive"] == pytest.approx(1.0)
    assert rows[0]["object_type_is_asset_blob"] == pytest.approx(0.0)
    assert rows[2]["object_type_is_embedded_archive"] == pytest.approx(0.0)
    assert rows[2]["object_type_is_asset_blob"] == pytest.approx(1.0)


def test_groups_c_and_f_default_off_preserves_pass2a_vector():
    """Group C / F are additive and default-off: the default config
    still emits 15 keys and none of the Group-C / Group-F keys leak
    in."""
    rows = [_mk_row_with_type("apk", "o", "classes.dex", 0, 7.0, "dex")]

    def loader(row):
        return b"dex\n035\x00" + b"\x00" * 200

    extract_handcrafted_features(rows, byte_loader=loader)
    assert "has_dex_magic" not in rows[0]
    assert "object_size_log2" not in rows[0]
    assert "object_type_is_embedded_archive" not in rows[0]
