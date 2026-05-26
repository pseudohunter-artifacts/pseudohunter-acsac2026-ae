"""Unit tests for android_packer.features.entropy_delta.

The tests construct small synthetic region sequences where we can
compute the expected delta values by hand, so that a drift between
the module and the already-deployed
``scripts/validate_entropy_delta_auroc.py`` stays visible.
"""

from __future__ import annotations

import pytest

from android_packer.features import (
    EntropyDeltaConfig,
    EntropyDeltaKeys,
    add_entropy_deltas_inplace,
    compute_entropy_deltas_for_group,
    entropy_delta_feature_names,
)


def _mk_row(apk_id, object_id, object_path, offset, entropy):
    return {
        "apk_id": apk_id,
        "object_id": object_id,
        "object_path": object_path,
        "offset_start": offset,
        "entropy": entropy,
    }


def test_feature_names_default_order():
    names = entropy_delta_feature_names()
    assert names == [
        "entropy_delta_neighbor",
        "entropy_delta_entry",
        "entropy_delta_apk",
    ]


def test_feature_names_with_toggles():
    cfg = EntropyDeltaConfig(include_neighbor=False, include_apk=False)
    assert entropy_delta_feature_names(cfg) == ["entropy_delta_entry"]


def test_single_object_neighbor_delta_matches_hand_compute():
    """5 regions within one object, half_window=2. For the middle
    region (i=2), the 4 neighbours are [1, 3, 5, 7] so mean = 4; its
    entropy is 4 so delta_neighbor = 0. Border cases clip the window.
    """

    rows = [
        _mk_row("apk", "obj", "assets/x", 0, 1.0),
        _mk_row("apk", "obj", "assets/x", 16, 3.0),
        _mk_row("apk", "obj", "assets/x", 32, 4.0),
        _mk_row("apk", "obj", "assets/x", 48, 5.0),
        _mk_row("apk", "obj", "assets/x", 64, 7.0),
    ]
    add_entropy_deltas_inplace(rows)

    # Middle region: neighbours [1, 3, 5, 7], mean = 4.0, delta = 0.
    assert rows[2]["entropy_delta_neighbor"] == pytest.approx(0.0)
    # First region: neighbours [3, 4] -> mean 3.5, delta = 1.0 - 3.5.
    assert rows[0]["entropy_delta_neighbor"] == pytest.approx(-2.5)
    # Last region: neighbours [4, 5] -> mean 4.5, delta = 7.0 - 4.5.
    assert rows[4]["entropy_delta_neighbor"] == pytest.approx(2.5)


def test_single_object_entropy_delta_entry():
    rows = [
        _mk_row("apk", "obj1", "assets/a", 0, 2.0),
        _mk_row("apk", "obj1", "assets/a", 16, 4.0),
        _mk_row("apk", "obj2", "assets/b", 0, 8.0),  # different entry
    ]
    add_entropy_deltas_inplace(rows)
    # assets/a mean = 3.0 -> deltas are -1.0 and +1.0.
    assert rows[0]["entropy_delta_entry"] == pytest.approx(-1.0)
    assert rows[1]["entropy_delta_entry"] == pytest.approx(1.0)
    # assets/b has only one region -> delta = 0.
    assert rows[2]["entropy_delta_entry"] == pytest.approx(0.0)


def test_entropy_delta_apk_uses_full_apk_mean():
    rows = [
        _mk_row("apk", "obj1", "x", 0, 2.0),
        _mk_row("apk", "obj1", "x", 16, 4.0),
        _mk_row("apk", "obj2", "y", 0, 6.0),
    ]
    add_entropy_deltas_inplace(rows)
    # APK mean = (2 + 4 + 6)/3 = 4.0.
    assert rows[0]["entropy_delta_apk"] == pytest.approx(-2.0)
    assert rows[1]["entropy_delta_apk"] == pytest.approx(0.0)
    assert rows[2]["entropy_delta_apk"] == pytest.approx(2.0)


def test_multi_apk_does_not_leak_across_apks():
    """The delta means must respect the apk_id bucketing; a row in
    apk_A must never be averaged with rows from apk_B.
    """

    rows = [
        _mk_row("A", "o", "p", 0, 1.0),
        _mk_row("A", "o", "p", 16, 3.0),
        _mk_row("B", "o", "p", 0, 100.0),
        _mk_row("B", "o", "p", 16, 102.0),
    ]
    add_entropy_deltas_inplace(rows)
    # A's entry mean = 2, B's entry mean = 101.
    assert rows[0]["entropy_delta_entry"] == pytest.approx(-1.0)
    assert rows[1]["entropy_delta_entry"] == pytest.approx(1.0)
    assert rows[2]["entropy_delta_entry"] == pytest.approx(-1.0)
    assert rows[3]["entropy_delta_entry"] == pytest.approx(1.0)


def test_disable_scope_does_not_write_its_key():
    rows = [
        _mk_row("apk", "o", "p", 0, 1.0),
        _mk_row("apk", "o", "p", 16, 3.0),
    ]
    cfg = EntropyDeltaConfig(include_neighbor=False, include_apk=False)
    add_entropy_deltas_inplace(rows, config=cfg)
    for r in rows:
        assert "entropy_delta_entry" in r
        assert "entropy_delta_neighbor" not in r
        assert "entropy_delta_apk" not in r


def test_custom_keys_rename_output_fields():
    rows = [
        _mk_row("apk", "o", "p", 0, 1.0),
        _mk_row("apk", "o", "p", 16, 3.0),
    ]
    ks = EntropyDeltaKeys(neighbor="n", entry="e", apk="a")
    add_entropy_deltas_inplace(rows, keys=ks)
    for r in rows:
        assert "n" in r
        assert "e" in r
        assert "a" in r


def test_compute_entropy_deltas_for_group_does_not_mutate_input():
    rows = [
        _mk_row("apk", "o", "p", 0, 1.0),
        _mk_row("apk", "o", "p", 16, 3.0),
    ]
    snapshot = [dict(r) for r in rows]
    out = compute_entropy_deltas_for_group(rows)
    assert rows == snapshot  # input untouched
    assert len(out) == 2
    assert out[0]["entropy_delta_entry"] == pytest.approx(-1.0)
    assert out[1]["entropy_delta_entry"] == pytest.approx(1.0)


def test_empty_input_is_noop():
    # Must not raise, regardless of config.
    add_entropy_deltas_inplace([])
    assert compute_entropy_deltas_for_group([]) == []


def test_parity_with_validate_script_default_window():
    """Regression guard: the default half_window must match the value
    used in scripts/validate_entropy_delta_auroc.py so that a
    precheck run and a module-based training run see identical
    features.
    """

    from android_packer.features.entropy_delta import DEFAULT_NEIGHBOR_HALF_WINDOW

    assert DEFAULT_NEIGHBOR_HALF_WINDOW == 2
