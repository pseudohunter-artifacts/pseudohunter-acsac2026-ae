"""Unit tests for the per-task RNG seed derivation formula.

Covers the three guarantees from the module docstring:

* Determinism: same inputs => same output.
* Decorrelation: differing input components produce different seeds
  even when the Knuth multiplier would otherwise collide.
* Input validation: empty/invalid inputs raise clear errors.
"""

from __future__ import annotations

import pytest

from android_packer.synthetic.seed_derivation import (
    KNUTH_MULTIPLIER,
    derive_task_rng_seed,
)


def _derive(**kwargs) -> int:
    defaults = {
        "base_seed": 42,
        "package_name": "com.example.app",
        "version_code": "1",
        "transform_family": "xor",
    }
    defaults.update(kwargs)
    return derive_task_rng_seed(**defaults)


def test_derive_is_deterministic():
    a = _derive()
    b = _derive()
    assert a == b


def test_derive_differs_when_package_changes():
    a = _derive(package_name="com.a.app")
    b = _derive(package_name="com.b.app")
    assert a != b


def test_derive_differs_when_version_code_changes():
    a = _derive(version_code="1")
    b = _derive(version_code="2")
    assert a != b


def test_derive_differs_when_transform_family_changes():
    a = _derive(transform_family="xor")
    b = _derive(transform_family="base64")
    assert a != b


def test_derive_differs_when_base_seed_changes():
    a = _derive(base_seed=42)
    b = _derive(base_seed=43)
    assert a != b


def test_derive_fits_in_32_bits():
    # Exercise many combinations and check every output is a valid
    # 32-bit unsigned int (suitable for random.Random(...)).
    for base in (0, 1, 42, 2**32 - 1, 2**40):
        for family in ("xor", "base64", "dex_method_inlined", "so_embedded"):
            for pkg in ("a", "com.example", "x" * 200):
                s = derive_task_rng_seed(
                    base_seed=base,
                    package_name=pkg,
                    version_code="1",
                    transform_family=family,
                )
                assert isinstance(s, int)
                assert 0 <= s < (1 << 32)


def test_derive_rejects_empty_package():
    with pytest.raises(ValueError):
        derive_task_rng_seed(
            base_seed=42,
            package_name="",
            version_code="1",
            transform_family="xor",
        )


def test_derive_rejects_empty_transform_family():
    with pytest.raises(ValueError):
        derive_task_rng_seed(
            base_seed=42,
            package_name="com.example",
            version_code="1",
            transform_family="",
        )


def test_derive_rejects_non_int_base_seed():
    with pytest.raises(TypeError):
        derive_task_rng_seed(
            base_seed="42",  # type: ignore[arg-type]
            package_name="com.example",
            version_code="1",
            transform_family="xor",
        )


def test_knuth_multiplier_matches_known_constant():
    # Pin the constant so future refactors cannot silently change the
    # derivation formula without a failing test.
    assert KNUTH_MULTIPLIER == 2_654_435_761


def test_variance_across_gen3_dex_method_inlined_tasks():
    """Regression test for the Gen3 ``dex_method_inlined`` layout-collapse bug.

    Before the derivation was introduced, 7 Gen3 tasks sharing the
    same global ``rng_seed=42`` produced identical segment layouts.
    This test synthesises 7 distinct seed APK identities, derives
    their effective seeds, and asserts that all 7 are distinct —
    guarding against any future refactor that accidentally makes the
    derivation degenerate back to a constant.
    """

    packages = [
        ("org.example.keepass", "13"),
        ("org.example.signal", "45"),
        ("org.example.tasks", "8"),
        ("org.example.newpipe", "23"),
        ("org.example.opentracks", "11"),
        ("org.example.briar", "19"),
        ("org.example.fennec", "2025"),
    ]
    seeds = {
        derive_task_rng_seed(
            base_seed=42,
            package_name=pkg,
            version_code=ver,
            transform_family="dex_method_inlined",
        )
        for pkg, ver in packages
    }
    assert len(seeds) == 7
