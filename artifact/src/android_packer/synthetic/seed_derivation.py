"""Per-task RNG seed derivation for synthetic APK generation.

Centralises the formula used by the ``run_synthetic_*_baseline`` CLIs to
turn a **global** ``rng_seed`` from the experiment config into a
**per-task** effective seed. Having this live in one place guarantees
both CLIs agree on the mapping, and makes it trivial to unit-test the
determinism / decorrelation properties.

Problem motivating this module
------------------------------
Before this derivation landed, every task was built with
``random.Random(rng_seed)`` where ``rng_seed = 42`` globally. The
result: Gen3 ``dex_method_inlined`` produced identical segment layouts
across 7 different seed APKs (``[3293, 1259, 827]`` byte split every
single time). Reviewers correctly pointed out that 7 tasks * identical
layout = 1 task repeated 7 times. See
:mod:`docs/method/threat_model.md` §"B2 实装后遗留：task 间 layout
一致性" for the full diagnosis.

Derivation formula (per 2026-04-29 decision, user-approved option 1)
---------------------------------------------------------------------

.. code-block::

    tag = f"{package_name}|{version_code}|{transform_family}".encode()
    h = int(sha256(tag).hexdigest()[:8], 16)            # 32-bit tag hash
    effective_seed = (base_seed * 2654435761 + h) & 0xFFFFFFFF

Properties:

* **Deterministic + reproducible**: same ``(base, package, version,
  transform)`` always yields the same ``effective_seed``.
* **Decorrelated across tasks**: Knuth multiplicative hash
  (``2654435761`` is Knuth's integer-hash multiplier) mixes
  ``base_seed`` with the SHA-256 prefix of the task tag, so two tasks
  that differ in any of the four inputs get uncorrelated RNG states.
* **No file I/O**: the derivation reads only pure-Python strings /
  ints, so it is cheap to compute and cannot race with the seed APK
  file on disk.

Manifests emitted by the CLIs should record **both** ``rng_seed_base``
(the config-level global) and ``rng_seed`` (the derived per-task
value) so that an individual task can be reproduced in isolation
while a full sweep is reproducible end-to-end.
"""

from __future__ import annotations

from hashlib import sha256


__all__ = ["derive_task_rng_seed", "KNUTH_MULTIPLIER"]


# Knuth's integer hash multiplier (phi-based, well-known constant from
# The Art of Computer Programming vol. 3 §6.4). Any non-trivial odd
# 32-bit integer works; this one has good dispersion properties.
KNUTH_MULTIPLIER: int = 2_654_435_761


def derive_task_rng_seed(
    *,
    base_seed: int,
    package_name: str,
    version_code: str,
    transform_family: str,
) -> int:
    """Return a deterministic 32-bit per-task RNG seed.

    Parameters
    ----------
    base_seed:
        Global ``synthetic.rng_seed`` from the experiment config. Can
        be any Python int; only the low 32 bits of the product
        ``base_seed * KNUTH_MULTIPLIER`` affect the result.
    package_name:
        Seed APK's Android package name, e.g. ``"org.keepassdx.keepassdx"``.
    version_code:
        Seed APK's Android ``versionCode`` (stringified). We accept
        strings because the seed manifest uses raw JSON values.
    transform_family:
        One of the registered synthetic transform families (``"xor"``,
        ``"base64"``, ``"dex_method_inlined"``, ...).

    Returns
    -------
    int
        A 32-bit non-negative integer suitable as an argument to
        :class:`random.Random`.
    """

    if not isinstance(base_seed, int):
        raise TypeError(
            f"base_seed must be int, got {type(base_seed).__name__}"
        )
    if not package_name:
        raise ValueError("package_name must be a non-empty string")
    if not transform_family:
        raise ValueError("transform_family must be a non-empty string")

    tag = f"{package_name}|{version_code}|{transform_family}".encode("utf-8")
    tag_hash = int(sha256(tag).hexdigest()[:8], 16)
    return (base_seed * KNUTH_MULTIPLIER + tag_hash) & 0xFFFFFFFF
