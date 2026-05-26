"""Entropy-delta features for PayloadHunter-Lite (F-Lite-b).

The raw per-region Shannon entropy is a well-known weak detector for
Android APKs because the whole byte stream is already near the
theoretical 8.0 ceiling once ZIP-deflated resources dominate. See
``docs/method/baseline_numbers.md`` §"Entropy" for the reversed-AUROC
(0.388) observation from Week-1.

This module provides the **contrast-entropy** fix: instead of scoring
the absolute entropy of a region, we score **how unusual its entropy
is relative to its local context**. Three complementary locality
scopes are implemented, each motivated by a different packer threat
model:

* ``entropy_delta_neighbor``: region entropy minus the mean of the
  ``neighbor_window`` preceding + following regions within the **same
  object**. Honest to the Gen3 ``dex_method_inlined`` threat, where
  encrypted payload bytes sit adjacent to real bytecode.
* ``entropy_delta_entry``: region entropy minus the mean of all
  regions that share its ``(apk_id, object_path)``. Honest to the
  Gen3 ``embedded_asset`` / ``embedded_archive`` threat, where the
  payload is a sub-range inside an otherwise-benign-looking ZIP entry.
* ``entropy_delta_apk``: region entropy minus the mean across every
  region in the APK. Honest to the Gen1 / Gen2 whole-object packing
  threat, where the entire payload object is an outlier against the
  rest of the APK.

Literature anchors:

* Lyda & Hamrock, S&P 2007. *Using Entropy Analysis to Find Encrypted
  and Packed Malware* — local entropy profiling in PE binaries.
* Ugarte-Pedrero, Santos, Bringas, CCS 2012. *Countering entropy
  measure attacks on packed software detection* — multi-scale entropy.

Both prior works target single-file PE binaries; the APK setting
introduces a natural three-level hierarchy (region / ZIP entry / APK)
that the neighbor / entry / apk deltas exploit.

API shape
---------
This module intentionally does **not** compute per-region entropy
itself. It expects each input row to already carry an ``entropy``
field (as produced by :mod:`android_packer.regioning` / ``iter_regions``),
and only computes the three deltas. That keeps the module ~100 loc and
lets tests run without constructing real bytes.

Everything is pure stdlib (``defaultdict``, ``bisect``-free, no
``math.*`` tricks beyond basic arithmetic). Numpy / torch are the
consumer's problem.

The canonical reference implementation is
:func:`add_entropy_deltas_inplace`, which mutates a sequence of region
rows. A streaming variant :func:`compute_entropy_deltas_for_group` is
also exposed for callers that already have regions bucketed by
``(apk_id, object_id)``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


__all__ = [
    "DEFAULT_NEIGHBOR_HALF_WINDOW",
    "EntropyDeltaConfig",
    "EntropyDeltaKeys",
    "add_entropy_deltas_inplace",
    "compute_entropy_deltas_for_group",
    "entropy_delta_feature_names",
]


# Match the gate verified in
# ``scripts/validate_entropy_delta_auroc.py`` for parity.
DEFAULT_NEIGHBOR_HALF_WINDOW: int = 2


@dataclass(frozen=True)
class EntropyDeltaConfig:
    """Knobs for entropy-delta extraction.

    Attributes
    ----------
    neighbor_half_window:
        ``entropy_delta_neighbor`` compares each region with up to this
        many neighbours on each side (exclusive of self). Default 2 =
        up to 4 neighbours total.
    entropy_key:
        Field name that holds the per-region raw entropy in every
        input row. Default ``"entropy"``, matching the output of
        :func:`android_packer.regioning.iter_regions`.
    include_neighbor / include_entry / include_apk:
        Toggles for the three locality scopes, kept for ablation
        studies. Disabling a scope skips its computation and does not
        add the corresponding output key.
    """

    neighbor_half_window: int = DEFAULT_NEIGHBOR_HALF_WINDOW
    entropy_key: str = "entropy"
    include_neighbor: bool = True
    include_entry: bool = True
    include_apk: bool = True


@dataclass(frozen=True)
class EntropyDeltaKeys:
    """Output field names written onto each region row.

    Kept as a dataclass rather than plain constants so consumers can
    optionally remap them (e.g. for a different feature naming
    convention in a downstream model) without monkey-patching.
    """

    neighbor: str = "entropy_delta_neighbor"
    entry: str = "entropy_delta_entry"
    apk: str = "entropy_delta_apk"


def entropy_delta_feature_names(
    config: EntropyDeltaConfig | None = None,
    keys: EntropyDeltaKeys | None = None,
) -> List[str]:
    """Return the list of output feature names in a stable order.

    Useful for building a feature vocabulary that must match the one
    baked into a trained model checkpoint.
    """

    cfg = config or EntropyDeltaConfig()
    ks = keys or EntropyDeltaKeys()
    out: List[str] = []
    if cfg.include_neighbor:
        out.append(ks.neighbor)
    if cfg.include_entry:
        out.append(ks.entry)
    if cfg.include_apk:
        out.append(ks.apk)
    return out


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def add_entropy_deltas_inplace(
    rows: Sequence[MutableMapping[str, object]],
    *,
    config: EntropyDeltaConfig | None = None,
    keys: EntropyDeltaKeys | None = None,
) -> None:
    """Populate the three entropy-delta fields on every row in-place.

    The input rows must carry at minimum:

    * ``apk_id`` — identifies the APK
    * ``object_id`` — identifies the ZIP entry (one layer below APK)
    * ``object_path`` — POSIX path of the entry within the APK
    * ``offset_start`` — byte offset of the region within the entry
    * ``entropy`` — per-region Shannon entropy (float in [0, 8])

    All other fields are ignored. The function appends
    ``entropy_delta_neighbor`` / ``_entry`` / ``_apk`` (subject to the
    ``include_*`` toggles) and returns ``None``.

    Complexity: O(N) for ``_entry`` / ``_apk`` scopes (two passes), and
    O(N log N) for ``_neighbor`` (one sort per object). No numpy.
    """

    if not rows:
        return

    cfg = config or EntropyDeltaConfig()
    ks = keys or EntropyDeltaKeys()
    ent_key = cfg.entropy_key

    # ---- entry / apk means ----
    if cfg.include_entry:
        entry_sum: Dict[Tuple[str, str], float] = defaultdict(float)
        entry_cnt: Dict[Tuple[str, str], int] = defaultdict(int)
    if cfg.include_apk:
        apk_sum: Dict[str, float] = defaultdict(float)
        apk_cnt: Dict[str, int] = defaultdict(int)

    for row in rows:
        ent = float(row[ent_key])
        if cfg.include_entry:
            key = (str(row["apk_id"]), str(row["object_path"]))
            entry_sum[key] += ent
            entry_cnt[key] += 1
        if cfg.include_apk:
            apk_key = str(row["apk_id"])
            apk_sum[apk_key] += ent
            apk_cnt[apk_key] += 1

    if cfg.include_entry:
        entry_mean = {k: entry_sum[k] / entry_cnt[k] for k in entry_sum}
    if cfg.include_apk:
        apk_mean = {k: apk_sum[k] / apk_cnt[k] for k in apk_sum}

    # ---- neighbor delta: bucket by (apk_id, object_id), sort by
    # offset_start, produce left+right window mean. ----
    if cfg.include_neighbor:
        per_object: Dict[Tuple[str, str], List[MutableMapping[str, object]]] = defaultdict(list)
        for row in rows:
            per_object[(str(row["apk_id"]), str(row["object_id"]))].append(row)
        half = cfg.neighbor_half_window
        for group in per_object.values():
            group.sort(key=lambda r: int(r["offset_start"]))
            entropies = [float(r[ent_key]) for r in group]
            n = len(group)
            for i, row in enumerate(group):
                left = max(0, i - half)
                right = min(n, i + half + 1)
                neighbours = entropies[left:i] + entropies[i + 1 : right]
                if neighbours:
                    row[ks.neighbor] = float(row[ent_key]) - (
                        sum(neighbours) / len(neighbours)
                    )
                else:
                    # Single-region object: by convention the delta
                    # carries no signal, store 0.0 to keep the feature
                    # vector dense.
                    row[ks.neighbor] = 0.0

    # ---- entry / apk delta writebacks ----
    for row in rows:
        ent = float(row[ent_key])
        if cfg.include_entry:
            key = (str(row["apk_id"]), str(row["object_path"]))
            row[ks.entry] = ent - entry_mean[key]
        if cfg.include_apk:
            row[ks.apk] = ent - apk_mean[str(row["apk_id"])]


def compute_entropy_deltas_for_group(
    group: Sequence[Mapping[str, object]],
    *,
    config: EntropyDeltaConfig | None = None,
) -> List[Dict[str, float]]:
    """Functional variant: return a list of delta dicts.

    ``group`` is expected to already share a single ``apk_id``; the
    function computes neighbor / entry / apk deltas restricted to this
    group. Useful for streaming pipelines that feed the aggregator one
    APK at a time and want a pure function without mutating the input.

    ``_apk`` is computed against the group's own mean, i.e. "what this
    function calls 'apk' is really 'this-group's-apk'". The caller is
    responsible for making sure the group is exactly one APK's worth
    of regions; passing a mixed-APK sequence produces wrong numbers.
    """

    cfg = config or EntropyDeltaConfig()
    if not group:
        return []
    # Use a deep-enough shallow copy so we can reuse the in-place
    # helper without mutating the caller's data.
    mutable_rows: List[MutableMapping[str, object]] = [dict(r) for r in group]
    add_entropy_deltas_inplace(mutable_rows, config=cfg)
    feat_names = entropy_delta_feature_names(cfg)
    out: List[Dict[str, float]] = []
    for r in mutable_rows:
        out.append({name: float(r[name]) for name in feat_names})
    return out
