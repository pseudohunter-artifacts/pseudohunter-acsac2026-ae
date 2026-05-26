"""Dataset split construction utilities.

Splits are central to evaluating *generalization* in this project:

- **seen vs. unseen packer** — train on some transform families, test on
  others, to estimate how well a detector transfers to new packers.
- **seen vs. unseen package** — hold out entire Android packages so the
  detector can't cheat by memorising per-app artifacts.

Every split returns a deterministic, sorted tuple of IDs for ``train`` /
``val`` / ``test`` partitions. The construction is pure Python and does not
touch the file system so it can be combined freely with other pipelines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence

StrategyName = str  # ``by_transform`` or ``by_package``.


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for :func:`build_split`.

    - ``id_field`` is the record key whose values become the partition IDs
      (default ``apk_id``). Every record must contain this field.
    - ``group_field`` is the record key used to form split groups
      (``transform_family`` or ``package_name``); a single record must carry
      one consistent group value.
    - ``train_groups`` / ``val_groups`` / ``test_groups`` list the group
      values that belong to each partition. A group value listed in multiple
      partitions is a configuration error.
    """

    id_field: str
    group_field: str
    train_groups: tuple[str, ...] = ()
    val_groups: tuple[str, ...] = ()
    test_groups: tuple[str, ...] = ()

    def partitions(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": self.train_groups,
            "val": self.val_groups,
            "test": self.test_groups,
        }


@dataclass(frozen=True)
class DatasetSplit:
    """Concrete assignment of record IDs to partitions.

    ``unassigned`` lists IDs whose group didn't match any partition; they are
    excluded from train/val/test but reported so callers can spot silent data
    loss.
    """

    strategy: StrategyName
    train: tuple[str, ...] = ()
    val: tuple[str, ...] = ()
    test: tuple[str, ...] = ()
    unassigned: tuple[str, ...] = ()
    group_to_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "unassigned": list(self.unassigned),
            "group_to_ids": {
                group: list(ids)
                for group, ids in sorted(self.group_to_ids.items())
            },
        }

    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "unassigned": len(self.unassigned),
        }


def build_split(
    records: Iterable[Mapping],
    strategy: StrategyName,
    config: SplitConfig,
) -> DatasetSplit:
    """Build a :class:`DatasetSplit` from record-level metadata.

    Typical usage joins a synthetic-label JSONL (for the ``transform_family``
    group) or a seed-manifest (for the ``package_name`` group) with an APK
    identifier list. ``strategy`` is used purely as a label attached to the
    returned split; the actual partitioning logic is driven by ``config``.
    """

    _validate_partitions(config)

    group_to_ids: dict[str, set[str]] = {}
    for record in records:
        record_id = _require_field(record, config.id_field)
        group = _require_field(record, config.group_field)
        group_to_ids.setdefault(str(group), set()).add(str(record_id))

    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    known_group_values = set()
    for partition, groups in config.partitions().items():
        for group in groups:
            known_group_values.add(group)
            for record_id in group_to_ids.get(group, ()):  # empty group is OK
                assigned[partition].append(record_id)

    unassigned: list[str] = []
    for group, ids in group_to_ids.items():
        if group not in known_group_values:
            unassigned.extend(ids)

    return DatasetSplit(
        strategy=strategy,
        train=_sorted_unique(assigned["train"]),
        val=_sorted_unique(assigned["val"]),
        test=_sorted_unique(assigned["test"]),
        unassigned=_sorted_unique(unassigned),
        group_to_ids={
            group: _sorted_unique(ids)
            for group, ids in group_to_ids.items()
        },
    )


def by_transform_split(
    records: Iterable[Mapping],
    *,
    train_families: Sequence[str],
    val_families: Sequence[str] = (),
    test_families: Sequence[str] = (),
    id_field: str = "apk_id",
    group_field: str = "transform_family",
) -> DatasetSplit:
    """Convenience wrapper for the common ``by_transform`` split.

    Example: ``train_families=("xor", "base64")`` + ``test_families=("split_xor",)``
    yields a "seen vs. unseen packer" evaluation.
    """

    config = SplitConfig(
        id_field=id_field,
        group_field=group_field,
        train_groups=tuple(train_families),
        val_groups=tuple(val_families),
        test_groups=tuple(test_families),
    )
    return build_split(records, "by_transform", config)


def by_package_split(
    records: Iterable[Mapping],
    *,
    train_packages: Sequence[str],
    val_packages: Sequence[str] = (),
    test_packages: Sequence[str] = (),
    id_field: str = "apk_id",
    group_field: str = "package_name",
) -> DatasetSplit:
    """Convenience wrapper for the ``by_package`` split."""

    config = SplitConfig(
        id_field=id_field,
        group_field=group_field,
        train_groups=tuple(train_packages),
        val_groups=tuple(val_packages),
        test_groups=tuple(test_packages),
    )
    return build_split(records, "by_package", config)


def _validate_partitions(config: SplitConfig) -> None:
    seen: dict[str, str] = {}
    for partition, groups in config.partitions().items():
        for group in groups:
            if group in seen and seen[group] != partition:
                raise ValueError(
                    f"group '{group}' cannot appear in both "
                    f"{seen[group]!r} and {partition!r}"
                )
            seen[group] = partition
    if not any(config.partitions().values()):
        raise ValueError("at least one of train/val/test must contain a group")


def _require_field(record: Mapping, field_name: str) -> object:
    if field_name not in record:
        raise KeyError(
            f"record is missing required field '{field_name}': keys={sorted(record.keys())}"
        )
    return record[field_name]


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


__all__ = [
    "DatasetSplit",
    "SplitConfig",
    "build_split",
    "by_package_split",
    "by_transform_split",
]
