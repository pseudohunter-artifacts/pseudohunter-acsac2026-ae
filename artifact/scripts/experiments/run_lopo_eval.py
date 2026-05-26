"""Leave-One-Packer-Out (LOPO) evaluation for Pseudo-code BERT v4.

For each fold, hold out one packer family as test packed set.
Train on all OTHER packers + all benign. Evaluate on held-out packer + unseen benign.

This gives a fair cross-packer generalization metric.

Usage:
    python scripts/experiments/run_lopo_eval.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from android_packer.apkio.objects import iter_apk_objects
from android_packer.decoders.pseudo_tokenizer import (
    BYTE_REPRESENTATIONS,
    BYTE_REPRESENTATION_LEGACY_RAW,
    PseudoCodeTokenizer,
)
from android_packer.features.full_feature_extractor import (
    SCALAR_FEATURE_DIM,
    extract_apk_context,
    extract_region_features,
)
from android_packer.labeling.happer_diff import compute_paired_diff, parse_inject_labels
from android_packer.models.entry_aggregator import (
    APKMILConfig,
    EntryAggregatorConfig,
    build_apk_mil,
    build_entry_aggregator,
)
from android_packer.models.fusion_encoder import FusionEncoderConfig, build_fusion_encoder
from android_packer.regioning.typed_slicer import ENTRY_COARSE_TYPES, iter_typed_regions

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HAPPER = ROOT / "data" / "happer_dataset" / "FSet"
TRACK_B = ROOT / "data" / "real_world" / "track_b"
TRACK_B_V2 = ROOT / "data" / "real_world" / "track_b_v2"
PRETRAIN_CKPT = ROOT / "outputs" / "experiments" / "pseudo_bert_v3" / "pretrained_bert_v2.pt"
OUT_DIR = ROOT / "outputs" / "experiments" / "lopo_eval"
PATH_ABLATION_DIR = ROOT / "outputs" / "experiments" / "path_ablation"
BAG_CACHE_DIR = ROOT / "outputs" / "experiments" / "lopo_eval" / "bag_cache"

VALID_PATHS = ("dalvik", "arm64", "byte")
STRICT_DPT_CONTROL_MODES = (
    "non_dpt",
    "add_old_dpt",
    "add_old_dpt_benign",
    "other_positive_replay",
)
BAG_CACHE_VERSION = 3
UNKNOWN_ENTRY_TYPE_ID = ENTRY_COARSE_TYPES.index("unknown")
APKID_DIRTY_STRICT_BENIGN = {
    "005AF753A03FA7D753FD2C8988E91B47966187A504F24E4187BDD19AF5797B00",
}


def parse_active_paths(value: str) -> Tuple[str, ...]:
    paths = tuple(p.strip().lower() for p in value.split(",") if p.strip())
    if not paths:
        raise argparse.ArgumentTypeError("--paths must select at least one path")
    unknown = sorted(set(paths) - set(VALID_PATHS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown path(s): {unknown}; valid paths are {', '.join(VALID_PATHS)}"
        )
    return paths


def paired_key_from_apk_id(apk_id: str) -> str:
    """Return a stable origin key for known packed/unpacked bag ID patterns."""
    for prefix in (
        "benign_",
        "test_benign_",
        "az_benign_",
        "track_b_v2_benign_",
        "track_b_v2_dpt_",
    ):
        if apk_id.startswith(prefix):
            return apk_id[len(prefix):]
    if "_flat_" in apk_id:
        return apk_id.split("_flat_", 1)[1]
    parts = apk_id.split("_", 1)
    return parts[1] if len(parts) == 2 else apk_id


def is_apkid_dirty_strict_packed_path(path: Path) -> bool:
    stem = path.stem.upper()
    if not stem.startswith("DPT__"):
        return False
    packed_prefix = stem.split("DPT__", 1)[1]
    return any(dirty.startswith(packed_prefix) for dirty in APKID_DIRTY_STRICT_BENIGN)


def group_bags_by_pair_key(bags: Sequence[Mapping]) -> Dict[str, Dict[int, List[Mapping]]]:
    grouped: Dict[str, Dict[int, List[Mapping]]] = {}
    for bag in bags:
        key = paired_key_from_apk_id(str(bag.get("apk_id", "")))
        label = int(bag["apk_label"])
        grouped.setdefault(key, {0: [], 1: []})[label].append(bag)
    return grouped


def lookup_pair_bags(
    grouped: Mapping[str, Mapping[int, List[Mapping]]],
    bag: Mapping,
    target_label: int,
) -> List[Mapping]:
    key = paired_key_from_apk_id(str(bag.get("apk_id", "")))
    return list(grouped.get(key, {}).get(target_label, []))


def select_equal_size_other_positive_replay(
    families: Sequence[PackerFamily],
    held_out_name: str,
    target_count: int,
) -> List[Mapping]:
    """Select deterministic non-held-out positive replay bags for B1.1 control D."""
    if target_count <= 0:
        return []
    candidates: List[Mapping] = []
    for fam in sorted(families, key=lambda item: item.name):
        if fam.name == held_out_name:
            continue
        candidates.extend(fam.bags)
    return candidates[:target_count]


def build_strict_dpt_control_training_set(
    benign_bags: Sequence[Mapping],
    families: Sequence[PackerFamily],
    *,
    control_mode: str,
    dpt_benign_bags: Sequence[Mapping] = (),
) -> Tuple[List[Mapping], List[str], Dict[str, object]]:
    """Build B1.1 strict-DPT control train bags without changing test labels.

    The control modes are diagnostic-only:
    - non_dpt: train on non-DPT packers + benign.
    - add_old_dpt: add old Track B DPT positives.
    - add_old_dpt_benign: add old Track B DPT positives and paired benign bags.
    - other_positive_replay: add the same number of non-DPT positives instead
      of DPT positives, testing whether gains come from positive count alone.
    """
    if control_mode not in STRICT_DPT_CONTROL_MODES:
        raise ValueError(f"unknown strict DPT control mode: {control_mode}")

    train_families = [fam for fam in families if fam.name != "DPT"]
    dpt_families = [fam for fam in families if fam.name == "DPT"]
    dpt_positive_bags: List[Mapping] = []
    for fam in dpt_families:
        dpt_positive_bags.extend(fam.bags)

    train_bags: List[Mapping] = list(benign_bags)
    for fam in train_families:
        train_bags.extend(fam.bags)

    added_dpt_positive = 0
    added_dpt_benign = 0
    added_other_positive = 0

    if control_mode in {"add_old_dpt", "add_old_dpt_benign"}:
        train_bags.extend(dpt_positive_bags)
        added_dpt_positive = len(dpt_positive_bags)
    elif control_mode == "other_positive_replay":
        replay = select_equal_size_other_positive_replay(
            train_families,
            held_out_name="DPT",
            target_count=len(dpt_positive_bags),
        )
        train_bags.extend(replay)
        added_other_positive = len(replay)

    if control_mode == "add_old_dpt_benign":
        train_bags.extend(dpt_benign_bags)
        added_dpt_benign = len(dpt_benign_bags)

    info: Dict[str, object] = {
        "mode": control_mode,
        "train_families": [fam.name for fam in train_families],
        "old_dpt_positive_available": len(dpt_positive_bags),
        "added_old_dpt_positive": added_dpt_positive,
        "added_old_dpt_benign": added_dpt_benign,
        "added_equal_size_other_positive": added_other_positive,
    }
    return train_bags, [fam.name for fam in train_families], info


def normalize_scores_from_train_benign(
    scores: Sequence[float],
    train_benign_scores: Sequence[float],
    mode: str,
) -> Tuple[List[float], Dict[str, float]]:
    """Normalize scores using only train-benign score statistics."""
    if mode == "none":
        return list(scores), {}
    if not train_benign_scores:
        return list(scores), {"warning": "no_train_benign_scores"}
    mu = float(np.mean(train_benign_scores))
    sigma = float(np.std(train_benign_scores))
    if mode == "train_benign_center":
        normalized = [float(s - mu) for s in scores]
    elif mode == "train_benign_z":
        denom = sigma if sigma > 1e-6 else 1.0
        normalized = [float((s - mu) / denom) for s in scores]
    else:
        raise ValueError(f"unknown score normalization mode: {mode}")
    return normalized, {
        "mode": mode,
        "train_benign_mean": mu,
        "train_benign_std": sigma,
    }


def fixed_fpr_tpr_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Compute fixed-FPR/TPR operating points for strict detection."""
    labels = [int(v) for v in y_true]
    scores = [float(v) for v in y_score]
    positives = sum(1 for v in labels if v == 1)
    negatives = sum(1 for v in labels if v == 0)
    if positives == 0 or negatives == 0 or len(labels) != len(scores):
        return {
            "auprc": None,
            "fpr_at_95_tpr": None,
            "tpr_at_1_fpr": None,
            "tpr_at_5_fpr": None,
        }

    points = []
    for threshold in sorted(set(scores), reverse=True):
        tp = fp = 0
        for label, score in zip(labels, scores):
            if score >= threshold:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
        points.append((tp / positives, fp / negatives))

    fpr_at_95 = min((fpr for tpr, fpr in points if tpr >= 0.95), default=None)
    tpr_at_1 = max((tpr for tpr, fpr in points if fpr <= 0.01), default=0.0)
    tpr_at_5 = max((tpr for tpr, fpr in points if fpr <= 0.05), default=0.0)
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "fpr_at_95_tpr": None if fpr_at_95 is None else float(fpr_at_95),
        "tpr_at_1_fpr": float(tpr_at_1),
        "tpr_at_5_fpr": float(tpr_at_5),
    }


def configure_cuda_performance(device: torch.device, *, allow_tf32: bool) -> None:
    """Enable safe CUDA throughput knobs for Ampere/Ada GPUs."""
    if device.type != "cuda" or not allow_tf32:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def _torch_load_local(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _diff_cache_fingerprint(diff_result) -> str:
    if diff_result is None:
        return "none"
    items = []
    for name, entry_diff in sorted(diff_result.entry_diffs.items()):
        items.append(f"{name}:{entry_diff.diff_score:.6f}")
    return hashlib.sha256("|".join(items).encode("utf-8")).hexdigest()[:16]


def _bag_cache_path(
    apk_path: Path,
    apk_label: int,
    apk_id: str,
    diff_result,
    cache_dir: Optional[Path],
    byte_representation: str = BYTE_REPRESENTATION_LEGACY_RAW,
) -> Optional[Path]:
    if cache_dir is None:
        return None
    stat = apk_path.stat()
    key = {
        "version": BAG_CACHE_VERSION,
        "path": str(apk_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "label": apk_label,
        "apk_id": apk_id,
        "diff": _diff_cache_fingerprint(diff_result),
        "byte_representation": byte_representation,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.pkl"


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _iter_manifest_records(path: Path) -> Iterable[Mapping]:
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        records = payload.get("records", payload.get("apks", []))
        if not isinstance(records, list):
            raise ValueError(f"{path}: expected records/apks list")
        yield from records
    else:
        raise ValueError(f"{path}: unsupported manifest shape {type(payload).__name__}")


def load_hard_benign_manifest_apks(
    manifest_paths: Sequence[Path],
    *,
    limit: int = 0,
    require_hard: bool = False,
) -> List[Path]:
    """Return APKiD-clean, train-allowed APKs from hard-benign manifests."""
    selected: List[Path] = []
    seen = set()
    for manifest_path in manifest_paths:
        for record in _iter_manifest_records(manifest_path):
            if not bool(record.get("train_allowed", False)):
                continue
            if record.get("apkid_clean") is not True:
                continue
            label_class = str(record.get("label_class", ""))
            if label_class not in {"benign-clean", "benign-hard-clean"}:
                continue
            if require_hard and label_class != "benign-hard-clean":
                continue
            raw_path = (
                record.get("local_path")
                or record.get("apk_path")
                or record.get("path")
            )
            if not raw_path:
                continue
            apk_path = _resolve_repo_path(str(raw_path))
            if not apk_path.exists():
                continue
            key = str(apk_path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append(apk_path)
            if limit > 0 and len(selected) >= limit:
                return selected
    return selected


# ---------------------------------------------------------------------------
# Model (same as v3)
# ---------------------------------------------------------------------------


class PseudoBERTModel(torch.nn.Module):
    def __init__(
        self,
        fusion_cfg: FusionEncoderConfig,
        ablation_mode: str = "full",
        active_paths: Sequence[str] = VALID_PATHS,
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.active_paths = tuple(active_paths)
        self.fusion_encoder = build_fusion_encoder(fusion_cfg)
        agg_cfg = EntryAggregatorConfig(
            region_dim=fusion_cfg.output_dim, entry_dim=fusion_cfg.output_dim,
            attn_hidden=128, dropout=0.1,
        )
        self.entry_aggregator = build_entry_aggregator(agg_cfg)
        mil_cfg = APKMILConfig(
            entry_dim=fusion_cfg.output_dim, attn_hidden=128,
            dropout=0.1, use_normality=True,
        )
        self.apk_mil = build_apk_mil(mil_cfg)

    def freeze_bert(self):
        for name, param in self.fusion_encoder.named_parameters():
            if "bert" in name:
                param.requires_grad = False

    def configure_bert_training(self, mode: str, last_n_layers: int = 0):
        bert = self.fusion_encoder.bert
        for param in bert.parameters():
            param.requires_grad = False

        if mode == "frozen":
            return
        if mode == "all":
            for param in bert.parameters():
                param.requires_grad = True
            return
        if mode == "last_n":
            if last_n_layers <= 0:
                return
            for layer in bert.layers[-last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in bert.cls_head.parameters():
                param.requires_grad = True
            return
        raise ValueError(f"Unknown BERT train mode: {mode}")

    def forward_bag(
        self,
        bag: Dict,
        device: torch.device,
        chunk_size: int = 64,
        active_paths: Optional[Sequence[str]] = None,
    ):
        n_regions = bag["stat_features"].shape[0]
        max_regions = 128
        if n_regions > max_regions:
            rng_sub = np.random.RandomState(n_regions)  # deterministic per bag size
            indices = rng_sub.choice(n_regions, max_regions, replace=False)
            indices.sort()
            idx_set = set(indices.tolist())
            new_idx_map = {old: new for new, old in enumerate(indices)}
            entry_boundaries = []
            entry_indices = []
            for original_entry_idx, (start, end) in enumerate(bag["entry_boundaries"]):
                kept_region_indices = [
                    new_idx_map[i] for i in range(start, end) if i in idx_set
                ]
                if kept_region_indices:
                    entry_boundaries.append((kept_region_indices[0], kept_region_indices[-1] + 1))
                    entry_indices.append(original_entry_idx)
        else:
            indices = np.arange(n_regions)
            entry_boundaries = bag["entry_boundaries"]
            entry_indices = list(range(len(entry_boundaries)))

        N = len(indices)
        if N == 0 or not entry_boundaries:
            return torch.zeros((), device=device), torch.zeros((0,), device=device)

        stat = torch.tensor(bag["stat_features"][indices], dtype=torch.float32, device=device)
        d_ids = torch.tensor(bag["dalvik_ids"][indices], dtype=torch.long, device=device)
        d_types = torch.tensor(bag["dalvik_types"][indices], dtype=torch.long, device=device)
        d_mask = torch.tensor(bag["dalvik_mask"][indices], dtype=torch.float32, device=device)
        n_ids = torch.tensor(bag["native_ids"][indices], dtype=torch.long, device=device)
        n_types = torch.tensor(bag["native_types"][indices], dtype=torch.long, device=device)
        n_mask = torch.tensor(bag["native_mask"][indices], dtype=torch.float32, device=device)
        b_ids = torch.tensor(bag["byte_ids"][indices], dtype=torch.long, device=device)
        b_types = torch.tensor(bag["byte_types"][indices], dtype=torch.long, device=device)
        b_mask = torch.tensor(bag["byte_mask"][indices], dtype=torch.float32, device=device)
        entry_type_array = bag.get("entry_type_ids")
        if entry_type_array is None:
            entry_type_array = np.full(n_regions, UNKNOWN_ENTRY_TYPE_ID, dtype=np.int64)
        entry_type_ids = torch.tensor(entry_type_array[indices], dtype=torch.long, device=device)
        paths = tuple(active_paths) if active_paths is not None else self.active_paths

        all_emb, all_susp, all_norm = [], [], []
        for cs in range(0, N, chunk_size):
            ce = min(cs + chunk_size, N)
            s = slice(cs, ce)
            emb, susp, norm = self.fusion_encoder(
                d_ids[s], d_types[s], d_mask[s],
                n_ids[s], n_types[s], n_mask[s],
                b_ids[s], b_types[s], b_mask[s], stat[s],
                active_paths=paths,
                entry_type_ids=entry_type_ids[s],
            )
            all_emb.append(emb)
            all_susp.append(susp)
            all_norm.append(norm)

        embeddings = torch.cat(all_emb)
        suspicion = torch.cat(all_susp)
        normality = torch.cat(all_norm)

        entry_emb, entry_susp, _ = self.entry_aggregator(embeddings, suspicion, entry_boundaries)
        entry_norms = []
        for start, end in entry_boundaries:
            if start < end and end <= len(normality):
                entry_norms.append(normality[start:end].mean())
            else:
                entry_norms.append(torch.tensor(0.5, device=device))
        entry_normality = torch.stack(entry_norms)

        bag_logit, entry_attn, entry_logits = self.apk_mil(entry_emb, entry_normality)

        # Return all localization signals
        # entry_susp: per-entry mean suspicion
        entry_susp_scores = []
        for start, end in entry_boundaries:
            if start < end and end <= len(suspicion):
                entry_susp_scores.append(suspicion[start:end].mean())
            else:
                entry_susp_scores.append(torch.tensor(0.0, device=device))
        entry_suspicion = torch.stack(entry_susp_scores)

        return {
            "bag_logit": bag_logit,
            "entry_attention": entry_attn,
            "entry_normality": entry_normality,
            "entry_suspicion": entry_suspicion,
            "entry_logits": entry_logits,
            "entry_indices": entry_indices,
            "region_normality": normality,
            "region_suspicion": suspicion,
        }


# ---------------------------------------------------------------------------
# Bag building (reuse from v3)
# ---------------------------------------------------------------------------


def build_bag(apk_path, apk_label, tokenizer, diff_result=None, apk_id="", cache_dir: Optional[Path] = None):
    apk_path = Path(apk_path)
    cache_path = _bag_cache_path(
        apk_path,
        apk_label,
        apk_id,
        diff_result,
        cache_dir,
        getattr(tokenizer, "byte_representation", BYTE_REPRESENTATION_LEGACY_RAW),
    )
    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    try:
        entries = []
        dex_counts = {}
        for obj_meta, obj_bytes in iter_apk_objects(apk_path, max_depth=1):
            if len(obj_bytes) >= 64:
                entries.append((obj_meta, obj_bytes))
                if obj_bytes[:4] == b"dex\n" and len(obj_bytes) >= 100:
                    try:
                        s = struct.unpack_from("<I", obj_bytes, 56)[0]
                        t = struct.unpack_from("<I", obj_bytes, 64)[0]
                        m = struct.unpack_from("<I", obj_bytes, 88)[0]
                        f = struct.unpack_from("<I", obj_bytes, 80)[0]
                        dex_counts[obj_meta.object_path] = (s, t, m, f)
                    except (struct.error, IndexError):
                        pass
        if not entries:
            return None

        apk_ctx = extract_apk_context([(m.object_path, b) for m, b in entries])
        all_stat, all_d_ids, all_n_ids, all_b_ids = [], [], [], []
        all_d_types, all_n_types, all_b_types = [], [], []
        all_d_mask, all_n_mask, all_b_mask = [], [], []
        all_entry_type_ids = []
        entry_boundaries, entry_names = [], []
        ridx = 0

        for eidx, (obj_meta, obj_bytes) in enumerate(entries):
            regions = iter_typed_regions(obj_meta, obj_bytes, entry_index=eidx)
            cr = obj_meta.compressed_size / max(obj_meta.size, 1)
            hc = dex_counts.get(obj_meta.object_path, (0, 0, 0, 0))
            start = ridx
            for region in regions:
                rd = obj_bytes[region.offset_start:region.offset_end]
                fv = extract_region_features(region, rd, len(obj_bytes), apk_ctx, cr)
                all_stat.append(fv.scalars)
                d_enc, n_enc, b_enc = tokenizer.encode_region(
                    rd, entry_type=region.entry_type, dex_header_counts=hc,
                )
                all_d_ids.append(d_enc.token_ids)
                all_n_ids.append(n_enc.token_ids)
                all_b_ids.append(b_enc.token_ids)
                all_d_types.append(d_enc.token_type_ids)
                all_n_types.append(n_enc.token_type_ids)
                all_b_types.append(b_enc.token_type_ids)
                all_d_mask.append(d_enc.attention_mask)
                all_n_mask.append(n_enc.attention_mask)
                all_b_mask.append(b_enc.attention_mask)
                all_entry_type_ids.append(region.entry_type_id)
                ridx += 1
            if ridx > start:
                entry_boundaries.append((start, ridx))
                entry_names.append(obj_meta.object_path)

        if not entry_boundaries:
            return None

        diff_targets = None
        if diff_result:
            diff_targets = np.zeros(len(entry_boundaries), dtype=np.float32)
            for i, name in enumerate(entry_names):
                if name in diff_result.entry_diffs:
                    diff_targets[i] = diff_result.entry_diffs[name].diff_score

        bag = {
            "apk_id": apk_id, "apk_label": apk_label,
            "pair_key": paired_key_from_apk_id(apk_id),
            "stat_features": np.array(all_stat, dtype=np.float32),
            "dalvik_ids": np.array(all_d_ids, dtype=np.int64),
            "native_ids": np.array(all_n_ids, dtype=np.int64),
            "byte_ids": np.array(all_b_ids, dtype=np.int64),
            "dalvik_types": np.array(all_d_types, dtype=np.int64),
            "native_types": np.array(all_n_types, dtype=np.int64),
            "byte_types": np.array(all_b_types, dtype=np.int64),
            "dalvik_mask": np.array(all_d_mask, dtype=np.float32),
            "native_mask": np.array(all_n_mask, dtype=np.float32),
            "byte_mask": np.array(all_b_mask, dtype=np.float32),
            "entry_type_ids": np.array(all_entry_type_ids, dtype=np.int64),
            "entry_boundaries": entry_boundaries,
            "entry_names": entry_names,
            "diff_targets": diff_targets,
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(bag, f, protocol=pickle.HIGHEST_PROTOCOL)
        return bag
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading: enumerate all packer families with their bags
# ---------------------------------------------------------------------------


@dataclass
class PackerFamily:
    name: str
    bags: List[Dict]


def load_all_data(
    tokenizer,
    max_per_family: int = 15,
    androzoo_benign: int = 0,
    bag_cache_dir: Optional[Path] = BAG_CACHE_DIR,
    hard_benign_manifests: Optional[Sequence[Path]] = None,
    hard_benign_limit: int = 0,
    hard_benign_only: bool = False,
):
    """Load all packer families and benign data."""
    print("  Loading all data...", flush=True)

    # --- Benign bags ---
    benign_bags = []
    origins = {}  # stem → path (for diff matching)

    # Happer Origin-16 (30 apps) — ONLY these go into training as benign
    origin16_apks = sorted((HAPPER / "Origin-16").glob("*.apk"))[:30]
    print(f"    Loading Origin-16 benign: {len(origin16_apks)} APKs", flush=True)
    for i, p in enumerate(origin16_apks, 1):
        origins[p.stem] = p
        bag = build_bag(p, 0, tokenizer, apk_id=f"benign_{p.stem}", cache_dir=bag_cache_dir)
        if bag:
            benign_bags.append(bag)
        if i % 5 == 0 or i == len(origin16_apks):
            print(f"      Origin-16 {i}/{len(origin16_apks)}", flush=True)

    # NOTE: Track B benign (9 apps) intentionally EXCLUDED from training
    # to avoid app-identity leakage: these are the same apps that Track B
    # packers pack, so including them as benign would let the model
    # "recognize the app" instead of learning generic packing features.
    # They are still used OFFLINE to compute diff_targets for L_align.

    # AndroZoo benign (modern 2020+ apps, no overlap with any test data)
    ANDROZOO_DIR = ROOT / "data" / "androzoo" / "benign_corpus"
    if androzoo_benign > 0 and ANDROZOO_DIR.exists():
        az_apks = sorted(ANDROZOO_DIR.rglob("*.apk"))
        # Deterministic random sample
        rng_az = np.random.RandomState(123)
        if len(az_apks) > androzoo_benign:
            az_indices = rng_az.choice(len(az_apks), androzoo_benign, replace=False)
            az_apks = [az_apks[i] for i in sorted(az_indices)]
        print(f"    Loading {len(az_apks)} AndroZoo benign APKs...", flush=True)
        az_count = 0
        for p in az_apks:
            bag = build_bag(p, 0, tokenizer, apk_id=f"az_benign_{p.stem[:20]}", cache_dir=bag_cache_dir)
            if bag:
                benign_bags.append(bag)
                az_count += 1
                if az_count % 10 == 0:
                    print(f"      ... {az_count} AndroZoo bags built", flush=True)
        print(f"    AndroZoo: {az_count} benign bags added")

    if hard_benign_manifests:
        hard_apks = load_hard_benign_manifest_apks(
            hard_benign_manifests,
            limit=hard_benign_limit,
            require_hard=hard_benign_only,
        )
        print(
            f"    Loading {len(hard_apks)} manifest hard benign APKs...",
            flush=True,
        )
        hard_count = 0
        for p in hard_apks:
            bag = build_bag(
                p,
                0,
                tokenizer,
                apk_id=f"hard_benign_{p.stem[:32]}",
                cache_dir=bag_cache_dir,
            )
            if bag:
                benign_bags.append(bag)
                hard_count += 1
                if hard_count % 5 == 0:
                    print(f"      ... {hard_count} hard benign bags built", flush=True)
        print(f"    Hard benign: {hard_count} benign bags added")

    # Happer Origin-18 (20 apps, used as additional test benign)
    benign_test_bags = []
    origin18_apks = sorted((HAPPER / "Oirgin-18").glob("*.apk"))[:15]
    print(f"    Loading Origin-18 test benign: {len(origin18_apks)} APKs", flush=True)
    for i, p in enumerate(origin18_apks, 1):
        bag = build_bag(p, 0, tokenizer, apk_id=f"test_benign_{p.stem}", cache_dir=bag_cache_dir)
        if bag:
            benign_test_bags.append(bag)
        if i % 5 == 0 or i == len(origin18_apks):
            print(f"      Origin-18 {i}/{len(origin18_apks)}", flush=True)

    print(f"    Benign: {len(benign_bags)} train + {len(benign_test_bags)} test")
    print(f"    (Track B benign excluded from training to avoid leakage)")

    # --- Packed families ---
    families = []

    # Happer families
    for family_name, dirname in [("Ali", "Ali-16"), ("Qihoo", "Qihoo-16"), ("Tencent", "Tencent-16")]:
        family_bags = []
        family_dir = HAPPER / dirname
        if not family_dir.exists():
            continue
        family_apks = sorted(family_dir.glob("*.apk"))[:max_per_family]
        print(f"    Loading {family_name}: {len(family_apks)} APKs", flush=True)
        for i, p in enumerate(family_apks, 1):
            origin_match = None
            for ostem, opath in origins.items():
                if p.stem.startswith(ostem):
                    origin_match = opath
                    break
            diff = compute_paired_diff(origin_match, p) if origin_match else None
            bag = build_bag(p, 1, tokenizer, diff, f"{family_name}_{p.stem}", cache_dir=bag_cache_dir)
            if bag:
                family_bags.append(bag)
            if i % 5 == 0 or i == len(family_apks):
                print(f"      {family_name} {i}/{len(family_apks)}", flush=True)
        if family_bags:
            families.append(PackerFamily(name=family_name, bags=family_bags))
            print(f"    {family_name}: {len(family_bags)} bags")

    # Track B families
    tb_packed = TRACK_B / "packed"
    tb_benign = TRACK_B / "benign"

    for packer_name, short_name in [
        ("cs1_360_jiagu", "360"),
        ("cs3_bangcle", "Bangcle"),
        ("s5_timscriptov_apkprotector_multiplatform", "APKProtector"),
        ("s6_dpt_shell", "DPT"),
    ]:
        family_bags = []
        packer_dir = tb_packed / packer_name
        if packer_dir.exists() and packer_dir.is_dir():
            for seed_dir in sorted(packer_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                packed_apk = seed_dir / "packed.apk"
                if not packed_apk.exists():
                    continue
                benign_apk = tb_benign / f"{seed_dir.name}.apk"
                inject_path = seed_dir / "inject_labels.jsonl"
                if inject_path.exists():
                    diff = parse_inject_labels(inject_path)
                elif benign_apk.exists():
                    diff = compute_paired_diff(benign_apk, packed_apk)
                else:
                    diff = None
                bag = build_bag(
                    packed_apk,
                    1,
                    tokenizer,
                    diff,
                    f"{short_name}_{seed_dir.name}",
                    cache_dir=bag_cache_dir,
                )
                if bag:
                    family_bags.append(bag)

        # Also flat-layout APKs
        for apk_file in sorted(tb_packed.glob(f"{packer_name}__*.apk")):
            jsonl = apk_file.with_name(apk_file.stem + ".inject_labels.jsonl")
            if jsonl.exists():
                diff = parse_inject_labels(jsonl)
            else:
                parts = apk_file.stem.split("__")
                if len(parts) == 2:
                    ba = tb_benign / f"{parts[1]}.apk"
                    diff = compute_paired_diff(ba, apk_file) if ba.exists() else None
                else:
                    diff = None
            bag = build_bag(
                apk_file,
                1,
                tokenizer,
                diff,
                f"{short_name}_flat_{apk_file.stem[:30]}",
                cache_dir=bag_cache_dir,
            )
            if bag:
                family_bags.append(bag)

        if family_bags:
            families.append(PackerFamily(name=short_name, bags=family_bags))
            print(f"    {short_name}: {len(family_bags)} bags")

    return benign_bags, benign_test_bags, families


def load_track_b_benign_counterpart_bags(
    tokenizer,
    bag_cache_dir: Optional[Path] = BAG_CACHE_DIR,
) -> List[Dict]:
    """Load old Track B benign apps for B1.1 DPT-boundary diagnostics only."""
    tb_benign = TRACK_B / "benign"
    counterpart_bags: List[Dict] = []
    if not tb_benign.exists():
        return counterpart_bags
    benign_apks = sorted(tb_benign.glob("*.apk"))
    print(
        f"    Loading Track B benign counterparts for control: {len(benign_apks)} APKs",
        flush=True,
    )
    for i, p in enumerate(benign_apks, 1):
        bag = build_bag(
            p,
            0,
            tokenizer,
            apk_id=f"track_b_control_benign_{p.stem[:32]}",
            cache_dir=bag_cache_dir,
        )
        if bag:
            counterpart_bags.append(bag)
        if i % 5 == 0 or i == len(benign_apks):
            print(f"      Track B control benign {i}/{len(benign_apks)}", flush=True)
    return counterpart_bags


def _match_track_b_v2_benign(packed_apk: Path, benign_by_stem: Dict[str, Path]) -> Optional[Path]:
    stem = packed_apk.stem
    prefix = stem[len("dpt__"):] if stem.startswith("dpt__") else stem
    matches = [path for key, path in benign_by_stem.items() if key.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return None


def load_track_b_v2_strict_data(
    tokenizer,
    bag_cache_dir: Optional[Path] = BAG_CACHE_DIR,
    exclude_dirty_benign: bool = False,
):
    """Load Track B v2 app-disjoint DPT packed APKs and paired benign APKs."""
    benign_dir = TRACK_B_V2 / "benign"
    packed_dir = TRACK_B_V2 / "packed" / "dpt_shell"
    if not benign_dir.exists() or not packed_dir.exists():
        raise FileNotFoundError(f"Track B v2 data not found under {TRACK_B_V2}")

    benign_apks = sorted(benign_dir.glob("*.apk"))
    if exclude_dirty_benign:
        benign_apks = [
            p for p in benign_apks
            if p.stem.upper() not in APKID_DIRTY_STRICT_BENIGN
        ]
    packed_apks = sorted(packed_dir.glob("*.apk"))
    if exclude_dirty_benign:
        packed_apks = [
            p for p in packed_apks
            if not is_apkid_dirty_strict_packed_path(p)
        ]
    benign_by_stem = {p.stem: p for p in benign_apks}

    test_benign = []
    print(f"    Loading Track B v2 benign: {len(benign_apks)} APKs", flush=True)
    for i, p in enumerate(benign_apks, 1):
        bag = build_bag(
            p,
            0,
            tokenizer,
            apk_id=f"track_b_v2_benign_{p.stem[:24]}",
            cache_dir=bag_cache_dir,
        )
        if bag:
            test_benign.append(bag)
        if i % 5 == 0 or i == len(benign_apks):
            print(f"      Track B v2 benign {i}/{len(benign_apks)}", flush=True)

    test_packed = []
    print(f"    Loading Track B v2 DPT packed: {len(packed_apks)} APKs", flush=True)
    for i, p in enumerate(packed_apks, 1):
        origin = _match_track_b_v2_benign(p, benign_by_stem)
        diff = compute_paired_diff(origin, p) if origin else None
        bag = build_bag(
            p,
            1,
            tokenizer,
            diff,
            apk_id=f"track_b_v2_dpt_{p.stem[:24]}",
            cache_dir=bag_cache_dir,
        )
        if bag:
            test_packed.append(bag)
        if i % 5 == 0 or i == len(packed_apks):
            print(f"      Track B v2 DPT {i}/{len(packed_apks)}", flush=True)

    return test_benign, test_packed


# ---------------------------------------------------------------------------
# Training (simplified from v3: frozen BERT + 4-loss)
# ---------------------------------------------------------------------------


def _checkpoint_payload(
    model,
    optimizer,
    scheduler,
    rng,
    epoch: int,
    args,
    fold_name: str,
):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_numpy": rng.get_state(),
        "rng_python": random.getstate(),
        "rng_torch": torch.get_rng_state(),
        "rng_torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None
        ),
        "args": vars(args) if args is not None else {},
        "fold_name": fold_name,
    }


def _load_fold_checkpoint(model, optimizer, scheduler, rng, checkpoint_path: Optional[Path], device):
    if checkpoint_path is None or not checkpoint_path.exists():
        return 0
    ckpt = _torch_load_local(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if "rng_numpy" in ckpt:
        rng.set_state(ckpt["rng_numpy"])
    if "rng_python" in ckpt:
        random.setstate(ckpt["rng_python"])
    if "rng_torch" in ckpt:
        torch.set_rng_state(ckpt["rng_torch"].cpu())
    if torch.cuda.is_available() and ckpt.get("rng_torch_cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["rng_torch_cuda"]])
    start_epoch = int(ckpt.get("epoch", 0))
    print(f"  Resuming fold from epoch {start_epoch}: {checkpoint_path}", flush=True)
    return start_epoch


def train_fold(
    model,
    train_bags,
    device,
    epochs=50,
    train_batch_size=4,
    bag_chunk_size=64,
    lr=5e-4,
    lr_bert: Optional[float] = None,
    args=None,
    checkpoint_dir: Optional[Path] = None,
    fold_name: str = "fold",
    save_every: int = 1,
    resume: bool = False,
    paired_ranking_weight: float = 0.0,
    paired_ranking_margin: float = 1.0,
):
    """Train one LOPO fold."""
    if train_batch_size <= 0:
        raise ValueError(f"train_batch_size must be positive, got {train_batch_size}")
    if bag_chunk_size <= 0:
        raise ValueError(f"bag_chunk_size must be positive, got {bag_chunk_size}")
    model.train()
    if lr_bert is not None:
        bert_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ".bert." in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr})
        if bert_params:
            param_groups.append({"params": bert_params, "lr": lr_bert})
        trainable_params = other_params + bert_params
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = trainable_params
    if not trainable_params:
        raise ValueError("No trainable parameters are enabled for this fold")
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.RandomState(42)
    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "latest.pt"
    start_epoch = (
        _load_fold_checkpoint(model, optimizer, scheduler, rng, checkpoint_path, device)
        if resume else 0
    )

    packed_logits, benign_logits = [], []
    paired_groups = group_bags_by_pair_key(train_bags) if paired_ranking_weight > 0 else {}

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        n_steps = 0
        order = rng.permutation(len(train_bags))
        packed_logits.clear()
        benign_logits.clear()

        for i in range(0, len(train_bags), train_batch_size):
            batch_indices = order[i:i + train_batch_size]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for idx in batch_indices:
                bag = train_bags[idx]
                out = model.forward_bag(bag, device, chunk_size=bag_chunk_size)
                bag_logit = out["bag_logit"]
                label = bag["apk_label"]
                target = torch.tensor(float(label), device=device)

                l_bag = F.binary_cross_entropy_with_logits(bag_logit, target)

                # Rank loss
                l_rank = torch.tensor(0.0, device=device)
                margin = 2.0
                if label == 1 and benign_logits:
                    for bl in benign_logits[-5:]:
                        l_rank = l_rank + F.relu(margin + bl - bag_logit)
                    l_rank = l_rank / len(benign_logits[-5:])
                elif label == 0 and packed_logits:
                    for pl in packed_logits[-5:]:
                        l_rank = l_rank + F.relu(margin + bag_logit - pl)
                    l_rank = l_rank / len(packed_logits[-5:])

                # Align loss
                l_align = torch.tensor(0.0, device=device)
                # (simplified: skip align for speed in LOPO)

                l_pair = torch.tensor(0.0, device=device)
                if paired_ranking_weight > 0:
                    target_label = 0 if label == 1 else 1
                    counterparts = lookup_pair_bags(paired_groups, bag, target_label)
                    if counterparts:
                        counterpart = counterparts[int(rng.randint(len(counterparts)))]
                        other_logit = model.forward_bag(
                            counterpart, device, chunk_size=bag_chunk_size
                        )["bag_logit"]
                        if label == 1:
                            l_pair = F.relu(paired_ranking_margin + other_logit - bag_logit)
                        else:
                            l_pair = F.relu(paired_ranking_margin + bag_logit - other_logit)

                loss = l_bag + 0.5 * l_rank + paired_ranking_weight * l_pair
                batch_loss = batch_loss + loss / len(batch_indices)

                logit_val = bag_logit.detach().item()
                if label == 1:
                    packed_logits.append(logit_val)
                else:
                    benign_logits.append(logit_val)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()
            n_steps += 1

        scheduler.step()
        completed_epoch = epoch + 1
        if checkpoint_path is not None and (
            save_every <= 1 or completed_epoch % save_every == 0 or completed_epoch == epochs
        ):
            torch.save(
                _checkpoint_payload(
                    model, optimizer, scheduler, rng, completed_epoch, args, fold_name
                ),
                checkpoint_path,
            )

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_fold(model, test_bags, device, bag_chunk_size=64):
    """Evaluate one fold with APK-level + entry-level localization metrics."""
    if bag_chunk_size <= 0:
        raise ValueError(f"bag_chunk_size must be positive, got {bag_chunk_size}")
    model.eval()
    y_true, y_score = [], []
    # Entry-level localization (only for packed bags with diff_targets)
    all_entry_true, all_entry_scores_norm, all_entry_scores_susp, all_entry_scores_attn = [], [], [], []
    mrr_list_norm, mrr_list_susp, mrr_list_attn = [], [], []
    inference_times_ms = []

    with torch.no_grad():
        for bag in test_bags:
            t0 = time.time()
            out = model.forward_bag(bag, device, chunk_size=bag_chunk_size)
            inference_times_ms.append((time.time() - t0) * 1000)

            bag_logit = out["bag_logit"]
            score = torch.sigmoid(bag_logit).item()
            y_true.append(bag["apk_label"])
            y_score.append(score)

            # Entry-level localization (only for packed bags with ground truth)
            if bag["apk_label"] == 1 and bag.get("diff_targets") is not None:
                dt = bag["diff_targets"]
                n_entries = len(out["entry_normality"])
                dt_trunc = dt[:n_entries]

                if len(dt_trunc) > 0 and np.any(dt_trunc > 0.3):
                    # Ground truth: entry is "packed" if diff_score > 0.5
                    entry_gt = (dt_trunc > 0.5).astype(np.float32)

                    # Three localization scores
                    norm_scores = (1.0 - out["entry_normality"].cpu().numpy()[:len(dt_trunc)])
                    susp_scores = out["entry_suspicion"].cpu().numpy()[:len(dt_trunc)]
                    attn_scores = out["entry_attention"].cpu().numpy()[:len(dt_trunc)]

                    all_entry_true.extend(entry_gt.tolist())
                    all_entry_scores_norm.extend(norm_scores.tolist())
                    all_entry_scores_susp.extend(susp_scores.tolist())
                    all_entry_scores_attn.extend(attn_scores.tolist())

                    # MRR: rank of first true packed entry
                    for scores_arr, mrr_list in [
                        (norm_scores, mrr_list_norm),
                        (susp_scores, mrr_list_susp),
                        (attn_scores, mrr_list_attn),
                    ]:
                        ranked_idx = np.argsort(-scores_arr)
                        for rank, idx in enumerate(ranked_idx, 1):
                            if idx < len(entry_gt) and entry_gt[idx] > 0.5:
                                mrr_list.append(1.0 / rank)
                                break
                        else:
                            mrr_list.append(0.0)

    # Compute entry-level AUROCs
    loc_metrics = {}
    if len(set(all_entry_true)) >= 2:
        loc_metrics["entry_auroc_normality"] = float(roc_auc_score(all_entry_true, all_entry_scores_norm))
        loc_metrics["entry_auroc_suspicion"] = float(roc_auc_score(all_entry_true, all_entry_scores_susp))
        loc_metrics["entry_auroc_attention"] = float(roc_auc_score(all_entry_true, all_entry_scores_attn))
    if mrr_list_norm:
        loc_metrics["entry_mrr_normality"] = float(np.mean(mrr_list_norm))
        loc_metrics["entry_mrr_suspicion"] = float(np.mean(mrr_list_susp))
        loc_metrics["entry_mrr_attention"] = float(np.mean(mrr_list_attn))
    loc_metrics["n_entries_evaluated"] = len(all_entry_true)
    loc_metrics["mean_inference_ms"] = float(np.mean(inference_times_ms)) if inference_times_ms else 0

    return y_true, y_score, loc_metrics


def score_bags(model, bags, device, bag_chunk_size=64) -> List[float]:
    if bag_chunk_size <= 0:
        raise ValueError(f"bag_chunk_size must be positive, got {bag_chunk_size}")
    model.eval()
    scores = []
    with torch.no_grad():
        for bag in bags:
            out = model.forward_bag(bag, device, chunk_size=bag_chunk_size)
            scores.append(torch.sigmoid(out["bag_logit"]).item())
    return scores


def build_model(args):
    fusion_cfg = FusionEncoderConfig(
        bert_hidden_dim=args.bert_dim, bert_n_layers=args.bert_layers,
        bert_max_length=128,
        bert_n_heads=8, bert_intermediate_dim=args.bert_dim * 2,
        use_gated_fusion=True, gate_hidden_dim=128,
        use_bert_features=args.ablation != "stat_only",
        use_stat_features=args.ablation != "bert_only",
        path_dropout_prob=args.path_dropout,
        use_region_type_routing=args.region_type_routing,
        batch_bert_streams=not args.no_batch_bert_streams,
        routing_dex_byte_weight=args.routing_dex_byte_weight,
        routing_elf_byte_weight=args.routing_elf_byte_weight,
        routing_byte_entry_weight=args.routing_byte_entry_weight,
        routing_unknown_weight=args.routing_unknown_weight,
        byte_representation=getattr(args, "byte_representation", BYTE_REPRESENTATION_LEGACY_RAW),
    )
    return PseudoBERTModel(
        fusion_cfg,
        ablation_mode=args.ablation,
        active_paths=args.paths,
    )


def load_pretrained_bert(model, ckpt_path: Path) -> int:
    if not ckpt_path.exists():
        return 0
    state = _torch_load_local(ckpt_path, map_location="cpu")
    model_dict = model.state_dict()
    first_key = next(iter(state.keys()))
    if first_key.startswith("bert.") or first_key.startswith("fusion."):
        pretrained = {
            f"fusion_encoder.{k}": v for k, v in state.items()
            if f"fusion_encoder.{k}" in model_dict
            and v.shape == model_dict[f"fusion_encoder.{k}"].shape
        }
    else:
        pretrained = {
            k: v for k, v in state.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    return len(pretrained)


def checkpoint_root_for_output(out_dir: Path, out_file: str) -> Path:
    return out_dir / "checkpoints" / Path(out_file).stem


def safe_output_suffix(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def resolve_output_path(output_suffix: Optional[str], track_b_v2_strict: bool) -> Tuple[Path, str]:
    out_dir = OUT_DIR
    out_file = "lopo_results.json"
    if output_suffix:
        out_dir = PATH_ABLATION_DIR
        out_file = f"lopo_results_{safe_output_suffix(output_suffix)}.json"
    if track_b_v2_strict:
        out_dir = ROOT / "outputs" / "experiments" / "track_b_v2_strict_dpt"
        out_file = "results.json"
        if output_suffix:
            out_file = f"results_{safe_output_suffix(output_suffix)}.json"
    return out_dir, out_file


# ---------------------------------------------------------------------------
# Main LOPO
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=4,
                        help="Number of APK bags per gradient step")
    parser.add_argument("--bag-chunk-size", type=int, default=64,
                        help="Number of regions processed per model forward chunk")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-bert", type=float, default=None,
                        help="Learning rate for unfrozen BERT parameters (default: same optimizer lr)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-tf32", action="store_true",
                        help="Disable TF32 matmul on CUDA devices")
    parser.add_argument("--no-batch-bert-streams", action="store_true",
                        help="Disable batching Dalvik/ARM64/byte streams into one shared-BERT forward")
    parser.add_argument("--byte-representation", type=str,
                        default=BYTE_REPRESENTATION_LEGACY_RAW,
                        choices=BYTE_REPRESENTATIONS,
                        help="Byte-path representation: legacy raw bytes or B3 typed byte patterns")
    parser.add_argument("--max-per-family", type=int, default=15)
    parser.add_argument("--androzoo-benign", type=int, default=0,
                        help="Add N random AndroZoo benign APKs to training (0=none)")
    parser.add_argument("--hard-benign-manifest", type=Path, action="append", default=[],
                        help="JSON/JSONL manifest of APKiD-clean benign APKs to add to training")
    parser.add_argument("--hard-benign-limit", type=int, default=0,
                        help="Maximum hard-benign APKs to load from manifests (0=all)")
    parser.add_argument("--hard-benign-only", action="store_true",
                        help="From manifests, load only records labeled benign-hard-clean")
    parser.add_argument("--pretrain-ckpt", type=str, default=None,
                        help="Override pretrained BERT checkpoint path")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "bert_only", "stat_only"],
                        help="Ablation mode: full, bert_only (disable stat branch), stat_only (disable BERT branch)")
    parser.add_argument("--bert-layers", type=int, default=4)
    parser.add_argument("--bert-dim", type=int, default=256)
    parser.add_argument("--paths", type=parse_active_paths, default=VALID_PATHS,
                        help="Comma-separated active BERT paths: dalvik,arm64,byte")
    parser.add_argument("--bert-train-mode", type=str, default="frozen",
                        choices=["frozen", "last_n", "all"],
                        help="BERT fine-tuning mode after loading the pretrained checkpoint")
    parser.add_argument("--bert-last-n-layers", type=int, default=2,
                        help="Number of final BERT layers to unfreeze with --bert-train-mode last_n")
    parser.add_argument("--path-dropout", type=float, default=0.0,
                        help="Training-only probability of dropping each active BERT path")
    parser.add_argument("--region-type-routing", action="store_true",
                        help="Apply fixed DEX/ELF/blob path routing weights before path aggregation")
    parser.add_argument("--routing-dex-byte-weight", type=float, default=0.25,
                        help="Byte-path weight for DEX entries under --region-type-routing")
    parser.add_argument("--routing-elf-byte-weight", type=float, default=0.25,
                        help="Byte-path weight for ELF entries under --region-type-routing")
    parser.add_argument("--routing-byte-entry-weight", type=float, default=1.0,
                        help="Byte-path weight for asset/archive/resource/manifest entries under --region-type-routing")
    parser.add_argument("--routing-unknown-weight", type=float, default=0.25,
                        help="Uniform path weight for unknown entries under --region-type-routing")
    parser.add_argument("--output-suffix", type=str, default=None,
                        help="Write LOPO output as lopo_results_<suffix>.json")
    parser.add_argument("--track-b-v2-strict", action="store_true",
                        help="Evaluate a DPT-held-out model on Track B v2 app-disjoint DPT APKs")
    parser.add_argument("--strict-dpt-control-mode", type=str, default="non_dpt",
                        choices=STRICT_DPT_CONTROL_MODES,
                        help="B1.1 diagnostic train-set control for --track-b-v2-strict")
    parser.add_argument("--no-bag-cache", action="store_true",
                        help="Disable cached APK bag extraction under outputs/experiments/lopo_eval/bag_cache")
    parser.add_argument("--resume", action="store_true",
                        help="Resume each fold from its latest checkpoint when available")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save a fold checkpoint every N epochs")
    parser.add_argument("--paired-ranking-weight", type=float, default=0.0,
                        help="Weight for same-origin packed/unpacked margin ranking loss")
    parser.add_argument("--paired-ranking-margin", type=float, default=1.0,
                        help="Logit margin for --paired-ranking-weight")
    parser.add_argument("--score-normalization", type=str, default="none",
                        choices=["none", "train_benign_center", "train_benign_z"],
                        help="Report AUROC after train-benign-only score normalization")
    parser.add_argument("--exclude-apkid-dirty-strict-benign", action="store_true",
                        help="Exclude Track B v2 benign APKs flagged as packed by APKiD")
    args = parser.parse_args()
    if args.path_dropout < 0.0 or args.path_dropout >= 1.0:
        parser.error("--path-dropout must be in [0.0, 1.0)")
    if args.train_batch_size <= 0:
        parser.error("--train-batch-size must be positive")
    if args.bag_chunk_size <= 0:
        parser.error("--bag-chunk-size must be positive")
    if args.paired_ranking_weight < 0.0:
        parser.error("--paired-ranking-weight must be non-negative")
    if args.hard_benign_limit < 0:
        parser.error("--hard-benign-limit must be non-negative")
    for name in (
        "routing_dex_byte_weight",
        "routing_elf_byte_weight",
        "routing_byte_entry_weight",
        "routing_unknown_weight",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    out_dir, out_file = resolve_output_path(args.output_suffix, args.track_b_v2_strict)
    if args.region_type_routing and args.output_suffix is None and not args.track_b_v2_strict:
        out_dir = PATH_ABLATION_DIR
        out_file = "lopo_results_region_type_routing.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    configure_cuda_performance(device, allow_tf32=not args.no_tf32)
    print(f"Device: {device}")
    print(
        f"Config: {args.epochs} epochs, batch={args.train_batch_size}, "
        f"chunk={args.bag_chunk_size}, lr={args.lr}, ablation={args.ablation}"
    )
    print(f"TF32 enabled: {device.type == 'cuda' and not args.no_tf32}")
    print(f"Batch BERT streams: {not args.no_batch_bert_streams}")
    print(f"Byte representation: {args.byte_representation}")
    print(f"Active paths: {','.join(args.paths)}")
    print(
        "BERT training: "
        f"{args.bert_train_mode}"
        + (f" last_n={args.bert_last_n_layers}" if args.bert_train_mode == "last_n" else "")
        + (f", lr_bert={args.lr_bert}" if args.lr_bert is not None else "")
    )
    print(f"Path dropout: {args.path_dropout}")
    print(f"Region-type routing: {args.region_type_routing}")
    print(
        "Routing weights: "
        f"dex_byte={args.routing_dex_byte_weight}, "
        f"elf_byte={args.routing_elf_byte_weight}, "
        f"byte_entry={args.routing_byte_entry_weight}, "
        f"unknown={args.routing_unknown_weight}"
    )
    print(
        f"Paired ranking: weight={args.paired_ranking_weight}, "
        f"margin={args.paired_ranking_margin}"
    )
    print(f"Score normalization: {args.score_normalization}")
    print(f"Strict DPT control mode: {args.strict_dpt_control_mode}")
    print(f"Exclude APKiD-dirty strict benign: {args.exclude_apkid_dirty_strict_benign}")
    print(f"AndroZoo benign: {args.androzoo_benign}")
    print(
        "Hard benign manifests: "
        f"{[str(p) for p in args.hard_benign_manifest]}"
        + (f", limit={args.hard_benign_limit}" if args.hard_benign_manifest else "")
        + (", hard-only" if args.hard_benign_only else "")
    )
    bag_cache_dir = None if args.no_bag_cache else BAG_CACHE_DIR
    print(f"Bag cache: {bag_cache_dir if bag_cache_dir is not None else 'disabled'}")

    tokenizer = PseudoCodeTokenizer(
        max_length=128,
        byte_representation=args.byte_representation,
    )

    # Load all data
    print("\n=== Loading Data ===", flush=True)
    t0 = time.time()
    benign_bags, benign_test_bags, families = load_all_data(
        tokenizer,
        args.max_per_family,
        args.androzoo_benign,
        bag_cache_dir,
        hard_benign_manifests=args.hard_benign_manifest,
        hard_benign_limit=args.hard_benign_limit,
        hard_benign_only=args.hard_benign_only,
    )
    print(f"  Total load time: {time.time()-t0:.0f}s")
    print(f"  Families: {[f.name for f in families]}")
    print(f"  Total packed bags: {sum(len(f.bags) for f in families)}")

    if args.track_b_v2_strict:
        print("\n=== Track B v2 Strict DPT Evaluation ===", flush=True)
        control_benign_bags = (
            load_track_b_benign_counterpart_bags(tokenizer, bag_cache_dir)
            if args.strict_dpt_control_mode == "add_old_dpt_benign"
            else []
        )
        train_bags, train_family_names, control_info = build_strict_dpt_control_training_set(
            benign_bags,
            families,
            control_mode=args.strict_dpt_control_mode,
            dpt_benign_bags=control_benign_bags,
        )
        test_benign, test_packed = load_track_b_v2_strict_data(
            tokenizer,
            bag_cache_dir,
            exclude_dirty_benign=args.exclude_apkid_dirty_strict_benign,
        )
        test_bags = list(test_benign) + list(test_packed)
        print(f"  Train families: {train_family_names}")
        print(f"  Control info: {control_info}")
        print(f"  Train: {len(train_bags)} bags")
        print(f"  Test: {len(test_bags)} ({len(test_packed)} packed, {len(test_benign)} benign)")

        model = build_model(args)
        ckpt_path = Path(args.pretrain_ckpt) if args.pretrain_ckpt else PRETRAIN_CKPT
        n_loaded = load_pretrained_bert(model, ckpt_path)
        print(f"  Pretrained tensors loaded: {n_loaded} from {ckpt_path}")
        model.configure_bert_training(args.bert_train_mode, args.bert_last_n_layers)
        model = model.to(device)

        t1 = time.time()
        model = train_fold(
            model,
            train_bags,
            device,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            bag_chunk_size=args.bag_chunk_size,
            lr=args.lr,
            lr_bert=args.lr_bert,
            args=args,
            checkpoint_dir=checkpoint_root_for_output(out_dir, out_file) / "strict_dpt",
            fold_name="strict_dpt",
            save_every=args.save_every,
            resume=args.resume,
            paired_ranking_weight=args.paired_ranking_weight,
            paired_ranking_margin=args.paired_ranking_margin,
        )
        train_time = time.time() - t1

        y_true, y_score, loc_metrics = evaluate_fold(
            model, test_bags, device, bag_chunk_size=args.bag_chunk_size
        )
        train_benign_scores = score_bags(
            model,
            [bag for bag in train_bags if bag["apk_label"] == 0],
            device,
            bag_chunk_size=args.bag_chunk_size,
        )
        y_score_norm, norm_info = normalize_scores_from_train_benign(
            y_score,
            train_benign_scores,
            args.score_normalization,
        )
        benign_scores = [s for s, l in zip(y_score, y_true) if l == 0]
        packed_scores = [s for s, l in zip(y_score, y_true) if l == 1]
        benign_scores_norm = [s for s, l in zip(y_score_norm, y_true) if l == 0]
        packed_scores_norm = [s for s, l in zip(y_score_norm, y_true) if l == 1]
        auroc = roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else 0.0
        auroc_norm = (
            roc_auc_score(y_true, y_score_norm)
            if args.score_normalization != "none" and len(set(y_true)) >= 2
            else None
        )
        operating_metrics = fixed_fpr_tpr_metrics(y_true, y_score)
        operating_metrics_norm = (
            fixed_fpr_tpr_metrics(y_true, y_score_norm)
            if args.score_normalization != "none"
            else None
        )
        detected = sum(1 for s in packed_scores if s > 0.5)
        detected_norm = (
            sum(1 for s in packed_scores_norm if s > 0.5)
            if args.score_normalization != "none"
            else None
        )

        result = {
            "method": "Pseudo-code BERT v4 Track B v2 strict DPT",
            "evaluation": "Track B v2 strict app-disjoint DPT",
            "held_out": "DPT",
            "n_train": len(train_bags),
            "n_test_packed": len(test_packed),
            "n_test_benign": len(test_benign),
            "auroc": float(auroc),
            "auroc_normalized": float(auroc_norm) if auroc_norm is not None else None,
            **operating_metrics,
            "operating_metrics_normalized": operating_metrics_norm,
            "detection_rate": detected / max(len(packed_scores), 1),
            "packed_mean": float(np.mean(packed_scores)) if packed_scores else 0.0,
            "benign_mean": float(np.mean(benign_scores)) if benign_scores else 0.0,
            "detection_rate_normalized": (
                detected_norm / max(len(packed_scores_norm), 1)
                if detected_norm is not None
                else None
            ),
            "packed_mean_normalized": (
                float(np.mean(packed_scores_norm))
                if args.score_normalization != "none" and packed_scores_norm
                else None
            ),
            "benign_mean_normalized": (
                float(np.mean(benign_scores_norm))
                if args.score_normalization != "none" and benign_scores_norm
                else None
            ),
            "train_time_s": round(train_time, 1),
            **loc_metrics,
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "train_batch_size": args.train_batch_size,
                "bag_chunk_size": args.bag_chunk_size,
                "tf32": device.type == "cuda" and not args.no_tf32,
                "batch_bert_streams": not args.no_batch_bert_streams,
                "byte_representation": args.byte_representation,
                "ablation": args.ablation,
                "active_paths": list(args.paths),
                "bert_layers": args.bert_layers,
                "bert_dim": args.bert_dim,
                "gated_fusion": True,
                "use_bert_features": args.ablation != "stat_only",
                "use_stat_features": args.ablation != "bert_only",
                "bert_train_mode": args.bert_train_mode,
                "bert_last_n_layers": args.bert_last_n_layers,
                "lr_bert": args.lr_bert,
                "path_dropout": args.path_dropout,
                "region_type_routing": args.region_type_routing,
                "routing_dex_byte_weight": args.routing_dex_byte_weight,
                "routing_elf_byte_weight": args.routing_elf_byte_weight,
                "routing_byte_entry_weight": args.routing_byte_entry_weight,
                "routing_unknown_weight": args.routing_unknown_weight,
                "paired_ranking_weight": args.paired_ranking_weight,
                "paired_ranking_margin": args.paired_ranking_margin,
                "score_normalization": args.score_normalization,
                "score_normalization_info": norm_info,
                "strict_dpt_control_mode": args.strict_dpt_control_mode,
                "strict_dpt_control_info": control_info,
                "exclude_apkid_dirty_strict_benign": args.exclude_apkid_dirty_strict_benign,
                "hard_benign_manifest": [str(p) for p in args.hard_benign_manifest],
                "hard_benign_limit": args.hard_benign_limit,
                "hard_benign_only": args.hard_benign_only,
                "pretrain": str(ckpt_path),
                "pretrained_tensors_loaded": n_loaded,
            },
        }
        with open(out_dir / out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  AUROC: {auroc:.4f}")
        if auroc_norm is not None:
            print(f"  AUROC ({args.score_normalization}): {auroc_norm:.4f}")
        print(f"  Detection: {detected}/{len(packed_scores)}")
        if detected_norm is not None:
            print(f"  Detection ({args.score_normalization}): "
                  f"{detected_norm}/{len(packed_scores_norm)}")
        print(f"  Saved: {out_dir / out_file}")
        return

    # LOPO evaluation
    print(f"\n=== LOPO Evaluation ({len(families)} folds) ===", flush=True)
    results = []

    for fold_idx, held_out in enumerate(families):
        print(f"\n--- Fold {fold_idx+1}/{len(families)}: held-out = {held_out.name} "
              f"({len(held_out.bags)} bags) ---", flush=True)

        # Build training set: all OTHER families + all benign
        train_bags = list(benign_bags)  # copy
        for fam in families:
            if fam.name != held_out.name:
                train_bags.extend(fam.bags)

        # Test set: held-out packed + test benign (Origin-18)
        test_bags = list(benign_test_bags) + held_out.bags

        n_train_packed = sum(1 for b in train_bags if b["apk_label"] == 1)
        n_train_benign = len(train_bags) - n_train_packed
        n_test_packed = len(held_out.bags)
        n_test_benign = len(benign_test_bags)
        print(f"  Train: {len(train_bags)} ({n_train_packed} packed, {n_train_benign} benign)")
        print(f"  Test:  {len(test_bags)} ({n_test_packed} packed, {n_test_benign} benign)")

        # Build fresh model for this fold
        model = build_model(args)

        # Load pretrained BERT
        ckpt_path = Path(args.pretrain_ckpt) if args.pretrain_ckpt else PRETRAIN_CKPT
        n_loaded = load_pretrained_bert(model, ckpt_path)
        print(f"  Pretrained tensors loaded: {n_loaded}")

        model.configure_bert_training(args.bert_train_mode, args.bert_last_n_layers)
        model = model.to(device)

        # Train
        t1 = time.time()
        fold_safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in held_out.name)
        model = train_fold(
            model,
            train_bags,
            device,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            bag_chunk_size=args.bag_chunk_size,
            lr=args.lr,
            lr_bert=args.lr_bert,
            args=args,
            checkpoint_dir=checkpoint_root_for_output(out_dir, out_file) / fold_safe,
            fold_name=held_out.name,
            save_every=args.save_every,
            resume=args.resume,
            paired_ranking_weight=args.paired_ranking_weight,
            paired_ranking_margin=args.paired_ranking_margin,
        )
        train_time = time.time() - t1
        print(f"  Training: {train_time:.0f}s")

        # Evaluate
        y_true, y_score, loc_metrics = evaluate_fold(
            model, test_bags, device, bag_chunk_size=args.bag_chunk_size
        )
        train_benign_scores = []
        auroc_norm = None
        norm_info = {}
        if args.score_normalization != "none":
            train_benign_scores = score_bags(
                model,
                [bag for bag in train_bags if bag["apk_label"] == 0],
                device,
                bag_chunk_size=args.bag_chunk_size,
            )
            y_score_norm, norm_info = normalize_scores_from_train_benign(
                y_score,
                train_benign_scores,
                args.score_normalization,
            )
            auroc_norm = roc_auc_score(y_true, y_score_norm) if len(set(y_true)) >= 2 else 0.0
        benign_scores = [s for s, l in zip(y_score, y_true) if l == 0]
        packed_scores = [s for s, l in zip(y_score, y_true) if l == 1]

        auroc = roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else 0.0
        detected = sum(1 for s in packed_scores if s > 0.5)

        print(f"  AUROC: {auroc:.4f}")
        if auroc_norm is not None:
            print(f"  AUROC ({args.score_normalization}): {auroc_norm:.4f}")
        print(f"  Detection: {detected}/{len(packed_scores)} "
              f"(packed_mean={np.mean(packed_scores):.3f}, "
              f"benign_mean={np.mean(benign_scores):.3f})")
        if loc_metrics.get("entry_auroc_normality"):
            print(f"  Entry AUROC (normality): {loc_metrics['entry_auroc_normality']:.4f}")
            print(f"  Entry MRR (normality):   {loc_metrics['entry_mrr_normality']:.4f}")
            print(f"  Inference: {loc_metrics['mean_inference_ms']:.0f} ms/APK")

        results.append({
            "fold": fold_idx + 1,
            "held_out": held_out.name,
            "n_test_packed": n_test_packed,
            "auroc": auroc,
            "auroc_normalized": float(auroc_norm) if auroc_norm is not None else None,
            "detection_rate": detected / max(len(packed_scores), 1),
            "packed_mean": float(np.mean(packed_scores)),
            "benign_mean": float(np.mean(benign_scores)),
            "train_time_s": round(train_time, 1),
            **loc_metrics,
            "score_normalization_info": norm_info,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"=== LOPO Results Summary ===")
    print(f"{'='*60}")
    print(f"{'Packer':<15} {'AUROC':>7} {'Det.Rate':>9} {'Packed':>7} {'Benign':>7} {'EntAUROC':>8} {'EntMRR':>7}")
    print(f"{'-'*70}")
    for r in results:
        e_auroc = r.get('entry_auroc_normality', 0)
        e_mrr = r.get('entry_mrr_normality', 0)
        print(f"{r['held_out']:<15} {r['auroc']:>7.4f} "
              f"{r['detection_rate']:>8.1%} "
              f"{r['packed_mean']:>7.3f} {r['benign_mean']:>7.3f} "
              f"{e_auroc:>8.4f} {e_mrr:>7.4f}")
    print(f"{'-'*70}")
    mean_auroc = np.mean([r["auroc"] for r in results])
    mean_det = np.mean([r["detection_rate"] for r in results])
    mean_entry_auroc = np.mean([r.get("entry_auroc_normality", 0) for r in results if r.get("entry_auroc_normality")])
    mean_entry_mrr = np.mean([r.get("entry_mrr_normality", 0) for r in results if r.get("entry_mrr_normality")])
    print(f"{'MEAN':<15} {mean_auroc:>7.4f} {mean_det:>8.1%} {'':>7} {'':>7} {mean_entry_auroc:>8.4f} {mean_entry_mrr:>7.4f}")

    # Localization comparison (3 scoring methods)
    print(f"\n--- Entry-Level Localization (mean across folds) ---")
    for method in ["normality", "suspicion", "attention"]:
        aurocs = [r.get(f"entry_auroc_{method}", 0) for r in results if r.get(f"entry_auroc_{method}")]
        mrrs = [r.get(f"entry_mrr_{method}", 0) for r in results if r.get(f"entry_mrr_{method}")]
        if aurocs:
            print(f"  {method:12s}: AUROC={np.mean(aurocs):.4f}  MRR={np.mean(mrrs):.4f}")

    # Inference time
    times = [r.get("mean_inference_ms", 0) for r in results if r.get("mean_inference_ms")]
    if times:
        print(f"\n  Mean inference time: {np.mean(times):.0f} ms/APK")

    print(f"\nComparison:")
    print(f"  Entropy baseline:    0.7246")
    print(f"  Stat-only (v2):      0.7543")
    print(f"  BERT v3b:            0.6761")
    print(f"  BERT v4 LOPO mean:   {mean_auroc:.4f}")

    # Save
    summary = {
        "method": "Pseudo-code BERT v4 (gated fusion + Track B training)",
        "evaluation": "Leave-One-Packer-Out (LOPO)",
        "n_folds": len(results),
        "mean_auroc": float(mean_auroc),
        "mean_detection_rate": float(mean_det),
        "folds": results,
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "train_batch_size": args.train_batch_size,
            "bag_chunk_size": args.bag_chunk_size,
            "tf32": device.type == "cuda" and not args.no_tf32,
            "batch_bert_streams": not args.no_batch_bert_streams,
            "byte_representation": args.byte_representation,
            "ablation": args.ablation,
            "active_paths": list(args.paths),
            "bert_layers": args.bert_layers,
            "bert_dim": args.bert_dim,
            "gated_fusion": True,
            "use_bert_features": args.ablation != "stat_only",
            "use_stat_features": args.ablation != "bert_only",
            "bert_train_mode": args.bert_train_mode,
            "bert_last_n_layers": args.bert_last_n_layers,
            "lr_bert": args.lr_bert,
            "path_dropout": args.path_dropout,
            "region_type_routing": args.region_type_routing,
            "routing_dex_byte_weight": args.routing_dex_byte_weight,
            "routing_elf_byte_weight": args.routing_elf_byte_weight,
            "routing_byte_entry_weight": args.routing_byte_entry_weight,
            "routing_unknown_weight": args.routing_unknown_weight,
            "paired_ranking_weight": args.paired_ranking_weight,
            "paired_ranking_margin": args.paired_ranking_margin,
            "score_normalization": args.score_normalization,
            "hard_benign_manifest": [str(p) for p in args.hard_benign_manifest],
            "hard_benign_limit": args.hard_benign_limit,
            "hard_benign_only": args.hard_benign_only,
            "pretrain": str(Path(args.pretrain_ckpt) if args.pretrain_ckpt else PRETRAIN_CKPT),
        },
    }
    with open(out_dir / out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_dir / out_file}")


if __name__ == "__main__":
    main()
