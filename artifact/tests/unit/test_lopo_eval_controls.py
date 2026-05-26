from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="requires [dl] extra")


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "experiments" / "run_lopo_eval.py"


def _load_lopo_module():
    spec = importlib.util.spec_from_file_location("run_lopo_eval_for_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_active_paths_validates_values() -> None:
    module = _load_lopo_module()

    assert module.parse_active_paths("dalvik,byte") == ("dalvik", "byte")
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_active_paths("dalvik,x86")


def test_bert_training_modes_control_trainable_layers() -> None:
    module = _load_lopo_module()
    args = argparse.Namespace(
        bert_dim=24,
        bert_layers=2,
        ablation="bert_only",
        paths=("dalvik", "byte"),
        path_dropout=0.0,
        region_type_routing=False,
        no_batch_bert_streams=False,
        routing_dex_byte_weight=0.25,
        routing_elf_byte_weight=0.25,
        routing_byte_entry_weight=1.0,
        routing_unknown_weight=0.25,
    )
    model = module.build_model(args)

    model.configure_bert_training("frozen")
    assert not any(p.requires_grad for p in model.fusion_encoder.bert.parameters())

    model.configure_bert_training("last_n", last_n_layers=1)
    first_layer_trainable = any(
        p.requires_grad for p in model.fusion_encoder.bert.layers[0].parameters()
    )
    last_layer_trainable = any(
        p.requires_grad for p in model.fusion_encoder.bert.layers[-1].parameters()
    )
    assert not first_layer_trainable
    assert last_layer_trainable

    model.configure_bert_training("all")
    assert all(p.requires_grad for p in model.fusion_encoder.bert.parameters())


def test_build_model_passes_routing_weights() -> None:
    module = _load_lopo_module()
    args = argparse.Namespace(
        bert_dim=24,
        bert_layers=2,
        ablation="bert_only",
        paths=("dalvik", "arm64", "byte"),
        path_dropout=0.0,
        region_type_routing=True,
        no_batch_bert_streams=False,
        routing_dex_byte_weight=0.05,
        routing_elf_byte_weight=0.10,
        routing_byte_entry_weight=0.25,
        routing_unknown_weight=0.05,
    )
    model = module.build_model(args)

    assert model.fusion_encoder.cfg.routing_dex_byte_weight == pytest.approx(0.05)
    assert model.fusion_encoder.cfg.routing_elf_byte_weight == pytest.approx(0.10)
    assert model.fusion_encoder.cfg.routing_byte_entry_weight == pytest.approx(0.25)
    assert model.fusion_encoder.cfg.routing_unknown_weight == pytest.approx(0.05)


def test_build_model_can_disable_batched_bert_streams() -> None:
    module = _load_lopo_module()
    args = argparse.Namespace(
        bert_dim=24,
        bert_layers=2,
        ablation="bert_only",
        paths=("dalvik", "arm64", "byte"),
        path_dropout=0.0,
        region_type_routing=False,
        no_batch_bert_streams=True,
        routing_dex_byte_weight=0.25,
        routing_elf_byte_weight=0.25,
        routing_byte_entry_weight=1.0,
        routing_unknown_weight=0.25,
    )
    model = module.build_model(args)

    assert model.fusion_encoder.cfg.batch_bert_streams is False


def test_train_fold_checkpoint_resume_restores_model(tmp_path: Path) -> None:
    module = _load_lopo_module()

    class _TinyFoldModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def forward_bag(self, bag, device, chunk_size=64):
            return {"bag_logit": self.weight + float(bag["apk_label"]) * 0.1}

    bags = [{"apk_label": 0}, {"apk_label": 1}]
    device = torch.device("cpu")
    model = _TinyFoldModel()
    module.train_fold(
        model,
        bags,
        device,
        epochs=1,
        lr=1e-3,
        checkpoint_dir=tmp_path,
        fold_name="tiny",
    )
    saved_weight = model.weight.detach().clone()
    assert (tmp_path / "latest.pt").exists()

    model.weight.data.fill_(123.0)
    module.train_fold(
        model,
        bags,
        device,
        epochs=1,
        lr=1e-3,
        checkpoint_dir=tmp_path,
        fold_name="tiny",
        resume=True,
    )

    assert torch.allclose(model.weight.detach(), saved_weight)


def test_train_fold_forwards_bag_chunk_size(tmp_path: Path) -> None:
    module = _load_lopo_module()

    class _ChunkAwareModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.seen_chunks = []

        def forward_bag(self, bag, device, chunk_size=64):
            self.seen_chunks.append(chunk_size)
            return {"bag_logit": self.weight + float(bag["apk_label"]) * 0.1}

    bags = [{"apk_label": 0}, {"apk_label": 1}]
    device = torch.device("cpu")
    model = _ChunkAwareModel()

    module.train_fold(
        model,
        bags,
        device,
        epochs=1,
        train_batch_size=2,
        bag_chunk_size=17,
        lr=1e-3,
        checkpoint_dir=tmp_path,
        fold_name="tiny",
    )

    assert model.seen_chunks == [17, 17]


def test_evaluate_and_score_forward_bag_chunk_size() -> None:
    module = _load_lopo_module()

    class _ChunkAwareModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_chunks = []

        def forward_bag(self, bag, device, chunk_size=64):
            self.seen_chunks.append(chunk_size)
            return {
                "bag_logit": torch.tensor(0.0),
                "entry_normality": torch.tensor([]),
                "entry_suspicion": torch.tensor([]),
                "entry_attention": torch.tensor([]),
            }

    bags = [{"apk_label": 0}, {"apk_label": 1, "diff_targets": None}]
    device = torch.device("cpu")
    model = _ChunkAwareModel()

    module.evaluate_fold(model, bags, device, bag_chunk_size=23)
    module.score_bags(model, bags, device, bag_chunk_size=29)

    assert model.seen_chunks == [23, 23, 29, 29]


def test_checkpoint_root_is_scoped_by_output_file() -> None:
    module = _load_lopo_module()

    root_a = module.checkpoint_root_for_output(
        Path("outputs/experiments/path_ablation"),
        "lopo_results_path_ablation_arm64.json",
    )
    root_b = module.checkpoint_root_for_output(
        Path("outputs/experiments/path_ablation"),
        "lopo_results_path_ablation_dalvik_byte.json",
    )

    assert root_a != root_b
    assert root_a.name == "lopo_results_path_ablation_arm64"
    assert root_b.name == "lopo_results_path_ablation_dalvik_byte"


def test_strict_output_suffix_preserves_baseline_results_file() -> None:
    module = _load_lopo_module()

    default_dir, default_file = module.resolve_output_path(None, track_b_v2_strict=True)
    suffix_dir, suffix_file = module.resolve_output_path(
        "strict dpt/pairrank",
        track_b_v2_strict=True,
    )

    assert default_dir == suffix_dir
    assert default_file == "results.json"
    assert suffix_file == "results_strict_dpt_pairrank.json"
    assert module.checkpoint_root_for_output(suffix_dir, suffix_file).name == (
        "results_strict_dpt_pairrank"
    )


def test_dirty_strict_packed_path_matches_sha_prefix() -> None:
    module = _load_lopo_module()

    assert module.is_apkid_dirty_strict_packed_path(
        Path("dpt__005AF753A03FA7D753FD.apk")
    )
    assert not module.is_apkid_dirty_strict_packed_path(
        Path("dpt__0008C3A85769C8082AF4.apk")
    )
    assert not module.is_apkid_dirty_strict_packed_path(
        Path("other__005AF753A03FA7D753FD.apk")
    )


def test_paired_key_from_apk_id_normalizes_known_patterns() -> None:
    module = _load_lopo_module()

    assert module.paired_key_from_apk_id("track_b_v2_benign_ABCDEF") == "ABCDEF"
    assert module.paired_key_from_apk_id("track_b_v2_dpt_ABCDEF") == "ABCDEF"
    assert module.paired_key_from_apk_id("DPT_appname") == "appname"


def test_pair_lookup_groups_bags_by_origin_key() -> None:
    module = _load_lopo_module()
    bags = [
        {"apk_id": "track_b_v2_benign_ABCDEF", "apk_label": 0},
        {"apk_id": "track_b_v2_dpt_ABCDEF", "apk_label": 1},
        {"apk_id": "track_b_v2_dpt_123456", "apk_label": 1},
    ]

    grouped = module.group_bags_by_pair_key(bags)

    assert module.lookup_pair_bags(grouped, bags[1], 0) == [bags[0]]
    assert module.lookup_pair_bags(grouped, bags[2], 0) == []


def test_strict_dpt_control_training_modes() -> None:
    module = _load_lopo_module()
    benign = [{"apk_id": "benign_a", "apk_label": 0}]
    dpt_benign = [{"apk_id": "test_benign_a", "apk_label": 0}]
    families = [
        module.PackerFamily(
            name="Ali",
            bags=[
                {"apk_id": "ali_1", "apk_label": 1},
                {"apk_id": "ali_2", "apk_label": 1},
            ],
        ),
        module.PackerFamily(
            name="Bangcle",
            bags=[{"apk_id": "bangcle_1", "apk_label": 1}],
        ),
        module.PackerFamily(
            name="DPT",
            bags=[
                {"apk_id": "dpt_1", "apk_label": 1},
                {"apk_id": "dpt_2", "apk_label": 1},
            ],
        ),
    ]

    train_a, names_a, info_a = module.build_strict_dpt_control_training_set(
        benign,
        families,
        control_mode="non_dpt",
        dpt_benign_bags=dpt_benign,
    )
    assert names_a == ["Ali", "Bangcle"]
    assert [bag["apk_id"] for bag in train_a] == ["benign_a", "ali_1", "ali_2", "bangcle_1"]
    assert info_a["added_old_dpt_positive"] == 0

    train_b, _, info_b = module.build_strict_dpt_control_training_set(
        benign,
        families,
        control_mode="add_old_dpt",
        dpt_benign_bags=dpt_benign,
    )
    assert [bag["apk_id"] for bag in train_b][-2:] == ["dpt_1", "dpt_2"]
    assert info_b["added_old_dpt_positive"] == 2

    train_c, _, info_c = module.build_strict_dpt_control_training_set(
        benign,
        families,
        control_mode="add_old_dpt_benign",
        dpt_benign_bags=dpt_benign,
    )
    assert [bag["apk_id"] for bag in train_c][-3:] == ["dpt_1", "dpt_2", "test_benign_a"]
    assert info_c["added_old_dpt_benign"] == 1

    train_d, _, info_d = module.build_strict_dpt_control_training_set(
        benign,
        families,
        control_mode="other_positive_replay",
        dpt_benign_bags=dpt_benign,
    )
    assert [bag["apk_id"] for bag in train_d] == [
        "benign_a",
        "ali_1",
        "ali_2",
        "bangcle_1",
        "ali_1",
        "ali_2",
    ]
    assert info_d["added_equal_size_other_positive"] == 2
    assert all(not bag["apk_id"].startswith("dpt_") for bag in train_d)


def test_train_benign_score_normalization() -> None:
    module = _load_lopo_module()

    centered, info = module.normalize_scores_from_train_benign(
        [0.5, 0.7],
        [0.2, 0.4],
        "train_benign_center",
    )
    assert centered == pytest.approx([0.2, 0.4])
    assert info["train_benign_mean"] == pytest.approx(0.3)

    z_scores, info = module.normalize_scores_from_train_benign(
        [0.5, 0.7],
        [0.2, 0.4],
        "train_benign_z",
    )
    assert z_scores == pytest.approx([2.0, 4.0])
    assert info["train_benign_std"] == pytest.approx(0.1)


def test_fixed_fpr_tpr_metrics() -> None:
    module = _load_lopo_module()

    metrics = module.fixed_fpr_tpr_metrics(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.8, 0.9],
    )

    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["fpr_at_95_tpr"] == pytest.approx(0.0)
    assert metrics["tpr_at_1_fpr"] == pytest.approx(1.0)
    assert metrics["tpr_at_5_fpr"] == pytest.approx(1.0)


def test_load_hard_benign_manifest_filters_to_clean_train_allowed(tmp_path: Path) -> None:
    module = _load_lopo_module()
    apk_a = tmp_path / "a.apk"
    apk_b = tmp_path / "b.apk"
    apk_a.write_bytes(b"PK\x03\x04dummy-a")
    apk_b.write_bytes(b"PK\x03\x04dummy-b")
    missing = tmp_path / "missing.apk"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
{
  "records": [
    {
      "local_path": "%s",
      "train_allowed": true,
      "apkid_clean": true,
      "label_class": "benign-hard-clean"
    },
    {
      "local_path": "%s",
      "train_allowed": true,
      "apkid_clean": true,
      "label_class": "benign-clean"
    },
    {
      "local_path": "%s",
      "train_allowed": false,
      "apkid_clean": true,
      "label_class": "benign-hard-clean"
    },
    {
      "local_path": "%s",
      "train_allowed": true,
      "apkid_clean": false,
      "label_class": "benign-hard-clean"
    }
  ]
}
"""
        % (
            apk_a.as_posix(),
            apk_b.as_posix(),
            apk_b.as_posix(),
            missing.as_posix(),
        ),
        encoding="utf-8",
    )

    selected = module.load_hard_benign_manifest_apks([manifest])
    selected_hard = module.load_hard_benign_manifest_apks(
        [manifest],
        require_hard=True,
    )

    assert selected == [apk_a, apk_b]
    assert selected_hard == [apk_a]
