from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOL = 5e-4


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise AssertionError(f"{path} did not contain a JSON object")
    return data


def _assert_close(name: str, actual: float, expected: float, tol: float = TOL) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def _claim1(args: argparse.Namespace) -> None:
    data = _read_json(Path(args.result))
    if data.get("n_folds") != 7:
        raise AssertionError(f"n_folds: expected 7, got {data.get('n_folds')}")
    _assert_close("mean_auroc", float(data["mean_auroc"]), 0.9582010582010582)
    _assert_close(
        "mean_detection_rate",
        float(data["mean_detection_rate"]),
        0.9904761904761905,
    )
    print("claim1 ok: LOPO mean AUROC and detection rate match expected values")


def _load_mean_auroc(path: Path) -> float:
    return float(_read_json(path)["mean_auroc"])


def _claim2(args: argparse.Namespace) -> None:
    root = Path(args.path_ablation_dir)
    expected = {
        "lopo_results_path_ablation_byte.json": 0.862,
        "lopo_results_path_ablation_arm64.json": 0.854,
        "lopo_results_path_ablation_dalvik.json": 0.889,
        "lopo_results_path_ablation_dalvik_byte.json": 0.903,
        "lopo_results_path_ablation_full.json": 0.828,
        "lopo_results_routing_path_dropout_full.json": 0.9582010582010582,
    }
    actual: dict[str, float] = {}
    for filename, value in expected.items():
        path = root / filename
        actual[filename] = _load_mean_auroc(path)
        _assert_close(filename, actual[filename], value, tol=0.001)

    if not actual["lopo_results_path_ablation_full.json"] < actual["lopo_results_path_ablation_dalvik_byte.json"]:
        raise AssertionError("naive three-path should underperform Dalvik+byte")
    if not actual["lopo_results_routing_path_dropout_full.json"] > actual["lopo_results_path_ablation_dalvik_byte.json"]:
        raise AssertionError("routed three-path + dropout should outperform Dalvik+byte")

    print("claim2 ok: path ablation ordering and AUROC values match expected values")


def _claim3(args: argparse.Namespace) -> None:
    baseline = _read_json(Path(args.baseline))
    hard = _read_json(Path(args.hard_benign))

    _assert_close("baseline.auroc", float(baseline["auroc"]), 0.6)
    _assert_close("baseline.benign_mean", float(baseline["benign_mean"]), 0.8520507149863988)
    _assert_close("hard.auroc", float(hard["auroc"]), 0.9279778393351801)
    _assert_close("hard.auprc", float(hard["auprc"]), 0.8972750457588325)
    _assert_close("hard.benign_mean", float(hard["benign_mean"]), 0.16610928969759312)
    _assert_close("hard.fpr_at_95_tpr", float(hard["fpr_at_95_tpr"]), 0.3157894736842105)

    if not float(hard["auroc"]) > float(baseline["auroc"]):
        raise AssertionError("hard-benign AUROC should exceed baseline AUROC")
    if not float(hard["benign_mean"]) < float(baseline["benign_mean"]):
        raise AssertionError("hard-benign training should lower benign mean score")

    print("claim3 ok: strict DPT-v2 hard-benign repair matches expected values")


def _claim4(_: argparse.Namespace) -> None:
    try:
        import android_packer  # noqa: F401
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "Could not import android_packer. Run bash install.sh first, or set "
            "PYTHONPATH=artifact/src before invoking this verifier."
        ) from exc

    print("claim4 ok: android_packer imports successfully")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="claim", required=True)

    p1 = sub.add_parser("claim1")
    p1.add_argument("--result", required=True)
    p1.set_defaults(func=_claim1)

    p2 = sub.add_parser("claim2")
    p2.add_argument("--path-ablation-dir", required=True)
    p2.set_defaults(func=_claim2)

    p3 = sub.add_parser("claim3")
    p3.add_argument("--baseline", required=True)
    p3.add_argument("--hard-benign", required=True)
    p3.set_defaults(func=_claim3)

    p4 = sub.add_parser("claim4")
    p4.set_defaults(func=_claim4)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
