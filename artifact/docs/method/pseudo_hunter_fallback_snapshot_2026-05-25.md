# PseudoHunter Fallback Snapshot

Date: 2026-05-25
Status: frozen fallback for paper writing if the next method track does not
produce stronger results in time.

This document preserves the current method, configurations, and major results.
Future method experiments must not overwrite these claims; they should be
reported as a separate candidate track until a decision is made.

## 1. Fallback Method

Fallback method name: PseudoHunter.

Core formulation:

- typed APK entry / region extraction;
- Dalvik, ARM64, and byte pseudo-code streams;
- shared PseudoCodeBERT pretrained on benign APKs;
- frozen encoder during supervised packer training;
- region-type routing and training-only path dropout;
- weak APK-level MIL detection with entry-level localization scores.

Strongest fallback configuration:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths dalvik,arm64,byte `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --output-suffix routing_path_dropout_full `
  --resume --save-every 1
```

Result file:
`outputs/experiments/path_ablation/lopo_results_routing_path_dropout_full.json`

## 2. Main LOPO Result

Leave-One-Packer-Out over seven packer families:

| Held-out packer | AUROC | Detection |
|---|---:|---:|
| Ali | 0.9111 | 14/15 |
| Qihoo | 0.9667 | 15/15 |
| Tencent | 0.9667 | 15/15 |
| 360 | 0.9667 | 1/1 |
| Bangcle | 0.9667 | 9/9 |
| APKProtector | 0.9630 | 18/18 |
| DPT Shell | 0.9667 | 18/18 |

Summary:

- mean AUROC: 0.9582;
- mean detection rate: 99.0%;
- mean normality MRR: 0.4367;
- mean attention MRR: 0.4679;
- mean inference: 84 ms/APK.

Interpretation: fixed region-type routing plus path dropout rescues the
multi-path model. It is the current best packer-disjoint LOPO result.

## 3. Path Ablation Results

| Configuration | Mean AUROC | Notes |
|---|---:|---|
| routed three-path + path dropout | 0.9582 | best fallback configuration |
| Dalvik + byte | 0.9025 | strongest simple non-routed path set |
| Dalvik only | 0.8887 | strongest single path in LOPO |
| byte only | 0.8618 | useful but coarse |
| ARM64 only | 0.8542 | useful on some native-heavy folds, unstable on Qihoo |
| naive three-path concat | 0.8281 | Qihoo collapses to 0.3333 |

Interpretation: all views contain signal, but naive concatenation is not
reliable. Path reliability modeling is necessary.

## 4. Strict DPT-v2 Results

Strict app-disjoint DPT-v2 baseline:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths dalvik,arm64,byte `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --track-b-v2-strict `
  --resume --save-every 1
```

Result file:
`outputs/experiments/track_b_v2_strict_dpt/results.json`

Result:

- test: 20 DPT-packed APKs + 20 benign counterparts;
- AUROC: 0.6000;
- detection: 20/20;
- packed mean score: 1.0000;
- benign mean score: 0.8521;
- entry normality MRR: 0.4306;
- entry attention AUROC: 0.7046.

Interpretation: the model detects DPT-packed APKs at threshold 0.5, but assigns
high scores to many benign counterparts. Strict DPT-v2 is a ranking/calibration
failure and must be described as a limitation, not as a positive main result.

## 4a. Hard-Benign Strict DPT Update (2026-05-26)

The strict DPT conclusion changed after adding APKiD-clean modern hard benign
training data. The original no-hard-benign strict DPT run remains important as
the failure that motivated the repair, but it is no longer the latest result.

F-Droid-expanded hard benign run:

- manifest:
  `outputs/experiments/hard_benign/manifest_fdroid_expanded_apkid.json`;
- hard benign training APKs: 24 APKiD-clean train-allowed samples;
- result:
  `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json`;
- AUROC: 0.9280;
- AUPRC: 0.8973;
- detection: 17/19 raw, 18/19 after train-benign z normalization;
- packed mean: 0.8831 raw;
- benign mean: 0.1661 raw;
- FPR@95TPR: 0.3158;
- TPR@1%FPR / TPR@5%FPR: 0.1579 / 0.1579.

Interpretation: hard-benign normality substantially reduces strict benign
over-scoring and repairs ranking compared with the original 0.600 AUROC
routed baseline. It is still not a production-low-FPR detector: fixed-FPR TPR
remains weak. Paper wording should therefore be "hard benign normality is the
highest-leverage repair so far" rather than "strict DPT is solved."

AndroZoo replication status:

- CSV index download is active as of 2026-05-26 13:06 CST;
- APK download has not started yet because the CSV gzip is incomplete;
- the detached pipeline will select/download 50 modern Play-market benign APKs,
  APKiD-audit them into
  `outputs/experiments/hard_benign/manifest_androzoo_modern_apkid.json`, and
  launch `strict_dpt_clean_hardbenign_androzoo50_lowbyte025` automatically.

## 5. Strict Clean Controls

APKiD flagged one strict benign seed as `SecNeo.B`:

`005AF753A03FA7D753FD2C8988E91B47966187A504F24E4187BDD19AF5797B00.apk`

After excluding this dirty pair, strict clean has 19 packed + 19 benign APKs.

| Configuration | AUROC | Packed mean | Benign mean | Result file |
|---|---:|---:|---:|---|
| routed full clean | 0.5789 | 1.0000 | 0.8557 | `results_strict_dpt_clean_z.json` |
| routed full + paired ranking | 0.5789 | 1.0000 | 0.8421 | `results_strict_dpt_pairrank_clean_z.json` |
| byte-only clean | 0.5526 | 1.0000 | 0.8947 | `results_strict_dpt_clean_byte.json` |
| Dalvik-only clean | 0.6953 | 0.9997 | 0.7741 | `results_strict_dpt_clean_dalvik.json` |

Interpretation:

- paired ranking did not improve strict clean DPT;
- byte-only is the worst strict clean path and over-scores benign APKs most;
- Dalvik-only is the strongest strict clean diagnostic so far;
- strict failure is likely a path reliability / hard-benign normality problem,
  not proof that pseudo-code normality is useless.

Pending strict clean controls:

- ARM64-only;
- Dalvik + byte;
- Dalvik + ARM64;
- ARM64 + byte;
- routed three-path with reduced byte weights.

## 6. Partial Unfreeze Result

Partial BERT unfreeze configuration:

- routed three-path + path dropout;
- `--bert-train-mode last_n`;
- `--bert-last-n-layers 2`;
- `--lr-bert 1e-5`.

Result file:
`outputs/experiments/path_ablation/lopo_results_routing_dropout_unfreeze_last2.json`

Result:

- mean AUROC: 0.8978;
- mean detection rate: 92.4%;
- Ali AUROC: 0.5844.

Interpretation: partial unfreeze is worse than frozen in the current supervised
data regime. The fallback method should keep BERT frozen unless a larger,
cleaner supervised corpus is available.

## 7. CPU / APKiD Strict Baselines

APKiD on strict DPT-v2:

- AUROC: 0.975;
- packed detection: 20/20;
- benign false positives: 1/20.

CPU structural/statistical scores:

| Score | AUROC |
|---|---:|
| `.so` entry count | 0.6513 |
| APK size log2 | 0.6225 |
| asset entry count | 0.6213 |
| max entry size log2 | 0.5800 |
| max weighted entropy | 0.5400 |
| max entropy | 0.5000 |

Interpretation: APKiD is a strong known-DPT signature detector but APK-only.
Simple entropy does not separate strict DPT; weak structural shifts exist but
should not become the core method claim.

## 8. Paper Fallback Positioning

If the new method track does not produce stronger results in time, the paper
should use this fallback positioning:

- main positive result: packer-disjoint LOPO, routed three-path + path dropout;
- honest limitation: strict app-disjoint DPT exposes high benign scores;
- diagnostic insight: Dalvik-only is stronger than byte/full on strict clean,
  so byte path reliability and hard benign normality are the next bottleneck;
- APKiD comparison: known-packer signatures are strong but do not solve
  unknown/signature-mutated localization.
