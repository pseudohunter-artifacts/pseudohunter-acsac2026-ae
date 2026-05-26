# 2026-05-25 LOPO Routing / Resume Follow-up

## 1. Current Task

- Resume after an interrupted session with partially modified files.
- Read `README.md` and align with repository rules before editing.
- Complete the LOPO controls requested after the path-ablation result:
  checkpoint/resume, BERT unfreeze modes, path dropout, region-type routing,
  and additional path combinations.
- Clarify whether AndroZoo can directly provide high-quality packed labels.

## 2. Findings

- `README.md` confirms the current workstream is Phase 2 / PseudoHunter
  with Track B packer-disjoint evaluation and ACSAC writing.
- The working tree had partial edits in:
  - `src/android_packer/models/fusion_encoder.py`
  - `scripts/experiments/run_lopo_eval.py`
  - `tests/unit/test_fusion_encoder.py`
- The partial `run_lopo_eval.py` edit passed `entry_type_ids` into the model
  but `build_bag()` did not persist that field. This would break routing and
  any normal forward pass after cached bags were rebuilt.
- AndroZoo is useful for benign pretraining / candidate mining but is not a
  direct source of packed/unpacked or packer-family labels. Its official API
  and CSV metadata expose APK hashes and metadata such as VirusTotal counts,
  dates, markets, and size, but not "packed" or "packer family" supervision.
  High-quality packed training still needs APKiD filtering, manual validation,
  commercial packer pairing, or runtime evidence.

## 3. Changes

- `src/android_packer/models/fusion_encoder.py`
  - Added training-only path dropout via `FusionEncoderConfig.path_dropout_prob`.
  - Added fixed region-type routing via `use_region_type_routing`.
  - Routing prior:
    - DEX: Dalvik + weak byte
    - ELF: ARM64 + weak byte
    - archive/asset/arsc/manifest: byte only
    - unknown: low uniform weight across paths
- `scripts/experiments/run_lopo_eval.py`
  - Bumped bag cache version to include `entry_type_ids`.
  - Added CLI:
    - `--bert-train-mode {frozen,last_n,all}`
    - `--bert-last-n-layers`
    - `--lr-bert`
    - `--path-dropout`
    - `--region-type-routing`
    - `--resume`
    - `--save-every`
  - Added fold-level checkpoints under the experiment output directory:
    `checkpoints/<fold>/latest.pt`.
  - Checkpoints include model, optimizer, scheduler, CLI args, numpy RNG,
    Python RNG, Torch CPU RNG, and Torch CUDA RNG state.
  - `--resume` restores from the fold checkpoint before continuing.
  - Result JSON configs now record the new controls.
- `tests/unit/test_fusion_encoder.py`
  - Added region routing and path dropout tests.
- `tests/unit/test_lopo_eval_controls.py`
  - Added tests for path parser validation, BERT training modes, and checkpoint
    resume restoration.

## 4. Validation

- `.\.venv\Scripts\python.exe -m pytest tests\unit\test_fusion_encoder.py tests\unit\test_lopo_eval_controls.py -q`
  - `10 passed`
- `.\.venv\Scripts\python.exe -m pytest tests\unit -q`
  - `703 passed, 1 skipped, 6 subtests passed`
- `.\.venv\Scripts\python.exe -m py_compile scripts\experiments\run_lopo_eval.py src\android_packer\models\fusion_encoder.py tests\unit\test_fusion_encoder.py tests\unit\test_lopo_eval_controls.py`
  - passed
- `git diff --check -- scripts/experiments/run_lopo_eval.py src/android_packer/models/fusion_encoder.py tests/unit/test_fusion_encoder.py tests/unit/test_lopo_eval_controls.py`
  - passed, with CRLF warnings only
- A full-data one-epoch smoke was attempted, but timed out after 10 minutes
  while refreshing/building cached APK bags. The spawned Python processes were
  stopped. This did not produce usable experimental numbers.

## 5. Experiment Results

### routed three-path + path dropout

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

Result file: `outputs/experiments/path_ablation/lopo_results_routing_path_dropout_full.json`

- Mean AUROC: 0.9582
- Mean detection rate: 99.0%
- Mean normality MRR: 0.4367
- Mean attention MRR: 0.4679
- Mean inference: 84 ms/APK
- Fold notes:
  - Ali AUROC: 0.9111
  - Qihoo AUROC: 0.9667
  - Tencent AUROC: 0.9667
  - Bangcle AUROC: 0.9667
  - APKProtector AUROC: 0.9630
  - DPT AUROC: 0.9667

Interpretation: typed routing plus path dropout rescues the multi-path story.
Naive full three-path was 0.8281 and failed on Qihoo (0.3333); routed full is
0.9582 and Qihoo rises to 0.9667. This supports claiming that path reliability
modeling is necessary, while still keeping DPT-v2 strict as a limitation.

### arm64_only

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths arm64 `
  --epochs 50 --device cuda `
  --output-suffix path_ablation_arm64 `
  --resume --save-every 1
```

Result file: `outputs/experiments/path_ablation/lopo_results_path_ablation_arm64.json`

- Mean AUROC: 0.8542
- Mean detection rate: 85.1%
- Mean normality MRR: 0.3584
- Mean inference: 32 ms/APK
- Fold notes:
  - Qihoo AUROC: 0.3156
  - DPT AUROC: 0.9519
  - Bangcle AUROC: 0.9481

Interpretation: ARM64 is not pure noise. It is strong on DPT and Bangcle, but
the signal is not safe to apply uniformly; Qihoo collapses.

### dalvik_byte

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths dalvik,byte `
  --epochs 50 --device cuda `
  --output-suffix path_ablation_dalvik_byte `
  --resume --save-every 1
```

Result file: `outputs/experiments/path_ablation/lopo_results_path_ablation_dalvik_byte.json`

- Mean AUROC: 0.9025
- Mean detection rate: 85.7%
- Mean normality MRR: 0.3170
- Mean attention MRR: 0.5672
- Mean inference: 55 ms/APK
- Fold notes:
  - Ali AUROC: 0.9622
  - Qihoo AUROC: 0.5556
  - Tencent AUROC: 0.9556
  - Bangcle AUROC: 0.9519
  - APKProtector AUROC: 0.9630
  - DPT AUROC: 0.9630

Interpretation: Dalvik+byte is the strongest simple non-routed configuration,
ahead of Dalvik-only (0.8887), byte-only (0.8618), ARM64-only (0.8542), and
naive three-path (0.8281). Byte adds complementary signal; ARM64 should be used
through routing rather than unconditional concatenation.

### Checkpoint Fix During Experiments

The first `dalvik_byte` run found a real resume bug: all path-ablation variants
shared `outputs/experiments/path_ablation/checkpoints/<fold>/latest.pt`, so a
new variant could accidentally resume from a different path configuration. This
was fixed in commit `1a8f9e2` by scoping checkpoints under
`checkpoints/<result-file-stem>/<fold>/latest.pt`.

## 6. Next Experiment Commands

User requested the next run stay frozen. Started the strongest frozen setup
first, targeting the current weak point (strict app-disjoint DPT):

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

Logs:

- `outputs/experiments/path_ablation/strict_dpt_routing_dropout.out.log`
- `outputs/experiments/path_ablation/strict_dpt_routing_dropout.err.log`

Result file: `outputs/experiments/track_b_v2_strict_dpt/results.json`

- AUROC: 0.6000
- Detection rate: 20/20 at threshold 0.5
- Packed mean score: 1.0000
- Benign mean score: 0.8521
- Entry normality MRR: 0.4306
- Entry attention AUROC: 0.7046
- Mean inference: 94 ms/APK

Interpretation: typed routing + path dropout improves strict DPT-v2 over the
earlier 0.5575/1-of-20 result and fixes threshold-scale detection, but AUROC is
still weak because benign scores are also high. This remains a limitation and
not main-table positive evidence.

Started partial unfreeze after the frozen strict run finished:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths dalvik,arm64,byte `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --bert-train-mode last_n --bert-last-n-layers 2 --lr-bert 1e-5 `
  --output-suffix routing_dropout_unfreeze_last2 `
  --resume --save-every 1
```

Logs:

- `outputs/experiments/path_ablation/routing_dropout_unfreeze_last2.out.log`
- `outputs/experiments/path_ablation/routing_dropout_unfreeze_last2.err.log`

Process command uses `--bert-train-mode last_n --bert-last-n-layers 2
--lr-bert 1e-5`; checkpoints are enabled with `--resume --save-every 1`.

Final result file:
`outputs/experiments/path_ablation/lopo_results_routing_dropout_unfreeze_last2.json`

- Mean AUROC: 0.8978
- Mean detection rate: 92.4%
- Ali fold AUROC: 0.5844, detection 7/15
- Qihoo/Tencent/360/Bangcle/APKProtector/DPT AUROC:
  0.9667 / 0.9333 / 0.9000 / 0.9667 / 0.9667 / 0.9667

Interpretation: partial BERT unfreeze is worse than the frozen routed/dropout
configuration (0.9582 mean AUROC). The Ali collapse indicates unstable
fine-tuning on the current small supervised LOPO training set, so the next
strict DPT run should stay frozen.

Additional local controls added after this result:

- `--paired-ranking-weight`
- `--paired-ranking-margin`
- `--score-normalization {none,train_benign_center,train_benign_z}`
- `--exclude-apkid-dirty-strict-benign`

Strict DPT output suffix handling was changed so the original frozen strict
baseline remains at `outputs/experiments/track_b_v2_strict_dpt/results.json`.
When `--track-b-v2-strict --output-suffix <name>` is used, the new run writes
`results_<name>.json` and gets an isolated checkpoint root.

Frozen strict DPT mitigation attempt:

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
  --paired-ranking-weight 0.5 `
  --paired-ranking-margin 1.0 `
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_pairrank_clean_z `
  --resume --save-every 1
```

Result file:
`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_pairrank_clean_z.json`

- Test size after APKiD-dirty seed exclusion: 19 packed + 19 benign
- AUROC: 0.5789
- AUROC after train-benign z-normalization: 0.5789
- Detection: 19/19 at raw threshold 0.5
- Packed mean: 1.0000
- Benign mean: 0.8421
- Normalized packed mean: 3.7417
- Normalized benign mean: 3.1087
- Entry normality MRR: 0.1303
- Entry attention AUROC: 0.6284

Interpretation: dirty-seed exclusion plus paired ranking does not fix strict
DPT app-disjoint ranking. The model still saturates all packed scores at 1.0
and keeps benign scores very high. Score normalization is monotonic, so it
does not change AUROC; it also does not repair the 0.5 threshold because both
packed and benign remain far above train-benign normality. This should be
reported as a negative mitigation attempt unless a separate clean/no-pairrank
control shows the paired objective caused the drop.

Clean-set control without paired ranking:

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
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_clean_z `
  --resume --save-every 1
```

Result file:
`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_z.json`

- Test size: 19 packed + 19 benign
- AUROC: 0.5789
- AUROC after train-benign z-normalization: 0.5789
- Detection: 19/19
- Packed mean: 1.0000
- Benign mean: 0.8557
- Entry normality MRR: 0.3328
- Entry suspicion MRR: 0.4702
- Entry attention AUROC: 0.6750

Interpretation: paired ranking did not cause the AUROC drop; the clean 19/19
split itself is slightly harder than the original 20/20 split. Paired ranking
mainly worsens localization/entry ranking in this setting and should not be
used as a claimed improvement. The core strict failure remains benign
over-scoring under app-disjoint DPT.

Strict clean path ablation, byte-only:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths byte `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --track-b-v2-strict `
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_clean_byte `
  --resume --save-every 1
```

Result file:
`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_byte.json`

- AUROC: 0.5526
- AUROC after train-benign z-normalization: 0.5526
- Detection: 19/19
- Packed mean: 1.0000
- Benign mean: 0.8947
- Entry normality MRR: 0.3764
- Entry attention AUROC: 0.6088
- Mean inference: 37.0 ms/APK

Interpretation: byte-only is worse than routed full on the same clean strict
split (0.5526 vs 0.5789) and pushes benign mean even higher (0.8947). This
supports the hypothesis that raw byte normality/high-entropy cues are a major
source of benign over-scoring under app-disjoint DPT.

Strict clean path ablation, Dalvik-only:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths dalvik `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --track-b-v2-strict `
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_clean_dalvik `
  --resume --save-every 1
```

Result file:
`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_dalvik.json`

- AUROC: 0.6953
- AUROC after train-benign z-normalization: 0.6953
- Detection: 19/19
- Packed mean: 0.9997
- Benign mean: 0.7741
- Entry normality MRR: 0.4168
- Entry attention AUROC: 0.6595
- Mean inference: 34.2 ms/APK

Interpretation: Dalvik-only is substantially better than full clean (0.6953
vs 0.5789) and byte-only (0.5526). It also lowers strict benign mean from
0.8557/0.8947 to 0.7741 while preserving packed detection. This is the
strongest evidence so far that strict app-disjoint DPT failure is aggravated by
byte-path over-scoring; the next controls should test ARM64-only and
Dalvik+byte.

## 7. Modern Packed/Unpacked Data Audit

Added an auditable pair-manifest tool for modern paired data:

- `scripts/data/build_packed_pair_manifest.py`
- `tests/unit/test_build_packed_pair_manifest.py`
- `docs/references/dataset_download_guide.md` §3.5

The tool stitches benign seed APKs together with externally packed outputs and
optionally runs APKiD on both sides. It accepts hierarchical layouts
(`<packed>/<packer>/<seed>/packed.apk`), flat layouts (`<packer>__<seed>.apk`),
and Track B v2 SHA-prefix layouts such as `dpt__<sha-prefix>.apk`.

Validation on existing Track B v2 AndroZoo/DPT data:

```powershell
.\.venv\Scripts\python.exe scripts\data\build_packed_pair_manifest.py `
  --unpacked-dirs data\real_world\track_b_v2\benign `
  --packed-dir data\real_world\track_b_v2\packed `
  --packers dpt_shell `
  --source androzoo `
  --run-apkid `
  --apkid-cmd .\.venv\Scripts\apkid.exe `
  --out-jsonl outputs\experiments\paired_packed_apks\track_b_v2_dpt_pairs.jsonl `
  --summary-out outputs\experiments\paired_packed_apks\track_b_v2_dpt_summary.json
```

Results:

- Records: 20 paired candidates
- Status: 19 `paired`, 1 `unpacked_not_clean`
- APKiD clean unpacked: 19/20
- APKiD packed-side packer/protector hit: 20/20
- Manual-review seed: `005AF753A03FA7D753FD2C8988E91B47966187A504F24E4187BDD19AF5797B00.apk`
  had an unpacked-side `SecNeo.B` packer hit, so it should not be treated as a
  clean benign seed without review.

APKiD environment note: installed APKiD is 3.1.0. It exposes `apkid -h` and
direct `apkid -j <apk>` scanning; `apkid --prepare` and `apkid --version` are
not valid in this environment. Docs were updated to stop requiring
`--prepare` for APKiD 3.x.

## 8. Deferred / Next Step

- CPU-only strict DPT baselines were added after the frozen strict result:
  - `scripts/experiments/run_track_b_v2_strict_cpu_baselines.py`
  - `scripts/experiments/summarize_track_b_v2_apkid.py`
- Results on the same 20 Track B v2 DPT-packed APKs + 20 benign counterparts:
  - APKiD packer/protector hit: AUROC 0.975, packed 20/20, benign false
    positive 1/20 (`005AF...`).
  - CPU structural/stat baselines: best simple score is `so_entry_count`
    AUROC 0.651; `apk_size_log2` AUROC 0.623; `asset_entry_count` AUROC
    0.621; entropy scores are 0.500.
- Interpretation: strict DPT-v2 is not separable by simple high-entropy
  heuristics, but known-signature APKiD is very strong on this exact DPT set.
  This reinforces the paper framing: APKiD is a strong known-packer detector,
  while PseudoHunter's value must come from localization and robustness to
  unknown/signature-mutated packers.
- Update `docs/progress/sessions/2026-05-24_path_ablation_track_b_v2.md` only
  if that historical log is promoted into a consolidated report; it remains
  accurate as of its own session.
- Partial unfreeze has finished and is worse than frozen: mean AUROC 0.8978
  vs 0.9582, with Ali collapsing to 0.5844. Do not run strict DPT under this
  partial-unfreeze setting unless a larger supervised corpus is added.
- Run the next strict DPT-v2 experiment frozen, using paired ranking,
  train-benign score normalization, and APKiD-dirty benign exclusion.
- Clean-set no-pairrank control is complete: AUROC is also 0.5789. The next
  useful experiment is path-specific strict clean ablation (`dalvik`, `byte`,
  `arm64`, `dalvik,byte`) to identify which stream drives benign over-scoring.
- Byte-only clean strict AUROC is 0.5526 and Dalvik-only is 0.6953. The next
  path controls are ARM64-only and Dalvik+byte.
- Exclude or manually review the `005AF...` seed before treating Track B v2 as
  a clean 20-pair modern benign/DPT corpus.

## 9. Method Snapshot and New Candidate Route

Current method is preserved as a paper fallback:

- `docs/method/pseudo_hunter_fallback_snapshot_2026-05-25.md`

This snapshot freezes the current PseudoHunter method and numbers:

- routed three-path + path dropout LOPO mean AUROC 0.9582;
- strict DPT baseline AUROC 0.6000 with high benign scores;
- strict clean full AUROC 0.5789;
- strict clean byte-only AUROC 0.5526;
- strict clean Dalvik-only AUROC 0.6953;
- partial unfreeze mean AUROC 0.8978, worse than frozen.

New candidate route is documented here:

- `docs/method/typed_multiview_normality_plan_2026-05-25.md`

Status: proposed. It should be treated as the next experiment design, not as a
replacement for the current paper method until results justify the switch.
Major deviations from the plan should be discussed before implementation.

Immediate next experiments under the new route:

1. Finish Stage 0 strict clean diagnostics:
   - ARM64-only;
   - Dalvik+byte;
   - top-scored strict benign entry-type statistics.
2. Run V1 byte downweight / gate controls.
3. Build APKiD-clean hard benign set and keep borderline APKs out of training.
4. Decide with the user before implementing corrupted-region pretraining,
   learned path confidence, or a larger objective rewrite.

V1 byte-downweight experiment started after the plan:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths dalvik,arm64,byte `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --routing-dex-byte-weight 0.05 `
  --routing-elf-byte-weight 0.05 `
  --routing-byte-entry-weight 0.25 `
  --routing-unknown-weight 0.05 `
  --track-b-v2-strict `
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_clean_lowbyte025 `
  --resume --save-every 1
```

Result file:
`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_lowbyte025.json`

- AUROC: 0.6053
- AUROC after train-benign z-normalization: 0.6053
- Detection: 19/19
- Packed mean: 1.0000
- Benign mean: 0.8588
- Entry normality MRR: 0.4206
- Entry suspicion MRR: 0.5495
- Entry attention AUROC: 0.6609

Interpretation: reducing byte contribution improves strict clean AUROC over
routed full clean (0.6053 vs 0.5789), but benign mean remains high. This is a
small ranking improvement, not a calibration fix. More aggressive byte
downweighting and entry-type false-positive diagnostics are still needed.

More aggressive V1 byte-downweight:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths dalvik,arm64,byte `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --routing-dex-byte-weight 0.01 `
  --routing-elf-byte-weight 0.01 `
  --routing-byte-entry-weight 0.05 `
  --routing-unknown-weight 0.01 `
  --track-b-v2-strict `
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_clean_lowbyte005 `
  --resume --save-every 1
```

Result file:
`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_lowbyte005.json`

- AUROC: 0.5789
- AUROC after train-benign z-normalization: 0.5789
- Detection: 19/19
- Packed mean: 1.0000
- Benign mean: 0.8431
- Entry normality MRR: 0.4667
- Entry attention AUROC: 0.7074
- Entry attention MRR: 0.5976

Interpretation: aggressive byte downweighting lowers the benign mean slightly
and improves attention-based localization, but APK AUROC falls back to the
routed-full clean baseline. The 0.25 byte-entry setting is currently the better
strict ranking tradeoff, while 0.05 is more useful as a localization/calibration
diagnostic.

## 10. Long Autonomous Run: A-E Normality Route Progress

User authorized a long autonomous run through A-E before considering F/G. The
current execution order is:

- A: strict clean path diagnostics;
- B: strict benign top-entry type statistics;
- C: fixed byte-gate sweep;
- D: APKiD-clean hard benign manifest;
- E: hard benign + fixed gate strict training.

### A. Additional strict clean path diagnostics

ARM64-only:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths arm64 `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --track-b-v2-strict `
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_clean_arm64 `
  --resume --save-every 1
```

Result file:
`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_arm64.json`

- AUROC: 0.6053
- Detection: 19/19
- Packed mean: 1.0000
- Benign mean: 0.8421
- Entry attention AUROC: 0.6495

Dalvik+byte:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths dalvik,byte `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --track-b-v2-strict `
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_clean_dalvik_byte `
  --resume --save-every 1
```

Result file:
`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_dalvik_byte.json`

- AUROC: 0.5526
- Detection: 19/19
- Packed mean: 1.0000
- Benign mean: 0.8947
- Entry normality AUROC: 0.6249
- Entry attention AUROC: 0.6892

Interpretation: Dalvik+byte is worse than Dalvik-only (0.5526 vs 0.6953) and
matches byte-only at APK AUROC, with the same high benign mean. This is direct
evidence that the byte path is damaging strict app-disjoint ranking when added
to the otherwise stronger Dalvik signal. ARM64-only is not useless (0.6053),
but still does not close the benign false-positive gap.

### B. Strict benign top-entry type statistics

Added:

- `scripts/experiments/analyze_strict_benign_entries.py`

The script reloads a strict checkpoint, rebuilds strict benign bags without
cache by default so entry names are guaranteed fresh, and aggregates top entries
by:

- `apk_contribution`;
- `attention`;
- `suspicion`;
- `anomaly = 1 - normality`.

Output for full clean strict:

- `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_z_benign_entry_types.json`
- `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_z_benign_entry_types.md`

Top findings:

- `apk_contribution`: `lib_so` 37/57 top-k hits, then `unknown_blob` 11/57.
- `attention`: `lib_so` 37/57, then `unknown_blob` 11/57.
- `suspicion`: `lib_so` 44/57 with mean 1.0000.
- `anomaly`: `lib_so` 44/57 with mean 1.0000.

Interpretation: the strict benign failure is not only "byte path sees
high-entropy assets." Native libraries in modern benign APKs are the dominant
false-positive source for the current full clean checkpoint. This makes
hard-benign normality and path reliability more important than only reducing
global byte weights.

### D. Hard benign manifest

Added:

- `scripts/data/build_hard_benign_manifest.py`

Generated local audit artefacts:

- `outputs/experiments/hard_benign/manifest.json`
- `outputs/experiments/hard_benign/manifest.jsonl`
- `outputs/experiments/hard_benign/manifest.md`

Summary:

- records: 29
- failures: 0
- `benign-hard-clean`: 28
- `benign-borderline`: 1 (`005AF...`, APKiD hit `SecNeo.B`)
- train-allowed clean APKs: 9
- train-allowed hard-clean APKs: 9

The 20 Track B v2 strict benign APKs are explicitly marked
`strict_dpt_test_set=true` and `train_allowed=false`, so they are not used for
E-stage training. The 9 train-allowed APKs are the existing Track B/F-Droid
benign set. This is a first hard-benign training seed, not yet a large
AndroZoo/F-Droid expansion.

### E. Hard benign + fixed gate training

Runner support added:

- `--hard-benign-manifest`
- `--hard-benign-limit`
- `--hard-benign-only`

The first E-stage run started:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\run_lopo_eval.py `
  --pretrain-ckpt outputs\experiments\pseudo_bert_v3\checkpoints\epoch_050.pt `
  --bert-layers 8 --bert-dim 512 `
  --ablation bert_only `
  --paths dalvik,arm64,byte `
  --epochs 50 --device cuda `
  --path-dropout 0.25 `
  --region-type-routing `
  --routing-dex-byte-weight 0.05 `
  --routing-elf-byte-weight 0.05 `
  --routing-byte-entry-weight 0.25 `
  --routing-unknown-weight 0.05 `
  --hard-benign-manifest outputs\experiments\hard_benign\manifest.json `
  --hard-benign-only `
  --track-b-v2-strict `
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_clean_hardbenign_lowbyte025 `
  --resume --save-every 1
```

Final result file:
`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_lowbyte025.json`

- AUROC: 0.9335
- AUPRC: 0.9397
- FPR@95TPR: 0.6316
- TPR@1%FPR: 0.4737
- TPR@5%FPR: 0.4737
- Detection: 18/19
- Packed mean: 0.9485
- Benign mean: 0.4892
- Entry normality AUROC: 0.7064
- Entry attention AUROC: 0.7707
- Entry normality MRR: 0.4385
- Entry attention MRR: 0.4403

The first full training invocation took about 450.7 s. The result JSON was
later regenerated with `--resume` after adding AUPRC / fixed-FPR fields; that
resume path starts from epoch 50 and therefore reports `train_time_s=1.1` for
the re-evaluation invocation.

Interpretation: this is the first strong positive strict app-disjoint DPT-v2
result. The change is not a new structural shortcut: it keeps BERT frozen and
uses the existing routed/dropout pseudo-code model, but adds a small
APKiD-clean hard-benign training seed. The main effect is the intended one:
strict benign scores drop sharply while packed scores remain high.

E-stage top-entry follow-up:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\analyze_strict_benign_entries.py `
  --device cuda `
  --use-bag-cache `
  --result outputs\experiments\track_b_v2_strict_dpt\results_strict_dpt_clean_hardbenign_lowbyte025.json `
  --top-k 3
```

Output files:

- `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_lowbyte025_benign_entry_types.json`
- `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_lowbyte025_benign_entry_types.md`

Top findings after hard benign:

- `apk_contribution`: `elf` 43/57 top-k hits, mean 0.0905, max 0.1866.
- `attention`: `dex` 28/57, `elf` 24/57.
- `suspicion`: `asset_entry` 48/57, mean 0.5347.
- `anomaly`: `elf` 31/57, `dex` 21/57.

Interpretation: native entries still appear frequently in top diagnostics, but
their APK contribution is much lower than before. The strict improvement is
therefore consistent with hard benign reducing native-heavy benign APK
over-scoring rather than merely shifting the threshold.

### Validation

Commands run after code changes:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  scripts\experiments\run_lopo_eval.py `
  scripts\experiments\analyze_strict_benign_entries.py `
  scripts\data\build_hard_benign_manifest.py

.\.venv\Scripts\python.exe -m pytest `
  tests\unit\test_lopo_eval_controls.py `
  tests\unit\test_fusion_encoder.py -q
```

Result after the final metric helper update: `19 passed`.

### Deferred

- `strict_dpt_clean_lowbyte010` was started as a V1 middle sweep but stopped
  after a previous concurrent CUDA initialization conflict. Re-run it after
  the E-stage hard-benign result is documented and committed only if a middle
  byte-weight point is still useful for ablation completeness.
- The current hard-benign pool is useful but too small. A separate
  AndroZoo/F-Droid expansion should add more train-allowed modern benign APKs
  that are not Track B v2 strict test apps.
- Do not start F/G immediately. A-E produced a strong positive result, so the
  next decision should first replicate/ablate hard benign rather than jumping
  to corrupted-region pretraining or learned path confidence.

### Parallel Execution Policy

Current safe parallelism:

- Keep only one CUDA training job active at a time. Earlier concurrent CUDA
  starts caused initialization conflict, and simultaneous strict runs would make
  runtime and checkpoint behavior harder to review.
- Run CPU/I/O work in parallel with the GPU slot: result parsing, strict benign
  entry diagnostics, hard-benign manifest scans, documentation, py_compile, and
  unit tests.
- Do not start a second large bag-building job while the E-stage run is holding
  ~17 GB RAM unless the active process stops progressing.

Status at the latest review: the E-stage hard-benign run has entered training
and is writing
`outputs/experiments/track_b_v2_strict_dpt/checkpoints/results_strict_dpt_clean_hardbenign_lowbyte025/strict_dpt/latest.pt`.
The GPU can show low memory because BERT is frozen, but utilization rises during
training steps. The run is resumable with the same command and `--resume`.
