# 2026-05-26 Hard Benign Expansion

## 1. Current Task

The next autonomous step is to expand hard benign data and rerun strict DPT-v2
training. The user clarified that AndroZoo CSV should be preferred when
possible because it is more reproducible and more modern; F-Droid is also
allowed as a complementary source.

## 2. Data Source Decision

Priority:

1. AndroZoo modern benign APKs selected by CSV fields (`sha256`, `dex_date`,
   `apk_size`, `vt_detection`, `markets`) and downloaded by SHA256 with the
   user-provided API key. The key is used only as a process environment value
   and is not written to repo files.
2. F-Droid modern hard benign APKs as an open-source complementary pool.

Rationale:

- AndroZoo SHA256 lists are easier for external reviewers to reproduce.
- F-Droid APKs are transparent and license-friendly, but the ecosystem is less
  representative of modern Play-store apps.
- Strict Track B v2 benign APKs remain `train_allowed=false`; they are never
  used as hard benign training data.

## 3. F-Droid Expansion

Downloaded 15 F-Droid candidates into:

`data/real_world/hard_benign/fdroid_expansion/`

Examples include large/native/asset-heavy apps such as Element, Nextcloud,
OsmAnd, StreetComplete, SuperTuxKart, Jellyfin, KDE Connect, FairEmail, and
Amaze.

Initial audit accidentally marked all samples as borderline because the shell
could not find `apkid`. Re-running with the explicit venv APKiD path fixed the
audit:

```powershell
.\.venv\Scripts\python.exe scripts\data\build_hard_benign_manifest.py `
  --apkid-cmd .\.venv\Scripts\apkid.exe `
  --apk-dir data\real_world\track_b\benign `
  --apk-dir data\real_world\track_b_v2\benign `
  --apk-dir data\real_world\hard_benign\fdroid_expansion `
  --out-json outputs\experiments\hard_benign\manifest_fdroid_expanded_apkid.json
```

Manifest summary:

- records: 44
- failures: 0
- `benign-hard-clean`: 43
- `benign-borderline`: 1 (`005AF...`, APKiD `SecNeo.B`)
- train-allowed clean APKs: 24
- train-allowed hard-clean APKs: 24

## 4. AndroZoo Expansion

The local machine did not have `data/androzoo/latest_with-added-date.csv.gz`.
A direct foreground download reset once. A resumable background curl download
was started:

```powershell
curl.exe -L --http1.1 -C - --retry 50 --retry-delay 10 --connect-timeout 30 `
  --output data\androzoo\latest_with-added-date.csv.gz `
  https://androzoo.uni.lu/static/lists/latest_with-added-date.csv.gz
```

Progress at the first review:

- CSV size on disk: about 0.02 GB / 3.56 GB.
- Download speed is slow and expected to take hours.
- After completion, select modern `vt_detection=0` candidates and fetch APKs
  by SHA256 using the API key only in the process environment.

### 4.1 Downloader Fix

Updated `scripts/data/download_androzoo_benign.py` before using it for the new
hard benign pool:

- candidate selection now scans the full CSV by default instead of stopping
  after `target * 3` matches;
- rows are ranked by preferred market (`play` by default), newest `dex_date`,
  then APK size;
- `--require-market`, `--min-dex-date`, `--min-size-mb`, `--candidate-multiplier`,
  and `--max-scan-rows` were added for reproducible curation;
- selected SHA256s and JSONL metadata are always written;
- downloads now verify SHA256 and use a `.part` file before replacing the final
  APK path.

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\unit\test_download_androzoo_benign.py `
  tests\unit\test_lopo_eval_controls.py `
  tests\unit\test_fusion_encoder.py -q
# 21 passed
```

### 4.2 Modern Benign Watcher

Started a detached watcher that does not use CUDA. It waits for the CSV curl
process to finish, validates the gzip stream, selects 50 modern benign APKs,
downloads them, and builds an APKiD-audited manifest:

- logs:
  - first attempt failed due PowerShell command quoting:
    `logs/androzoo_modern_benign_watcher_20260526-111808.err.log`
  - active watcher:
    `logs/androzoo_modern_benign_watcher2_20260526-111917.out.log`
  - active watcher stderr:
    `logs/androzoo_modern_benign_watcher2_20260526-111917.err.log`
- target data path: `data/androzoo/hard_benign_modern/`
- candidate metadata:
  - `data/androzoo/hard_benign_modern_candidates.sha256.txt`
  - `data/androzoo/hard_benign_modern_candidates.jsonl`
- manifest after audit:
  - `outputs/experiments/hard_benign/manifest_androzoo_modern_apkid.json`

The AndroZoo API key is used only through the watcher process environment and
is not written to repo files or logs.

### 4.3 Robust AndroZoo Pipeline

The watcher in §4.2 was superseded by a more autonomous detached pipeline. The
older `watcher2` process was stopped. The active pipeline:

- waits for the current CSV curl process;
- validates the gzip CSV;
- if the curl process exits before the gzip is valid, resumes the CSV download
  with `curl -C -`;
- selects and downloads 50 modern Play-market benign APKs;
- audits them with APKiD into:
  `outputs/experiments/hard_benign/manifest_androzoo_modern_apkid.json`;
- launches the same strict DPT-v2 protocol with:
  `--output-suffix strict_dpt_clean_hardbenign_androzoo50_lowbyte025`;
- runs strict benign entry-type diagnostics for the AndroZoo run.

Logs:

- `logs/androzoo_modern_benign_pipeline_20260526-114653.out.log`
- `logs/androzoo_modern_benign_pipeline_20260526-114653.err.log`

The stderr currently contains gzip EOF tracebacks while the CSV is incomplete;
that is expected during validation of a partial gzip and is not a pipeline
failure.

Status check at `2026-05-26T13:06:41+08:00`:

- CSV download is still active, not stalled.
- `curl` process: PID 26508, started `2026-05-26 10:57:24`.
- CSV path: `data/androzoo/latest_with-added-date.csv.gz`.
- Current size: about 0.667 GiB; curl stderr reports about 682.7 MiB of
  3.56 GiB.
- Last write age at check time: < 1 second.
- APK download has not started yet; `data/androzoo/hard_benign_modern/` is
  still empty because the CSV gzip is incomplete.
- Existing detached pipeline PID 24084 is waiting and should not be duplicated.

## 5. Training Run

Started strict DPT-v2 training with the expanded F-Droid manifest:

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
  --hard-benign-manifest outputs\experiments\hard_benign\manifest_fdroid_expanded_apkid.json `
  --hard-benign-only `
  --track-b-v2-strict `
  --score-normalization train_benign_z `
  --exclude-apkid-dirty-strict-benign `
  --output-suffix strict_dpt_clean_hardbenign_fdroid24_lowbyte025 `
  --resume --save-every 1
```

Final result:

- The first foreground process stopped after the tool timeout and did not write
  a result JSON or checkpoint.
- Relaunched detached at `2026-05-26 11:15` with the same command, `--resume`,
  and `--save-every 1`.
- Logs:
  - `logs/strict_dpt_fdroid24_lowbyte025_20260526-111551.out.log`
  - `logs/strict_dpt_fdroid24_lowbyte025_20260526-111551.err.log`
- Result:
  `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json`
- Training set: 127 bags (non-DPT packers + 54 train benign, including 24
  APKiD-clean train-allowed hard benign APKs).
- Strict test set: 19 DPT-v2 packed + 19 strict benign APKs.
- AUROC: 0.9280.
- AUPRC: 0.8973.
- Detection at threshold 0.5: 17/19 raw, 18/19 after train-benign z
  normalization.
- Packed mean score: 0.8831 raw, 9.7165 normalized.
- Benign mean score: 0.1661 raw, 1.6291 normalized.
- FPR@95TPR: 0.3158.
- TPR@1%FPR / TPR@5%FPR: 0.1579 / 0.1579.
- Train time after cache construction: 533.5 s.

Comparison against the prior 9-hard-benign result:

| Run | Hard benign train APKs | AUROC | AUPRC | Detection | Packed mean | Benign mean | FPR@95TPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9-hard-benign | 9 | 0.9335 | 0.9397 | 18/19 | 0.9485 | 0.4892 | 0.6316 |
| F-Droid expanded | 24 | 0.9280 | 0.8973 | 17/19 raw; 18/19 normalized | 0.8831 | 0.1661 | 0.3158 |

Interpretation:

- The larger F-Droid hard benign pool did not improve the AUROC peak over the
  9-hard-benign run, but it sharply lowered strict benign scores and cut
  FPR@95TPR by half.
- This is the expected direction if hard benign data is teaching normality for
  complex benign APKs.
- The low TPR at 1%-5% FPR remains a limitation; the detector is not yet
  production-low-FPR, even though strict app-disjoint ranking is much better
  than the original routed baseline.

Top-entry diagnostics from:

```powershell
.\.venv\Scripts\python.exe scripts\experiments\analyze_strict_benign_entries.py `
  --device cuda `
  --use-bag-cache `
  --result outputs\experiments\track_b_v2_strict_dpt\results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json `
  --top-k 3
```

- `apk_contribution`: `elf` 40/57 top-k hits, mean 0.1145, max 0.3611;
  `dex` 5/57, mean 0.1154; `asset_entry` 8/57, mean 0.0031.
- `attention`: `dex` 27/57 and `elf` 25/57.
- `suspicion`: `asset_entry` 51/57, mean 0.5237.
- `anomaly`: `elf` 26/57 and `dex` 26/57.

Native and DEX entries still dominate attention/anomaly, while asset entries
dominate the standalone suspicion head. The APK contribution is no longer
saturated, so hard benign training is reducing but not eliminating the
native/asset false-positive pressure.

## 6. Next Step

1. Let the AndroZoo CSV download and robust pipeline continue.
2. After `manifest_androzoo_modern_apkid.json` exists, run the same strict DPT
   protocol with the APKiD-clean AndroZoo hard benign pool. The active pipeline
   is already configured to do this automatically.
3. Compare F-Droid-expanded and AndroZoo-expanded results before moving to
   corrupted-region pretraining or learned path confidence.

## 7. AndroZoo Release CSV Fast Path

Status check at `2026-05-26T14:11:10+08:00`:

- GitHub Release split CSV was the winning route. The two release assets reached
  their exact expected sizes:
  - `androzoo_csv_part_aa`: `1,992,294,400` bytes.
  - `androzoo_csv_part_ab`: `1,815,703,169` bytes.
- The parts were concatenated and gzip-validated as
  `data/androzoo/latest_with-added-date.release.csv.gz` at
  `2026-05-26T13:47:43+08:00`.
- The standard pipeline input path now points to the complete release CSV:
  `data/androzoo/latest_with-added-date.csv.gz`, size `3,807,997,569` bytes.
- The slower official direct/proxy partial downloads were stopped after the
  release CSV was validated. Partial files are kept only as noncanonical
  leftovers:
  - `latest_with-added-date.direct_or_proxy.partial.csv.gz`
  - `latest_with-added-date.proxy.csv.gz`

Operational note:

- The release assemble watcher validated the gzip successfully, but then failed
  during its cleanup/promote block because the loop variable used PowerShell's
  read-only `$pid` automatic variable. The validated release CSV was promoted
  manually, and the slow competing downloads were stopped manually.
- This does not affect the data integrity of the standard CSV; the file was
  already gzip-validated before promotion.

Downstream pipeline status:

- Pipeline log:
  `logs/androzoo_modern_benign_pipeline_20260526-114653.out.log`.
- The pipeline accepted the standard CSV at `2026-05-26T13:50:17+08:00`.
- Candidate scan completed over `27,133,811` rows, with `13,415` matching the
  modern benign filters.
- Selected `50` AndroZoo hard-benign candidates and wrote:
  - `data/androzoo/hard_benign_modern_candidates.sha256.txt`
  - `data/androzoo/hard_benign_modern_candidates.jsonl`
- APK download is actively progressing. At the status check, `27/50` APK files
  were present under `data/androzoo/hard_benign_modern/`, totaling about
  `355.0` MiB, with the latest write at `2026-05-26T14:11:10+08:00`.

Next synchronized work while the pipeline runs:

1. Keep monitoring APK count until it reaches 50 or the downloader reports
   failures.
2. When downloads finish, confirm
   `outputs/experiments/hard_benign/manifest_androzoo_modern_apkid.json` exists
   and record APKiD-clean counts.
3. Let the configured strict DPT run continue, then compare the AndroZoo-expanded
   result against the F-Droid-expanded run above.

## 8. AndroZoo50 Final Result

The AndroZoo modern hard-benign pipeline completed at
`2026-05-26T15:45:55+08:00`.

Data and manifest:

- Downloaded `50/50` APKs, with `0` download failures.
- Built APKiD-audited manifest:
  `outputs/experiments/hard_benign/manifest_androzoo_modern_apkid.json`.
- Manifest summary:
  - `records`: 50.
  - `train_allowed`: 50.
  - `hard_train_allowed`: 50.
  - `label_class_counts`: `benign-hard-clean=50`.
  - `failures`: 0.

Strict DPT-v2 result:

- Result:
  `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_androzoo50_lowbyte025.json`.
- Training set: 153 bags.
- Strict test set: 19 DPT-v2 packed + 19 strict benign APKs.
- AUROC: 0.8864.
- AUPRC: 0.8869.
- Detection at threshold 0.5: 11/19 raw, 17/19 after train-benign z
  normalization.
- Packed mean score: 0.5481 raw, 49.4750 normalized.
- Benign mean score: 0.0824 raw, 7.3171 normalized.
- FPR@95TPR: 0.4211.
- TPR@1%FPR / TPR@5%FPR: 0.3158 / 0.3158.
- Train time after cache construction: 875.7 s.

Comparison against F-Droid24:

| Run | Hard benign train APKs | AUROC | AUPRC | Detection | Packed mean | Benign mean | FPR@95TPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| F-Droid expanded | 24 | 0.9280 | 0.8973 | 17/19 raw; 18/19 normalized | 0.8831 | 0.1661 | 0.3158 |
| AndroZoo modern | 50 | 0.8864 | 0.8869 | 11/19 raw; 17/19 normalized | 0.5481 | 0.0824 | 0.4211 |

Interpretation:

- AndroZoo50 further suppresses strict benign scores, so hard-benign normality is
  still a useful signal.
- The same run also suppresses packed DPT scores, reducing raw detection and
  worsening AUROC/FPR@95TPR. More hard benign is therefore not monotonic.
- The likely failure mode is normality over-calibration: the model is learning a
  stronger "normal" boundary, but it is not preserving enough margin for hidden
  payload residues.

Top-entry diagnostics:

- AndroZoo50 `apk_contribution`: `elf` 36/57 top-k hits, mean 0.0539,
  max 0.1396; `asset_entry` 10/57; `dex` 5/57.
- AndroZoo50 `attention`: `dex` 26/57 and `elf` 26/57.
- AndroZoo50 `suspicion`: `elf` 37/57, `dex` 13/57, `asset_entry` 4/57.
- AndroZoo50 `anomaly`: `elf` 30/57 and `dex` 22/57.

Relative to F-Droid24, suspicion is no longer asset-dominated, but DEX/ELF
normality remains the main pressure point.

## 9. Stage B Route Handoff

Created expert-review brief:
`docs/method/stage_b_expert_review_brief_2026-05-26.md`.

Created execution route:
`docs/method/stage_b_technical_route_2026-05-26.md`.

The recommended Stage B route is now frozen as B0-B9:

1. B0: reproduce V2 and run fixed-gate hard-benign ablation.
2. B1: DPT control table plus path reliability diagnosis.
3. B2: calibration-first metrics and bootstrap confidence intervals.
4. B3: fix Typed APK Object Pseudo-code.
5. B4: run minimal typed spMLM + corrupted-region + benign normality
   pretraining.
6. B5: compare fixed routing, learned gate, path confidence, and byte
   regularization.
7. B6: add benign entry suppression, byte regularization, calibration head, and
   PU auxiliary entry loss sequentially.
8. B7: use hard negative replay with bounded 30%-50% hard benign ratio.
9. B8: run counterfactual masking as a diagnostic loop.
10. B9: add paired ranking last, only after path reliability and normality are
    stable.

Core hypothesis for expert review:

> Stage B should teach the model what complex but normal APK objects look like,
> and only then ask which typed entry or region behaves like hidden executable
> payload residue.

Operational constraint: after this handoff, do not start V3/V4 or paired
ranking directly. B0/B1/B2 are the next required steps.
