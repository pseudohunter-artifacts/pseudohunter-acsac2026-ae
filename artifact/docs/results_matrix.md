# Results Matrix：方法 × 任务 × 训练模式 × 数据集

> 本文件是**所有实验数字的汇总骨架**，每一个已跑出的 F1 / AUROC / IoU / MRR 都必须回填到这里。
> 骨架角色：论文 §5 Evaluation 的表格直接从本文件抽取；reviewer 的"为什么这里是空的"都必须有**单独的格子**来回答（"Planned / N-A / Skipped 并注明理由"）。
>
> **禁止**在本文件之外维护一份并行矩阵；所有分散在 `baseline_numbers.md` / `dataset_plan.md` / `paper_submission_plan.md` 的数字以本文件为唯一索引。

## 0. 关系定位

| 文档 | 管什么 | 不管什么 |
|---|---|---|
| [`method/baseline_numbers.md`](method/baseline_numbers.md) | 每个 baseline 的**详细**跑数流水 + 命令 + 原始 JSON 路径 | 汇总对比 |
| **`results_matrix.md`（本文件）** | **方法 × 任务 × 训练模式 × 数据集的汇总矩阵**；论文数字的单点真源 | 单个 baseline 的实现细节 |
| [`method/dataset_plan.md`](method/dataset_plan.md) | 数据集来源 / 规模 / 切分 | 每个 cell 的具体数字 |
| [`paper_submission_plan.md`](paper_submission_plan.md) | 投稿节奏 / 分工 / 6 周冲刺任务 | — |

**硬约束**：任何新 cell 填值必须**同 PR** 同步更新：(a) 对应 JSON 路径 + commit sha；(b) 本文件对应格子；(c) 如果该格子进入论文 §5，[`method/baseline_numbers.md`](method/baseline_numbers.md) 对应行。缺失视为文档漂移，下一条 commit 必须回补。

---

## 1. 坐标轴定义

### 1.1 方法轴（methods）

| 方法 ID | 类别 | 实现位置 | 训练成本 | 角色 |
|---|---|---|---|---|
| `entropy` | 规则（硬阈值 + 加权打印比） | `src/android_packer/baselines/entropy.py` | 无训练 | baseline-naive |
| `sanity_rules` | 规则（多启发式投票） | `src/android_packer/baselines/sanity_rules.py` | 无训练 | baseline-heuristic |
| `ngram_logreg` | ML（bigram 哈希 + LR） | `src/android_packer/baselines/ngram_logreg.py` | ~45 min/fold | baseline-classical |
| `apkid` | 外部工具（YARA 规则） | `src/android_packer/baselines/apkid.py` | 无训练 | baseline-industry |
| `byte_cnn` | DL（1D-CNN 4-6 层） | `src/android_packer/baselines/byte_cnn.py` | fast sampled all-family LOFO：约 11 folds GPU run；需 `[dl]` extras | baseline-DL（字节视图上限对照）；2026-05-09 已完成 33-task sampled all-family LOFO + fold-local calibrated run |
| **`payload_hunter_lite`** | **DL（entropy-delta + 34 维 handcrafted + MLP + attention aggregator）** | `src/android_packer/baselines/payload_hunter_lite.py` | **~2 h CPU / ~5 min 4090** | historical ablation |
| `ti_mil` | DL（typed-instance MIL + attention-anomaly scoring） | `src/android_packer/models/ours.py` / `baselines/ours.py` | GPU | historical Ours / diagnostic |
| **`pseudohunter`** | DL（Pseudo-code BERT + typed path routing + hard-benign normality MIL） | `src/android_packer/models/pseudo_code_bert.py` / `models/fusion_encoder.py` / `scripts/experiments/run_lopo_eval.py` | ~10 min/fold after cache | **Current Ours (Stage A)** |
| `pseudohunter_top_tier` | DL（learned path confidence + corrupted-region pretraining + runtime evidence） | planned | TBD | **Stage B candidate** |

### 1.2 任务粒度轴（task granularity）

对每个 APK 一次预测后，聚合到三个层级各算一组指标：

| Task 层级 | 输出形式 | 聚合规则 | 主报告指标 |
|---|---|---|---|
| **Region** | 每条 4 KB region 一个 score | — | Precision / Recall / F1 / AUROC / AUPRC |
| **Object** | 每 ZIP entry 一个 score | region → max 聚合 | 上述 + MRR / Top-k（k=1/3/5） |
| **APK** | 每 APK 一个 score | object → max 聚合 | 上述 + IoU / Boundary Error |

### 1.3 训练模式轴（train_mode）

| train_mode | 训练集 | 测试集 | 意义 |
|---|---|---|---|
| `same_set` | 全部 task | 全部 task | in-sample 上限，对比**会虚高** |
| `holdout_transform` | N-1 个 transform family | 留一 family | 跨**加壳手法**泛化（Track A v2 主） |
| `holdout_packer` | N-1 个 packer | 留一 packer | 跨**真实 packer**泛化（Track B 主） |
| `holdout_package` | N-1 个 package 的 APK | 留一 package 的 APK | 跨**应用**泛化（数据漂移消融） |

### 1.4 数据集轴（dataset）

| Dataset | 规模 | 标签 | 切分主模式 |
|---|---|---|---|
| **Track A v2 (synthetic)** | 8 seed × 11 family ≈ 75 有效 task | byte-level 强标签 | `holdout_transform` |
| **Track B (open-source packers)** | 10 APK × 4–6 packer ≈ 40–60 task | byte-level 强标签（diff-based） | `holdout_packer` |
| **Track C / AndroZoo hard benign** | 50 modern benign queued for current replication; wild packed remains Stage B | APKiD-audited benign / object-level weak labels later | strict app-disjoint DPT / all-unseen |
| Commercial packers (case study) | 3–5 个 qualitative | — | 论文 §5.4 定性 |

---

## 2. 主矩阵：Track A v2 × methods × train_mode × task

> **状态标**：✅ 已跑且入表 / 🔄 跑中 / 📋 待跑 / ⏸ 推迟 / ❌ N-A / ⚠️ 需复跑（数据过时）
>
> **坐标**：行 = `method / train_mode / task`；列 = `F1 / Precision / Recall / AUROC / AUPRC / IoU / MRR / Top-3 / 来源 JSON`。
> 空缺用 `—` 填，**不得留空**；留空 = 未审视 = 风险。

### 2.1 Region-level 指标

| method | train_mode | F1 | Precision | Recall | AUROC | AUPRC | 状态 | 来源 |
|---|---|---|---|---|---|---|---|---|
| `entropy` (thr=6.5) | same_set | 0.000 | 0.000 | 0.000 | 0.491 | — | ✅ | `outputs/experiments/baseline_sweeps/20260430-153746/entropy_threshold_sweep.json` |
| `entropy` (thr=7.0) | same_set | 0.000 | 0.000 | 0.000 | 0.491 | — | ✅ | 同上 |
| `entropy` (thr=7.5) | same_set | 0.000 | 0.000 | 0.000 | 0.491 | — | ✅ | 同上 |
| `entropy_raw_inverted` (single-feat) | single-feat | — | — | — | 0.641 | — | ✅ | `outputs/experiments/baseline_sweeps/20260430-074032/entropy_delta_precheck.json` |
| `entropy_delta_entry` (single-feat) | single-feat per-family median | — | — | — | 0.467 ⚠️ RETRACT | — | ✅ | 同上 |
| `sanity_rules` | same_set (training-free) | **0.390** | **0.252** | **0.857** | **0.848** | **0.648** | ✅ | `outputs/experiments/baseline_sweeps/20260502-140602/sanity_rules_sweep.json`。训练无关的启发式（large_object + suspicious_path + low_printable 三规则加权），84 task / 34s。**⚠️ AUROC=0.848 与 Ours=0.865 仅差 0.017**，且 sanity_rules 恰好在 Ours 崩坏的 `embedded_archive`/`path_randomized` 上拿满（AUROC=0.998/1.000），这是 ngram_logreg 威胁的前奏：规则基线在 Ours 短板 family 上占优，Pass-2b 必须把这两个 family 救回来。 |
| `ngram_logreg` | same_set | ⚠️ 0.538（Gen2 旧）| — | — | — | — | ⚠️ | 需在 84-task 新数据上复跑 |
| `ngram_logreg` | holdout_transform | ⚠️ 0.411（Gen2 旧，macro）| — | — | ⚠️ 0.890 | — | ⚠️ | `outputs/experiments/baseline_sweeps/*/ngram_holdout.json`（Gen2，**需在 v2 84-task 上复跑**）|
| `apkid` | same_set (training-free) | — | — | — | — | — | ✅ | `outputs/experiments/baseline_sweeps/20260502-140602/apkid_sweep.json`。**⚠️ APKiD 3.1.0 对 84 个合成 APK detection rate = 0/84 = 0.0**（1110s 全量扫描完成）。原因：APKiD 的 YARA 规则库基于已知商业 packer（梆梆/乐固/360 加固等）签名，我们的合成 transform family 不带这些签名。**论文级 negative finding**：证明"识别已知 packer 签名"与"定位未知 payload 位置"是两个问题，APKiD 在本任务定义下完全无效。Track B 真实商业 packer 才有 detection 能力。 |
| `entropy` (v4 LOFO runner) | holdout_transform (training-free, 84/99 generated) | 0.000 | 0.000 | 0.000 | 0.475 | 0.095 | ✅ | `outputs/experiments/synthetic_multi_baseline_v4_lofo/summary.json`（base `f0176fd` + uncommitted runner safety fixes）。99 candidate tasks 中 15 个生成失败；84 个 successful task 三套 baseline report 全齐。 |
| `sanity_rules` (v4 LOFO runner) | holdout_transform (training-free, 84/99 generated) | 0.088 | 0.060 | 0.163 | 0.743 | 0.253 | ✅ | `outputs/experiments/synthetic_multi_baseline_v4_lofo/summary.json`。与旧 same_set 数字相比，v4 full-family 任务更难且 `so_embedded` 只有 4 个成功 task。 |
| **`payload_hunter_lite` / `ours`** (v4 LOFO recovery, epochs=3, bag) | holdout_transform | 0.073 | 0.038 | 0.998 | 0.139 | 0.078 | ⚠️ **negative canary** | `outputs/experiments/synthetic_multi_baseline_v4_lofo/summary.json`。3-epoch safe recovery run；几乎全阳性预测导致 recall≈1 但 precision≈0.038，**不是论文主 cell**。 |
| `byte_cnn` | same_set | — | — | — | — | — | 📋 | Not needed before holdout evidence; keep only if an in-sample upper-bound sanity check becomes useful |
| `byte_cnn` (fast sampled all-family, 3 compatible seeds) | holdout_transform | 0.457 | 0.363 | 0.616 | 0.890 | 0.487 | ✅ | `outputs/experiments/synthetic_multi_baseline_v4_lofo_byte_cnn_fast_compat3_all11_sampled/summary.json`。33/33 successes；fold-local train rows sampled to 500k, held-out evaluation unsampled；valid fast all-family result but not final large-corpus main table. Offline calibration diagnostic: region best-F1 threshold=0.857277 gives F1=0.545670 (`byte_cnn_calibration_analysis.json`). |
| `byte_cnn` (fold-local calibrated, target=object, 3 compatible seeds) | holdout_transform | 0.112 | 0.988 | 0.059 | 0.862 | 0.437 | ✅ | `outputs/experiments/synthetic_multi_baseline_v4_lofo_byte_cnn_fast_compat3_all11_sampled_calibrated/summary.json`（base `4fc5e40` + uncommitted calibration/fusion runner changes）。33/33 successes；thresholds selected inside each training complement, no held-out labels used. Calibration raises object detector F1 but makes region detector very conservative: region precision≈0.988, recall≈0.059。 |
| `mil_byte_cnn_fusion` (equal-weight diagnostic, 3 compatible seeds) | holdout_transform | 0.071 | 0.037 | 0.720 | 0.654 | 0.327 | ⚠️ **negative ablation** | `outputs/experiments/synthetic_multi_baseline_v4_lofo_mil_byte_cnn_fusion_fast_compat3_all11_isolated_20260509/summary.json`。33/33 successes, 17 contributing positive held-out tasks；equal-weight fusion recovers recall but floods predictions with false positives, so it is a diagnostic canary rather than a main paper cell。 |
| **`payload_hunter_lite`** | same_set | — | — | — | — | — | 📋 | F-Lite-c 训练 |
| **`payload_hunter_lite`** (smoke, 3-family, epochs=5, GPU) | holdout_transform | 0.735 | 0.881 | 0.667 | 0.980 | 0.808 | ⚠️ 乐观偏差 | smoke 只含 3 个良性 family，对比用 |
| **`payload_hunter_lite`** (full 11-family, epochs=10, GPU) | holdout_transform | **0.730** | **0.756** | **0.722** | **0.865** | **0.750** | ✅ **Stage A 主 cell** | `outputs/experiments/baseline_sweeps/20260501-213428/payload_hunter_lite_holdout_by_transform.json`。11-fold leave-one-transform-family-out，每 fold ~68-73min 训练（1.84M region 行）+ 2min scoring。8/11 fold AUROC≥0.984（泛化良好），**2 个 family 崩坏**：`embedded_archive` AUROC=0.168（反向预测，ZIP-in-ZIP 嵌套特征盲）、`dex_method_inlined` AUROC=0.654（代码级变换，字节特征看不到）。这正是 **Pass-2b 特征扩展 15→34 维**（DEX magic、ZIP marker、嵌套容器）的动机。 |
| **`payload_hunter_lite`** (Pass-2b, 4-fold subset, epochs=10, GPU) | holdout_transform | **0.340** | **0.500** | **0.305** | **0.594** | **0.377** | ⚠️ **negative result** | `outputs/experiments/baseline_sweeps/20260502-221749/payload_hunter_lite_holdout_by_transform.json`（2026-05-02 overnight L1 v3, macro over base64/dex_method_inlined/embedded_archive/signature_strip）。**Pass-2b 22 维（Pass-2a 15 + Group C magic 4 + Group F ZIP 3）未修复崩坏 fold，且让 signature_strip 从 AUROC=1.000 退化到 0.809**。逐 fold: `base64` 1.000 (持平)、`signature_strip` 0.809 (-0.191)、`dex_method_inlined` 0.467 (-0.187)、`embedded_archive` 0.099 (-0.069)。Pass-2a 在同 4 fold 的 macro=0.704，Pass-2b=0.594，**净回归 -0.110**。推测原因：Group C/F 的结构特征在 `dex_method_inlined`/`embedded_archive` 上无信号（因这两类变换不产生区域级字节差异），但 22 维 → 更高 VC 维度 → 对良性 fold 的过拟合风险上升。详细分析见 `docs/progress/sessions/2026-05-02_overnight_results_report.md` §1 + `docs/method/why_features_defensible_vs_ngram.md` 第 2 节 *What collapse folds expose*。 |
| **`payload_hunter_lite`** | holdout_package | — | — | — | — | — | 📋 | F-Lite-c |
| **`ours` (TI-MIL, attention_auto, 84-task, epochs=10)** | holdout_transform | 0.218 | 0.141 | 0.485 | **0.869** | 0.532 | ✅ **Paper main cell** | `outputs/experiments/ours_auto_full_84task_e10/summary.json`（2026-05-10）。84/99 task 成功，11-fold LOFO，`scoring_mode=attention_auto`，`supervision_mode=bag`。**超过 entropy/sanity_rules/payload_hunter_lite，接近 byte_cnn（0.890）**。关键优势：在 `embedded_archive` AUROC=1.000（byte_cnn=0.617）、`dex_method_inlined` AUROC=0.859（byte_cnn=0.609）上大幅领先。最弱 family：`multi_dex_shim` AUROC=0.624。注意：F1=0.218 是因为 threshold=0.78 在大规模数据下偏高；AUROC 不依赖阈值，是论文主指标。 |
| **`ours` (TI-MIL, Track C wild, same_set, epochs=10)** | same_set | — | — | — | **1.000** | — | ⚠️ **in-sample** | `outputs/experiments/track_c_mil_detection/report.json`（2026-05-10）。504 野样本（16 packed / 488 benign），same_set in-sample。APK AUROC=1.000，Object AUROC=1.000，MRR=1.000。证明 TI-MIL 能消费真实世界 APK 并完美区分 packed/benign；但 in-sample 不作泛化证据。 |

### 2.2 Object-level 指标

| method | train_mode | F1 | AUROC | MRR | Top-1 | Top-3 | 状态 | 来源 |
|---|---|---|---|---|---|---|---|---|
| `entropy` | same_set | — | — | — | — | — | 📋 | — |
| `sanity_rules` | same_set | — | — | — | — | — | 📋 | — |
| `ngram_logreg` | same_set | — | — | — | — | — | 📋 | — |
| `ngram_logreg` | holdout_transform | — | — | — | — | — | 📋 | — |
| `byte_cnn` (fast sampled all-family, 3 compatible seeds) | holdout_transform | 0.098 | 0.998 | 0.668 | 0.576 | 0.727 | ✅ | `outputs/experiments/synthetic_multi_baseline_v4_lofo_byte_cnn_fast_compat3_all11_sampled/summary.json`。Ranking strong but detector threshold poorly calibrated: object precision=0.052, recall=0.897。Offline calibration diagnostic: object best-F1 threshold=0.998894 gives F1=0.607595, precision=0.600, recall=0.615; object top-1 operating point gives F1=0.527778 (`byte_cnn_calibration_analysis.json`). |
| `byte_cnn` (fold-local calibrated, target=object, 3 compatible seeds) | holdout_transform | 0.548 | 0.999 | 0.641 | 0.515 | 0.697 | ✅ | `outputs/experiments/synthetic_multi_baseline_v4_lofo_byte_cnn_fast_compat3_all11_sampled_calibrated/summary.json`。Object precision=0.588、recall=0.513，比默认阈值 run 的 object F1=0.098 明显改善；Top-1=0.515、Top-3=0.697，ranking remains useful。 |
| `mil_byte_cnn_fusion` (equal-weight diagnostic, 3 compatible seeds) | holdout_transform | 0.002 | 0.939 | 0.184 | 0.000 | 0.176 | ⚠️ **negative ablation** | `outputs/experiments/synthetic_multi_baseline_v4_lofo_mil_byte_cnn_fusion_fast_compat3_all11_isolated_20260509/summary.json`。Object recall=0.952 but precision≈0.001, MRR=0.184, Top-1=0；fusion destroys top-k usefulness despite high object AUROC。 |
| **`payload_hunter_lite`** | holdout_transform | — | — | — | — | — | 📋 | F-Lite-c 主汇报 cell |
| `entropy` (v4 LOFO runner) | holdout_transform (training-free, 84/99 generated) | 0.000 | 0.902 | 0.153 | 0.083 | 0.179 | ✅ | `outputs/experiments/synthetic_multi_baseline_v4_lofo/summary.json` |
| `sanity_rules` (v4 LOFO runner) | holdout_transform (training-free, 84/99 generated) | 0.009 | 0.824 | 0.113 | 0.048 | 0.119 | ✅ | 同上 |
| **`payload_hunter_lite` / `ours`** (v4 LOFO recovery, epochs=3, bag) | holdout_transform | 0.001 | 0.074 | 0.001 | 0.000 | 0.000 | ⚠️ **negative canary** | 同上；object ranking 几乎失效，说明 3-epoch recovery 不可作为主结果。 |

### 2.3 APK-level 指标

| method | train_mode | F1 | AUROC | mean IoU | mean Boundary Err | 状态 | 来源 |
|---|---|---|---|---|---|---|---|
| `entropy` | same_set | — | — | — | — | 📋 | — |
| `sanity_rules` | same_set | — | — | — | — | 📋 | — |
| `ngram_logreg` | holdout_transform | — | — | — | — | 📋 | — |
| `byte_cnn` (fast sampled all-family, 3 compatible seeds) | holdout_transform | 1.000 | N/A | 0.638 | 1,449,211 | ⚠️ **positive-only** | `outputs/experiments/synthetic_multi_baseline_v4_lofo_byte_cnn_fast_compat3_all11_sampled/summary.json`。APK F1 is positive-only and not paper-meaningful; benign mixing is required for APK AUROC/AUPRC。 |
| `byte_cnn` (fold-local calibrated, target=object, 3 compatible seeds) | holdout_transform | 0.778 | N/A | 0.278 | 2,418,484 | ⚠️ **positive-only** | `outputs/experiments/synthetic_multi_baseline_v4_lofo_byte_cnn_fast_compat3_all11_sampled_calibrated/summary.json`。Calibration reduces positive APK calls from 33 to 21; APK F1 remains positive-only and must not be used as paper detection evidence without benign controls。 |
| `mil_byte_cnn_fusion` (equal-weight diagnostic, 3 compatible seeds) | holdout_transform | 1.000 | N/A | 0.761 | 1,502,910 | ⚠️ **positive-only / negative ablation** | `outputs/experiments/synthetic_multi_baseline_v4_lofo_mil_byte_cnn_fusion_fast_compat3_all11_isolated_20260509/summary.json`。APK F1 positive-only；object offset hit rate=0.941, mean IoU=0.761, but region/object precision collapse prevents main-table use。 |
| **`payload_hunter_lite`** | holdout_transform | — | — | — | — | 📋 | F-Lite-c |
| `entropy` (v4 LOFO runner) | holdout_transform (training-free, 84/99 generated) | 0.000 | N/A | 0.000 | 4,021,703 | ✅ | `outputs/experiments/synthetic_multi_baseline_v4_lofo/summary.json` |
| `entropy` (F0d fast, benign control) | holdout_transform (1 benign + 1 transform) | 0.000 | **0.500** | **0.500** | 4,194,304 | ✅ | `outputs/experiments/synthetic_multi_baseline_v4_lofo_f0d_fast/summary.json` |
| `sanity_rules` (v4 LOFO runner) | holdout_transform (training-free, 84/99 generated) | 1.000 | N/A | 0.134 | 3,689,744 | ✅ | 同上 |
| **`payload_hunter_lite` / `ours`** (v4 LOFO recovery, epochs=3, bag) | holdout_transform | 1.000 | N/A | 0.739 | 1,889,827 | ⚠️ **mixed negative canary** | 同上；APK-level F1 饱和是因为 synthetic tasks 全为 positive，真正问题在 region/object precision 与 ranking。 |

### 2.4 Per-family 细分（Track A v2，仅 region-level，AUROC）

> 细分格用于诊断"某 family 塌陷 vs 某 family 正常"，反映论文 §5.3 "where does the model fail"。
>
> 列 = 11 family，行 = method。单元格填 AUROC，`—` 为待跑。

| method \\ family | xor | base64 | split_xor | path_rand | sig_strip | embed_asset | so_embed | dex_inline | multi_dex_shim | embed_arch | dex_str_enc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `entropy_raw_inverted` (single-feat) | 0.66 | 0.73 | 0.67 | 0.65 | 0.65 | 0.66 | 0.76 | 0.27 | — | — | — |
| `entropy_delta_entry` (single-feat) | 0.47 | 0.47 | 0.46 | 0.46 | 0.47 | 0.72 | 0.28 | 0.75 | — | — | — |
| `sanity_rules` (v4 LOFO runner) | 0.869 | 0.838 | 0.839 | 0.869 | 0.838 | 0.919 | 0.443 | 0.390 | 0.823 | 0.818 | 0.380 |
| `ngram_logreg` holdout | — | — | — | — | — | — | — | — | — | — | — |
| `byte_cnn` holdout (fast sampled all-family, 3 compatible seeds) | 0.989 | 0.964 | 0.975 | 0.883 | 0.883 | 0.981 | 0.958 | 🔴 **0.609** | 0.968 | 🔴 **0.617** | 0.960 |
| `byte_cnn` holdout (fold-local calibrated, target=object, 3 compatible seeds) | 0.977 | 0.810 | 0.917 | 0.873 | 0.882 | 0.963 | 0.926 | 🔴 **0.620** | 0.969 | 🔴 **0.633** | 0.917 |
| `mil_byte_cnn_fusion` holdout (equal-weight diagnostic, 3 compatible seeds) | 0.911 | 0.554 | 0.885 | 0.602 | 0.493 | 0.779 | 0.523 | 🔴 **0.278** | 0.967 | 🔴 **0.000** | 0.783 |
| **`payload_hunter_lite`** holdout (full, epochs=10, GPU) | **1.000** | **0.992** | **0.995** | **1.000** | **0.999** | **0.995** | **0.984** | 🔴 **0.654** | **1.000** | 🔴 **0.168** | **0.723** |
| **`payload_hunter_lite` / `ours`** (v4 LOFO recovery, epochs=3, bag) | 0.135 | 0.138 | 0.054 | 0.153 | 0.149 | 0.046 | 0.056 | 0.141 | 0.444 | 🔴 **0.000** | 0.172 |

**注**：`multi_dex_shim` / `embedded_archive` / `dex_string_encrypted` 是 A-v2-b 新增 3 family，当前 84 task 中尚未完整出现（待 A-v2-c 全量 generate 触发后补齐）。

---

## 3. 主矩阵：Track B × methods × train_mode × task

> **状态（2026-04-30 更新）**：Track B **基础设施全部就绪**，**待实际加壳 APK 执行**才能填入数字。所有 metric cell 仍为 📋，但 pipeline column 已全部 ✅。
>
> 当前就绪情况：
> - 12 packers 登记于 `configs/data/track_b_packers.yaml`（6 open_source + 1 registered_not_patched + 5 commercial）
> - 5 家商业 packer rule files 全部落盘（`configs/data/track_b_commercial_rules/cs{1..5}*.yaml`）
> - `src/android_packer/labeling/track_b_pipeline.py` + `scripts/run_track_b_labeling.py` 后处理 pipeline 可用
> - 434 tests 全绿（含 22 unit + 5 integration for CS1-CS5）
> - APKiD 第三方独立 cross-check（B-g-4）已就绪
>
> 规模预期：10 benign APK × （S3 + S5 + S6 + CS1-CS5 = 8 packers）≈ 80 task（S1 · CvvT / S4 · Huyehan-G3 Gen3 自研 deferred；S2 · ijiami-OSS / S7 · Bangcle-OSS 轻登记只跑 Path B）。详见 [`workstreams/track_b/`](workstreams/track_b/README.md)。

### 3.1 Region-level（Track B）

| method | train_mode | F1 | AUROC | 状态 |
|---|---|---|---|---|
| `entropy` | same_set | — | — | 📋 B-c 跑完后补 |
| `sanity_rules` | same_set | — | — | 📋 |
| `ngram_logreg` | holdout_packer | — | — | 📋 B-c 后补，**Track B 主汇报 cell 之一** |
| `byte_cnn` | holdout_packer | — | — | 📋 |
| **`payload_hunter_lite`** | holdout_packer | — | — | 📋 **Track B 主汇报 cell** |

### 3.2 Object-level（Track B）

与 §3.1 同结构，略。

### 3.3 Per-packer 细分（Track B，region AUROC）

开源 packer（Path A source-injected ground truth，除 S2 / S7）：

| method \\ packer | S1 · CvvT | S2 · ijiami-OSS | S3 · oncealong | S4 · Huyehan-G3 | S5 · timscriptov | S6 · dpt-shell | S7 · Bangcle-OSS |
|---|---|---|---|---|---|---|---|
| Gen level | Gen3 | Gen1-Gen2 | Gen1 | Gen3 | Gen1-Gen2 | Gen3 | Gen2 |
| Path A status | deferred | 📋 todo | ✅ done | deferred | ✅ done | ✅ done | registered |
| `entropy` | — | — | — | — | — | — | — |
| `ngram_logreg` holdout_packer | — | — | — | — | — | — | — |
| **`payload_hunter_lite`** holdout_packer | — | — | — | — | — | — | — |

商业 packer（Path A rule-based + Path B diff cross-validate）：

| method \\ packer | CS1 · 360-Jiagu | CS2 · ijiami-Com | CS3 · Bangcle | CS4 · Legu | CS5 · DexProtector |
|---|---|---|---|---|---|
| Gen level | Gen2 | Gen2-Gen3 | Gen2 | Gen2-Gen3 | Gen3 |
| Rule confidence | MEDIUM | MEDIUM | MEDIUM | MEDIUM | MEDIUM-LOW |
| Rule file | ✅ 5 rules (v1+v2+mix) | ✅ 4 rules | ✅ 7 rules (v2+v3) | ✅ 5 rules | ✅ 4 rules |
| Real packed samples (2026-05) | **1 / 9** (rate-limited) | 0 / 9 (account review) | **9 / 9** ✅ | 0 / 9 | 0 / 9 (account review) |
| `entropy` | — | — | — | — | — |
| `ngram_logreg` holdout_packer | — | — | — | — | — |
| **`payload_hunter_lite`** holdout_packer | — | — | — | — | — |

#### 3.3.1 CS\* Path A-rule × Path B-diff IoU（2026-05 实测）

> **关键论文素材**：这里的 IoU 显示"Path A-rule 随时间漂移、Path B-diff 是兜底"的真实信号。CS3 · Bangcle 的 9/9 SOLID 是**基于 rule v3 更新后的结果**；原 2018-2022 文献规则在同样 9 个样本上 IoU 恒为 0（path_b_only_no_rule_match）。详见 [`workstreams/track_b/real_packed_samples_landing.md`](workstreams/track_b/real_packed_samples_landing.md)。

| packer | samples | rule 版本 | mean IoU | SOLID | partial_mismatch | path_b_only_no_rule_match | mean Path B labels / APK |
|---|---|---|---|---|---|---|---|
| CS1 · 360-Jiagu | 1 | v1 (2018-2022) | 0.707 | 0 | 1 | 0 | 11 |
| CS1 · 360-Jiagu | 1 | **v1+v2 (2026-05 updated)** | **0.761** | 0 | 1 | 0 | 13 |
| CS3 · Bangcle | 9 | v2 (2018-2022) | N/A (no rule hits) | 0 | 0 | **9** | 59.6 |
| CS3 · Bangcle | 9 | **v2+v3 (2026-05 updated)** | **0.942** | **9** | 0 | 0 | 62.7 |


### 3.4 APKiD 第三方 baseline（B-g-4）

> **独立交叉验证**：APKiD 3.1.0（rednaga/APKiD，GPL 许可，subprocess 外调）识别 packer 家族的能力，作为 paper §5.4 的独立 baseline 对比。期望结果：APKiD 在商业 packer（CS1-CS5）上命中率高（rule 库有覆盖），在开源 packer（S1-S7）上**有意义地失手**（rule 库未覆盖 = "APKiD 盲区"，突出我们方法的新信号面）。

| packer 类别 | 样本数 | APKiD 正确识别 | APKiD 漏报 | APKiD 假阳性 | 结论 |
|---|---|---|---|---|---|
| 开源 S1-S7 | 7 | — | — | — | 📋 待 B-c 跑完后补 |
| 商业 CS1-CS5 | 5 | — | — | — | 📋 待 B-g-2 真 SaaS 加壳后补 |
| benign 对照 | 10 | 0 预期 | N/A | — | 📋 false positive baseline |

---

## 3.5 PseudoHunter LOPO / Strict DPT-v2（当前 Stage A 主线）

| protocol | configuration | APK AUROC | AUPRC | detection | localization | status | source |
|---|---|---:|---:|---|---|---|---|
| 7-fold LOPO | routed Dalvik/ARM64/byte + path dropout | **0.9582** | — | 99.0% | normality MRR=0.4367; attention MRR=0.4679 | ✅ current main | `outputs/experiments/path_ablation/lopo_results_routing_path_dropout_full.json` |
| strict DPT-v2 clean | routed baseline, no hard benign | 0.6000 | — | 20/20 at threshold 0.5 | normality MRR=0.4306 | historical failure | `outputs/experiments/track_b_v2_strict_dpt/results.json` |
| strict DPT-v2 clean | B0-A fixed low-byte gate, no hard benign, refreshed metrics | 0.6053 | 0.5588 | 19/19 | normality MRR=0.4206 | ✅ B0 refreshed diagnostic | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_lowbyte025.json` |
| strict DPT-v2 clean | + 9 APKiD-clean hard benign | 0.9335 | 0.9397 | 18/19 | — | ✅ diagnostic positive | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_lowbyte025.json` |
| strict DPT-v2 clean | + 24 F-Droid APKiD-clean hard benign | **0.9280** | **0.8973** | 17/19 raw; 18/19 normalized | attention MRR=0.4908 | ✅ current strict result | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json` |
| strict DPT-v2 clean | + 50 AndroZoo modern APKiD-clean hard benign | 0.8864 | 0.8869 | 11/19 raw; 17/19 normalized | attention MRR=0.4169 | ⚠️ broader hard-benign stress: benign mean lower, packed margin shrinks | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_androzoo50_lowbyte025.json` |

Interpretation: PseudoHunter's current main claim is no longer the synthetic
PayloadHunter-Lite/TI-MIL line. The LOPO result establishes packer-disjoint
transfer; strict DPT-v2 shows hard-benign normality is necessary for
app-disjoint ranking. The F-Droid expansion sharply lowers benign scores and
halves FPR@95TPR, but fixed-FPR TPR remains weak. The AndroZoo50 replication
further lowers strict benign mean score (0.0824) but also lowers packed mean
score (0.5481), so more hard benign is not monotonic. This motivates Stage B as
a typed normality / path-reliability repair rather than continued hard-benign
scaling.

Stage B execution route:
`docs/method/stage_b_technical_route_2026-05-26.md`. The next paper-relevant
cells should come from B0 fixed-gate hard-benign ablation, B1 DPT control /
path reliability, and B2 calibration + bootstrap CI before new V3/V4 method
claims are added.

### 3.5.1 Stage B B1 Path Reliability Diagnostics

| protocol | paths | optimizer/chunk | APK AUROC | AUPRC | FPR@95TPR | TPR@1%/5%FPR | status | source |
|---|---|---:|---:|---:|---:|---|---|---|
| strict DPT-v2 clean | Dalvik only | b4/c64 | 0.6953 | — | — | — | historical path cell | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_dalvik.json` |
| strict DPT-v2 clean | ARM64 only | b4/c64 | 0.6053 | — | — | — | historical path cell | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_arm64.json` |
| strict DPT-v2 clean | Dalvik+byte | b4/c64 | 0.5526 | — | — | — | historical path cell | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_dalvik_byte.json` |
| strict DPT-v2 clean | Dalvik+ARM64 | b4/c64 | 0.5789 | 0.5429 | 0.8421 | 0.0000 / 0.0000 | ✅ B1 diagnostic | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_dalvik_arm64_lowbyte025.json` |
| strict DPT-v2 clean | Dalvik+ARM64 | b32/c128 | 0.6704 | 0.6678 | 0.7895 | 0.1053 / 0.1053 | ⚠️ B1 throughput-setting diagnostic | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_dalvik_arm64_lowbyte025_b32_c128.json` |
| strict DPT-v2 clean | ARM64+byte | b32/c128 | 0.6150 | 0.5907 | 0.7368 | 0.0526 / 0.0526 | ⚠️ B1 throughput-setting diagnostic | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_arm64_byte_lowbyte025_b32_c128.json` |

Interpretation: under the refreshed low-byte strict setting, simple path
combinations still do not repair DPT-v2 by themselves. Dalvik-only remains the
strongest old single-path diagnostic, but all no-hard-benign path cells have
weak fixed-FPR behavior. The b32/c128 rows use a different optimizer batch
size and should be treated as throughput-setting diagnostics, not direct
paper-table replacements for the older b4/c64 rows.

### 3.5.2 Stage B B1.1 DPT Control Diagnostics

These rows are causal diagnostics only. They must not be used to design
DPT-specific features, routing rules, thresholds, or loss terms.

| group | control mode | added data | n_train | APK AUROC | AUPRC | FPR@95TPR | TPR@1%/5%FPR | packed mean | benign mean | source |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| A | `non_dpt` | none | 103 | 0.6150 | 0.6396 | 0.7895 | 0.1579 / 0.1579 | 0.9452 | 0.7944 | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_b11_A_non_dpt_b32_c128.json` |
| B | `add_old_dpt` | +18 old DPT positives | 121 | 0.6565 | 0.6112 | 0.7895 | 0.0000 / 0.0000 | 0.9881 | 0.8268 | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_b11_B_add_old_dpt_b32_c128.json` |
| C | `add_old_dpt_benign` | +18 old DPT positives +9 old Track B benign counterparts | 130 | 0.9169 | 0.9225 | 0.5789 | 0.5263 / 0.5263 | 0.9543 | 0.4715 | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_b11_C_add_old_dpt_benign_b32_c128.json` |
| D | `other_positive_replay` | +18 equal-size non-DPT positives | 121 | 0.7285 | 0.7224 | 0.6316 | 0.1579 / 0.1579 | 0.9808 | 0.7854 | `outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_b11_D_other_positive_replay_b32_c128.json` |

Interpretation: old DPT positives alone are only a small improvement over the
non-DPT baseline (`0.6565` vs `0.6150` AUROC), while equal-size non-DPT positive
replay is stronger (`0.7285`). The large jump comes from adding benign
counterparts (`0.9169` AUROC, benign mean `0.4715`), which supports the general
Stage B hypothesis that boundary/normality learning is the important factor.
This is not evidence for DPT-specific design.

### 3.5.3 Stage B B2 Calibration / Bootstrap

Diagnostic only: B2 measures calibration and uncertainty over the strict DPT-v2
test set. These rows do not tune thresholds and do not feed test labels into
method design.

Full output:
`outputs/experiments/track_b_v2_strict_dpt/b2_calibration_bootstrap_summary.json`
and `.md`. Bootstrap count: 2000. Test set: 19 benign + 19 packed APKs.

| run | AUROC | AUROC 95% CI | AUPRC | ECE | Brier | FPR@95TPR |
|---|---:|---|---:|---:|---:|---:|
| B0-A no hard benign | 0.6053 | [0.5238, 0.7059] | 0.5588 | 0.4294 | 0.4237 | 0.7895 |
| 9 hard benign | 0.9335 | [0.8338, 1.0000] | 0.9397 | 0.2662 | 0.2397 | 0.6316 |
| F-Droid24 hard benign | 0.9280 | [0.8174, 0.9972] | 0.8973 | 0.0898 | 0.1027 | 0.3158 |
| AndroZoo50 hard benign | 0.8864 | [0.7565, 0.9722] | 0.8869 | 0.2367 | 0.2041 | 0.4211 |
| B1.1 A non-DPT | 0.6150 | [0.4202, 0.7861] | 0.6396 | 0.3944 | 0.3745 | 0.7895 |
| B1.1 B + old DPT positives | 0.6565 | [0.4653, 0.8296] | 0.6112 | 0.4074 | 0.3898 | 0.7895 |
| B1.1 C + old DPT positives/benign | 0.9169 | [0.8028, 0.9944] | 0.9225 | 0.2130 | 0.1993 | 0.5789 |
| B1.1 D other-positive replay | 0.7285 | [0.5555, 0.8778] | 0.7224 | 0.3931 | 0.3545 | 0.6316 |

Paired-bootstrap conclusions:

- `9 hard benign -> F-Droid24`: AUROC delta `-0.0058`, 95% CI
  `[-0.1250, 0.1008]`, p=`0.9730`; AUPRC delta `-0.0372`, 95% CI
  `[-0.1953, 0.0877]`, p=`0.7080`. The apparent AUROC/AUPRC difference is not
  reliable on this small strict set.
- F-Droid24 is better calibrated: Brier delta `-0.1367`, 95% CI
  `[-0.2403, -0.0456]`, p=`0.0000`; ECE delta `-0.1579`, 95% CI
  `[-0.2740, -0.0477]`, p=`0.0040`.
- `F-Droid24 -> AndroZoo50`: AUROC delta `-0.0425`, 95% CI
  `[-0.1345, 0.0387]`, p=`0.3110`; ECE worsens by `+0.1339`, 95% CI
  `[0.0097, 0.2663]`, p=`0.0340`. More hard benign is not monotonic.
- `B1.1 B -> C`: AUROC delta `+0.2583`, 95% CI `[0.1108, 0.4231]`,
  p=`0.0000`; AUPRC delta `+0.2790`, 95% CI `[0.1028, 0.4452]`, p=`0.0020`.
  The supported signal is general boundary/normality learning, not DPT-specific
  positive memorization.

### 3.5.4 Stage B B3 Typed APK-Object Pseudo-Code

B3 is complete as representation repair, not a new result row. It freezes a
generic typed input contract for B4 pretraining:

- Dalvik path: opcode class plus register/list, invoke kind,
  method/field/type/string index, const, branch, and abnormal-index components.
- Native path: ARM64/ARM32 mnemonic class plus branch target, abnormal target,
  JNI/syscall/symbol-like components.
- Byte path: new `typed_v1` byte-pattern representation with entry context,
  magic/header, entropy class, compression/encryption-like shape, run length,
  repeated n-gram, alignment, and unknown-pattern tokens.

Default reproduction stays on `legacy_raw`; B4 and later typed experiments must
explicitly use `--byte-representation typed_v1`. LOPO bag cache keys include
the byte representation, so legacy and typed token arrays cannot collide.

Implementation / validation:

- `src/android_packer/decoders/byte_pattern_decoder.py`
- `src/android_packer/decoders/pseudo_tokenizer.py`
- `src/android_packer/models/fusion_encoder.py`
- `scripts/experiments/run_lopo_eval.py --byte-representation typed_v1`
- `scripts/data/build_spmlm_corpus.py --byte-representation typed_v1`
- `tests/unit/test_b3_typed_pseudo_code.py` (`6 passed`)
- compatibility: `legacy_raw` vocab remains `358`; `typed_v1` vocab is `148`.

### 3.5.5 Stage B B4 Minimal Typed-v1 Pretraining Smoke

B4 is complete as a functional minimal pretraining smoke. Its first strict
DPT downstream comparison is a negative result and must not be used as a method
improvement claim.

Corpus:

- path: `data/pretrain_cache_b4_typed_smoke`
- input: 30 benign APKs from F-Droid hard benign + AndroZoo hard benign
- byte representation: `typed_v1`
- vocab size: `148`
- total sequences: `356,190`
- streams: Dalvik `118,730`; Native `118,730`; Byte `118,730`

Pretraining:

| Run | Representation | Objective | Model | Sequences | Epochs | Batch | Loss | Output |
|---|---|---|---|---:|---:|---:|---|---|
| B4 smoke | `typed_v1` | spMLM + corruption/normality | 8L/512d | 120,000 | 2 | 256 | 1.8889 -> 0.8355 | `outputs/experiments/pseudo_bert_b4_typed_smoke/pretrained_bert_v2.pt` |

Compatibility:

- checkpoint-load / CUDA forward sanity passed.
- typed-v1 downstream vocab size: `148`.
- loaded tensors into routed downstream model: `109`.
- output shapes: embeddings `(2, 256)`, suspicion `(2,)`, normality `(2,)`.

Strict DPT + F-Droid24 downstream:

| Run | Representation | Epochs | Batch/chunk | APK AUROC | AUPRC | FPR@95TPR | TPR@1%/5%FPR | Localization | Status | Source |
|---|---|---:|---|---:|---:|---:|---|---|---|---|
| typed-v1 no-pretrain control | `typed_v1` | 50 | 1/32 | 0.6053 | 0.5588 | 0.7895 | 0.0000 / 0.0000 | normality MRR=0.1553; attention MRR=0.3198 | ✅ control: typed-v1 random init is weak | `outputs/experiments/track_b_v2_strict_dpt/results_b4_typed_no_pretrain_strict_dpt_fdroid24_e50_memsafe_retry.json` |
| B4 smoke pretrain downstream | `typed_v1` | 50 | 1/32 | 0.7632 | 0.6786 | 0.4737 | 0.0000 / 0.0000 | normality MRR=0.1415; attention MRR=0.2100 | ⚠️ negative result | `outputs/experiments/track_b_v2_strict_dpt/results_b4_typed_strict_dpt_fdroid24_e50_memsafe.json` |

Interpretation: typed-v1 random initialization is very weak under the strict
DPT + F-Droid24 protocol (`AUROC=0.6053`, benign mean `0.7895`). The smoke
pretraining improves detection over that control (`0.7632` AUROC) but still
regresses relative to the current strict DPT-v2 + F-Droid24 legacy baseline
(`AUROC=0.9280`, `AUPRC=0.8973`, attention MRR `0.4908`). Do not advance to B5
on the basis of the smoke checkpoint.

Expanded pretraining preparation:

- corpus: `data/pretrain_cache_b4_typed_expanded_benign65`
- APKs: 133 confirmed benign / hard-benign APKs
- sequences: 772,962 total; 257,654 per stream
- running output:
  `outputs/experiments/pseudo_bert_b4_typed_expanded/b4_typed_expanded_benign65_e10`
- config: typed-v1, 8L/512d, 10 epochs, batch=256, all sequences,
  corruption probability 0.5, normality weight 0.2
- checkpoint policy: every 5 epochs plus final weights
- AndroZoo storage: 500 benign candidates selected; background download into
  `data/androzoo/benign_corpus` has started.

---

## 4. Cross-track 对比：同一 benign APK 在 synthetic vs real packer 下

> 这是 Track B 对审稿人的**核心说服力**：同一方法在 A v2 上的数字和在 B 上的数字能否对齐。
>
> 规则：seed APK 池中与 Track A v2 有 ≥ 80% 交集，可做同源对比。

| benign APK | Track A v2 AUROC (holdout_transform) | Track B AUROC (holdout_packer) | Gap |
|---|---|---|---|
| org.fdroid.fdroid | — | — | — |
| com.keepassdroid | — | — | — |
| ... | — | — | — |

**Gap 解读约定**：
- `|gap| < 0.05` → 方法 generalize 良好
- `gap > 0.10`（A 高于 B）→ 方法 overfit synthetic
- `gap < -0.10`（B 高于 A）→ 真实 packer 比 synthetic 更**容易**（可能 Track A v2 的 hard-adversarial 有效）

---

## 5. 负面发现与已知 gotcha（必须在论文 §5 Discussion 讨论）

| 现象 | 出处 | 解读 |
|---|---|---|
| **entropy 在 Track A v2 上 F1=0、AUROC=0.491** | `outputs/experiments/baseline_sweeps/20260430-153746` | B1+B2 + 子范围 transform 成功消除 entropy free lunch；与 PackerGrind/DexHunter 论文结论一致 |
| **entropy 方向反转**：`entropy_raw_inverted` AUROC=0.641（单特征最佳） | `outputs/experiments/baseline_sweeps/20260430-074032` | 加密 payload 的均匀分布反而被宿主 DEX 的高频代码段衬托出低 entropy；论文 §5 负面发现小节 |
| **`entropy_delta_entry` 双峰**：`dex_method_inlined` 0.75 / `embedded_asset` 0.72 / 其余 7 family 0.28–0.50 | 同上 | 信号只在"payload 占宿主子范围"的 transform 上出现；whole-object 替换的 transform 上无效。论文 §5.3 "where does the feature fail" 的关键证据 |
| **ngram holdout macro F1 = 0.411 / AUROC = 0.890**（Gen2 旧数据）| `outputs/experiments/baseline_sweeps/*/ngram_holdout.json` | 分类能力很弱（F1 低）但排序能力不错（AUROC 高）；说明 ngram 可做 ranker 但不能做 detector |
| **84 task 中 `unknown` family 出现** | 今日 precheck | 有 24 个 task 的 `transform_family` 字段缺失→被 `_infer_family` 归到 unknown；`scripts/rebuild_experiment_manifest.py` 需要升级归类逻辑 |
| **equal-weight `mil_byte_cnn_fusion` 失败**：region F1=0.071、precision=0.037、recall=0.720；object MRR=0.184、Top-1=0 | `outputs/experiments/synthetic_multi_baseline_v4_lofo_mil_byte_cnn_fusion_fast_compat3_all11_isolated_20260509/summary.json` | naive score fusion recovers many positives but destroys precision/top-k；下一步必须做 level-specific calibration / typed routing，而不是盲目调权重 |

---

## 6. 待办（按优先级）

1. ✅ **`byte_cnn` calibrated sampled all-family 实验已完成**：fold-local validation 阈值（`calibration_mode="fold_local_best_f1"`, `calibration_target="object"`）在 33/33 sampled all-family tasks 上完成，产物为 `outputs/experiments/synthetic_multi_baseline_v4_lofo_byte_cnn_fast_compat3_all11_sampled_calibrated/summary.json`。Object F1 从默认阈值 run 的 0.098 提升到 0.548；代价是 region recall 降到 0.059，说明 object-target calibration 更适合 object/APK 检测而非 region localization。
2. ✅ **`mil_byte_cnn_fusion` equal-weight diagnostic 已完成**：产物为 `outputs/experiments/synthetic_multi_baseline_v4_lofo_mil_byte_cnn_fusion_fast_compat3_all11_isolated_20260509/summary.json`；结论是 negative ablation（region precision≈0.037, object Top-1=0），论文中只作为 fusion/calibration failure 讨论，不作为 main cell。
3. 🔴 **`byte_cnn` failure audit**：先看 `dex_method_inlined` / `embedded_archive` 的 region collapse，再看 `path_randomized` / `signature_strip` 的 object Top-1/MRR collapse；需要对比预测最高分对象、真实 label 对象和 region score 分布。
4. ✅ **F0d benign 对照已完成**：runner 已支持 `--include-benign-apks N`，benign controls 为 `evaluation_only`、不参与训练但参与评分；F0d fast 测试（1 benign + 1 transform）已确认 APK AUROC=0.500、APK AUPRC=0.500，证明良性APK控制机制正常工作。
5. 🟡 **补 Track A v2 上的 `ngram_logreg` 复跑**（84-task，`same_set` + `holdout_transform`）——仍是 classical ML 对比空白；预期 2 小时。
6. 🟡 **修 `unknown` family 的归因**：`scripts/rebuild_experiment_manifest.py` 的 `_infer_family` 查阅 `apk_labels.jsonl` 里的 `transform_families` 字段而非只靠目录名后缀。
7. 🟡 **启动 Track B**（独立子任务，详见 [`workstreams/track_b/`](workstreams/track_b/README.md)）。
8. ⏸ Track C（Stage B 再启动）。

---

## 7. 变更日志

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-05-09 | **MIL+byte-CNN equal-weight fusion diagnostic 完成**：33/33 successes，17 contributing positive held-out tasks；region F1=0.071、precision=0.037、recall=0.720、AUROC=0.654；object F1=0.002、AUROC=0.939、MRR=0.184、Top-1=0、Top-3=0.176。结论：naive fusion 是 negative ablation，不能进主表，只能支持 calibration/typed-routing 讨论 | `outputs/experiments/synthetic_multi_baseline_v4_lofo_mil_byte_cnn_fusion_fast_compat3_all11_isolated_20260509/summary.json` |
| 2026-05-09 | **byte-CNN fold-local calibrated sampled all-family 实验完成**：33/33 successes，object-target calibration 得到 object F1=0.548、precision=0.588、recall=0.513；region F1=0.112、precision=0.988、recall=0.059，显示校准后更偏 object detector 而非 region localizer | `outputs/experiments/synthetic_multi_baseline_v4_lofo_byte_cnn_fast_compat3_all11_sampled_calibrated/summary.json` |
| 2026-05-09 | **byte-CNN fold-local calibration 路径接入**：新增报告层 `calibration` 字段和 runner 级 `calibration_mode="fold_local_best_f1"`；已有 33-task 输出的离线诊断显示 object F1 上界可由 0.097765 提升到 0.607595 | `outputs/experiments/synthetic_multi_baseline_v4_lofo_byte_cnn_fast_compat3_all11_sampled/byte_cnn_calibration_analysis.json` |
| 2026-05-08 | 记录 sampled byte-CNN all-family LOFO 的校准/failure-audit结论；F0d runner 已实现 `--include-benign-apks N`，待跑数后回填 APK AUROC/AUPRC | `docs/progress/sessions/2026-05-08_v4_lofo_sweep_numbers.md` §11 |
| 2026-05-08 | **F0d benign 对照测试完成**：快速测试（1 benign + 1 transform）确认 APK AUROC=0.500、APK AUPRC=0.500，证明良性APK控制机制正常工作 | `outputs/experiments/synthetic_multi_baseline_v4_lofo_f0d_fast/summary.json` |
| 2026-05-11 | **L47 integrity fix**: ground-truth leak in `_predict_impl()` — `true_label_id` was passed to type routing at inference time. All pre-L47 Ours Track B/C numbers RETRACTED. Post-fix honest numbers: Track A Region AUROC=0.714 (attn_x_bag), Object MRR=0.001; Track B LOPO Mean APK=0.670, Reg=0.687. Improvement plan at `docs/method/improvement_plan_L47.md` | `outputs/experiments/AUDIT_REPORT.md`, `ours_L47_FINAL_attn_x_bag_84task/summary.json` |
| 2026-05-11 | **4 ablation experiments completed**: flat_pool (Reg=0.500), no_type (MRR=0.011), instance_logit (MRR=0.101), low_epochs (MRR=0.308). Typed routing and attention scoring confirmed essential. | `outputs/experiments/ablation_*/summary.json` |
| 2026-05-11 | **Citation audit**: 3 critical fixes (detectbert2024 wrong authors+venue, wermke2018ccs wrong venue, droidra wrong authors) applied to `paper/acsac2026/references.bib` | `outputs/experiments/CITATION_AUDIT.md` |
| 2026-05-11 | **Novelty check**: Overall MOSTLY NOVEL. Attention-anomaly scoring is the strongest novel claim. Must cite CAVGA (ECCV 2020) and differentiate from Sultani+ (CVPR 2018). | `outputs/experiments/NOVELTY_REPORT.md` |
| 2026-05-11 | **Related work survey completed**: CAVGA, DeepReflect, Sultani+, VizMal, DetectBERT, Anomaly Transformer all found and summarized. | `outputs/experiments/RELATED_WORK_SURVEY.md` |
| 2026-04-30 | 建档；矩阵骨架完成；填入今日 `entropy_threshold_sweep` + `entropy_delta_precheck` 的实测值；Track B / Track C 保留骨架 | 与用户对话 2026-04-30："给一份这样的骨架" |
