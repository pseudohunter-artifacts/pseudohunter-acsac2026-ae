# AndroidPacker

**Weakly-Supervised Static Localization of Hidden Executable Payloads in Packed Android Apps** 研究原型。给定加壳 APK，输出 **APK / entry / region** 三层检测与定位结果，主要依赖 APK 级弱标签、packed/unpacked 差分标签和零成本结构标签（无人工 region 标注）。

> **当前核心方法（2026-05-26）**：**PseudoHunter / Pseudo-code BERT**。系统把 APK entry/region 解码成 Dalvik、ARM64、raw-byte 三路伪指令流，使用 benign spMLM 预训练的共享 Transformer、region-type routing、path dropout 和 normality-conditioned MIL 做 APK 检测与 entry-level localization。TI-MIL / PayloadHunter-Lite 已转为历史方法与消融 baseline（详见 [`docs/method/ds_amil_spec.md`](docs/method/ds_amil_spec.md)、[`docs/method/pseudo_hunter_fallback_snapshot_2026-05-25.md`](docs/method/pseudo_hunter_fallback_snapshot_2026-05-25.md)）。

> 本仓库当前处于 **DEX-only MVP** 阶段：只关注静态存放于 APK、运行时被恢复为 DEX 并被加载执行的 hidden executable payload。SO 协同与 runtime-grounded 标签增强属于后续阶段。

## 0. 文档导航（阅读路径）

> 本文件是**人类阅读主入口**。不同角色按下表数字顺序阅读即可。5 分钟完成 onboarding。

| 我是… | 必读顺序 | 预计用时 |
|---|---|---|
| 新加入的合作者（首次 onboarding）| 本文件 §1“§2 → [`docs/research_framing.md`](docs/research_framing.md) §1“§2 → [`docs/project_constraints.md`](docs/project_constraints.md) | 15 分钟 |
| 要写代码 / 交 PR 的 agent | [`AGENTS.md`](AGENTS.md) 全文 → [`docs/method/ours_method_spec.md`](docs/method/ours_method_spec.md) §1“§2 → [`docs/paper_submission_plan.md`](docs/paper_submission_plan.md) §2 周任务 | 30 分钟 |
| 要跟进度 / 冲刺任务的用户 | [`docs/paper_submission_plan.md`](docs/paper_submission_plan.md) §1“§2 → [`docs/progress/README.md`](docs/progress/README.md) → [`docs/progress/sessions/`](docs/progress/sessions/)（日期命名的会话日志，最细颗粒进度源）→ [`docs/project_progress.md`](docs/project_progress.md) 附录 A（第 1–10 周历史时间线）| 10 分钟 |
| 要拾取 Track B 子任务的 agent | [`docs/workstreams/track_b/README.md`](docs/workstreams/track_b/README.md) → [`tasks.md`](docs/workstreams/track_b/tasks.md) → [`conventions.md`](docs/workstreams/track_b/conventions.md) | 10 分钟 |
| 审稿人 / 外部读者想快速了解项目 | 本文件全文 + [`docs/method/problem_definition.md`](docs/method/problem_definition.md) | 10 分钟 |
| 想看最新结构性修复 / leakage 闭合结果 | [`docs/workstreams/2026-05-06-evening-handoff.md`](docs/workstreams/2026-05-06-evening-handoff.md)（Track B v2 + F-MIL 三件套落地）→ [`docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md`](docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md)（v4 合成语料 leakage 闭合 + L41-L45 MIL 结构性修复） | 15 分钟 |

**文档层次（权威源单一化，避免漂移）**：

- 研究叙事（RP / Phase / 卖点） → [`docs/research_framing.md`](docs/research_framing.md)
- 投稿执行（会议 / deadline / 周任务） → [`docs/paper_submission_plan.md`](docs/paper_submission_plan.md)
- **当前方法规格（PseudoHunter = Pseudo-code BERT）** → [`docs/method/ds_amil_spec.md`](docs/method/ds_amil_spec.md)（DS-AMIL 基础 + Pseudo-code BERT 扩展）
- **当前可回退论文快照** → [`docs/method/pseudo_hunter_fallback_snapshot_2026-05-25.md`](docs/method/pseudo_hunter_fallback_snapshot_2026-05-25.md)（若新路线未跑出更强结果，保留此方法与数字写论文）
- **下一候选技术路线** → [`docs/method/typed_multiview_normality_plan_2026-05-25.md`](docs/method/typed_multiview_normality_plan_2026-05-25.md)（Typed multi-view APK normality learning；proposed，重大调整需与用户确认）
- 历史方法细节（TI-MIL / 批次 F0–F8） → [`docs/method/ours_method_spec.md`](docs/method/ours_method_spec.md)
- 数据集计划（3 条 Track） → [`docs/method/dataset_plan.md`](docs/method/dataset_plan.md)
- 威胁模型（transform 对标） → [`docs/method/threat_model.md`](docs/method/threat_model.md)
- **环境与数据迁移方案（一键复现）** → [`docs/environment_setup.md`](docs/environment_setup.md)
- **最新进展报告（2026-05-16, 含投稿分析）** → [`docs/progress/archive/2026-05-16_pseudo_bert_v4_progress_report.md`](docs/progress/archive/2026-05-16_pseudo_bert_v4_progress_report.md)
- **当前最强 LOPO 结果（routed three-path + path dropout, AUROC=0.9582）** → [`outputs/experiments/path_ablation/lopo_results_routing_path_dropout_full.json`](outputs/experiments/path_ablation/lopo_results_routing_path_dropout_full.json)
- **Strict DPT-v2 + F-Droid hard benign 结果（AUROC=0.9280）** → [`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json`](outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json)
- **Track B 数据 Release** -> external dataset release metadata is not redistributed in this artifact; see `artifact/paper_results/hard_benign/` manifests.
- 合成语料 leakage 审计 + MIL 特征诊断 → [`docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md`](docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md)
- 文档生态审计 → [`docs/progress/sessions/2026-05-07_doc_audit.md`](docs/progress/sessions/2026-05-07_doc_audit.md)
- **全框架进展报告（方法对比 + 实验数字）** → [`docs/progress/archive/2026-05-13_full_framework_progress_report.md`](docs/progress/archive/2026-05-13_full_framework_progress_report.md)
- Track C wild malware corpus → [`docs/workstreams/track_c/corpus_schema.md`](docs/workstreams/track_c/corpus_schema.md)
- Agent 协作规范 → [`AGENTS.md`](AGENTS.md)

## 0.1 新 agent / 新协作者任意时刻上手清单（**强约束**）

> **本节是任何 agent 任意时刻进入本仓库时的 onboarding 契约**：读完本节 + AGENTS.md 即具备向后推进的完整资格。无论项目推进到 Phase 几、无论 deadline 剩多少天，**这五步是进入工作前的最小充分动作**，不可跳过。
>
> **契约效力**：如果跳过本节任何一步就开始 commit / push，PR 将被判定为未 onboard 直接回滚。

### Step 1 · 选择自己的角色并按路径读文档（10–30 分钟）

| 我是… | 按序阅读 | 目标 |
|---|---|---|
| **新合作者** | 本文件全文 → [`docs/research_framing.md`](docs/research_framing.md) §1–§2 → [`docs/project_constraints.md`](docs/project_constraints.md) | 理解研究定位、Phase 路线、目录/数据/commit 约束 |
| **要写代码 / 交 PR 的 agent** | [`AGENTS.md`](AGENTS.md) 全文 → [`docs/method/ours_method_spec.md`](docs/method/ours_method_spec.md) §1–§2 + §12 → [`docs/paper_submission_plan.md`](docs/paper_submission_plan.md) §2 当周任务 | 掌握方法契约 + 当周冲刺任务 |
| **要跟进度 / 冲刺任务的用户** | [`docs/paper_submission_plan.md`](docs/paper_submission_plan.md) §1–§2 → [`docs/progress/README.md`](docs/progress/README.md) → [`docs/progress/sessions/`](docs/progress/sessions/) 最新 2–3 份 → [`docs/project_progress.md`](docs/project_progress.md) 附录 A | 知道现在在哪、下一步做什么 |
| **Track B 子任务执行 agent** | [`docs/workstreams/track_b/README.md`](docs/workstreams/track_b/README.md) → [`tasks.md`](docs/workstreams/track_b/tasks.md) → [`conventions.md`](docs/workstreams/track_b/conventions.md) | 拾取一单 B-* 工单 |
| **审稿人 / 外部读者** | 本文件全文 + [`docs/method/problem_definition.md`](docs/method/problem_definition.md) | 论文级别理解 |

### Step 2 · 拿到"当前最新情况"（5 分钟，**每次会话必做**）

权威信息源的先后顺序（上游优先，下游追认）：

1. `git log --oneline -15` — 最近 15 个 commit 是最新批次的落地点。
2. [`docs/progress/sessions/`](docs/progress/sessions/) 下按日期**最新的 2–3 份会话日志** — 这是最细颗粒度的真实进度。
3. [`docs/workstreams/2026-05-06-evening-handoff.md`](docs/workstreams/2026-05-06-evening-handoff.md) — Track B v2 + F-MIL 综合 handoff，跨多日累积。
4. [`docs/workstreams/track_b/tasks.md`](docs/workstreams/track_b/tasks.md) §1 — 拾取未认领且可独立完成的 📋 工单。
5. [`docs/paper_submission_plan.md`](docs/paper_submission_plan.md) §2 — 对照当前 ISO 日历周确认是否偏离冲刺计划。

**不要**通过"猜最近改了什么"来判断进度；**不要**凭印象；**一律**从 git log + sessions/ 查起。

### Step 3 · 验证本地环境（5 分钟）

```powershell
python -m pip install -e ".[dev,metrics]"
python -m pytest tests/unit/ -q --tb=no
# 预期：623 passed, 1 skipped, zero failure（数字可能随批次增加而上调）
```

**任何失败 -> 先修环境再做其他任何事。** Windows 会话的 PowerShell 有 PATH / UTF-8 陷阱；artifact reviewer path 不依赖本地固定路径。

### Step 4 · 会话开始时立刻新建会话日志（自动记录自己的工作）

**规则**：每一次较长的工作会话都必须产出一份 `docs/progress/sessions/<YYYY-MM-DD>_<topic>.md`。

- 命名：`<YYYY-MM-DD>_<snake_case_topic>.md`；一天多份就在 topic 上区分。
- 结构：参考最近两份日志（`2026-05-07_leakage_audit_and_mil_feature_review.md` / `2026-05-07_doc_audit.md`）。必含：§1 当前任务、§2 诊断/发现、§3 修改/动作、§4 验收（测试/实验数字）、§7（或末段）签收/deferred。
- 写作规约：[`docs/progress/sessions/README.md`](docs/progress/sessions/README.md)；归档规则：[`docs/progress/archive/README.md`](docs/progress/archive/README.md)。
- **不要**把会话结论直接写到 `project_progress.md` 或 `progress/README.md`；先进 `sessions/`，经过一次 review 后再由 `project_progress.md` §18+ 追加批次条目做 summary。

### Step 5 · 动手前的硬契约自检（commit 前必须全部 ✅）

以下**任一项未通过**就不能 commit / push：

1. **角色问题（AGENTS.md §0 四问）**：
   - 本次改动对应哪个 Phase？哪条 deliverable？
   - 若涉及方法创新：`research_framing.md` §4.1 顶会自检 6 条是否全部能勾选？
   - 若涉及数据/实验：对应 RP1 / RP2 / RP3 / RP4 哪一条研究问题？
   - 若是新卖点：有没有先提 docs PR 再提 feat PR？
2. **方法诚信（L43，AGENTS.md §8.1 第 7 条）**：任何 paper-quotable 的 Ours 训练必须显式 `supervision_mode="bag"`；`instance_aided` 仅作诊断 / 上界，不得作为 Tier-A 数字。
3. **Leakage 守护（L41/L42/L45，`sessions/2026-05-07_leakage_audit_and_mil_feature_review.md` §7）**：
   - 新加 transform family 必须同步更新 `baselines.ours.FAMILY_TO_PAYLOAD_KIND`（否则 `test_family_mapping_is_exhaustive_over_registered_transforms` 失败）。
   - 使用 v4 synthetic 语料的 train/val split 必须来自 `data/synthetic/splits_v4_lofo/`（LOFO），不得随机 70/14 切分。
   - 训练 loop 必须走 `subsample_bag_for_training`（默认 `train_max_bag_size=256`）；不要直接把 1500+ 实例的 raw bag 灌进去。
4. **工程硬约束（`docs/project_constraints.md`）**：
   - 核心管线保持零依赖；torch / transformers 懒加载在函数体内。
   - MLM 语料仅限 benign seed APK；检测到 synthetic 样本混入必须抛 `BenignCorpusError`。
   - `data/real_world/` / `outputs/` / `thirdparty/` 已被 `.gitignore` 排除；不得 `git add -f` 它们下面的大文件。
   - 不得删除 `.codebuddy/` 与 `.workbuddy/memory/` 目录（分别为 IDE 状态和记忆库）。
5. **commit message 规范**：
   - 前缀：`feat: / fix: / refactor: / docs: / test: / experiment: / data: / chore:`。
   - 方法/数字相关的 commit body 必须引用具体数字来源（`baseline_numbers.md` 或 `sessions/<date>_*.md` §x）。

### Step 6 · 把已完成工作推进到下一个拾取点（交接）

一次会话结束前必须留下可被**下一个 agent / 下一次会话**零成本继续推进的钩子：

- `sessions/<date>_<topic>.md` 末尾写明 §Deferred / §Next Step（哪些未做、下一个 agent 从哪里接）。
- 代码有结构性改动 → `project_progress.md` §进行中 追加一行。
- 新发现的阻塞 → 在 `workstreams/track_b/tasks.md` 或 `paper_submission_plan.md` §2 当周任务里补 📋 工单。

---

**总原则**：任何时刻，读完 README §0 + §0.1 + AGENTS.md 的 agent 都能无缝接上当前进度并开始产出；偏离本清单的 PR 视作 onboarding 失败直接回滚。

## 1. 研究定位

- **研究叙事唯一源**：[`docs/research_framing.md`](docs/research_framing.md)（顺层研究问题、四阶段路线 Phase 1→4、MVP 论文四段式、方法卖点顶会水准自检）。本节仅做摘要。
- **投稿执行唯一源**：[`docs/paper_submission_plan.md`](docs/paper_submission_plan.md)（两阶段打法、6 周冲刺任务、会议时间表、你/agent 分工）。当前策略：**Stage A** 冲 ACSAC 2026（deadline 约 2026-06-03）保底发一篇，**Stage B** 扩写冲 USENIX Security '27 Cycle 2（约 2026-10）。
- 长期瞑向安全四大顶会（S&P / USENIX Security / CCS / NDSS）投稿。
- 当前阶段（Phase 2）先达到软件工程 A 会水平的工程完整性：问题定义清晰、数据链路可复现、实验协议可验证、系统原型可运行。
- 项目级约束（目录结构、数据处置、Git 提交、验证要求）集中在 [`docs/project_constraints.md`](docs/project_constraints.md)。
- 阶段进度与未完成项见 [`docs/project_progress.md`](docs/project_progress.md) 与 [`docs/progress/README.md`](docs/progress/README.md)。
- 方法与实验协议见 [`docs/method/`](docs/method/)（`ours_method_spec.md` §1–§2 与 `research_framing.md` §1–§4 必须保持一致）。
## 2. 当前能力矩阵

| 能力 | 入口 | 输出 | 状态 |
| --- | --- | --- | --- |
| APK 对象抽取 + region 切分 | `android-packer-extract-regions` | `objects.jsonl` / `regions.jsonl` | ✅ |
| Synthetic packed APK 生成（可注册新 transform） | `android-packer-generate-packed` / `android_packer.synthetic.register_transform` | generated APK / manifest JSON / strong-label JSONL | ✅ **11 种 transform family 可用**（Track A v2 及 3 fix + 3 新 transform 已落地）；✅ **v4 corpus（108 task）leakage L1-L26 闭合**（sanity_rules region AUROC 0.848→0.743，见 [`progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md`](docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md)） |
| 训练标签对齐 (region/object/APK) | `android-packer-build-labels` | 三份 `*_labels.jsonl` | ✅ |
| 熵阈值基线 + AUROC/MRR/IoU 评测 | `android-packer-run-entropy` | 三层 predictions + JSON 报告（含 AUROC / AUPRC / MRR / Top-k / IoU / Boundary Error） | ✅ |
| 端到端批量 synthetic + entropy 实验 | `android-packer-experiment-entropy` | `experiment_manifest.json` + `summary.json`（含 AUROC / ranking / localization 聚合） | ✅ |
| APKiD reference baseline（APK-level，社区公认的规则 baseline） | `android-packer-run-apkid` | APK-level predictions + JSON 报告（`localization_granularity = "apk_only"`） | ✅（需 `pip install -e ".[apkid]"`；APKiD 3.x 可直接运行） |
| Sanity-check 启发式 baseline（内部诊断用，非论文 baseline） | `android-packer-run-sanity-rules` | 三层 predictions + JSON 报告 | ✅ |
| Byte-level n-gram + LR baseline（学习型字节视图 baseline） | `android-packer-train-ngram` / `android-packer-run-ngram` | 训练 pickle 模型 + 三层 predictions + JSON 报告 | ✅ （需 `[metrics]` extras；已改用 FeatureHasher 解决 OOM） |
| **PayloadHunter-Lite**（历史消融 baseline：entropy-delta + 手工 15 维 + 浅 MLP + attention 聚合） | `android_packer.baselines.payload_hunter_lite` / `android_packer.models.payload_hunter_lite` | 三层 predictions + JSON 报告 | 🟡 **历史 ablation baseline**；不再作为当前 Ours |
| **Typed-Instance MIL Localization**（历史 Ours：typed encoder + MIL pooling + attention-anomaly scoring） | `android-packer-train-ours` / `android-packer-run-ours` | 三层 predictions + bag→instance attention 解释 | 🟡 **历史方法**；84-task LOFO Region AUROC=0.869，Object MRR=0.521；真实壳 Decision Gate 后转入 PseudoHunter 主线 |
| **Grammar-aware byte pretraining**（历史 TI-MIL 预训练组件） | `android_packer.models.item_type_head` + `android_packer.training.pretrain_mlm` / `android-packer-pretrain-mlm` | byte encoder ckpt + per-token item-type 监督 | 🟡 已落地并保留；当前论文主线改用 Pseudo-code BERT spMLM |
| **Packed/Unpacked Differential Contrastive Pretraining**（历史 Track B 18 pair InfoNCE） | `android_packer.training.contrastive` | encoder ckpt（`h_app` app-semantic + `h_pack` packing residual） | 🟡 已落地并保留；当前 PseudoHunter 监督使用 packed/unpacked differential labels + MIL |
| 多 baseline 批量实验（entropy / sanity_rules / ngram / 可选 apkid） | `android-packer-experiment-multi-baseline` | `experiment_manifest.json` + `summary.json`（按 baseline × transform 聚合） | ✅ |
| 统一评测指标模块 (AUROC / MRR / Top-k / IoU 等) | `android_packer.evaluation` | — | ✅ |
| seen / unseen transform split + seen / unseen package split | `android_packer.splits` (`by_transform_split` / `by_package_split`) + [`configs/splits/transforms_holdout.yaml`](configs/splits/transforms_holdout.yaml) 已固化 11-fold | `DatasetSplit` 对象（train/val/test/unassigned）| ✅ |
| JSON Schema 运行时校验 (dev-only) | `android_packer.utils.schema` (`validate_record` / `validate_jsonl`) | 异常或 ``None`` | ✅ |
| Leave-One-Packer-Out (LOPO) 跨 packer 家族评测协议 | `android_packer.splits.by_packer_family`（planned） | LOPO `DatasetSplit` × 3 fold (S5/S6/S3) | ⏳ F-MIL-eval-1（Tier B-3） |
| **Leave-One-Family-Out (LOFO) 合成 split**（L45 修复，11 fold × 76/8 train/val）| `scripts/split_synthetic_by_family.py` | `data/synthetic/splits_v4_lofo/split_<family>.json` + `index.json` | ✅ 2026-05-07 落地，5 单测全绿；runner 接线待 F-MIL-eval-1 配套完成 |
| **MIL bag subsampling**（L41 修复：正例保全 + 负例均匀下采样，解决 bag size 1574 / 正例占比 0.1% 退化）| `android_packer.training.mil_trainer.subsample_bag_for_training` + `MILTrainerConfig.train_max_bag_size / train_min_positive_fraction` | 每 epoch × bag 的 deterministic 子采样 | ✅ 2026-05-07；8 单测全绿 |
| **Supervision mode 诚信开关**（L43 修复：区分 weakly-supervised bag 模式 vs instance-aided 模式，避免论文声明漂移）| `MILTrainerConfig.supervision_mode ∈ {"bag","instance_aided"}` + `OursBaselineConfig.supervision_mode` | 训练时 strictly 忽略 per-instance label（bag 模式） | ✅ 2026-05-07；3 单测全绿 |
| **Typed-instance ground-truth 路由**（L42 修复：`transform_family → kind` 权威映射，新增 `benign_other` 第 7 类）| `android_packer.baselines.ours.FAMILY_TO_PAYLOAD_KIND` + `_object_instance_type(..., transform_families, label_id)` | 合成数据按 family 精确路由；benign object 有独立 head；真 packer 推理走 legacy 启发式 | ✅ 2026-05-07；14 单测全绿 |
| Byte-CNN 消融 baseline（字节视图上限对照） | — | — | ⏳ Week 2 D6 待开发（L44 follow-up） |
| PseudoHunter top-tier version（learned path confidence + corrupted-region pretraining + runtime evidence；Stage B） | planned | encoder + strict app-disjoint evaluation | ⏳ Stage B（USENIX）|
| Runtime-grounded 标签增强（emulator hook DexClassLoader） | — | — | ⏳ Stage B M6 |
| **全框架 Region Encoder + Entry MIL**（improved_packed_apk_framework 实现）| `regioning/typed_slicer` + `features/full_feature_extractor` + `models/full_encoder` + `models/entry_aggregator` | 318维特征 + 双层聚合 + normality-conditioned MIL | ✅ **2026-05-13 落地**；跨域 AUROC=0.7543（首次超越 entropy baseline 0.7246）|
| **Stage 3 差分训练**（Happer 配对 APK attention alignment）| `labeling/happer_diff` + `training/differential_trainer` | L_bag + L_rank + L_align + L_normality 四组份 loss | ✅ **2026-05-13 落地**；Happer LOPO domain-内 APK AUROC=1.0 |
| **PseudoHunter / Pseudo-code BERT**（当前主方法：Dalvik/ARM64/Byte 三路伪指令 BERT + region-type routing + path dropout + normality MIL）| `decoders/dalvik_decoder` + `decoders/native_decoder` + `decoders/pseudo_tokenizer` + `models/pseudo_code_bert` + `models/fusion_encoder` + `scripts/experiments/run_lopo_eval.py` | APK 检测 + entry-level localization + strict DPT hard-benign 评估 | ✅ **当前最强 LOPO**：routed three-path + dropout AUROC=0.9582，detection=99.0%，normality MRR=0.4367；✅ **strict DPT + 24 F-Droid hard benign**：AUROC=0.9280，AUPRC=0.8973，FPR@95TPR=0.3158 |
| **spMLM 预训练**（structured pseudo-code masked language modeling）| `training/pretrain_spmlm` + `scripts/data/build_spmlm_corpus.py` + `scripts/experiments/run_spmlm_pretrain_v2.py` | 8.7M benign sequences (1408 APKs), pretrain loss 2.0→1.07 | ✅ **2026-05-15 完成** |
| **AndroZoo benign 语料下载**（1340 APK spMLM 预训练语料）| `scripts/data/download_androzoo_benign.py` | 1340 APKs 已下载 + 8.7M 序列语料库 | ✅ **完成** |
| **Gated Fusion**（ABMIL-style 自适应 BERT vs Stat 加权）| `models/fusion_encoder.py` (use_gated_fusion=True) | 学习型门控：per-sample 选择 BERT 或 stat features 权重 | ✅ **2026-05-15 落地** |
| **LOPO 跨壳评估**（Leave-One-Packer-Out 7-fold）| `scripts/experiments/run_lopo_eval.py` | 7 壳家族轮流 held-out | ✅ **当前最强配置 2026-05-25**：routed three-path + dropout 平均 AUROC=0.9582 |
| **Track B inject_labels 解析**（s5/s6 注入式 ground truth）| `labeling/happer_diff.py::parse_inject_labels()` | JSONL → DiffResult，支持 ground-truth 差分标签 | ✅ **2026-05-15 落地** |
| **Modern hard benign 扩展**（F-Droid + AndroZoo 复杂 benign normality） | `scripts/data/download_androzoo_benign.py` + `scripts/data/build_hard_benign_manifest.py` + `scripts/experiments/run_lopo_eval.py --track-b-v2-strict` | APKiD-audited hard benign manifest + strict DPT-v2 结果 | 🔄 **F-Droid 24 clean 已完成**；AndroZoo CSV 正在下载，APK 下载/审计/训练已由后台 pipeline 排队 |

以上 CLI 入口由 `pyproject.toml` 通过 `project.scripts` 暴露，等价脚本仍保留在 `scripts/` 目录下方便阅读。

## 3. 快速上手

```powershell
python -m pip install -e .                       # 可编辑安装（核心管线 = 零依赖）
python -m pip install -e ".[dev,metrics]"        # 追加 pytest + numpy + scikit-learn
python -m pytest                                  # 运行全部单测 + 集成 smoke test
```

> 核心 pipeline（对象抽取、region 切分、synthetic packer、熵阈值 baseline）只依赖标准库；`[metrics]` extras 为后续 AUROC / MRR / AUPRC 等指标模块准备，`[dev]` 只为开发工具。

### 3.1 从一个 APK 抽取对象与 region

```powershell
android-packer-extract-regions path\to\sample.apk `
  --objects-out data\processed\objects\sample.objects.jsonl `
  --regions-out data\processed\regions\sample.regions.jsonl `
  --window-size 4096 --stride 2048
```

产出：

- **object metadata**：APK 内对象的路径、类型、大小、sha256、压缩方式、嵌套深度等。
- **region metadata**：按 byte window 切分后的 region，含 offset、sha256、entropy、printable_ratio。

### 3.2 生成一个 synthetic packed APK

```powershell
android-packer-generate-packed path\to\seed.apk `
  --transform-family xor `
  --generated-apk-out data\synthetic\generated_apks\seed.xor.apk `
  --manifest-out     data\synthetic\manifests\seed.xor.manifest.json `
  --labels-out       data\synthetic\labels\seed.xor.labels.jsonl
```

当前支持的 transform family（Track A v2 完成 2026-04-30，共 **11 个**）：`path_randomized` / `xor` / `base64` / `split_xor`（每段独立 key，Fix-1）/ `signature_strip`（扩展至 80 字节 header 扰动，Fix-2）/ `embedded_asset` / `so_embedded`（simplified ELF overlay，Fix-3）/ `dex_method_inlined` / `multi_dex_shim` / `embedded_archive` / `dex_string_encrypted`。若不传 `--payload`，默认从 seed APK 中选取一个顶层 DEX 作为 payload 来源，保证 synthetic 样本自带强标签。

### 3.3 对齐训练标签

```powershell
android-packer-build-labels `
  --regions            data\processed\regions\sample.regions.jsonl `
  --synthetic-labels   data\synthetic\labels\seed.xor.labels.jsonl `
  --region-labels-out  data\processed\labels\sample.region_labels.jsonl `
  --object-labels-out  data\processed\labels\sample.object_labels.jsonl `
  --apk-labels-out     data\processed\labels\sample.apk_labels.jsonl
```

### 3.4 跑熵阈值基线

```powershell
android-packer-run-entropy `
  --region-labels           data\processed\labels\sample.region_labels.jsonl `
  --region-predictions-out  outputs\predictions\entropy.region_predictions.jsonl `
  --object-predictions-out  outputs\predictions\entropy.object_predictions.jsonl `
  --apk-predictions-out     outputs\predictions\entropy.apk_predictions.jsonl `
  --report-out              outputs\reports\entropy_baseline_report.json `
  --entropy-threshold       7.0
```

### 3.5 批量实验（synthetic + entropy）

```powershell
android-packer-experiment-entropy `
  --config configs\eval\synthetic_entropy_baseline.json
```

产出 `outputs/experiments/synthetic_entropy_baseline/experiment_manifest.json` 和 `summary.json`，按 transform family 聚合 micro-averaged 指标。

### 3.6 跑 APKiD reference baseline

APKiD 是 RedNaga 社区维护的 YARA 规则引擎，用于识别 Android packer / protector / obfuscator / compiler 家族；本项目的**论文 baseline**就是它，因为规则由社区而非作者维护，避免自造 baseline 引发的方法论质疑。APKiD 只提供 APK 粒度的结果（`localization_granularity = "apk_only"`），object/region-level 指标需要我们的方法来补足——这正是论文的 motivation。

首次使用需要安装 extra 并确认 CLI 可用：

```powershell
python -m pip install -e ".[apkid]"
apkid -h
```

APKiD 3.x 自带可直接加载的规则集；旧版 APKiD 2.x 环境如果提示规则未编译，再运行 `apkid --prepare`。

输入 JSONL 支持两种形状自动识别：

- **显式**：每行 `{"apk_id": ..., "apk_path": ..., "true_label_id": 0|1}`
- **synthetic manifest**：携带 `generated_apk_path` 字段的行会被视作 manifest 行，`true_label_id` 默认 1

```powershell
android-packer-run-apkid `
  --apk-entries           data\synthetic\manifests\packed.manifest.jsonl `
  --apk-predictions-out   outputs\predictions\apkid.apk_predictions.jsonl `
  --report-out            outputs\reports\apkid_baseline_report.json `
  --min-hits              1
```

未安装 APKiD 时 CLI 以退出码 2 返回，并打印可执行的安装提示。

### 3.7 跑 sanity-check 启发式 baseline（内部诊断用）

> ⚠️ **这不是论文 baseline**。它是一个自造的、基于 object 路径 / 扩展名 / 大小 / printable ratio 的 heuristic，用于验证标签 / 评测链路在端到端上是否自洽，以及与 APKiD 形成互补的 object / region 级消融视图。向审稿人汇报时请使用 APKiD。

```powershell
android-packer-run-sanity-rules `
  --region-labels           data\processed\labels\sample.region_labels.jsonl `
  --region-predictions-out  outputs\predictions\sanity_rules.region_predictions.jsonl `
  --object-predictions-out  outputs\predictions\sanity_rules.object_predictions.jsonl `
  --apk-predictions-out     outputs\predictions\sanity_rules.apk_predictions.jsonl `
  --report-out              outputs\reports\sanity_rules_baseline_report.json
```

### 3.8 训练并跑字节级 byte-level baseline（n-gram + LogisticRegression）

byte-level baseline 代表"只看字节的学习型方法"——消融实验要证明"结构视图 + 对象级聚合"带来真实增量就必须有它。依赖：`scikit-learn`（在 `[metrics]` extras 里）。

**apk-index 是一份 JSONL**，每行给出 `apk_id -> apk_path` 的映射；也直接支持 synthetic manifest 形状（带 `generated_apk_path` 的行会被自动识别）。

```powershell
# 训练：读 region_labels.jsonl + apk_index，保存 pickle 模型
android-packer-train-ngram `
  --region-labels  data\processed\labels\sample.region_labels.jsonl `
  --apk-index      data\synthetic\manifests\packed.manifest.jsonl `
  --model-out      outputs\models\ngram_logreg.pkl `
  --report-out     outputs\reports\ngram_logreg_train_report.json

# 推理：载入模型并对 region_labels 打分，输出三层 predictions + 报告
android-packer-run-ngram `
  --model                    outputs\models\ngram_logreg.pkl `
  --region-labels            data\processed\labels\sample.region_labels.jsonl `
  --apk-index                data\synthetic\manifests\packed.manifest.jsonl `
  --region-predictions-out   outputs\predictions\ngram_logreg.region_predictions.jsonl `
  --object-predictions-out   outputs\predictions\ngram_logreg.object_predictions.jsonl `
  --apk-predictions-out      outputs\predictions\ngram_logreg.apk_predictions.jsonl `
  --report-out               outputs\reports\ngram_logreg_baseline_report.json
```

默认特征为 256-维字节频率直方图 + 1024-bucket hashed bigram + 9 条 scalar（熵 / 分块熵均值与方差 / printable / zero / high-byte / 最长 zero-run / 重复 16-byte block / log1p 长度）。支持 `--no-bigram` / `--no-scalars` / `--bigram-hash-dim` / `--C` / `--class-weight` 等开关做消融。

### 3.9 一次跑通多 baseline 对比表（entropy / sanity_rules / ngram，可选 apkid）

批量 runner：给它一份 seed manifest，同时跑 N 个 baseline 在同一批 synthetic 任务上，产出一份 `summary.json`，按 `baseline × transform_family` 聚合 AUROC / MRR / IoU 等指标。

```powershell
android-packer-experiment-multi-baseline `
  --config configs\eval\synthetic_multi_baseline.json
```

- 当前支持的 synthetic baseline：`entropy / sanity_rules / ngram_logreg / apkid`（4 档）。PayloadHunter-Lite 与 Typed-Instance MIL 保留为历史消融/诊断入口；**当前论文主方法是 PseudoHunter**，走 [`scripts/experiments/run_lopo_eval.py`](scripts/experiments/run_lopo_eval.py) + `models/pseudo_code_bert.py` / `models/fusion_encoder.py` 的 LOPO 与 strict DPT 评估链路。
- 默认启用 `entropy / sanity_rules / ngram_logreg`；APKiD 需在 config 里手工加入 `enabled` 列表（并装好 `[apkid]` extras）。若 APKiD 未装，runner 会把该 baseline 记为 `skipped` 并在 `summary.warnings` 给出安装提示，而不是中断整个实验。
- `ngram_logreg` 当前只支持 `train_mode = "same_set"`：在所有任务的 region labels 上训一个模型然后在线打分，数字是 in-sample 的，`summary.warnings` 会显式标注此偏差。`holdout_transform` / `holdout_package` 模式列在"推进顺序"里。

> 脚本仍可用 `python scripts/...` 直接调用；如果未 `pip install -e .`，请先 `$env:PYTHONPATH='src'`。

## 4. 目录结构

```text
AndroidPacker/
├── docs/                        # 研究笔记、方法、实验记录、阶段进度
│   ├── research_framing.md      # 研究叙事唯一源（RP / Phase / 卖点）
│   ├── paper_submission_plan.md # 投稿执行唯一源（Stage A/B / 6 周冲刺）
│   ├── project_constraints.md   # 项目级工程与提交约束
│   ├── project_progress.md      # 批次级进度与风险
│   ├── results_matrix.md        # baseline × Track 数字矩阵
│   ├── method/                  # 问题定义、威胁模型、数据集计划、Ours 方法规格书
│   ├── progress/                # 月度与周度进度报告索引
│   └── workstreams/             # 子工作流（当前有 track_b/：真实 packer 评测池）
├── paper/                       # 论文草稿、图、表
├── references/                  # 参考文献与相关工作
├── configs/                     # data / model / train / eval / splits / runtime 配置
│   ├── data/schemas/            # JSON Schema（数据契约）
│   └── splits/                  # seen/unseen transform holdout split 配置（11-fold）
├── thirdparty/                  # 真实 packer 源码 clone + 我们的 source-injection patch
│   └── patches/                 # per-packer apply.py + *.patched.java + pinned sha256
├── data/                        # APK / DEX / JSONL 等运行产物（默认不入 Git）
│   ├── raw/                     # 原始 benign 与真实加壳 APK
│   ├── synthetic/               # synthetic packer 生成样本 + 强标签
│   ├── processed/               # 对象、region、特征、split
│   ├── real_world/              # Track B / 真实加壳 APK（track_b/ + commercial_packers/）
│   └── runtime/                 # Stage B 保留：运行时 trace / recovered DEX / 对齐
├── scripts/                     # Thin CLI 入口（pipeline 拼装）；baselines / experiments / ingest / labeling / synthetic 子目录
├── src/android_packer/
│   ├── apkio/                   # APK / ZIP 对象抽取
│   ├── regioning/               # Byte-window region 切分
│   ├── synthetic/               # Synthetic packer：transform + 注入 + manifest（11 family）
│   ├── labeling/                # Synthetic / diff-alignment / injected-packer / commercial-rule 标签路径
│   ├── features/                # Byte-level（unigram/bigram/scalars + LRU）+ DEX-aware（dex_item_parser / entropy_delta）
│   ├── baselines/               # entropy / apkid / sanity_rules / ngram_logreg / payload_hunter_lite
│   ├── evaluation/              # AUROC / MRR / Top-k / IoU / Boundary Error 等指标
│   ├── splits/                  # seen / unseen transform & package split 构造器
│   ├── experiments/             # 多 baseline 批量 runner 共享的聚合逻辑
│   ├── models/                  # PayloadHunter-Lite + tokenizer（Stage B 再入 RoBERTa encoder）
│   ├── cli/                     # CLI 入口模块（一对一映射 scripts/）
│   ├── aggregation/ candidates/ runtime/   # 预留空模块（待后续阶段填充）
│   └── utils/                   # 共享工具（jsonl / paths / schema）
├── outputs/                     # checkpoints / predictions / reports / logs
├── tests/                       # unit + integration
├── notebooks/                   # 探索性分析（不承载 pipeline）
└── samples/                     # 小规模调试样本
```

## 5. 数据 schema

所有中间产物都是 **JSONL**（一行一条记录），schema 定义在 `configs/data/schemas/*.json`：

- `object_metadata.schema.json` — `iter_apk_objects()` 输出。
- `region_metadata.schema.json` — `iter_regions()` 输出。
- `synthetic_label.schema.json` — synthetic packer 输出的 strong label。
- `synthetic_manifest.schema.json` — synthetic packer 的生成清单。
- `region_training_label.schema.json` / `object_training_label.schema.json` / `apk_training_label.schema.json` — `build_training_labels()` 输出。

> 数据 schema 目前由文档和 dataclass 双向约束；运行时校验仍待补齐（见 `docs/project_progress.md` 风险项）。

## 6. 推进顺序（近期）

> **当前冲刺以 [`docs/paper_submission_plan.md`](docs/paper_submission_plan.md) §2 为准**（Week 1–5 任务细分、人机分工、ACSAC 2026 deadline 对齐）。本节仅保留"工程视角的推进顺序"，投稿时间线与周级任务以投稿计划为唯一源。

### 当前主线方法：Pseudo-code BERT v4 + Gated Fusion (2026-05-15)

> **方向转换 (2026-05-12)**：TI-MIL (15维特征 + bag-only ABMIL) 在真实壳上 APK AUROC=0.67, Object MRR≈0。经过 Decision Gate 实验确认特征过拟合 app 分布后，切换为全框架方法。详见 [`docs/method/ds_amil_spec.md`](docs/method/ds_amil_spec.md) 和 [`docs/progress/archive/2026-05-13_full_framework_progress_report.md`](docs/progress/archive/2026-05-13_full_framework_progress_report.md)。

**当前架构**：
```
APK → typed_slicer (DEX/ELF section-aware regions)
    → 318维统计特征 + 三路伪指令 token (Dalvik/Native/Byte)
    → Pseudo-code BERT (4L/256d, 共享权重, spMLM 预训练)
    → Gated Fusion (ABMIL-style 自适应 BERT vs Stat 加权)
    → Entry聚合 (attention + maxpool) → APK MIL (normality-conditioned)
```

**实验数字**（LOPO = Leave-One-Packer-Out，无 leakage，最严格跨壳泛化评估）：

| 方法 | APK AUROC | Entry MRR | 评估协议 | 状态 |
|------|:---------:|:---------:|:--------:|:----:|
| Entropy threshold (无训练) | 0.7246 | — | Cross-dataset | baseline |
| XGBoost + 256维字节直方图 | 0.6232 | — | Cross-dataset | 过拟合 app 分布 |
| Stat-only + Stage 3 差分训练 | 0.7543 | — | Cross-dataset | 纯统计特征 |
| **BERT v4 (gated + pretrain v3 + AZ)** | **0.9251** | **0.5367** | **7-fold LOPO** | historical |
| **PseudoHunter routed three-path + dropout** | **0.9582** | **0.4367** | **7-fold LOPO** | ✅ 当前最优 |

**LOPO 7-fold 逐壳结果**（2026-05-16, pretrain v3 1M×10ep + 50 AndroZoo benign, 无 leakage）：

| 壳家族 (held-out) | APK AUROC | 检出率 | Entry AUROC | Entry MRR | 说明 |
|---|:---:|:---:|:---:|:---:|---|
| Ali (Happer) | 0.9689 | 80.0% | 0.7148 | 0.8400 | 良好泛化 |
| Qihoo (Happer) | 0.6000 | 0.0% | 0.6823 | 0.3772 | 强加密最难 |
| Tencent (Happer) | 0.9733 | 86.7% | 0.7700 | 0.8810 | 定位准确 |
| 360加固 (Track B†) | 1.0000 | 100% | — | — | 1 样本 |
| 梆梆 Bangcle (Track B†) | 0.9852 | 88.9% | 0.6293 | 0.3280 | app-disjoint |
| APKProtector (Track B†) | 0.9704 | 88.9% | — | — | app-disjoint |
| DPT Shell (Track B†) | 0.9778 | 100% | 0.4189 | 0.2576 | app-disjoint |
| **平均** | **0.9251** | **77.8%** | **0.6431** | **0.5367** | — |

†Track B: seed apps 未出现在训练 benign 中，无 app-identity leakage。推理 121 ms/APK。

**关键技术贡献**：
1. **Pseudo-instruction naturalness** — 加密数据线性反汇编产生异常伪指令，LM 可检测
2. **三路伪指令 BERT** — Dalvik (55 vocab) + Native ARM64 (51 vocab) + Raw Byte (256 vocab) 共享 Transformer
3. **Gated Fusion** — ABMIL-style gating 自适应选择 BERT vs stat features 权重
4. **Entry-level Localization** — normality score 定位 payload 所在 entry (MRR=0.54)
5. **差分标签** — Happer paired APK 统计差分 + Track B inject_labels 注入式 ground truth
6. **Frozen BERT + 4-loss** — 避免灾难遗忘；L_bag + L_rank + L_align + L_normality

**当前执行项**：

1. ✅ **全框架实现**（2026-05-13）：typed_slicer + full_feature_extractor + full_encoder + entry_aggregator + happer_diff + differential_trainer
2. ✅ **Pseudo-code BERT 实现**（2026-05-14）：dalvik_decoder + native_decoder + pseudo_tokenizer + pseudo_code_bert + fusion_encoder + pretrain_spmlm
3. ✅ **AndroZoo 1340 benign 下载 + 语料库构建**（2026-05-14）：8.7M 序列 from 1408 APKs
4. ✅ **spMLM 预训练 v3**（2026-05-16）：1M seq × 10 epochs, loss 2.0→0.92
5. ✅ **Gated Fusion 实现**（2026-05-15）：ABMIL-style 自适应 BERT/Stat 加权
6. ✅ **Track B 配对数据 + inject_labels**（2026-05-15）：cs3+s5+s6 共 27 对
7. ✅ **LOPO v3 (无 leakage + localization)**（2026-05-16）：APK AUROC=0.9251, Entry MRR=0.5367
8. ✅ **Path routing + dropout 复现**（2026-05-25）：当前 fallback 主结果 APK AUROC=0.9582，detection=99.0%，normality MRR=0.4367
9. ✅ **Strict DPT-v2 hard benign 修复**（2026-05-26）：24 个 F-Droid APKiD-clean hard benign 后 strict clean DPT AUROC=0.9280，AUPRC=0.8973，benign mean 0.1661，FPR@95TPR=0.3158
10. 🔄 **AndroZoo modern hard benign replication**（2026-05-26）：CSV 索引下载中；后台 pipeline 将在 gzip 校验通过后下载 50 个 modern Play-market benign APK、APKiD 审计，并自动启动 `strict_dpt_clean_hardbenign_androzoo50_lowbyte025`
11. ✅ **实验审计 + leakage 修复**（2026-05-15）：移除 TB benign, 修复 RNG
12. ⏳ 论文写作 + 消融实验矩阵：当前叙事已从“strict DPT 失败”更新为“hard benign normality 显著修复，但低 FPR 工作点仍不足”

**关键文档**：
- 方法规格：[`docs/method/ds_amil_spec.md`](docs/method/ds_amil_spec.md)
- 环境迁移：[`docs/environment_setup.md`](docs/environment_setup.md)
- 进展报告：[`docs/progress/archive/2026-05-13_full_framework_progress_report.md`](docs/progress/archive/2026-05-13_full_framework_progress_report.md)
- 当前 LOPO 结果：[`outputs/experiments/path_ablation/lopo_results_routing_path_dropout_full.json`](outputs/experiments/path_ablation/lopo_results_routing_path_dropout_full.json)
- strict DPT hard benign 结果：[`outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json`](outputs/experiments/track_b_v2_strict_dpt/results_strict_dpt_clean_hardbenign_fdroid24_lowbyte025.json)

### 历史方法演进 (2026-05-06 ~ 2026-05-12)

<details>
<summary>点击展开历史推进记录</summary>

批次 A～E + Track A v2 + Track B v2 (18/18 green) + F-Lite Pass-2a 已落地：熵法 / APKiD / sanity-rules / byte-level (ngram+LR) 四档 baseline + 统一批量 runner + 11 transform family + PayloadHunter-Lite + 18 packed/unpacked 配对。**2026-05-06 起，"Ours" 整体升级为 Typed-Instance MIL + Grammar-aware aux + Packed/Unpacked Contrastive 三件套**（详见 [`docs/research_framing.md`](docs/research_framing.md) §3.2 与 [`docs/method/ours_method_spec.md`](docs/method/ours_method_spec.md) §12）。当前重点：

1. **F-MIL-a/b 已落地**（2026-05-06）：`models/mil_head.py` (top-k / noisy-or / gated-attention) + `models/typed_encoder.py`（6 typed instance：`encrypted_dex / extracted_method_body / metadata_table / compressed_payload / shim / native_stub`，**标签零成本来自 [`labeling/injected_packer_adapter.py`](src/android_packer/labeling/injected_packer_adapter.py)**） + `models/ours.py` 编排器；39 个新单测已绿，无 schema 漂移。
2. **F-MIL-c 已落地**（2026-05-06）：`training/pretrain_mlm.py` + `models/item_type_head.py` 实现 byte-MLM (80/10/10 Devlin 2018) + item-type span 辅助 CE + benign-only 硬约束（解析失败即排除，超 5% 阈值抛 `BenignCorpusError`）+ `item_type_aux_weight=0` 完全跳过辅助分支（§12.5 ablation 合约，单测断言）；CLI `android-packer-pretrain-mlm` CI-safe dry-run；26 单测全绿。
3. **F-MIL-d 已落地**（2026-05-06）：`training/contrastive.py` 用 Track B 18 packed/unpacked pair 做 InfoNCE，双头 `h_app` (app-semantic invariant, symmetric InfoNCE) + `h_pack` (supervised residual InfoNCE on `z_p - z_b`, grouped by `packer_id`)；修复关键 NaN 数值稳定性 bug（`0 * -inf` at masked 对角）；20 单测全绿。**这是把 Track B v2 18/18 基础设施交付首次变成方法交付。**
4. **F-MIL-e 已落地**（2026-05-06）：`training/mil_trainer.py`（`train_ours` = BCE + λ_diff 伪标签 + λ_sparsity attention 熵）+ `baselines/ours.py`（report dataclass 与 `NgramLogRegResult` 逐字段 parity，§8 第 1 条硬契约）+ CLI `android-packer-train-ours` / `android-packer-run-ours`；21 单测全绿。`experiments/synthetic_multi_baseline.py::SUPPORTED_BASELINES += ("ours",)` 接入作为后续小优化。
5. **PayloadHunter-Lite 保留为消融 baseline**（"no MIL / no typed / no grammar aux" 对照点，见 [`docs/method/ours_method_spec.md`](docs/method/ours_method_spec.md) §11 / §12.5）。
6. **Tier B 顶会弹药（next）**：Leave-One-Packer-Out (LOPO) 评测（F-MIL-eval-1）、hard-negative mining (DexClassLoader-positive benign)（F-MIL-eval-2）、native-aware lightweight typed instances (ELF entropy + JNI_OnLoad)（F-MIL-eval-3）。
7. **2026-05-07 v4 合成语料 leakage 闭合 + MIL 结构性修复**（见 [`docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md`](docs/progress/sessions/2026-05-07_leakage_audit_and_mil_feature_review.md)）：
   - `synthetic_multi_baseline_v4`（11 family × 108 task）重生成，闭合 L1-L26 leakage；sanity_rules region AUROC 0.848→0.743（过高估被还原）。
   - `scripts/diag_synthetic_leakage_v{1..4_extra}.py` + `scripts/compare_v3_v4_auroc.py` + `scripts/audit_mil_bag_size.py` 全套审计脚本。
   - MIL 结构性修复 P0 四件：L41 bag subsampling / L42 ground-truth type routing / L43 `supervision_mode` / L45 LOFO split generator（11 fold，`data/synthetic/splits_v4_lofo/`）。
   - 单测 593→623 passed，zero regression。Ours paper-quotable run 必须 `supervision_mode="bag"`。
8. **F0d benign-混入 APK AUROC**：保证 APK 级检测性能可算。
9. **Stage B 预留**：Runtime-grounded 最小闭环 + learned path confidence / corrupted-region pretraining + Track C / AndroZoo 扩展。已显式声明为 Stage A 的 Limitations / Future Work。

</details>

## 7. 提交与复现约束

- 每次改动完成后需 `git commit`，提交信息使用 `feat: / fix: / refactor: / chore: / docs:` 等短前缀。
- APK / DEX / ZIP / 生成 APK / 运行时 trace 等二进制产物 **不进 Git**。
- Python 代码改动后运行 `python -m pytest`；CLI/脚本改动至少跑一次 smoke test。
- 影响数据格式、评测指标、实验协议的改动必须同步更新 `docs/`。

详见 [`docs/project_constraints.md`](docs/project_constraints.md) 与 [`AGENTS.md`](AGENTS.md)。
