# PseudoHunter 方法规格书

**Date**: 2026-05-20 (updated from DS-AMIL spec 2026-05-13)
**Status**: Active — 当前主方法
**系统名**: PseudoHunter
**论文标题**: "Detecting Packed Android Applications via Pseudo-Instruction Language Models"

---

## 1. 定位

本文档描述 **PseudoHunter** 的完整技术规格。它在 DS-AMIL 框架（318 维统计特征 + MIL）上增加了 **Pseudo-code BERT** 和 **Gated Fusion**，实现了更强的跨壳检测和 entry-level 定位。

**演进路径**: TI-MIL (15 dim) → DS-AMIL (318 dim, stat-only) → **PseudoHunter** (BERT + stat + gated fusion)

---

## 2. 核心结果

### 当前论文报告版本（2026-05-26）

最强 fallback 配置为 **8L/512d BERT-only pseudo-code backbone + Dalvik/ARM64/byte routed three-path + path dropout 0.25**。BERT 在 supervised 阶段冻结，region-type routing 显式降低不可靠 byte 视图权重。

| 方法 | APK AUROC | Entry MRR | 评估协议 |
|------|:---:|:---:|:---:|
| Entropy baseline | 0.725 | — | LOPO / cross-dataset baseline |
| Byte histogram + XGBoost | 0.620 | — | LOPO / cross-dataset baseline |
| Naive three-path pseudo-code | 0.828 | 0.328 | 7-fold LOPO |
| Dalvik + byte | 0.903 | 0.317 | 7-fold LOPO |
| **Routed three-path + path dropout** | **0.958** | **0.437** | **7-fold LOPO** |

Strict app-disjoint DPT-v2 当前口径：

| 配置 | AUROC | AUPRC | Detection | Benign mean | FPR@95TPR |
|------|:---:|:---:|:---:|:---:|:---:|
| routed baseline, no hard benign | 0.600 | — | 20/20 | 0.852 | — |
| + 9 APKiD-clean hard benign | 0.9335 | 0.9397 | 18/19 | 0.4892 | 0.6316 |
| + 24 F-Droid APKiD-clean hard benign | 0.9280 | 0.8973 | 17/19 raw; 18/19 normalized | 0.1661 | 0.3158 |

Interpretation: hard-benign normality is now part of the method story, not a minor data cleanup. It sharply reduces strict benign over-scoring, while low-FPR operating points remain a limitation. AndroZoo modern Play-market benign replication is queued behind the CSV index download and will run the same strict DPT protocol automatically after APK download and APKiD audit.

### 4L/256d (historical pretrain v3)

| 方法 | APK AUROC | Entry MRR | 评估协议 |
|------|:---:|:---:|:---:|
| Entropy baseline | 0.725 | 0.095 | LOPO |
| Byte histogram + XGBoost | 0.620 | — | LOPO |
| Stat-only (318 dim + MIL) | 0.949 | — | LOPO 无 leakage |
| BERT-only (4L/256d) | 0.938 | — | LOPO 无 leakage |
| Concat fusion | 0.914 | — | LOPO 无 leakage |
| **PseudoHunter (gated fusion)** | **0.948** | **0.54** | **LOPO 无 leakage** |
| APKiD (signature) | 100% det. | — | 所有壳已有签名 |

Mean AUROC = 0.913 (6 folds excl. 360 n=1)

### 8L/512d (epoch_020, 预训练中)

| 方法 | APK AUROC | Entry MRR | 说明 |
|------|:---:|:---:|:---:|
| 8L/512d BERT-only | 0.843 | 0.421 | epoch_020, 仍在下降 |
| 8L/512d Gated fusion | 0.863 | 0.556 | epoch_020, 预训练未完成 |

8L/512d 模型预训练中 (epoch 45/50, 预计 2026-05-24 16:30 完成)。

---

## 3. 架构总览

```
APK (ZIP archive)
  │
  ├─ Typed Region Slicer ─────────────────────────────────────────┐
  │   DEX: 按 section (header/string_ids/code/data)               │
  │   ELF: 按 section (.text/.rodata/.data)                       │
  │   Other: uniform 4KB windows                                   │
  │                                                                │
  ▼                                                                ▼
Region bytes (variable size)                              Region bytes
  │                                                        │
  ├─ 318-dim Stat Features ─┐                              ├─ 3-path Pseudo-instruction Decoding
  │  (byte hist, entropy,   │                              │  ├─ Dalvik: 55-token vocab
  │   structural, type)     │                              │  ├─ Native ARM64: 51-token vocab
  │                         │                              │  └─ Byte: 256+5 token vocab
  │                         │                              │
  ▼                         │                              ▼
stat_proj (318→256)         │                   Shared BERT (4L/256d or 8L/512d)
  │                         │                     3 forward passes (token_type_id 区分)
  │                         │                              │
  ▼                         ▼                              ▼
  h_stat [N, 256]                               bert_agg([h_dalvik|h_native|h_byte]) → h_bert [N, 256]
  │                                                        │
  └────────────── Gated Fusion (ABMIL gating) ─────────────┘
                           │
                    h_fused [N, 256]
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         suspicion    normality    embedding
         head [N]     head [N]     [N, 256]
                                       │
                           Entry Aggregator (attention + maxpool per ZIP entry)
                                       │
                           APK MIL (normality-conditioned gated attention)
                                       │
                               ┌───────┼───────┐
                               ▼       ▼       ▼
                          bag_logit  entry_attn  entry_logits
                          (detection) (localization)
```

---

## 4. 伪指令解码器

### 4.1 Dalvik Path (55 tokens)

线性解码 raw bytes 为 Dalvik bytecode（16-bit aligned）：

| Token 类别 | 数量 | 示例 |
|:---:|:---:|---|
| Special | 5 | PAD, BOS, EOS, MASK, UNK |
| Opcode classes | 30 | nop, move, return, const, goto, if_*, invoke_*, iget, iput... |
| Operand types | 10 | REG, CONST, STRING_IDX, METHOD_IDX, BRANCH... |
| Validity markers | 7 | VALID_OP, INVALID_OP, IDX_OK, IDX_BAD, BRANCH_OK, BRANCH_BAD... |
| Meta | 3 | INSTRUCTION_END, PAD_NORMAL, PAD_ABNORMAL |

**核心信号**: 加密数据解码后产生大量 INVALID_OP 和 IDX_BAD tokens。

### 4.2 Native ARM64 Path (51 tokens)

线性解码为 ARM64 instructions（32-bit aligned）：

| Token 类别 | 数量 | 示例 |
|:---:|:---:|---|
| Special | 5 | PAD, BOS, EOS, MASK, UNK |
| Instruction classes | 20 | load, store, branch, call, arith, logic, simd, system... |
| Operand types | 10 | REG, IMM, MEM_OK, MEM_BAD, TARGET_OK, TARGET_BAD... |
| Validity/Meta | 16 | INVALID_INSN, DATA_LITERAL... |

### 4.3 Byte Path (261 tokens)

Raw byte values [0-255] + 5 special tokens。直接编码，无解码。

### 4.4 统一词汇表

Unified vocab = 358 tokens (5 special + 50 Dalvik + 46 Native + 256 Byte + 1 reserved)
Token type ID: 0=Dalvik, 1=Native, 2=Byte

---

## 5. Pseudo-code BERT

### 5.1 模型配置

| 参数 | 4L/256d (当前) | 8L/512d (新) | 12L/768d (可选) |
|------|:---:|:---:|:---:|
| Layers | 4 | 8 | 12 |
| Hidden dim | 256 | 512 | 768 |
| Heads | 8 | 8 | 12 |
| Intermediate | 512 | 1024 | 3072 |
| Params | 2.3M | 17.3M | 85.9M |
| Max length | 128 | 128 | 128 |
| Pretrain loss | 0.92 | 0.79 | (待测) |

### 5.2 spMLM 预训练

- **语料**: 8.7M sequences from 1408 benign APKs (AndroZoo 1340 + Happer 68)
- **任务**: Masked Language Modeling (15% random mask, instruction-boundary-aligned)
- **配置**: FP16, batch=32-64, lr=1e-4, 10 epochs
- **输出**: pretrained_bert_v2.pt

### 5.3 冻结 BERT 微调

Fine-tune 时 **BERT 参数冻结**（避免灾难遗忘）。只训练：
- Gated fusion network (~100K params)
- Entry aggregator (~200K params)
- APK MIL head (~66K params)
- Suspicion/normality heads (~1K params)

Total trainable: ~543K params (4L/256d) 或 ~900K params (8L/512d)

---

## 6. 318 维统计特征

沿用 DS-AMIL 的特征设计，分三层：

### Layer 1: byte_summary (274 dim)
- 字节频率直方图 [256]
- Shannon entropy [1]
- Rolling window entropy stats (mean, std, max, min) [4]
- Byte bigram top-20 [13]

### Layer 2: structural_context (14 dim)
- Region 在 entry 中的位置 (relative offset, log size) [4]
- Entry 在 APK 中的位置 (index, total entries, size ratio) [4]
- Compression ratio [1]
- Magic byte indicators (DEX, ELF, ZIP, XML, PNG) [5]

### Layer 3: type_specific (30 dim)
- DEX features (7): string/type/method/field counts (if DEX header)
- ELF features (7): section flags, entry point offset
- Asset features (4): path depth, extension type
- Padding (12): reserved

**实现**: `src/android_packer/features/full_feature_extractor.py`

---

## 7. Gated Fusion

### 7.1 动机

Simple concat (896-dim = 768 BERT + 128 stat) 让 BERT 噪声淹没 stat 信号：
- Concat AUROC = 0.914
- Stat-only = 0.949
- **Gated = 0.948** (恢复到 stat 水平，同时提供 localization)

### 7.2 机制

```python
h_bert = bert_aggregation([h_dalvik | h_native | h_byte])  # [N, 768] → [N, 256]
h_stat = stat_proj(features)                                 # [N, 318] → [N, 256]

# ABMIL-style gating
gate_input = [h_bert | h_stat]                              # [N, 512]
gated = tanh(W1 · gate_input) ⊙ sigmoid(W2 · gate_input)  # [N, 128]
[α_bert, α_stat] = softmax(W_g · gated)                    # [N, 2]

h_fused = α_bert · h_bert + α_stat · h_stat                # [N, 256]
```

**特性**: 每个 region 独立决定 BERT vs stat 的权重。
- 高熵加密 region: gate 倾向 stat (entropy 信号强)
- 代码结构异常 region: gate 倾向 BERT (指令序列信号)

---

## 8. 训练策略

### 8.1 数据组成

| 用途 | 来源 | 数量 |
|------|------|:---:|
| 训练 benign | Happer Origin-16 + APKiD-clean hard benign（F-Droid 24；AndroZoo 50 queued） | 80+ bags |
| 训练 packed | Happer (Ali/Qihoo/Tencent × 15) + Track B (cs3/s5/s6) | 73-91 bags |
| 测试 benign | Happer Origin-18 (app-disjoint) | 15 bags |
| 测试 packed | LOPO held-out packer | 1-18 bags |
| 差分标签 | compute_paired_diff() + inject_labels.jsonl | per-entry scores |

### 8.2 四组份 Loss

```
L = L_bag + 0.5·L_rank + 0.3·L_align + 0.2·L_normality
```

| Loss | 公式 | 作用 |
|------|------|------|
| L_bag | BCE(σ(bag_logit), y) | APK-level 分类 |
| L_rank | max(0, margin + benign_logit - packed_logit) | 强制 packed > benign |
| L_align | KL(attention ∥ softmax(diff_targets/τ)) | 注意力对齐差分标签 |
| L_normality | MSE(normality, 1-diff_targets) | benign entry→1, packed entry→0 |

### 8.3 训练配置

- Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=50)
- Epochs: 50
- Batch: 4 bags (accumulate gradients)
- Max regions/bag: 128 (random subsample)
- Gradient clipping: 1.0
- **BERT: frozen** (no gradient)
- 训练时间: ~10 min/fold on RTX 5060

---

## 9. 评估协议

### 9.1 LOPO (Leave-One-Packer-Out)

7 壳家族轮流 held-out:
- Happer: Ali, Qihoo, Tencent (同 app pool 问题—已声明)
- Track B†: 360, Bangcle, APKProtector, DPT (app-disjoint, 无 leakage)

†Track B benign seeds 不出现在训练 benign 中。

### 9.2 指标

| 层级 | 指标 | 含义 |
|------|------|------|
| APK | AUROC | packed vs benign 区分能力 |
| APK | Detection Rate | score > 0.5 的比例 |
| Entry | Entry AUROC | 定位 packed entry 的能力 |
| Entry | Entry MRR | 第一个 packed entry 排在第几 |
| 性能 | ms/APK | 推理耗时（含特征提取） |

### 9.3 Localization scoring 方法

| 方法 | Score | 来源 |
|------|------|------|
| 1 - normality | L_normality 直接监督 | 最稳定 (MRR=0.54) |
| attention | MIL attention weight | 间接（L_align 监督） |
| suspicion | suspicion head output | 弱（无直接监督） |

---

## 10. 实验审计要点

| 问题 | 状态 | 修复 |
|------|:---:|------|
| Track B benign app-identity leakage | ✅ 已修 | 从训练集移除 |
| 360 fold n=1 | ⚠️ 已注明 | 论文标注参考值 |
| Happer 同 app pool (先天) | ⚠️ 已注明 | 论文 Discussion |
| 全局 RNG 不确定性 | ✅ 已修 | deterministic seed |
| pretrain v4 过拟合 | ✅ 已识别 | 回退至 v3 配置；增大模型而非 epoch |

---

## 11. 文件索引

| 组件 | 文件 |
|------|------|
| Dalvik decoder | `src/android_packer/decoders/dalvik_decoder.py` |
| Native decoder | `src/android_packer/decoders/native_decoder.py` |
| Unified tokenizer | `src/android_packer/decoders/pseudo_tokenizer.py` |
| BERT model | `src/android_packer/models/pseudo_code_bert.py` |
| Gated fusion encoder | `src/android_packer/models/fusion_encoder.py` |
| Entry aggregator + MIL | `src/android_packer/models/entry_aggregator.py` |
| 318-dim features | `src/android_packer/features/full_feature_extractor.py` |
| Typed region slicer | `src/android_packer/regioning/typed_slicer.py` |
| Differential labels | `src/android_packer/labeling/happer_diff.py` |
| spMLM pretraining | `src/android_packer/training/pretrain_spmlm.py` |
| Pretrain script | `scripts/experiments/run_spmlm_pretrain_v2.py` |
| LOPO evaluation | `scripts/experiments/run_lopo_eval.py` |
| Ablation | `scripts/experiments/run_ablation_lopo.py` |
| Entropy baseline | `scripts/experiments/run_entropy_localization_baseline.py` |
| Synthetic localization | `scripts/experiments/run_synthetic_localization.py` |

---

## 12. 向更大模型扩展

| 模型 | 参数 | 预训练显存 | Fine-tune 显存 | RTX 5060 (8GB) |
|------|:---:|:---:|:---:|:---:|
| 4L/256d | 2.3M | 61 MB | 320 MB | ✅ |
| 8L/512d | 17.3M | 342 MB | 375 MB | ✅ |
| 12L/768d | 85.9M | 1333 MB | 555 MB | ✅ |

所有模型规模均可在 RTX 5060 上完成预训练和推理。瓶颈是 CPU 的 APK 解压时间。
