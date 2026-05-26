#!/bin/bash
# Quick launcher for Pseudo-code BERT v3 (frozen BERT + 4-loss)
# Run after v1 training completes and GPU is freed.
#
# Usage:
#   bash scripts/experiments/run_v3_quick.sh
#
# Check progress (while running):
#   python -c "import json; d=json.load(open('outputs/experiments/pseudo_bert_v3/progress.json')); print(f'Epoch {d[\"epoch\"]}/{d[\"total_epochs\"]} loss={d[\"loss\"]:.4f} sep={d.get(\"logit_separation\",0):.4f}')"

cd "$(dirname "$0")/../.."
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

echo "=== Pseudo-code BERT v3: Frozen BERT + Full 4-loss ==="
echo "Expected runtime: 5-10 minutes on RTX 5060"
echo ""

python scripts/experiments/run_pseudo_bert_v3.py \
    --skip-pretrain \
    --epochs-finetune 50 \
    --bert-layers 4 \
    --bert-dim 256 \
    --max-length 128 \
    --lr 1e-3 \
    --save-every 10 \
    --device cuda

echo ""
echo "=== Done. Results: outputs/experiments/pseudo_bert_v3/results_v3.json ==="
