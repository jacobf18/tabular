#!/usr/bin/env bash
set -euo pipefail

cd /root/tabular/mcpfn

export REPO_DIR=/root/tabular/mcpfn
export PYTHONPATH=/root/tabular/mcpfn/src:${PYTHONPATH:-}
export WANDB_PROJECT=${WANDB_PROJECT:-tabimpute}

python scripts/train_direct_pure_rabel_causal_panel_v2.py \
  --run-name tabimpute-direct-pure-rabel64-causal-panel-v2-hardmix-from-v1best-50k \
  --device cuda \
  --steps 50000 \
  --batch-size 16 \
  --lr 3e-5 \
  --min-lr 7.5e-6 \
  --weight-decay 0.03 \
  --grad-clip-norm 1.0 \
  --warmup-steps 1000 \
  --stable-steps 39000 \
  --decay-steps 10000 \
  --checkpoint-every 5000 \
  --init-checkpoint /root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-direct-pure-rabel64-causal-panel-bs16-from-rank10-100k/best_checkpoint.pth \
  --wandb-project "${WANDB_PROJECT}"
