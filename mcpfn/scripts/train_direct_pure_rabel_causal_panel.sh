#!/usr/bin/env bash
set -euo pipefail

cd /root/tabular/mcpfn

export REPO_DIR=/root/tabular/mcpfn
export PYTHONPATH=/root/tabular/mcpfn/src:${PYTHONPATH:-}
export WANDB_PROJECT=${WANDB_PROJECT:-tabimpute}

python scripts/train_direct_pure_rabel_causal_panel.py \
  --run-name tabimpute-direct-pure-rabel64-causal-panel-bs16-from-rank10-100k \
  --device cuda \
  --steps 100000 \
  --batch-size 16 \
  --lr 5e-5 \
  --min-lr 1e-5 \
  --weight-decay 0.03 \
  --grad-clip-norm 1.0 \
  --warmup-steps 2000 \
  --stable-steps 78000 \
  --decay-steps 20000 \
  --checkpoint-every 5000 \
  --init-checkpoint /root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-direct-pure-rabel64-rank10-curriculum-160k/checkpoint_160000.pth \
  --wandb-project "${WANDB_PROJECT}"
