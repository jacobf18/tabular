"""Phase 1 (M2b): joint backbone + RFM-head fine-tuning.

Loads the dwip backbone (skipping its bar-head decoder), attaches an RFMHead,
and trains both jointly under the closed-form kernel-ridge loss on the
synthetic prior described by the checkpoint's sibling config.json.

Usage:
    python -m tabimpute.train_new_rfm \
        --base_checkpoint /root/tabular/mcpfn/src/tabimpute/workdir/dwip_v3r1_t002_ctl-trial_001/checkpoint_17000.pth \
        --run_name m2b_t002_ctl_step17k \
        --max_steps 5000 --batch_size 4 --lr 1e-5 \
        --warmup_steps 250 --num_anchors 1024 --lambda_init 10.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tabimpute.model.model_new_stable import build_best_transfer_model
from tabimpute.model.rfm_head import RFMHead
from tabimpute.prior.training_set_generation import MissingnessPrior


_ARCH_KEYS = {
    "embedding_size", "num_attention_heads", "mlp_hidden_size", "num_layers",
    "num_cls", "use_rope", "rope_base", "rope_fraction",
    "use_absolute_positional_embeddings", "positional_damping_factor",
    "attention_dropout", "ffn_dropout", "drop_path_rate",
    "residual_scale_init", "rms_norm_eps", "embedding_dropout",
}


def load_sibling_config(checkpoint: Path) -> dict:
    cfg_path = checkpoint.parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"expected config.json next to checkpoint at {cfg_path}")
    with cfg_path.open() as f:
        return json.load(f)


def build_warmup_cosine(optimizer, total_steps: int, warmup_steps: int, min_lr_ratio: float):
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(step: int) -> float:
        s = step + 1
        if warmup_steps > 0 and s <= warmup_steps:
            return s / warmup_steps
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def load_backbone_only(model: nn.Module, checkpoint: Path, device: str):
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt.get("state_dict"))
    sd = {k: v for k, v in sd.items() if not k.startswith("decoder.")}
    res = model.load_state_dict(sd, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"unexpected keys when loading backbone: {res.unexpected_keys}")
    if res.missing_keys:
        raise RuntimeError(f"missing keys when loading backbone: {res.missing_keys}")
    return len(sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", type=Path, required=True)
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--work_dir_root", type=Path,
                    default=Path("/root/tabular/mcpfn/src/tabimpute/workdir"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    ap.add_argument("--grad_clip_norm", type=float, default=0.5)
    ap.add_argument("--warmup_steps", type=int, default=250)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--num_anchors", type=int, default=1024)
    ap.add_argument("--lambda_init", type=float, default=10.0)
    ap.add_argument("--agop_rank", type=int, default=None,
                    help="If set, use M=L L^T with L of shape (embed_dim, agop_rank); else dot-product kernel")
    ap.add_argument("--checkpoint_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prior_overrides", type=str, default=None,
                    help="JSON dict; overrides prior.config keys (e.g. '{\"latent_rank_high\":30,\"num_rows_high\":100}')")
    ap.add_argument("--from_scratch", action="store_true",
                    help="Skip backbone resume; init transformer from random. Still reads --base_checkpoint's sibling config.json for arch/prior.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_sibling_config(args.base_checkpoint)
    arch = {k: v for k, v in cfg["model"].items() if k in _ARCH_KEYS}
    prior_cfg = cfg["prior"]
    if args.prior_overrides:
        overrides = json.loads(args.prior_overrides)
        prior_cfg = {**prior_cfg, "config": {**prior_cfg["config"], **overrides}}
        print(f"[setup] prior overrides applied: {overrides}")
    print(f"[setup] arch overrides from config.json: {arch}")
    print(f"[setup] prior: generator={prior_cfg['generator_type']} miss={prior_cfg['missingness_type']} "
          f"config={prior_cfg['config']}")

    work_dir = args.work_dir_root / args.run_name
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "run_config.json").write_text(json.dumps(vars(args) | {"_arch": arch, "_prior": prior_cfg},
                                                         default=str, indent=2))

    device = args.device
    compute_dtype = torch.bfloat16 if "cuda" in device else torch.float32

    model = build_best_transfer_model(num_outputs=None, **arch)
    if args.from_scratch:
        model = model.to(device=device, dtype=compute_dtype)
        model.train()
        print(f"[setup] from-scratch: random init (decoder skipped); "
              f"{sum(p.numel() for p in model.parameters()):,} params")
    else:
        n_loaded = load_backbone_only(model, args.base_checkpoint, device=device)
        model = model.to(device=device, dtype=compute_dtype)
        model.train()
        print(f"[setup] backbone-only resume: {n_loaded} tensors loaded; decoder skipped")

    head = RFMHead(
        lambda_init=args.lambda_init,
        agop_rank=args.agop_rank,
        embed_dim=arch["embedding_size"] if args.agop_rank else None,
    ).to(device)
    head.train()
    if args.agop_rank:
        print(f"[setup] AGOP head: rank={args.agop_rank}, "
              f"embed_dim={arch['embedding_size']}, L params={head.L.numel():,}")

    n_backbone_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] backbone params: {n_backbone_params:,}; head params: {sum(p.numel() for p in head.parameters())}")

    prior = MissingnessPrior(
        generator_type=prior_cfg["generator_type"],
        missingness_type=prior_cfg["missingness_type"],
        config=prior_cfg["config"],
        batch_size=args.batch_size,
        verbose=False,
        entry_wise_features=False,
    )

    optim_params = list(model.parameters()) + list(head.parameters())
    try:
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr,
                                      weight_decay=args.weight_decay, fused=True)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(optim_params, lr=args.lr,
                                      weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine(optimizer, args.max_steps, args.warmup_steps, args.min_lr_ratio)

    metrics_path = work_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w")
    print(f"[setup] writing per-step metrics to {metrics_path}")

    pbar = tqdm(range(1, args.max_steps + 1), desc=args.run_name)
    for step in pbar:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        (batch_X, batch_target, _, _, _), _ = prior.get_batch()
        batch_X = batch_X.to(device=device, dtype=compute_dtype)
        batch_target = batch_target.to(device=device, dtype=compute_dtype)

        if not torch.isfinite(batch_target).all():
            print(f"[warn] step {step}: non-finite target, skipping")
            continue

        feats = model(batch_X, return_embeddings=True)         # (B, R, C, D)
        observed = ~torch.isnan(batch_X)
        loss, mae = head(feats, batch_target, observed, num_anchors=args.num_anchors)

        if not torch.isfinite(loss):
            print(f"[warn] step {step}: non-finite loss, skipping backward")
            continue

        loss.backward()
        if args.grad_clip_norm and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(optim_params, args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        log = {
            "step": step,
            "loss_mse_std": float(loss.item()),
            "mae_std": float(mae.item()),
            "lambda": float(head.lam.item()),
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": elapsed,
        }
        metrics_f.write(json.dumps(log) + "\n")
        metrics_f.flush()
        if step % 10 == 0:
            pbar.set_postfix(loss=f"{log['loss_mse_std']:.4f}",
                             mae=f"{log['mae_std']:.4f}",
                             lam=f"{log['lambda']:.3g}",
                             lr=f"{log['lr']:.2e}")

        if step % args.checkpoint_every == 0 or step == args.max_steps:
            ck = {
                "step": step,
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "_arch": arch,
                "_prior": prior_cfg,
            }
            ck_path = work_dir / f"checkpoint_{step}.pth"
            torch.save(ck, ck_path)
            tqdm.write(f"[ckpt] step {step} -> {ck_path}")

    metrics_f.close()
    print(f"\n[done] run_dir = {work_dir}")


if __name__ == "__main__":
    main()
