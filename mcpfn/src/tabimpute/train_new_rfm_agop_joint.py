"""Step 2 (post-training): joint backbone + RFMHeadAGOP fine-tuning.

Loads a tabimpute backbone (decoder weights stripped) and a warm-start
RFMHeadAGOP from Step 1, unfreezes the backbone, and trains both jointly
on the synthetic prior under kernel-ridge MSE loss.

Conceptually this is "head-swap fine-tuning" / LLM-style post-training:
the pretrained backbone is adapted from "make MLP-decoder bar-distribution
classification work well" to "make AGOP-kernel-ridge work well".

Usage:
    python -m tabimpute.train_new_rfm_agop_joint \
        --base_checkpoint /mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth \
        --warm_start_head_ckpt /root/tabular/mcpfn/src/tabimpute/workdir/agop_head_dwip_v2_t002_step1/head_final.pt \
        --run_name agop_joint_dwip_v2_t002 \
        --max_steps 20000 \
        --batch_size 16 \
        --backbone_lr 3e-5 \
        --head_lr 1e-2 \
        --warmup_steps 500 \
        --iters 1 \
        --checkpoint_every 500
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tabimpute.model.model_new_stable import build_best_transfer_model
from tabimpute.model.rfm_head_agop import RFMHeadAGOP
from tabimpute.prior.training_set_generation import MissingnessPrior


_ARCH_KEYS = {
    "embedding_size", "num_attention_heads", "mlp_hidden_size", "num_layers",
    "num_cls", "use_rope", "rope_base", "rope_fraction",
    "use_absolute_positional_embeddings", "positional_damping_factor",
    "attention_dropout", "ffn_dropout", "drop_path_rate",
    "residual_scale_init", "rms_norm_eps", "embedding_dropout",
}


# Same hardcoded prior as train_new_rfm_agop.py. We're not relying on
# whatever the backbone was originally pretrained against; this is our
# pick for what synthetic distribution to post-train on.
DEFAULT_PRIOR_CFG = {
    "generator_type": "latent_factor",
    "missingness_type": "mcar",
    "config": {
        "apply_feature_warping_prob": 0.0,
        "apply_quantization_prob": 0.0,
        "latent_rank_high": 11,
        "latent_rank_low": 1,
        "latent_slab_sigma": 2.0,
        "latent_spike_p": 0.3,
        "num_cols_high": 50,
        "num_cols_low": 5,
        "num_rows_high": 50,
        "num_rows_low": 10,
        "p_missing": 0.4,
    },
}


def load_sibling_config(checkpoint: Path) -> dict:
    cfg_path = checkpoint.parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"expected config.json next to checkpoint at {cfg_path}")
    with cfg_path.open() as f:
        return json.load(f)


def build_warmup_cosine(optimizer, total_steps, warmup_steps, min_lr_ratio):
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(step):
        s = step + 1
        if warmup_steps > 0 and s <= warmup_steps:
            return s / warmup_steps
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def load_backbone_only(model, checkpoint, device):
    """Load backbone weights from the dwip-style checkpoint. Decoder weights
    are stripped (the MLP decoder is replaced by RFMHeadAGOP)."""
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
    # --- checkpoints ---
    ap.add_argument("--base_checkpoint", type=Path, required=True,
                    help="Backbone warm-start (e.g. dwip_v2-trial_002 ckpt).")
    ap.add_argument("--warm_start_head_ckpt", type=Path, default=None,
                    help="Optional head warm-start (e.g. Step 1's head_final.pt). "
                         "If omitted, head starts from defaults.")
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--work_dir_root", type=Path,
                    default=Path("/root/tabular/mcpfn/src/tabimpute/workdir"))
    # --- training ---
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--backbone_lr", type=float, default=3e-5,
                    help="lr for backbone params (typically much lower than head_lr)")
    ap.add_argument("--head_lr", type=float, default=1e-2,
                    help="lr for the two head scalars (high; 2 params only)")
    ap.add_argument("--backbone_wd", type=float, default=0.02,
                    help="weight decay on backbone")
    ap.add_argument("--head_wd", type=float, default=0.0,
                    help="weight decay on head (usually 0 for the 2 scalars)")
    ap.add_argument("--grad_clip_norm", type=float, default=0.5)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    # --- head/inference settings ---
    ap.add_argument("--num_anchors", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=1,
                    help="AGOP iterations during TRAINING (1 is cheap; gradient "
                         "only flows through last solve anyway with detach_M)")
    ap.add_argument("--lambda_init", type=float, default=1e-2)
    ap.add_argument("--gamma_scale_init", type=float, default=1.0)
    # --- ckpts / validation ---
    ap.add_argument("--checkpoint_every", type=int, default=500)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--val_batches", type=int, default=4,
                    help="number of held-out synthetic batches for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prior_overrides", type=str, default=None)
    # --- wandb ---
    ap.add_argument("--wandb_project", type=str, default="tabimpute",
                    help="wandb project name (set to 'none' or use --no_wandb to disable)")
    ap.add_argument("--no_wandb", action="store_true",
                    help="Disable wandb logging.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_sibling_config(args.base_checkpoint)
    arch = {k: v for k, v in cfg["model"].items() if k in _ARCH_KEYS}

    prior_cfg = dict(DEFAULT_PRIOR_CFG)
    prior_cfg["config"] = dict(DEFAULT_PRIOR_CFG["config"])
    if args.prior_overrides:
        overrides = json.loads(args.prior_overrides)
        prior_cfg["config"].update(overrides)
        print(f"[setup] prior overrides applied: {overrides}")

    print(f"[setup] arch from sibling config.json: {arch}")
    print(f"[setup] prior: generator={prior_cfg['generator_type']} "
          f"miss={prior_cfg['missingness_type']} config={prior_cfg['config']}")

    work_dir = args.work_dir_root / args.run_name
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "run_config.json").write_text(
        json.dumps(vars(args) | {"_arch": arch, "_prior": prior_cfg},
                   default=str, indent=2)
    )

    # ---- wandb init ----
    use_wandb = (
        _HAS_WANDB
        and not args.no_wandb
        and args.wandb_project.lower() not in {"none", ""}
    )
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            dir=str(work_dir),
            config={**vars(args), "_arch": arch, "_prior": prior_cfg},
            mode="online",
        )
        print(f"[setup] wandb: project={args.wandb_project} run={args.run_name}")
    else:
        print("[setup] wandb disabled.")

    device = args.device
    compute_dtype = torch.bfloat16 if "cuda" in device else torch.float32

    # ---- Build backbone, load weights, leave UNFROZEN. ----
    model = build_best_transfer_model(num_outputs=None, **arch)
    n_loaded = load_backbone_only(model, args.base_checkpoint, device=device)
    model = model.to(device=device, dtype=compute_dtype)
    model.train()
    for p in model.parameters():
        p.requires_grad = True
    print(f"[setup] backbone loaded ({n_loaded} tensors), TRAINABLE.")

    # ---- Build head; warm-start from Step 1 if provided. ----
    head = RFMHeadAGOP(
        lambda_init=args.lambda_init,
        gamma_scale_init=args.gamma_scale_init,
        iters=args.iters,
        detach_M=True,
    ).to(device)
    if args.warm_start_head_ckpt is not None:
        warm = torch.load(str(args.warm_start_head_ckpt),
                          map_location="cpu", weights_only=False)
        head.load_state_dict(warm["head"])
        print(f"[setup] head warm-started from {args.warm_start_head_ckpt}: "
              f"lambda={float(head.lam.item()):.4f}, "
              f"gamma_scale={float(torch.exp(head.log_gamma_scale).item()):.3f}")
    else:
        print(f"[setup] head cold-init: "
              f"lambda={float(head.lam.item()):.4f}, "
              f"gamma_scale={float(torch.exp(head.log_gamma_scale).item()):.3f}")
    head.train()

    n_backbone = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"[setup] backbone params: {n_backbone:,}; head params: {n_head}")
    print(f"[setup] iters_train={args.iters} (inference can use higher)")

    # ---- Synthetic data ----
    prior = MissingnessPrior(
        generator_type=prior_cfg["generator_type"],
        missingness_type=prior_cfg["missingness_type"],
        config=prior_cfg["config"],
        batch_size=args.batch_size,
        verbose=False,
        entry_wise_features=False,
    )

    # ---- Two-group optimizer: different LRs for backbone and head ----
    param_groups = [
        {"params": list(model.parameters()), "lr": args.backbone_lr,
         "weight_decay": args.backbone_wd, "name": "backbone"},
        {"params": list(head.parameters()), "lr": args.head_lr,
         "weight_decay": args.head_wd, "name": "head"},
    ]
    try:
        optimizer = torch.optim.AdamW(param_groups, fused=True)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(param_groups)
    scheduler = build_warmup_cosine(
        optimizer, args.max_steps, args.warmup_steps, args.min_lr_ratio,
    )

    # ---- Held-out val batches (fixed) ----
    val_rng = np.random.Generator(np.random.PCG64(args.seed + 12345))
    val_batches = []
    val_prior = MissingnessPrior(
        generator_type=prior_cfg["generator_type"],
        missingness_type=prior_cfg["missingness_type"],
        config=prior_cfg["config"],
        batch_size=args.batch_size,
        verbose=False,
        entry_wise_features=False,
    )
    for _ in range(args.val_batches):
        (vx, vy, _, _, _), _ = val_prior.get_batch()
        val_batches.append((vx.detach().clone(), vy.detach().clone()))

    @torch.no_grad()
    def _run_validation():
        model.eval()
        head.eval()
        total_loss = 0.0
        total_count = 0
        for (vx, vy) in val_batches:
            vx_d = vx.to(device=device, dtype=compute_dtype)
            vy_d = vy.to(device=device, dtype=compute_dtype)
            if not torch.isfinite(vy_d).all():
                continue
            feats = model(vx_d, return_embeddings=True)
            obs = ~torch.isnan(vx_d)
            loss, _ = head(feats, vy_d, obs, num_anchors=args.num_anchors)
            if torch.isfinite(loss):
                total_loss += float(loss.item()) * vx_d.shape[0]
                total_count += vx_d.shape[0]
        model.train()
        head.train()
        return total_loss / max(total_count, 1)

    # ---- Train ----
    metrics_path = work_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("w")
    print(f"[setup] writing per-step metrics to {metrics_path}")

    best_val = float("inf")
    pbar = tqdm(range(1, args.max_steps + 1), desc=args.run_name)
    for step in pbar:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        (batch_X, batch_target, _, _, _), _ = prior.get_batch()
        batch_X = batch_X.to(device=device, dtype=compute_dtype)
        batch_target = batch_target.to(device=device, dtype=compute_dtype)

        if not torch.isfinite(batch_target).all():
            continue

        feats = model(batch_X, return_embeddings=True)
        observed = ~torch.isnan(batch_X)
        loss, mae = head(feats, batch_target, observed, num_anchors=args.num_anchors)

        if not torch.isfinite(loss):
            tqdm.write(f"[warn] step {step}: non-finite loss, skipping backward")
            continue

        loss.backward()

        # Guard: skip the step if any gradient is non-finite. Even ONE NaN/Inf
        # poisons Adam state irreversibly (m_t, v_t become NaN, params become
        # NaN, then forward forever produces NaN).
        all_params = list(model.parameters()) + list(head.parameters())
        bad_grad = False
        for p in all_params:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad_grad = True
                break
        if bad_grad:
            tqdm.write(f"[warn] step {step}: non-finite gradient, skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        if args.grad_clip_norm and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        log = {
            "step": step,
            "loss_mse_std": float(loss.item()),
            "mae_std": float(mae.item()),
            "lambda": float(head.lam.item()),
            "gamma_scale": float(torch.exp(head.log_gamma_scale).item()),
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "time_s": elapsed,
        }

        # Periodic validation.
        if step % args.val_every == 0 or step == args.max_steps:
            val_loss = _run_validation()
            log["val_loss"] = val_loss
            if val_loss < best_val:
                best_val = val_loss
                ck = {
                    "step": step,
                    "model": model.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                    "_arch": arch,
                    "_prior": prior_cfg,
                    "val_loss": val_loss,
                }
                torch.save(ck, work_dir / "checkpoint_best.pth")
                tqdm.write(f"[best] step {step}: val_loss={val_loss:.4f}")

        metrics_f.write(json.dumps(log) + "\n")
        metrics_f.flush()
        if use_wandb:
            wandb.log(log, step=step)
        if step % 10 == 0:
            pbar.set_postfix(
                loss=f"{log['loss_mse_std']:.4f}",
                mae=f"{log['mae_std']:.4f}",
                lam=f"{log['lambda']:.3g}",
                gs=f"{log['gamma_scale']:.3g}",
                blr=f"{log['backbone_lr']:.1e}",
                hlr=f"{log['head_lr']:.1e}",
            )

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
    final = work_dir / "checkpoint_final.pth"
    torch.save({
        "step": args.max_steps,
        "model": model.state_dict(),
        "head": head.state_dict(),
        "args": vars(args),
        "_arch": arch,
        "_prior": prior_cfg,
    }, final)
    print(f"\n[done] final -> {final}")
    print(f"[done] best  -> {work_dir / 'checkpoint_best.pth'} (val_loss={best_val:.4f})")
    print(f"[done] run_dir = {work_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
