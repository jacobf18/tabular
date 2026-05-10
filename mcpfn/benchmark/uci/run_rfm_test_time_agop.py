"""Test-time AGOP: per UCI patch, fit a fresh AGOP head on the patch's observed
cells via leave-K-out cross-validation, then score on the patch's originally-
missing cells. Backbone is fully frozen.

Per patch:
  1. Forward the frozen backbone -> Phi (R, C, D).
  2. Identify observed cells (non-NaN in missing.npy) and missing cells.
  3. Init fresh AGOPHead (L random ~ N(0, 1/r); log_lambda init from CLI).
  4. Inner loop for T steps:
       a. Random split of observed -> fit_anchors (1-q frac) + heldout (q frac).
       b. Closed-form solve on fit_anchors, predict heldout, MSE loss.
       c. Backprop into L and log_lambda only (Phi is detached).
  5. Final solve: use all observed cells (capped at num_anchors) as anchors,
     predict the originally-missing cells, score RMSE/MAE.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_rfm_frozen import load_backbone, random_subsample, standardize
from tabimpute.model.rfm_head import RFMHead


def per_patch_test_time_fit(
    Phi: torch.Tensor,         # (n_cells, D), already detached, fp32
    target: torch.Tensor,      # (n_cells,), fp32
    obs_idx: torch.Tensor,     # (n_obs,)
    mis_idx: torch.Tensor,     # (n_mis,)
    agop_rank: int,
    embed_dim: int,
    lambda_init: float,
    inner_steps: int,
    inner_lr: float,
    heldout_frac: float,
    num_anchors: int,
    device: str,
    rng_seed: int,
):
    head = RFMHead(lambda_init=lambda_init, agop_rank=agop_rank, embed_dim=embed_dim).to(device)
    head.train()
    optim = torch.optim.Adam(head.parameters(), lr=inner_lr)

    n_obs = int(obs_idx.numel())
    n_held = max(1, int(round(heldout_frac * n_obs)))
    g = torch.Generator(device=device).manual_seed(rng_seed)

    for _ in range(inner_steps):
        perm = torch.randperm(n_obs, device=device, generator=g)
        held = obs_idx[perm[:n_held]]
        fit = obs_idx[perm[n_held:]]
        if int(fit.numel()) > num_anchors:
            sub = torch.randperm(int(fit.numel()), device=device, generator=g)[:num_anchors]
            fit = fit[sub]

        Phi_a = Phi.index_select(0, fit)            # (S, D)
        Phi_h = Phi.index_select(0, held)            # (H, D)
        y_a = target.index_select(0, fit)            # (S,)
        y_h = target.index_select(0, held)           # (H,)

        L = head.L
        Phi_aL = Phi_a @ L
        Phi_hL = Phi_h @ L
        eye = torch.eye(int(Phi_a.shape[0]), device=device, dtype=Phi.dtype)
        K_ss = Phi_aL @ Phi_aL.transpose(-1, -2) + head.lam * eye
        K_ts = Phi_hL @ Phi_aL.transpose(-1, -2)
        alpha = torch.linalg.solve(K_ss, y_a.unsqueeze(-1)).squeeze(-1)
        y_hat_h = K_ts @ alpha
        loss = (y_hat_h - y_h).pow(2).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    head.eval()
    with torch.no_grad():
        anchor_idx = obs_idx
        if int(anchor_idx.numel()) > num_anchors:
            sub = torch.randperm(int(anchor_idx.numel()), device=device, generator=g)[:num_anchors]
            anchor_idx = anchor_idx[sub]
        Phi_s = Phi.index_select(0, anchor_idx)
        Phi_t = Phi.index_select(0, mis_idx)
        y_s = target.index_select(0, anchor_idx)
        y_t = target.index_select(0, mis_idx)

        L = head.L
        Phi_sL = Phi_s @ L
        Phi_tL = Phi_t @ L
        eye = torch.eye(int(Phi_s.shape[0]), device=device, dtype=Phi.dtype)
        K_ss = Phi_sL @ Phi_sL.transpose(-1, -2) + head.lam * eye
        K_ts = Phi_tL @ Phi_sL.transpose(-1, -2)
        alpha = torch.linalg.solve(K_ss, y_s.unsqueeze(-1)).squeeze(-1)
        y_hat_t = K_ts @ alpha
        sq = (y_hat_t - y_t).pow(2).sum().item()
        ab = (y_hat_t - y_t).abs().sum().item()
        return float(sq), float(ab), int(y_t.numel()), float(head.lam.item()), float(loss.item())


def evaluate_dataset(
    model, missing, true, args, device, dtype,
):
    R, C = missing.shape
    X_std, Y_std, _, _ = standardize(missing, true)

    rng = np.random.default_rng(args.seed)
    sum_sq = 0.0
    sum_ab = 0.0
    total = 0
    inner_losses = []
    skipped = 0

    for i in range(args.num_subsamples):
        rows, cols = random_subsample(rng, R, C, args.max_rows, args.max_cols)
        x_sub = X_std[np.ix_(rows, cols)]
        y_sub = Y_std[np.ix_(rows, cols)]
        observed = ~np.isnan(x_sub)
        n_obs = int(observed.sum())
        n_mis = int((~observed).sum())
        if n_obs < args.min_obs_for_fit or n_mis == 0:
            skipped += 1
            continue

        x_t = torch.from_numpy(x_sub).to(device=device, dtype=dtype).unsqueeze(0)
        y_t = torch.from_numpy(y_sub).to(device=device, dtype=dtype).unsqueeze(0)
        with torch.no_grad():
            feats = model(x_t, return_embeddings=True)              # (1, R, C, D)
        Phi = feats.squeeze(0).reshape(-1, feats.shape[-1]).float().detach()  # (R*C, D)
        target = y_t.squeeze(0).reshape(-1).float()                            # (R*C,)
        obs_flat = torch.from_numpy(observed.reshape(-1)).to(device)
        obs_idx = torch.nonzero(obs_flat, as_tuple=False).squeeze(-1)
        mis_idx = torch.nonzero(~obs_flat, as_tuple=False).squeeze(-1)

        sq, ab, n, lam, last_loss = per_patch_test_time_fit(
            Phi=Phi, target=target,
            obs_idx=obs_idx, mis_idx=mis_idx,
            agop_rank=args.agop_rank, embed_dim=int(Phi.shape[-1]),
            lambda_init=args.lambda_init,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            heldout_frac=args.heldout_frac,
            num_anchors=args.num_anchors,
            device=device, rng_seed=args.seed * 1000 + i,
        )
        sum_sq += sq
        sum_ab += ab
        total += n
        inner_losses.append(last_loss)

    if total == 0:
        return None
    return {
        "rmse_std": math.sqrt(sum_sq / total),
        "mae_std": sum_ab / total,
        "n_missing": total,
        "skipped": skipped,
        "n_subsamples_used": args.num_subsamples - skipped,
        "mean_inner_final_loss": float(np.mean(inner_losses)) if inner_losses else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data_root", type=Path,
                    default=Path("/root/tabular/mcpfn/benchmark/datasets/uci"))
    ap.add_argument("--pattern", default="MCAR")
    ap.add_argument("--missingness", type=float, default=0.4)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--num_subsamples", type=int, default=20)
    ap.add_argument("--max_rows", type=int, default=50)
    ap.add_argument("--max_cols", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_anchors", type=int, default=1024)
    ap.add_argument("--agop_rank", type=int, default=32)
    ap.add_argument("--lambda_init", type=float, default=10.0)
    ap.add_argument("--inner_steps", type=int, default=100)
    ap.add_argument("--inner_lr", type=float, default=5e-2)
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--min_obs_for_fit", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"[setup] loading frozen backbone from {args.checkpoint}")
    model, _head_sd = load_backbone(args.checkpoint, device=args.device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    datasets = args.datasets or sorted(d.name for d in args.data_root.iterdir() if d.is_dir())

    rows = {}
    rmses = []
    for name in datasets:
        cfg = args.data_root / name / args.pattern / f"missingness-{args.missingness}" / f"repeat-{args.repeat}"
        miss_p = cfg / "missing.npy"
        true_p = cfg / "true.npy"
        if not miss_p.is_file() or not true_p.is_file():
            continue
        missing = np.load(miss_p).astype(np.float32)
        true = np.load(true_p).astype(np.float32)
        out = evaluate_dataset(model, missing, true, args, args.device, dtype)
        if out is None:
            continue
        out["shape"] = list(missing.shape)
        rows[name] = out
        if math.isfinite(out["rmse_std"]):
            rmses.append(out["rmse_std"])
        print(f"  {name:30s} shape={tuple(missing.shape)} rmse_std={out['rmse_std']:.4f}  "
              f"mae_std={out['mae_std']:.4f}  inner_final_loss={out['mean_inner_final_loss']:.3f}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "pattern": args.pattern,
        "missingness": args.missingness,
        "agop_rank": args.agop_rank,
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "heldout_frac": args.heldout_frac,
        "num_anchors": args.num_anchors,
        "results": rows,
        "median_rmse_std": float(np.median(rmses)) if rmses else float("nan"),
        "mean_rmse_std": float(np.mean(rmses)) if rmses else float("nan"),
        "n_datasets": len(rmses),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] median={summary['median_rmse_std']:.4f}  mean={summary['mean_rmse_std']:.4f}  "
          f"n={summary['n_datasets']}")
    print(f"[summary] wrote {args.out}")


if __name__ == "__main__":
    main()
