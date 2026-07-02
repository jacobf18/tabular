"""Phase 0.5: frozen-everything RFM eval on UCI.

Loads a dwip backbone, drops the bar head, runs each UCI matrix through 50x50
random subsamples, fits a closed-form kernel-ridge head per subsample, and
scores RMSE/MAE on the masked cells (in column-standardized space).

Usage:
    python run_rfm_frozen.py \
        --checkpoint /root/tabular/mcpfn/src/tabimpute/workdir/dwip_v3r1_t002_ctl-trial_001/checkpoint_17000.pth \
        --pattern MCAR --missingness 0.4 --repeat 0 \
        --num_subsamples 20 --num_anchors 1024 --lambda_init 1e-2 \
        --out runs/rfm_frozen_t002_ctl_step17k.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from tabimpute.model.model_new_stable import build_best_transfer_model
from tabimpute.model.rfm_head import RFMHead


_ARCH_KEYS = {
    "embedding_size", "num_attention_heads", "mlp_hidden_size", "num_layers",
    "num_cls", "use_rope", "rope_base", "rope_fraction",
    "use_absolute_positional_embeddings", "positional_damping_factor",
    "attention_dropout", "ffn_dropout", "drop_path_rate",
    "residual_scale_init", "rms_norm_eps", "embedding_dropout",
}


def load_arch(config_path: Path) -> dict:
    with config_path.open() as f:
        full = json.load(f)
    return {k: v for k, v in full["model"].items() if k in _ARCH_KEYS}


def load_backbone(checkpoint: Path, device: str = "cuda", dtype=torch.bfloat16):
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    # Arch from the checkpoint itself (M2b path) or from sibling config.json (dwip path)
    if "_arch" in ckpt:
        arch = ckpt["_arch"]
    else:
        arch = load_arch(checkpoint.parent / "config.json")
    model = build_best_transfer_model(num_outputs=None, **arch).to(device).to(dtype)
    sd = ckpt.get("model", ckpt.get("state_dict"))
    sd = {k: v for k, v in sd.items() if not k.startswith("decoder.")}
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(
            f"backbone load mismatch: missing={res.missing_keys} unexpected={res.unexpected_keys}"
        )
    model.eval()
    head_sd = ckpt.get("head")  # present for M2b checkpoints; None for dwip
    return model, head_sd


def standardize(missing: np.ndarray, true: np.ndarray):
    mu = np.nanmean(missing, axis=0)
    sigma = np.nanstd(missing, axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return (missing - mu) / sigma, (true - mu) / sigma, mu, sigma


def random_subsample(rng: np.random.Generator, R: int, C: int, max_R: int, max_C: int):
    rows = rng.permutation(R)[:min(R, max_R)]
    cols = rng.permutation(C)[:min(C, max_C)]
    rows.sort()
    cols.sort()
    return rows, cols


@torch.no_grad()
def evaluate_dataset(
    model,
    head,
    missing: np.ndarray,
    true: np.ndarray,
    device: str,
    dtype,
    num_subsamples: int,
    num_anchors: int,
    max_rows: int,
    max_cols: int,
    seed: int,
):
    R, C = missing.shape
    X_std, Y_std, _, _ = standardize(missing, true)

    rng = np.random.default_rng(seed)
    sum_sq_err = 0.0
    sum_abs_err = 0.0
    total_missing = 0
    skipped = 0

    for _ in range(num_subsamples):
        rows, cols = random_subsample(rng, R, C, max_rows, max_cols)
        x_sub = X_std[np.ix_(rows, cols)]
        y_sub = Y_std[np.ix_(rows, cols)]

        observed = ~np.isnan(x_sub)
        n_obs = int(observed.sum())
        n_mis = int((~observed).sum())
        if n_obs == 0 or n_mis == 0:
            skipped += 1
            continue

        x_t = torch.from_numpy(x_sub).to(device=device, dtype=dtype).unsqueeze(0)
        y_t = torch.from_numpy(y_sub).to(device=device, dtype=dtype).unsqueeze(0)
        obs_t = torch.from_numpy(observed).to(device=device).unsqueeze(0)

        feats = model(x_t, return_embeddings=True)             # (1, r, c, D)
        loss_mse, mae = head(feats, y_t, obs_t, num_anchors=num_anchors)

        # accumulate sums (head returns means; convert back to sums via n_mis)
        sum_sq_err += float(loss_mse.item()) * n_mis
        sum_abs_err += float(mae.item()) * n_mis
        total_missing += n_mis

    if total_missing == 0:
        return {"rmse_std": float("nan"), "mae_std": float("nan"),
                "n_missing": 0, "n_subsamples": num_subsamples, "skipped": skipped}

    rmse = math.sqrt(sum_sq_err / total_missing)
    mae = sum_abs_err / total_missing
    return {
        "rmse_std": rmse,
        "mae_std": mae,
        "n_missing": total_missing,
        "n_subsamples": num_subsamples - skipped,
        "skipped": skipped,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data_root", type=Path,
                   default=Path("/root/tabular/mcpfn/benchmark/datasets/uci"))
    p.add_argument("--pattern", default="MCAR", choices=["MCAR", "MAR", "MNAR"])
    p.add_argument("--missingness", type=float, default=0.4)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--datasets", nargs="*", default=None,
                   help="dataset names; default = all under data_root")
    p.add_argument("--num_subsamples", type=int, default=20)
    p.add_argument("--num_anchors", type=int, default=1024)
    p.add_argument("--lambda_init", type=float, default=1e-2)
    p.add_argument("--max_rows", type=int, default=50)
    p.add_argument("--max_cols", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--head", choices=["rfm", "agop"], default="rfm",
                   help="Kernel head: 'rfm' (linear + low-rank L) or 'agop' "
                        "(Laplace + closed-form AGOP iteration).")
    p.add_argument("--iters", type=int, default=3,
                   help="AGOP iterations per patch (only used when --head agop).")
    p.add_argument("--gamma_scale_init", type=float, default=1.0,
                   help="Initial multiplier on adaptive bandwidth (only used when --head agop).")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model, head_sd = load_backbone(args.checkpoint, device=args.device, dtype=dtype)
    if args.head == "agop":
        from tabimpute.model.rfm_head_agop import RFMHeadAGOP
        head = RFMHeadAGOP(
            lambda_init=args.lambda_init,
            gamma_scale_init=args.gamma_scale_init,
            iters=args.iters,
        ).to(args.device)
        print(f"[setup] AGOP head: iters={args.iters}, "
              f"lambda={float(head.lam.item()):.4f}, "
              f"gamma_scale={float(torch.exp(head.log_gamma_scale).item()):.3f}")
    elif head_sd is not None and "L" in head_sd:
        embed_dim, agop_rank = head_sd["L"].shape
        head = RFMHead(lambda_init=args.lambda_init,
                       agop_rank=agop_rank, embed_dim=embed_dim).to(args.device)
        head.load_state_dict(head_sd)
        print(f"[setup] loaded AGOP head: rank={agop_rank}, lambda={float(head.lam.item()):.4f}")
    else:
        head = RFMHead(lambda_init=args.lambda_init).to(args.device)
        if head_sd is not None:
            head.load_state_dict(head_sd)
            print(f"[setup] loaded plain head: lambda={float(head.lam.item()):.4f}")
    head.eval()

    if args.datasets is None:
        datasets = sorted(d.name for d in args.data_root.iterdir() if d.is_dir())
    else:
        datasets = args.datasets

    results = {}
    rmses = []
    for name in datasets:
        cfg_dir = args.data_root / name / args.pattern / f"missingness-{args.missingness}" / f"repeat-{args.repeat}"
        miss_p = cfg_dir / "missing.npy"
        true_p = cfg_dir / "true.npy"
        if not miss_p.is_file() or not true_p.is_file():
            results[name] = {"error": f"missing files at {cfg_dir}"}
            continue
        missing = np.load(miss_p).astype(np.float32)
        true = np.load(true_p).astype(np.float32)
        out = evaluate_dataset(
            model, head, missing, true,
            device=args.device, dtype=dtype,
            num_subsamples=args.num_subsamples,
            num_anchors=args.num_anchors,
            max_rows=args.max_rows, max_cols=args.max_cols,
            seed=args.seed,
        )
        out["shape"] = list(missing.shape)
        results[name] = out
        if not math.isnan(out["rmse_std"]):
            rmses.append(out["rmse_std"])
        print(f"  {name:30s} shape={tuple(missing.shape)} rmse_std={out['rmse_std']:.4f} "
              f"mae_std={out['mae_std']:.4f} n_missing={out['n_missing']}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "pattern": args.pattern,
        "missingness": args.missingness,
        "repeat": args.repeat,
        "num_subsamples": args.num_subsamples,
        "num_anchors": args.num_anchors,
        "lambda_init": args.lambda_init,
        "head": args.head,
        "iters": args.iters if args.head == "agop" else None,
        "gamma_scale_init": args.gamma_scale_init if args.head == "agop" else None,
        "max_rows": args.max_rows,
        "max_cols": args.max_cols,
        "seed": args.seed,
        "results": results,
        "mean_rmse_std": float(np.mean(rmses)) if rmses else float("nan"),
        "median_rmse_std": float(np.median(rmses)) if rmses else float("nan"),
        "n_datasets": len(rmses),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] mean_rmse_std={summary['mean_rmse_std']:.4f}  "
          f"median_rmse_std={summary['median_rmse_std']:.4f}  n={summary['n_datasets']}")
    print(f"[summary] wrote {args.out}")


if __name__ == "__main__":
    main()
