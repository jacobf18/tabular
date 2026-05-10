"""Phase B: sweep lambda on the frozen RFM head over UCI.

Loads the backbone once, then evaluates at each lambda value. Reuses the
helpers in run_rfm_frozen.py so the protocol stays identical.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from run_rfm_frozen import load_backbone, evaluate_dataset
from tabimpute.model.rfm_head import RFMHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data_root", type=Path,
                   default=Path("/root/tabular/mcpfn/benchmark/datasets/uci"))
    p.add_argument("--pattern", default="MCAR")
    p.add_argument("--missingness", type=float, default=0.4)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--lambdas", nargs="*", type=float,
                   default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    p.add_argument("--num_subsamples", type=int, default=20)
    p.add_argument("--num_anchors", type=int, default=1024)
    p.add_argument("--max_rows", type=int, default=50)
    p.add_argument("--max_cols", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(f"[setup] loading backbone from {args.checkpoint}")
    model, _ = load_backbone(args.checkpoint, device=args.device, dtype=dtype)

    datasets = sorted(d.name for d in args.data_root.iterdir() if d.is_dir())

    sweep_results = {}
    medians = {}
    for lam in args.lambdas:
        head = RFMHead(lambda_init=lam).to(args.device)
        head.eval()

        per_dataset = {}
        rmses = []
        for name in datasets:
            cfg = args.data_root / name / args.pattern / f"missingness-{args.missingness}" / f"repeat-{args.repeat}"
            miss_p = cfg / "missing.npy"
            true_p = cfg / "true.npy"
            if not miss_p.is_file() or not true_p.is_file():
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
            per_dataset[name] = out
            if math.isfinite(out["rmse_std"]):
                rmses.append(out["rmse_std"])

        med = float(np.median(rmses)) if rmses else float("nan")
        mean = float(np.mean(rmses)) if rmses else float("nan")
        medians[lam] = med
        sweep_results[lam] = {
            "per_dataset": per_dataset,
            "median_rmse_std": med,
            "mean_rmse_std": mean,
            "n_datasets": len(rmses),
        }
        print(f"  lambda={lam:>10.4g}  median_rmse_std={med:.4f}  mean_rmse_std={mean:.4f}  n={len(rmses)}")

    out = {
        "checkpoint": str(args.checkpoint),
        "pattern": args.pattern,
        "missingness": args.missingness,
        "repeat": args.repeat,
        "num_subsamples": args.num_subsamples,
        "num_anchors": args.num_anchors,
        "max_rows": args.max_rows,
        "max_cols": args.max_cols,
        "seed": args.seed,
        "sweep": {f"{lam}": v for lam, v in sweep_results.items()},
        "summary": {f"{lam}": medians[lam] for lam in args.lambdas},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    # final ranking
    print("\n=== sweep summary ===")
    for lam in sorted(args.lambdas, key=lambda l: medians.get(l, float("inf"))):
        print(f"  lambda={lam:>10.4g}  median_rmse_std={medians[lam]:.4f}")
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
