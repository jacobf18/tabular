#!/usr/bin/env python
"""Compare pre/post-curriculum TabImpute checkpoints against SoftImpute."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from fancyimpute import SoftImpute

from tabimpute.interface import ImputePFN


DEFAULT_ORIGINAL_RUN = Path(
    "/root/tabular/mcpfn/src/tabimpute/workdir/"
    "tabimpute-new-stable_v2-rank_10_10-50x50"
)
DEFAULT_CURRICULUM_RUN = Path(
    "/root/tabular/mcpfn/src/tabimpute/workdir/"
    "tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000"
)


def parse_size(size: str) -> tuple[int, int]:
    if "x" not in size.lower():
        raise argparse.ArgumentTypeError(f"Expected ROWSxCOLS, got {size}")
    rows, cols = size.lower().split("x", maxsplit=1)
    return int(rows), int(cols)


def load_model_config(run_dir: Path) -> dict:
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model" in cfg:
        return cfg["model"]
    if "config" in cfg:
        return cfg["config"]
    raise ValueError(f"Could not find model config in {run_dir / 'config.json'}")


def load_prior_config(run_dir: Path) -> dict:
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return dict(cfg.get("prior", {}))


def make_low_rank(
    rows: int,
    cols: int,
    rank: int,
    missing_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rank > min(rows, cols):
        raise ValueError(f"rank={rank} must be <= min(rows, cols)={min(rows, cols)}")
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(rows, rank))
    right = rng.normal(size=(rank, cols))
    x_true = left @ right
    x_true += 0.05 * rng.normal(size=x_true.shape)
    x_true = x_true.astype(np.float32)
    mask = rng.random(size=x_true.shape) < missing_rate
    x_missing = x_true.copy()
    x_missing[mask] = np.nan
    return x_true, x_missing, mask


def score_imputation(
    x_true: np.ndarray,
    x_imputed: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    err = x_imputed[mask] - x_true[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.std(x_true[mask]))
    nrmse = rmse / denom if denom > 1e-12 else float("nan")
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse}


def summarize_attention(diagnostics: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for layer in diagnostics:
        for axis in ("features", "datapoints"):
            stats = layer[axis]["stats"]
            if stats is not None:
                rows.append({"layer": layer["layer"], "axis": axis, **stats})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def eval_softimpute(
    datasets: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]],
    rank: int,
    max_iters: int,
) -> list[dict[str, object]]:
    rows = []
    for size_label, seed, x_true, x_missing, mask in datasets:
        model = SoftImpute(
            max_rank=rank,
            max_iters=max_iters,
            convergence_threshold=1e-5,
            verbose=False,
        )
        x_imputed = model.fit_transform(x_missing)
        score = score_imputation(x_true, x_imputed, mask)
        rows.append(
            {
                "method": "softimpute",
                "size": size_label,
                "seed": seed,
                "rank": rank,
                **score,
            }
        )
    return rows


def eval_tabimpute(
    name: str,
    checkpoint_path: Path,
    model_config: dict,
    datasets: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]],
    device: str,
    capture_attention: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    imputer = ImputePFN(
        device=device,
        entry_wise_features=False,
        checkpoint_path=str(checkpoint_path),
        json_config={"config": model_config},
        model_family="new_stable",
    )
    rmse_rows = []
    attention_rows = []
    for size_label, seed, x_true, x_missing, mask in datasets:
        x_imputed = imputer.impute(x_missing.copy())
        score = score_imputation(x_true, x_imputed, mask)
        rmse_rows.append(
            {
                "method": name,
                "size": size_label,
                "seed": seed,
                **score,
            }
        )
        if capture_attention:
            diagnostics = imputer.diagnose_attention(x_missing, store_weights=False)
            for row in summarize_attention(diagnostics):
                attention_rows.append(
                    {
                        "method": name,
                        "size": size_label,
                        "seed": seed,
                        **row,
                    }
                )
    del imputer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rmse_rows, attention_rows


def print_rmse_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["size"]), []).append(row)
    print("method\tsize\trmse\tnrmse\tmae")
    for method, size in sorted(groups, key=lambda item: (item[0][0], item[0][1])):
        group = groups[(method, size)]
        print(
            f"{method}\t{size}\t"
            f"{np.mean([r['rmse'] for r in group]):.6f}\t"
            f"{np.mean([r['nrmse'] for r in group]):.6f}\t"
            f"{np.mean([r['mae'] for r in group]):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-run", type=Path, default=DEFAULT_ORIGINAL_RUN)
    parser.add_argument("--curriculum-run", type=Path, default=DEFAULT_CURRICULUM_RUN)
    parser.add_argument("--original-checkpoint", default="checkpoint_30000.pth")
    parser.add_argument("--curriculum-checkpoint", default="checkpoint_45000.pth")
    parser.add_argument("--sizes", nargs="+", type=parse_size, default=[(50, 50), (100, 100), (200, 200)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--missing-rate", type=float)
    parser.add_argument("--softimpute-max-iters", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("/root/tabular/mcpfn/runs/curriculum_eval"))
    parser.add_argument("--skip-attention", action="store_true")
    args = parser.parse_args()

    prior_config = load_prior_config(args.original_run)
    missing_rate = args.missing_rate
    if missing_rate is None:
        missing_rate = float(prior_config.get("p_missing", 0.2))

    datasets = []
    for rows, cols in args.sizes:
        for seed in args.seeds:
            x_true, x_missing, mask = make_low_rank(
                rows=rows,
                cols=cols,
                rank=args.rank,
                missing_rate=missing_rate,
                seed=seed,
            )
            datasets.append((f"{rows}x{cols}", seed, x_true, x_missing, mask))

    all_rmse_rows = []
    all_attention_rows = []

    all_rmse_rows.extend(
        eval_softimpute(
            datasets=datasets,
            rank=args.rank,
            max_iters=args.softimpute_max_iters,
        )
    )

    for name, run_dir, checkpoint_name in [
        ("tabimpute_50x50_rank10_step30000", args.original_run, args.original_checkpoint),
        ("tabimpute_curriculum_100x100_rank10_step45000", args.curriculum_run, args.curriculum_checkpoint),
    ]:
        rmse_rows, attention_rows = eval_tabimpute(
            name=name,
            checkpoint_path=run_dir / checkpoint_name,
            model_config=load_model_config(run_dir),
            datasets=datasets,
            device=args.device,
            capture_attention=not args.skip_attention,
        )
        all_rmse_rows.extend(rmse_rows)
        all_attention_rows.extend(attention_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "low_rank_rmse.csv", all_rmse_rows)
    write_csv(args.output_dir / "attention_stats.csv", all_attention_rows)

    print_rmse_summary(all_rmse_rows)
    print(f"Wrote {args.output_dir / 'low_rank_rmse.csv'}")
    if not args.skip_attention:
        print(f"Wrote {args.output_dir / 'attention_stats.csv'}")


if __name__ == "__main__":
    main()
