#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DWIP_V3R1_MANIFEST_PATH = Path(
    "/root/tabular/mcpfn/src/tabimpute/workdir/"
    "dwip_v3r1_9run_downstream_eval_manifest_20260415.json"
)

MCPFN_SHARED_KWARGS = {
    "device": "cuda",
    "nhead": 2,
    "preprocessors": None,
    "entry_wise_features": False,
    "max_num_rows": 200,
    "max_num_chunks": 4,
}

ROUND2_T2_CONFIG = {
    "config": {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "mlp_hidden_size": 1536,
        "num_layers": 8,
        "num_outputs": 5000,
        "num_cls": 12,
        "use_rope": True,
        "rope_base": 10000.0,
        "rope_fraction": 0.5,
        "use_absolute_positional_embeddings": False,
        "positional_damping_factor": 0.1,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.05,
        "drop_path_rate": 0.01,
        "residual_scale_init": 0.5,
        "embedding_dropout": 0.0,
        "rms_norm_eps": 1e-6,
    }
}

MASTERS_MCAR_CHECKPOINT = "/root/step-117000.ckpt"


ROUND2_T2_SPECS: list[dict] = [
    {
        "name": "round2_t2-rank_1_30",
        "model_family": "new_stable",
        "checkpoint_path": "/root/tabular/mcpfn/src/tabimpute/workdir/"
                           "tabimpute-new-stable_v2-rank_1_30-50x50/checkpoint_30000.pth",
        "json_config": ROUND2_T2_CONFIG,
    },
    {
        "name": "round2_t2-rank_10_10",
        "model_family": "new_stable",
        "checkpoint_path": "/root/tabular/mcpfn/src/tabimpute/workdir/"
                           "tabimpute-new-stable_v2-rank_10_10-50x50/checkpoint_30000.pth",
        "json_config": ROUND2_T2_CONFIG,
    },
    {
        "name": "round2_t2-rank_15_15_from_10_10",
        "model_family": "new_stable",
        "checkpoint_path": "/root/tabular/mcpfn/src/tabimpute/workdir/"
                           "tabimpute-new-stable_v2-rank_15_15-50x50_from_10_10/checkpoint_60000.pth",
        "json_config": ROUND2_T2_CONFIG,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CPS causal benchmark by imputing the treated-post block."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Directory created by create_cps_causal_benchmark.py.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="knn,softimpute,tabimpute_all",
        help=(
            "Comma-separated method ids. The special token 'tabimpute_all' expands "
            "to every trial_run_name in the v3r1 manifest."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DWIP_V3R1_MANIFEST_PATH,
        help="Path to the dwip_v3r1 manifest listing trial checkpoints.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Skip column-wise standardization (default: standardize, mirroring UCI/OpenML).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of draw folders to evaluate.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Overwrite existing per-method outputs.",
    )
    return parser.parse_args()


def fill_all_nan_columns(x_missing: np.ndarray) -> np.ndarray:
    x_processed = x_missing.copy()
    for col_idx in range(x_processed.shape[1]):
        if np.all(np.isnan(x_processed[:, col_idx])):
            x_processed[:, col_idx] = 0.0
    return x_processed


def fill_all_nan_rows(x_missing: np.ndarray) -> np.ndarray:
    x_processed = x_missing.copy()
    col_means = np.nanmean(x_processed, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for row_idx in range(x_processed.shape[0]):
        if np.all(np.isnan(x_processed[row_idx, :])):
            x_processed[row_idx, :] = col_means
    return x_processed


def make_knn_imputer() -> Callable[[np.ndarray], np.ndarray]:
    model = KNNImputer(n_neighbors=5)
    return lambda x: model.fit_transform(fill_all_nan_columns(x.copy()))


def load_manifest_mcpfn_specs(manifest_path: Path) -> list[dict]:
    if not manifest_path.is_file():
        return []

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    specs = []
    for run in manifest.get("runs", []):
        model_cfg = dict(run.get("model_cfg", {}))
        checkpoint_path = run.get("selection", {}).get("selected_checkpoint_path")
        trial_run_name = run.get("trial_run_name")
        source_model_file = str(run.get("source_model_file", ""))
        if not model_cfg or not checkpoint_path or not trial_run_name:
            continue

        model_family = "new_stable"
        if source_model_file.endswith("model_new.py"):
            model_family = "new"
        elif source_model_file.endswith("model_new_stable.py"):
            model_family = "new_stable"

        specs.append(
            {
                "name": trial_run_name,
                "model_family": model_family,
                "checkpoint_path": checkpoint_path,
                "json_config": {"config": model_cfg},
            }
        )

    return specs


def make_mcpfn_imputer(
    spec: dict, device: str
) -> Callable[[np.ndarray], np.ndarray]:
    from tabimpute.interface import ImputePFN

    init_kwargs = dict(MCPFN_SHARED_KWARGS)
    init_kwargs["device"] = device
    for key in ("precision_mode", "prediction_stat"):
        if key in spec:
            init_kwargs[key] = spec[key]
    model = ImputePFN(
        checkpoint_path=spec["checkpoint_path"],
        json_config=spec["json_config"],
        model_family=spec.get("model_family", "new_stable"),
        **init_kwargs,
    )
    return lambda x: model.impute(x.copy())


def make_mcpfn_entrywise_imputer(
    checkpoint_path: str, device: str
) -> Callable[[np.ndarray], np.ndarray]:
    from tabimpute.interface import ImputePFN

    model = ImputePFN(
        device=device,
        nhead=2,
        preprocessors=None,
        checkpoint_path=checkpoint_path,
        max_num_rows=200,
        max_num_chunks=4,
        entry_wise_features=True,
    )
    return lambda x: model.impute(x.copy())


def make_hyperimpute_imputer(plugin_name: str) -> Callable[[np.ndarray], np.ndarray]:
    from hyperimpute.plugins.imputers import Imputers

    def _run(x: np.ndarray) -> np.ndarray:
        plugin = Imputers().get(plugin_name, random_state=0)
        out_df = plugin.fit_transform(pd.DataFrame(fill_all_nan_rows(fill_all_nan_columns(x.copy()))))
        return out_df.to_numpy(dtype=np.float32)

    return _run


def expand_method_list(requested_methods: list[str], specs: list[dict]) -> list[str]:
    spec_names = [s["name"] for s in specs]
    hyperimpute_all = list(HYPERIMPUTE_PLUGINS.keys())
    baselines_all = ["knn"] + hyperimpute_all

    expanded: list[str] = []
    for m in requested_methods:
        if m == "tabimpute_all":
            expanded.extend(spec_names)
        elif m == "hyperimpute_all":
            expanded.extend(hyperimpute_all)
        elif m == "baselines_all":
            expanded.extend(baselines_all)
        elif m == "all":
            expanded.extend(baselines_all + spec_names)
        else:
            expanded.append(m)
    seen: set[str] = set()
    deduped = []
    for m in expanded:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


HYPERIMPUTE_PLUGINS = {
    "hyperimpute_mean": "mean",
    "softimpute": "softimpute",
    "hyperimpute": "hyperimpute",
    "missforest": "missforest",
    "mice": "mice",
    "ice": "ice",
    "gain": "gain",
    "miwae": "miwae",
    "em": "em",
    "miracle": "miracle",
    "sinkhorn": "sinkhorn",
}


def build_method_registry(
    args: argparse.Namespace, requested_methods: list[str], specs: list[dict]
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    spec_by_name = {s["name"]: s for s in specs}

    builders: dict[str, Callable[[], Callable[[np.ndarray], np.ndarray]]] = {
        "knn": make_knn_imputer,
        "masters_mcar": lambda: make_mcpfn_entrywise_imputer(
            MASTERS_MCAR_CHECKPOINT, args.device
        ),
    }
    for method_id, plugin_name in HYPERIMPUTE_PLUGINS.items():
        builders[method_id] = lambda p=plugin_name: make_hyperimpute_imputer(p)
    for name, spec in spec_by_name.items():
        builders[name] = lambda spec=spec: make_mcpfn_imputer(spec, args.device)

    unknown = [m for m in requested_methods if m not in builders]
    if unknown:
        raise SystemExit(
            f"Unknown methods requested: {unknown}. "
            f"Known: {sorted(builders.keys())}"
        )

    return {method: builders[method]() for method in requested_methods}


def compute_metrics(true: np.ndarray, completed: np.ndarray, treatment_mask: np.ndarray) -> dict[str, float]:
    treated = treatment_mask.astype(bool)
    observed = ~treated

    treated_true = true[treated]
    treated_pred = completed[treated]

    treated_block_rmse = float(np.sqrt(np.mean((treated_true - treated_pred) ** 2)))
    treated_block_mae = float(np.mean(np.abs(treated_true - treated_pred)))
    tau_hat = float(np.mean(treated_true) - np.mean(treated_pred))

    observed_rmse = float(np.sqrt(np.mean((true[observed] - completed[observed]) ** 2)))

    return {
        "treated_block_rmse": treated_block_rmse,
        "treated_block_mae": treated_block_mae,
        "tau_hat": tau_hat,
        "tau_sq_error": float(tau_hat**2),
        "tau_abs_error": float(abs(tau_hat)),
        "observed_block_rmse": observed_rmse,
        "n_treated_cells": int(treated.sum()),
    }


def main() -> None:
    args = parse_args()

    if not args.benchmark_dir.exists():
        raise SystemExit(f"Benchmark directory does not exist: {args.benchmark_dir}")

    specs = load_manifest_mcpfn_specs(args.manifest_path)
    specs.extend(ROUND2_T2_SPECS)

    raw_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    requested_methods = expand_method_list(raw_methods, specs)
    if not requested_methods:
        raise SystemExit("No methods requested after expansion.")
    registry = build_method_registry(args, requested_methods, specs)

    draw_dirs = sorted(
        p for p in args.benchmark_dir.iterdir() if p.is_dir() and p.name.startswith("draw_")
    )
    if args.limit is not None:
        draw_dirs = draw_dirs[: args.limit]

    results_dir = args.benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    standardize = not args.no_standardize

    total_work = len(draw_dirs) * len(requested_methods)
    outer_pbar = tqdm(
        draw_dirs,
        desc=f"Draws ({len(requested_methods)} methods)",
        unit="draw",
        total=len(draw_dirs),
    )
    work_pbar = tqdm(total=total_work, desc="Method×Draw", unit="step", leave=False)

    for draw_dir in outer_pbar:
        true = np.load(draw_dir / "true.npy")
        observed = np.load(draw_dir / "observed.npy")
        treatment_mask = np.load(draw_dir / "treatment_mask.npy")
        metadata = json.loads((draw_dir / "metadata.json").read_text(encoding="utf-8"))

        missing_mask = np.isnan(observed)

        if standardize:
            col_mean = np.nanmean(observed, axis=0)
            col_std = np.nanstd(observed, axis=0)
            col_std = np.where((col_std == 0) | np.isnan(col_std), 1.0, col_std)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            observed_in = (observed - col_mean) / col_std
        else:
            col_mean = np.zeros(observed.shape[1], dtype=observed.dtype)
            col_std = np.ones(observed.shape[1], dtype=observed.dtype)
            observed_in = observed

        for method in requested_methods:
            work_pbar.set_postfix_str(f"{draw_dir.name}:{method}")
            out_path = draw_dir / f"{method}.npy"
            if out_path.exists() and not args.force_rerun:
                completed = np.load(out_path)
                runtime_seconds = np.nan
            else:
                start = time.time()
                try:
                    completed_std = registry[method](observed_in.copy()).astype(np.float32)
                except Exception as exc:
                    failures.append(
                        {"draw_name": draw_dir.name, "method": method, "error": repr(exc)}
                    )
                    work_pbar.update(1)
                    continue
                runtime_seconds = time.time() - start
                completed = (completed_std * col_std) + col_mean
                completed = completed.astype(np.float32)
                completed[~missing_mask] = observed[~missing_mask]
                np.save(out_path, completed)

            metrics = compute_metrics(true, completed, treatment_mask)
            metrics_rows.append(
                {
                    "draw_name": draw_dir.name,
                    "draw_id": int(metadata["draw_id"]),
                    "method": method,
                    "runtime_seconds": runtime_seconds,
                    **metrics,
                }
            )
            work_pbar.update(1)

    work_pbar.close()
    outer_pbar.close()

    per_draw_path = results_dir / "per_draw_metrics.csv"
    new_df = pd.DataFrame(metrics_rows)
    if per_draw_path.exists() and not new_df.empty:
        old_df = pd.read_csv(per_draw_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["draw_name", "method"], keep="last"
        )
        merged.to_csv(per_draw_path, index=False)
    else:
        new_df.to_csv(per_draw_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(results_dir / "failures.csv", index=False)

    print(f"Wrote per-draw metrics to {per_draw_path}")
    if failures:
        print(f"Encountered {len(failures)} method failures; see {results_dir / 'failures.csv'}")


if __name__ == "__main__":
    main()
