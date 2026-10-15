#!/usr/bin/env python3
"""Evaluate newer TabImpute architectures on causal-panel benchmarks.

This script mirrors the existing CPS/Kansas/Prop99 causal eval convention:
standardize observed columns, impute with an interface class, de-standardize,
preserve observed entries, and append per-draw rows to results/per_draw_metrics.csv.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


BENCHMARK_DIR = Path("/root/tabular/mcpfn/benchmark")
ROOT = Path("/root/tabular/mcpfn")
SRC_DIR = ROOT / "src"
CPS_DIR = BENCHMARK_DIR / "CPS"
for extra in (SRC_DIR, CPS_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cps_causal import compute_metrics  # noqa: E402
from tabimpute.interface import CompressedCrossImputePFN, ImputePFN  # noqa: E402


WORKDIR = ROOT / "src/tabimpute/workdir"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str
    checkpoint: Path
    model_family: str = ""
    chunk_rows: int | None = None
    chunk_cols: int | None = None
    precision_mode: str = "full_bfloat16"
    prediction_stat: str = "median"


DWIP_SPECTRAL_TM128K_CHECKPOINT = (
    WORKDIR
    / "tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5"
    / "best_validation_checkpoint.pth"
)

METHOD_SPECS: dict[str, MethodSpec] = {
    "direct_pure_rabel_rank10": MethodSpec(
        name="direct_pure_rabel_rank10",
        kind="direct",
        checkpoint=WORKDIR
        / "tabimpute-direct-pure-rabel64-rank10-curriculum-160k"
        / "checkpoint_160000.pth",
        model_family="pure_rabel_memory_efficient",
    ),
    "direct_pure_rabel_spectrum": MethodSpec(
        name="direct_pure_rabel_spectrum",
        kind="direct",
        checkpoint=WORKDIR
        / "tabimpute-direct-pure-rabel64-spectrum-decay-top40-curriculum-160k-lr1e4-drop5e5"
        / "best_checkpoint.pth",
        model_family="pure_rabel_memory_efficient",
    ),
    "direct_pure_rabel_causal_panel": MethodSpec(
        name="direct_pure_rabel_causal_panel",
        kind="direct",
        checkpoint=WORKDIR
        / "tabimpute-direct-pure-rabel64-causal-panel-bs16-from-rank10-100k"
        / "best_checkpoint.pth",
        model_family="pure_rabel_memory_efficient",
    ),
    "direct_pure_rabel_causal_panel_v2": MethodSpec(
        name="direct_pure_rabel_causal_panel_v2",
        kind="direct",
        checkpoint=WORKDIR
        / "tabimpute-direct-pure-rabel64-causal-panel-v2-hardmix-from-v1best-50k"
        / "best_checkpoint.pth",
        model_family="pure_rabel_memory_efficient",
    ),
    "compressed_pure_rabel_k20": MethodSpec(
        name="compressed_pure_rabel_k20",
        kind="compressed",
        checkpoint=WORKDIR
        / "tabimpute-compressed-pure-rabel64-rank10-k20-curriculum-160k"
        / "best_checkpoint.pth",
        chunk_rows=100,
        chunk_cols=100,
    ),
    "compressed_stable_k20_200": MethodSpec(
        name="compressed_stable_k20_200",
        kind="compressed",
        checkpoint=WORKDIR
        / "tabimpute-comp-rank10-200x200-100x100chunks-random-k20-100k-from-100comp"
        / "checkpoint_100000.pth",
        chunk_rows=100,
        chunk_cols=100,
    ),
    # Trained/selected with amp_bfloat16 + mean point target; the -median
    # variant matches how the older dwip anchors were evaluated.
    "dwip-spectral-tm128k": MethodSpec(
        name="dwip-spectral-tm128k",
        kind="direct",
        checkpoint=DWIP_SPECTRAL_TM128K_CHECKPOINT,
        model_family="new_stable",
        precision_mode="amp_bfloat16",
        prediction_stat="mean",
    ),
    "dwip-spectral-tm128k-median": MethodSpec(
        name="dwip-spectral-tm128k-median",
        kind="direct",
        checkpoint=DWIP_SPECTRAL_TM128K_CHECKPOINT,
        model_family="new_stable",
        precision_mode="full_bfloat16",
        prediction_stat="median",
    ),
}


BENCHMARK_PRESETS: dict[str, list[Path]] = {
    "smoke": [
        BENCHMARK_DIR / "kansas/causal_eval/smoke",
        BENCHMARK_DIR / "prop99/causal_eval/smoke",
        BENCHMARK_DIR / "CPS/causal_eval/log_wage_min_wage",
    ],
    "small": [
        BENCHMARK_DIR / "kansas/causal_eval/smoke",
        BENCHMARK_DIR / "prop99/causal_eval/smoke",
        BENCHMARK_DIR / "CPS/causal_eval/log_wage_min_wage",
        BENCHMARK_DIR / "CPS/causal_eval/hours_min_wage",
    ],
    "all": sorted(
        [
            p
            for root in (
                BENCHMARK_DIR / "CPS/causal_eval",
                BENCHMARK_DIR / "kansas/causal_eval",
                BENCHMARK_DIR / "prop99/causal_eval",
            )
            for p in root.iterdir()
            if p.is_dir() and any(d.is_dir() and d.name.startswith("draw_") for d in p.iterdir())
        ]
    ),
}


def load_model_config(checkpoint: Path) -> dict:
    cfg_path = checkpoint.parent / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if "model" in config:
        return dict(config["model"])
    if "config" in config:
        return dict(config["config"])
    return dict(config)


def make_imputer(spec: MethodSpec, device: str) -> Callable[[np.ndarray], np.ndarray]:
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(spec.checkpoint)

    if spec.kind == "direct":
        model = ImputePFN(
            device=device,
            entry_wise_features=False,
            checkpoint_path=str(spec.checkpoint),
            json_config={"config": load_model_config(spec.checkpoint)},
            model_family=spec.model_family,
            max_num_rows=None,
            max_num_cols=None,
            preprocessors=None,
            precision_mode=spec.precision_mode,
            prediction_stat=spec.prediction_stat,
        )
    elif spec.kind == "compressed":
        model = CompressedCrossImputePFN(
            compressed_checkpoint_path=str(spec.checkpoint),
            chunk_rows=spec.chunk_rows,
            chunk_cols=spec.chunk_cols,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported method kind: {spec.kind}")

    return lambda x: model.impute(x.copy())


def standardize_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(x, axis=0)
        col_std = np.nanstd(x, axis=0)
    col_std = np.where((col_std == 0) | np.isnan(col_std), 1.0, col_std)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    return (x - col_mean) / col_std, col_mean, col_std


def evaluate_benchmark_dir(
    benchmark_dir: Path,
    registry: dict[str, Callable[[np.ndarray], np.ndarray]],
    *,
    limit: int | None,
    force_rerun: bool,
    standardize: bool,
) -> pd.DataFrame:
    draw_dirs = sorted(
        p for p in benchmark_dir.iterdir() if p.is_dir() and p.name.startswith("draw_")
    )
    if limit is not None:
        draw_dirs = draw_dirs[:limit]

    results_dir = benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    work = tqdm(
        total=len(draw_dirs) * len(registry),
        desc=benchmark_dir.name,
        unit="step",
        leave=False,
    )

    for draw_dir in draw_dirs:
        true = np.load(draw_dir / "true.npy")
        observed = np.load(draw_dir / "observed.npy")
        treatment_mask = np.load(draw_dir / "treatment_mask.npy")
        metadata = json.loads((draw_dir / "metadata.json").read_text(encoding="utf-8"))
        missing_mask = np.isnan(observed)

        if standardize:
            observed_in, col_mean, col_std = standardize_columns(observed)
        else:
            observed_in = observed
            col_mean = np.zeros(observed.shape[1], dtype=observed.dtype)
            col_std = np.ones(observed.shape[1], dtype=observed.dtype)

        for method, impute_fn in registry.items():
            work.set_postfix_str(f"{draw_dir.name}:{method}")
            out_path = draw_dir / f"{method}.npy"
            if out_path.exists() and not force_rerun:
                completed = np.load(out_path)
                runtime_seconds = float("nan")
            else:
                start = time.time()
                try:
                    completed_std = impute_fn(observed_in.copy()).astype(np.float32)
                    completed = (completed_std * col_std) + col_mean
                    completed = completed.astype(np.float32)
                    completed[~missing_mask] = observed[~missing_mask]
                    np.save(out_path, completed)
                    runtime_seconds = time.time() - start
                except Exception as exc:
                    failures.append(
                        {
                            "draw_name": draw_dir.name,
                            "method": method,
                            "error": repr(exc),
                        }
                    )
                    work.update(1)
                    continue

            metrics = compute_metrics(true, completed, treatment_mask)
            extra = {
                k: metadata.get(k)
                for k in ("design", "Nt", "T0", "N", "T", "outcome", "treatment")
                if k in metadata
            }
            rows.append(
                {
                    "draw_name": draw_dir.name,
                    "draw_id": int(metadata.get("draw_id", 0)),
                    "method": method,
                    "runtime_seconds": runtime_seconds,
                    **metrics,
                    **extra,
                }
            )
            work.update(1)

    work.close()
    new_df = pd.DataFrame(rows)
    per_draw_path = results_dir / "per_draw_metrics.csv"
    if per_draw_path.exists() and not new_df.empty:
        old_df = pd.read_csv(per_draw_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["draw_name", "method"], keep="last")
        merged.to_csv(per_draw_path, index=False)
    else:
        new_df.to_csv(per_draw_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(
            results_dir / "failures_new_architectures.csv",
            index=False,
        )
        print(
            f"{benchmark_dir}: {len(failures)} failures, see "
            f"{results_dir / 'failures_new_architectures.csv'}"
        )

    print(f"{benchmark_dir}: wrote {len(rows)} rows to {per_draw_path}")
    return new_df


def summarize(rows: pd.DataFrame, out_path: Path) -> None:
    if rows.empty:
        print("No rows to summarize.")
        return
    summary = (
        rows.groupby(["benchmark_dir", "method"], dropna=False)
        .agg(
            n=("treated_block_rmse", "size"),
            treated_block_rmse_mean=("treated_block_rmse", "mean"),
            treated_block_rmse_sem=("treated_block_rmse", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
            tau_abs_error_mean=("tau_abs_error", "mean"),
            tau_sq_error_mean=("tau_sq_error", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Wrote summary to {out_path}")
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--benchmark-preset",
        choices=sorted(BENCHMARK_PRESETS),
        default="smoke",
    )
    p.add_argument(
        "--benchmark-dir",
        action="append",
        type=Path,
        default=[],
        help="Additional/override causal_eval subgroup directory with draw_* children.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="direct_pure_rabel_rank10,direct_pure_rabel_spectrum,compressed_pure_rabel_k20,compressed_stable_k20_200",
        help=f"Comma-separated subset of {sorted(METHOD_SPECS)}.",
    )
    p.add_argument(
        "--extra-direct-method",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT[:MODEL_FAMILY]",
        help=(
            "Add an ad-hoc direct ImputePFN method. MODEL_FAMILY defaults to "
            "pure_rabel_memory_efficient. Example: "
            "my_run=/path/best_checkpoint.pth:pure_rabel_memory_efficient"
        ),
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument(
        "--summary-path",
        type=Path,
        default=BENCHMARK_DIR / "runs/causal_new_architectures_smoke_summary.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    method_specs = dict(METHOD_SPECS)
    for raw in args.extra_direct_method:
        if "=" not in raw:
            raise SystemExit(
                "--extra-direct-method must have form NAME=CHECKPOINT[:MODEL_FAMILY]"
            )
        name, rest = raw.split("=", 1)
        pieces = rest.rsplit(":", 1)
        if len(pieces) == 2 and pieces[1] and not pieces[1].startswith("/"):
            checkpoint_raw, model_family = pieces
        else:
            checkpoint_raw, model_family = rest, "pure_rabel_memory_efficient"
        checkpoint = Path(checkpoint_raw).expanduser()
        method_specs[name] = MethodSpec(
            name=name,
            kind="direct",
            checkpoint=checkpoint,
            model_family=model_family,
        )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in method_specs]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Known: {sorted(method_specs)}")

    benchmark_dirs = args.benchmark_dir or BENCHMARK_PRESETS[args.benchmark_preset]
    missing_dirs = [str(p) for p in benchmark_dirs if not p.is_dir()]
    if missing_dirs:
        raise SystemExit(f"Benchmark directories do not exist: {missing_dirs}")

    print("Loading methods:")
    registry: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for method in methods:
        spec = method_specs[method]
        print(f"  {method}: {spec.checkpoint}")
        registry[method] = make_imputer(spec, args.device)

    all_rows: list[pd.DataFrame] = []
    for benchmark_dir in benchmark_dirs:
        df = evaluate_benchmark_dir(
            benchmark_dir,
            registry,
            limit=args.limit,
            force_rerun=args.force_rerun,
            standardize=not args.no_standardize,
        )
        if not df.empty:
            df.insert(0, "benchmark_dir", str(benchmark_dir))
            all_rows.append(df)

    if all_rows:
        summarize(pd.concat(all_rows, ignore_index=True), args.summary_path)

    del registry
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
