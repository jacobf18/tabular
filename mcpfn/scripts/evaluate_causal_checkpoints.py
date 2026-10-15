"""Causal-panel evaluation of the larger-matrix and recommender-specialist checkpoints.

Runs the paper's step-98k MatrixFormer (control), the September 30 larger-matrix
checkpoint and the recommender specialist on the 12 paper causal designs with the
paper's causal protocol: column standardization, full-matrix amp_bfloat16 mean
inference, observed cells restored. The saved draws are only read; predictions,
metrics and timings are written to OUTPUT, never into the benchmark folders.
Scores reuse the paper exporter, swapping each checkpoint in for MatrixFormer.
"""

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_dwip_missbench_uci import REPO, sha256, write_json

PAPER = REPO / "69d7e55fd826e0325c3115d0"
sys.path[:0] = [str(REPO / "benchmark"), str(PAPER / "scripts")]
import eval_causal_new_architectures as causal  # noqa: E402  (also puts src/ and CPS/ on sys.path)
import export_causal_results as paper  # noqa: E402
from eval_cps_causal import compute_metrics  # noqa: E402


OUTPUT = REPO / "runs/benchmark_evaluations/causal_large_and_recsys_specialist_20261003"
PARENT = paper.MATRIXFORMER
CHECKPOINTS = {
    PARENT: (paper.CHECKPOINT, "1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d"),
    "matrixformer-large": (
        REPO / "runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth",
        "c0ddf75274c8ff92ca5fd22c1263369ffa7e55eb7cb85fe80961b0afa101f7a9"),
    "recsys-specialist": (
        REPO / "runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth",
        "1fcadc0c5846f9c35fbbae4e3b519b9244b608272cb1eda68c0850132321f770"),
}
METRIC_NAMES = {"treated_block_rmse": "Treated-block RMSE", "tau_abs_error": "Effect error |tau_hat - tau|"}
LABELS = {PARENT: "MatrixFormer 98k (paper)", "matrixformer-large": "MatrixFormer large (Sep 30)",
          "recsys-specialist": "Recsys specialist"}
DESIGNS = [(family, label, path) for family, designs in paper.FAMILIES.items() for label, path in designs]


def draw_dirs(path):
    return sorted(d for d in (paper.BENCHMARK / path).iterdir() if d.is_dir() and d.name.startswith("draw_"))


def evaluate(method, impute, family, label, path):
    """Impute every draw of one design; time only the synchronized model call."""
    draws = draw_dirs(path)
    warm, _, _ = causal.standardize_columns(np.load(draws[0] / "observed.npy"))
    impute(warm.copy())
    rows, predictions, identical, max_diff = [], {}, 0, 0.0
    for draw in draws:
        observed = np.load(draw / "observed.npy")
        missing = np.isnan(observed)
        observed_in, col_mean, col_std = causal.standardize_columns(observed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        completed_std = impute(observed_in.copy())
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        completed = ((completed_std.astype(np.float32) * col_std) + col_mean).astype(np.float32)
        completed[~missing] = observed[~missing]
        predictions[draw.name] = completed
        if method == PARENT:
            saved = np.load(draw / f"{PARENT}.npy")
            identical += int(np.array_equal(completed, saved))
            max_diff = max(max_diff, float(np.nanmax(np.abs(completed - saved))))
        metadata = json.loads((draw / "metadata.json").read_text(encoding="utf-8"))
        metrics = compute_metrics(np.load(draw / "true.npy"), completed, np.load(draw / "treatment_mask.npy"))
        rows.append({"family": family, "design": label, "path": path, "draw_name": draw.name,
                     "draw_id": int(metadata.get("draw_id", 0)), "method": method,
                     "runtime_seconds": runtime, **metrics})
    out = OUTPUT / "predictions" / method / f"{path.replace('/', '__')}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **predictions)
    parity = {"draws": len(draws), "identical": identical, "max_abs_diff": max_diff} if method == PARENT else None
    return rows, parity


def scores(new):
    """Paper normalization with each checkpoint in MatrixFormer's place."""
    out = []
    for method in CHECKPOINTS:
        for family, designs in paper.FAMILIES.items():
            frames = []
            for _, path in designs:
                saved, _ = paper.load(path)
                swap = new[(new["path"] == path) & (new["method"] == method)].assign(method=PARENT)
                saved = saved[saved["method"].isin(paper.CORE) & (saved["method"] != PARENT)]
                frames.append(pd.concat([saved, swap], ignore_index=True))
            for metric in paper.METRICS:
                mean, se, n = paper.normalized_scores(frames, metric)
                rank = int(mean.rank().loc[PARENT])
                out.append({"checkpoint": method, "family": family, "metric": metric,
                            "score": mean[PARENT], "se": se[PARENT], "ci95": 1.96 * se[PARENT],
                            "rank_of_7": rank, "n_draws": n})
    return pd.DataFrame(out)


def paired(new):
    """Raw per-design errors and paired differences against the 98k parent."""
    out = []
    for (family, label, path), group in new.groupby(["family", "design", "path"], sort=False):
        wide = {m: g.set_index("draw_name") for m, g in group.groupby("method")}
        base = wide[PARENT]
        for method in CHECKPOINTS:
            row = {"family": family, "design": label, "checkpoint": method, "draws": len(wide[method])}
            for metric in paper.METRICS:
                diff = wide[method][metric] - base[metric]
                row |= {f"{metric}_mean": wide[method][metric].mean(),
                        f"{metric}_diff": diff.mean(), f"{metric}_diff_se": diff.std(ddof=1) / np.sqrt(len(diff)),
                        f"{metric}_pct_change": 100 * diff.mean() / base[metric].mean(),
                        f"{metric}_win_rate": float((diff < 0).mean())}
            out.append(row)
    return pd.DataFrame(out)


def runtimes(new):
    return (new.groupby(["family", "design", "method"], sort=False)["runtime_seconds"]
            .agg(median="median", mean="mean", p95=lambda s: s.quantile(0.95), max="max")
            .reset_index())


def report(score_df, paired_df, runtime_df, parity, manifest):
    lines = ["# Causal panels: larger-matrix and recommender-specialist checkpoints", "",
             f"Completed {manifest['completed_utc']}. 12 paper designs, 3,000 draws, stocks excluded.",
             "Same protocol and scoring as the paper; baselines reused from the saved per-draw metrics.",
             "The 98k parent was rerun as a control and as the paired reference.", "",
             "## Control", "",
             f"98k parent rerun: {sum(p['identical'] for p in parity.values())}/"
             f"{sum(p['draws'] for p in parity.values())} draws bit-identical to the saved paper predictions "
             f"(max abs diff {max(p['max_abs_diff'] for p in parity.values()):.3g}).", "",
             "## Normalized score (paper method set, 0 = best in draw, lower is better)", ""]
    for metric, name in METRIC_NAMES.items():
        lines += [f"### {name}", "",
                  "| Benchmark | " + " | ".join(LABELS[m] for m in CHECKPOINTS) + " |",
                  "|---|" + "---:|" * len(CHECKPOINTS)]
        for family in paper.FAMILIES:
            cells = []
            for method in CHECKPOINTS:
                r = score_df[(score_df["checkpoint"] == method) & (score_df["family"] == family)
                             & (score_df["metric"] == metric)].iloc[0]
                cells.append(f"{r['score']:.3f} ± {r['ci95']:.3f} (#{r['rank_of_7']})")
            lines.append(f"| {family} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Paired raw change against the 98k parent (negative is better)", "",
              "| Design | Checkpoint | RMSE change | RMSE wins | Effect-error change | Effect-error wins |",
              "|---|---|---:|---:|---:|---:|"]
    for _, r in paired_df[paired_df["checkpoint"] != PARENT].iterrows():
        lines.append(f"| {r['family']}: {r['design']} | {LABELS[r['checkpoint']]} | "
                     f"{r['treated_block_rmse_pct_change']:+.2f}% | {r['treated_block_rmse_win_rate']:.0%} | "
                     f"{r['tau_abs_error_pct_change']:+.2f}% | {r['tau_abs_error_win_rate']:.0%} |")
    lines += ["", "## Runtime per draw (warm, CUDA-synchronized model call only)", "",
              f"GPU: {manifest['environment']['gpu']}.", "",
              "| Design | " + " | ".join(f"{LABELS[m]} median s" for m in CHECKPOINTS) + " |",
              "|---|" + "---:|" * len(CHECKPOINTS)]
    for (family, label), group in runtime_df.groupby(["family", "design"], sort=False):
        med = group.set_index("method")["median"]
        lines.append(f"| {family}: {label} | " + " | ".join(f"{med[m]:.4f}" for m in CHECKPOINTS) + " |")
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for method, (checkpoint, expected) in CHECKPOINTS.items():
        if sha256(checkpoint) != expected:
            raise ValueError(f"{method}: checkpoint hash differs from {expected}")
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {m: {"path": str(c), "sha256": h} for m, (c, h) in CHECKPOINTS.items()},
        "designs": [{"family": f, "label": l, "path": p, "draws": len(draw_dirs(p))} for f, l, p in DESIGNS],
        "protocol": {"model_family": "new_stable", "precision": "amp_bfloat16", "prediction_stat": "mean",
                     "max_num_rows": None, "max_num_cols": None, "standardize_columns": True,
                     "cudnn_sdpa": "torch default, as in the paper causal eval", "timing": "one warm-up call per design; perf_counter around "
                     "the model call with torch.cuda.synchronize() on both sides"},
        "sources_sha256": {str(p): sha256(p) for p in (Path(__file__), Path(causal.__file__),
                                                      Path(paper.__file__), REPO / "src/tabimpute/interface.py")},
        "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "python": platform.python_version()},
    }
    write_json(OUTPUT / "manifest.json", manifest)

    rows, parity = [], {}
    for method, (checkpoint, _) in CHECKPOINTS.items():
        spec = causal.MethodSpec(name=method, kind="direct", checkpoint=checkpoint, model_family="new_stable",
                                 precision_mode="amp_bfloat16", prediction_stat="mean")
        impute = causal.make_imputer(spec, "cuda")
        for family, label, path in DESIGNS:
            new_rows, check = evaluate(method, impute, family, label, path)
            rows += new_rows
            if check:
                parity[path] = check
            print(f"{method} {path}: {len(new_rows)} draws", flush=True)
        del impute
        torch.cuda.empty_cache()

    new = pd.DataFrame(rows)
    new.to_csv(OUTPUT / "per_draw_metrics.csv", index=False)
    write_json(OUTPUT / "control_parity.json", parity)
    score_df, paired_df, runtime_df = scores(new), paired(new), runtimes(new)
    score_df.to_csv(OUTPUT / "normalized_scores.csv", index=False)
    paired_df.to_csv(OUTPUT / "paired_vs_parent.csv", index=False)
    runtime_df.to_csv(OUTPUT / "runtime_summary.csv", index=False)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(OUTPUT / "manifest.json", manifest)
    report(score_df, paired_df, runtime_df, parity, manifest)
    print((OUTPUT / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
