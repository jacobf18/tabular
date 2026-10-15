"""Causal-panel evaluation of MatrixFormer+ (the observed-error-fitted convex blend).

MatrixFormer+ is TwoExpertRouter(policy="observed_blend") over the September 30
larger-matrix checkpoint (general) and the recommender specialist; there is no
blend checkpoint. Same protocol as evaluate_causal_checkpoints.py: column
standardization, full-matrix amp_bfloat16 mean inference, observed cells restored,
cuDNN SDPA left at the torch default. Each draw's expert outputs must match that
run's saved predictions bit for bit. Standard orientation only. The benchmark
folders are only read; everything is written to OUTPUT.
"""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_causal_checkpoints import (CHECKPOINTS, DESIGNS, OUTPUT as SAVED, PARENT, REPO, causal,
                                         compute_metrics, draw_dirs, paper, sha256, write_json)
from tabimpute.interface import ImputePFN
from tabimpute.routing import TwoExpertRouter


OUTPUT = SAVED.parent / "causal_matrixformer_plus_20261005"
METHOD = "matrixformer-plus"
GENERAL, SPECIALIST = "matrixformer-large", "recsys-specialist"
ROLES = {GENERAL: "general", SPECIALIST: "specialist"}
SHOWN = {PARENT: "MatrixFormer (paper)", GENERAL: "MatrixFormer large", f"{GENERAL}-avg": "MatrixFormer large, avg",
         SPECIALIST: "Recsys specialist", METHOD: "MatrixFormer+"}
METRIC_NAMES = {"treated_block_rmse": "Treated-block RMSE", "tau_abs_error": "Effect error |tau_hat - tau|"}


def make_expert(method):
    """The ImputePFN that causal.make_imputer builds, kept as an object so the router can ask for return_full."""
    checkpoint = CHECKPOINTS[method][0]
    return ImputePFN(device="cuda", entry_wise_features=False, checkpoint_path=str(checkpoint),
                     json_config={"config": causal.load_model_config(checkpoint)}, model_family="new_stable",
                     max_num_rows=None, max_num_cols=None, preprocessors=None,
                     precision_mode="amp_bfloat16", prediction_stat="mean")


def evaluate(router, family, label, path):
    """Blend every draw of one design; time only the synchronized router call (both experts plus the fit)."""
    draws = draw_dirs(path)
    key = f"{path.replace('/', '__')}.npz"
    saved = {name: np.load(SAVED / "predictions" / name / key) for name in (GENERAL, SPECIALIST)}
    warm, _, _ = causal.standardize_columns(np.load(draws[0] / "observed.npy"))
    router.impute(warm.copy())
    rows, predictions, parity = [], {}, {GENERAL: 0, SPECIALIST: 0}
    for draw in draws:
        observed = np.load(draw / "observed.npy")
        missing = np.isnan(observed)
        observed_in, col_mean, col_std = causal.standardize_columns(observed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        completed_std = router.impute(observed_in.copy())
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start

        def restore(x):
            out = ((x.astype(np.float32) * col_std) + col_mean).astype(np.float32)
            out[~missing] = observed[~missing]
            return out

        completed = restore(completed_std)
        experts = {name: restore(router.last_result.predictions[role]) for name, role in ROLES.items()}
        for name in parity:
            parity[name] += int(np.array_equal(experts[name], saved[name][draw.name]))
        true, mask = np.load(draw / "true.npy"), np.load(draw / "treatment_mask.npy").astype(bool)
        e_g = experts[GENERAL][mask].astype(np.float64) - true[mask]
        e_s = experts[SPECIALIST][mask].astype(np.float64) - true[mask]
        d = e_s - e_g
        decision = router.last_result.decisions["observed"]
        predictions[draw.name] = completed
        metadata = json.loads((draw / "metadata.json").read_text(encoding="utf-8"))
        rows.append({"family": family, "design": label, "path": path, "draw_name": draw.name,
                     "draw_id": int(metadata.get("draw_id", 0)), "method": METHOD, "runtime_seconds": runtime,
                     **compute_metrics(true, completed, mask),
                     "specialist_weight": decision["specialist_weight"], "selected": decision["selected"],
                     "general_observed_mse": decision["general_mse"],
                     "specialist_observed_mse": decision["specialist_mse"],
                     "evidence_cells": decision["evidence_cells"],
                     "oracle_weight": float(np.clip(-e_g @ d / (d @ d), 0, 1)) if d @ d > 0 else 0.0})
    out = OUTPUT / "predictions" / METHOD / key
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **predictions)
    return rows, {"draws": len(draws), **{f"{name}_identical": count for name, count in parity.items()}}


def anchored_scores(new):
    """Paper normalization: per-draw min-max over the seven core methods; other runs placed on that scale."""
    out = []
    for family, designs in paper.FAMILIES.items():
        for metric in paper.METRICS:
            means, ses = [], []
            for _, path in designs:
                saved, n_draws = paper.load(path)
                frame = pd.concat([saved, new[new["path"] == path]], ignore_index=True)
                wide = frame.pivot_table(index="draw_name", columns="method", values=metric)
                wide = wide[list(dict.fromkeys([*paper.CORE, *SHOWN]))]
                if len(wide) != n_draws or wide.isna().any().any():
                    raise ValueError(f"{path} {metric}: missing draws")
                lo, hi = wide[list(paper.CORE)].min(axis=1), wide[list(paper.CORE)].max(axis=1)
                norm = wide.sub(lo, axis=0).div(hi - lo, axis=0)[hi > lo]
                means.append(norm.mean())
                ses.append(norm.std(ddof=1) / np.sqrt(len(norm)))
            mean = pd.concat(means, axis=1).mean(axis=1)
            se = np.sqrt((pd.concat(ses, axis=1) ** 2).sum(axis=1)) / len(designs)
            for method in mean.index:
                out.append({"family": family, "metric": metric, "method": method,
                            "score": mean[method], "ci95": 1.96 * se[method]})
    return pd.DataFrame(out)


def paired(new):
    """Raw per-design change against the general expert and the paper's MatrixFormer, same draws."""
    saved = pd.read_csv(SAVED / "per_draw_metrics.csv")
    out = []
    for (family, label, path), group in new.groupby(["family", "design", "path"], sort=False):
        ours = group.set_index("draw_name")
        for reference in (GENERAL, PARENT):
            base = saved[(saved["path"] == path) & (saved["method"] == reference)].set_index("draw_name")
            base = base.loc[ours.index]
            row = {"family": family, "design": label, "reference": reference, "draws": len(ours)}
            for metric in paper.METRICS:
                diff = ours[metric] - base[metric]
                row |= {f"{metric}_mean": ours[metric].mean(), f"{metric}_reference_mean": base[metric].mean(),
                        f"{metric}_pct_change": 100 * diff.mean() / base[metric].mean(),
                        f"{metric}_diff_se": diff.std(ddof=1) / np.sqrt(len(diff)),
                        f"{metric}_win_rate": float((diff < 0).mean()),
                        f"{metric}_loss_rate": float((diff > 0).mean())}
            out.append(row)
    return pd.DataFrame(out)


def weights(new):
    """Fitted specialist weight per design, against the treated-block oracle weight (which uses the truth)."""
    def summary(g):
        return pd.Series({"draws": len(g), "mean_w": g["specialist_weight"].mean(),
                          "median_w": g["specialist_weight"].median(),
                          "zero_w": float((g["specialist_weight"] == 0).mean()),
                          "hard_specialist": float((g["selected"] == "specialist").mean()),
                          "median_oracle_w": g["oracle_weight"].median(),
                          "spearman_with_oracle": g["specialist_weight"].corr(g["oracle_weight"], method="spearman")})
    columns = ["specialist_weight", "selected", "oracle_weight"]
    return new.groupby(["family", "design"], sort=False)[columns].apply(summary).reset_index()


def runtimes(new):
    saved = pd.read_csv(SAVED / "per_draw_metrics.csv")
    both = pd.concat([new, saved[saved["method"] == GENERAL]], ignore_index=True)
    return (both.groupby(["family", "design", "method"], sort=False)["runtime_seconds"]
            .agg(median="median", mean="mean", p95=lambda s: s.quantile(0.95), max="max").reset_index())


def report(score_df, paired_df, weight_df, runtime_df, parity, manifest):
    total = sum(p["draws"] for p in parity.values())
    lines = ["# Causal panels: MatrixFormer+", "",
             f"Completed {manifest['completed_utc']}. 12 paper designs, {total:,} draws, stocks excluded, "
             "standard orientation.",
             "MatrixFormer+ = (1 - w) x MatrixFormer large (Sep 30) + w x Recsys specialist, w fitted per panel on "
             "observed-cell reconstruction error (in-sample; the treated block is never seen).",
             "Same protocol as evaluate_causal_checkpoints.py; baselines and expert runs reused from saved results.", "",
             "## Control", "",
             f"Expert outputs inside the router vs the saved Oct 3 predictions: general "
             f"{sum(p[f'{GENERAL}_identical'] for p in parity.values())}/{total}, specialist "
             f"{sum(p[f'{SPECIALIST}_identical'] for p in parity.values())}/{total} bit-identical.", "",
             "## Normalized score (0 = best, 1 = worst of the paper's seven core methods in each draw; lower is better)",
             "", "Core scores are unchanged; the other runs are placed on the same per-draw scale. "
             "\"MatrixFormer large, avg\" is the orientation the paper shows for that checkpoint.", ""]
    for metric, name in METRIC_NAMES.items():
        lines += [f"### {name}", "", "| Benchmark | " + " | ".join(SHOWN.values()) + " |",
                  "|---|" + "---:|" * len(SHOWN)]
        for family in paper.FAMILIES:
            s = score_df[(score_df["family"] == family) & (score_df["metric"] == metric)].set_index("method")
            lines.append(f"| {family} | " + " | ".join(f"{s.loc[m, 'score']:.3f} ± {s.loc[m, 'ci95']:.3f}"
                                                      for m in SHOWN) + " |")
        lines.append("")
    lines += ["## Paired raw change of MatrixFormer+ (negative is better; wins/losses over draws, rest are ties)", "",
              "| Design | Reference | RMSE change | RMSE wins/losses | Effect-error change | Effect-error wins/losses |",
              "|---|---|---:|---:|---:|---:|"]
    for _, r in paired_df.iterrows():
        lines.append(f"| {r['family']}: {r['design']} | {SHOWN[r['reference']]} | "
                     f"{r['treated_block_rmse_pct_change']:+.2f}% | "
                     f"{r['treated_block_rmse_win_rate']:.0%}/{r['treated_block_rmse_loss_rate']:.0%} | "
                     f"{r['tau_abs_error_pct_change']:+.2f}% | "
                     f"{r['tau_abs_error_win_rate']:.0%}/{r['tau_abs_error_loss_rate']:.0%} |")
    lines += ["", "## Fitted specialist weight", "",
              "Oracle weight minimizes each draw's treated-block squared error using the truth (diagnostic only).", "",
              "| Design | Mean w | Median w | w = 0 | Hard router picks specialist | Median oracle w | "
              "Spearman(w, oracle) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in weight_df.iterrows():
        lines.append(f"| {r['family']}: {r['design']} | {r['mean_w']:.3f} | {r['median_w']:.3f} | {r['zero_w']:.0%} | "
                     f"{r['hard_specialist']:.0%} | {r['median_oracle_w']:.3f} | {r['spearman_with_oracle']:+.2f} |")
    lines += ["", "## Runtime per draw (warm, CUDA-synchronized call only)", "",
              f"GPU: {manifest['environment']['gpu']}. MatrixFormer+ runs both experts and the weight fit.", "",
              "| Design | MatrixFormer+ median s | MatrixFormer large median s |", "|---|---:|---:|"]
    for (family, label), group in runtime_df.groupby(["family", "design"], sort=False):
        med = group.set_index("method")["median"]
        lines.append(f"| {family}: {label} | {med[METHOD]:.4f} | {med[GENERAL]:.4f} |")
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    if (OUTPUT / "per_draw_metrics.csv").exists():
        raise SystemExit(f"{OUTPUT} already has results; use a fresh output directory")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for method in (GENERAL, SPECIALIST):
        checkpoint, expected = CHECKPOINTS[method]
        if sha256(checkpoint) != expected:
            raise ValueError(f"{method}: checkpoint hash differs from {expected}")
    sources = [Path(__file__), Path(causal.__file__), Path(paper.__file__), REPO / "scripts/evaluate_causal_checkpoints.py",
               REPO / "src/tabimpute/routing.py", REPO / "src/tabimpute/interface.py"]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "experts": {m: {"path": str(CHECKPOINTS[m][0]), "sha256": CHECKPOINTS[m][1]} for m in (GENERAL, SPECIALIST)},
        "router": {"policy": "observed_blend", "normalization": "column"},
        "designs": [{"family": f, "label": l, "path": p, "draws": len(draw_dirs(p))} for f, l, p in DESIGNS],
        "protocol": {"model_family": "new_stable", "precision": "amp_bfloat16", "prediction_stat": "mean",
                     "max_num_rows": None, "max_num_cols": None, "standardize_columns": True, "orientation": "standard",
                     "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
                     "timing": "one warm-up call per design; perf_counter around router.impute with "
                               "torch.cuda.synchronize() on both sides"},
        "reused": {"expert_and_parent_runs": str(SAVED), "baselines": "benchmark results/per_draw_metrics.csv"},
        "sources_sha256": {str(p): sha256(p) for p in sources},
        "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "numpy": np.__version__, "python": platform.python_version()},
    }
    write_json(OUTPUT / "manifest.json", manifest)

    router = TwoExpertRouter(make_expert(GENERAL), make_expert(SPECIALIST), normalization="column",
                             policy="observed_blend")
    rows, parity = [], {}
    for family, label, path in DESIGNS:
        new_rows, parity[path] = evaluate(router, family, label, path)
        rows += new_rows
        print(f"{path}: {len(new_rows)} draws, parity {parity[path]}", flush=True)

    new = pd.DataFrame(rows)
    new.to_csv(OUTPUT / "per_draw_metrics.csv", index=False)
    write_json(OUTPUT / "control_parity.json", parity)
    broken = {p: c for p, c in parity.items() if c[f"{GENERAL}_identical"] != c["draws"]
              or c[f"{SPECIALIST}_identical"] != c["draws"]}
    if broken:
        raise SystemExit(f"Expert outputs differ from the saved predictions, not scoring: {broken}")
    score_df, paired_df, weight_df, runtime_df = anchored_scores(new), paired(new), weights(new), runtimes(new)
    score_df.to_csv(OUTPUT / "normalized_scores.csv", index=False)
    paired_df.to_csv(OUTPUT / "paired.csv", index=False)
    weight_df.to_csv(OUTPUT / "weights.csv", index=False)
    runtime_df.to_csv(OUTPUT / "runtime_summary.csv", index=False)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(OUTPUT / "manifest.json", manifest)
    report(score_df, paired_df, weight_df, runtime_df, parity, manifest)
    print((OUTPUT / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
