"""Paired saved-prediction and optimizer evidence; no training or GPU usage."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_rowcol_controlled_experiments import REPO, WORK, write_json
from analyze_rowcol_controlled_evidence import read_jsonl


def main():
    root = REPO / "runs/information_flow_diagnostics/cell_query_duration_audit_20260910"
    root.mkdir(parents=True, exist_ok=True)
    queue = REPO / "runs/objective_queues/cell_query_20260909_checked"
    cache = REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth"
    cases = torch.load(cache, weights_only=True)["cases"]
    groups = sorted({c["group"] for c in cases if not c["monitor_only"]})
    specs = sorted({g.removesuffix("_sparse_query").removesuffix("_mixed") for g in groups})
    paired = {}
    rng = np.random.default_rng(20260910)
    for label in ("12000", "24000", "36000", "selected"):
        predictions = {tag: torch.load(queue / f"{tag}_{label}_holdout_predictions.pth", weights_only=True)["predictions"]
                       for tag in ("control", "query2")}
        summaries = {tag: json.loads((queue / f"{tag}_{label}_holdout.json").read_text()) for tag in predictions}
        sse = {}
        n = {}
        for g in groups:
            for rep in range(4):
                key = f"{g}_{rep}"
                a, b = (predictions[tag][key] for tag in ("control", "query2"))
                torch.testing.assert_close(a["target"], b["target"], rtol=0, atol=0)
                assert torch.equal(a["flat_indices"], b["flat_indices"])
                sse[g, rep] = [float((x["mean"].double() - x["target"].double()).square().sum()) for x in (a, b)]
                n[g, rep] = a["target"].numel()
        boot = []
        for _ in range(1000):
            ratios, differences = [], []
            for spec in specs:
                reps = rng.integers(0, 4, 4)
                for mode in ("mixed", "sparse_query"):
                    g = f"{spec}_{mode}"
                    rmse = np.sqrt(np.sum([sse[g, int(r)] for r in reps], axis=0) / sum(n[g, int(r)] for r in reps))
                    ratios.append(rmse[1] / rmse[0])
                    differences.append(rmse[1] - rmse[0])
            boot.append((np.mean(ratios), np.mean(differences)))
        ratios = {g: summaries["query2"]["groups"][g]["rmse"] / summaries["control"]["groups"][g]["rmse"] for g in groups}
        paired[label] = {"query_rmse": summaries["query2"]["macro_rmse"],
            "control_rmse": summaries["control"]["macro_rmse"],
            "mean_group_ratio": float(np.mean(list(ratios.values()))),
            "group_ratios": ratios, "groups_improved": sum(v < 1 for v in ratios.values()),
            "stratified_matrix_bootstrap_95_percentiles_ratio_and_difference": np.quantile(boot, [.025, .975], axis=0).tolist()}
    figures, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    evidence = {"paired_holdout": paired, "parameter_changes": {}, "training": {}}
    for tag in ("control", "query2", "query1"):
        work = WORK / f"tabimpute-cell_query_20260909_checked-{tag}"
        rows, vals = read_jsonl(work / "metrics_stream.jsonl"), read_jsonl(work / "validation_stream.jsonl")
        axes[0, 0].plot([r["step"] for r in vals], [r["macro_rmse"] for r in vals], "o-", label=tag)
        axes[0, 1].plot([r["step"] for r in vals], [r["macro_nll"] for r in vals], "o-", label=tag)
        windows = []
        for start in range(0, 36000, 4000):
            batch = [r for r in rows if start < r["step"] <= start + 4000]
            if batch:
                windows.append({"end": batch[-1]["step"], "nll_median": float(np.median([r["nll_missing"] for r in batch])),
                    "point_median": float(np.median([r["point_loss_missing"] for r in batch]))})
        evidence["training"][tag] = windows
        axes[1, 0].plot([w["end"] for w in windows], [w["point_median"] for w in windows], "o-", label=tag)
        if tag != "query1":
            axes[1, 1].plot([12000, 24000, 36000], [paired[str(s)]["query_rmse" if tag == "query2" else "control_rmse"]
                          for s in (12000, 24000, 36000)], "o-", label=tag)
    for ax, title in zip(axes.flat, ("Fixed selection RMSE", "Fixed selection NLL", "Sampled-batch point-loss median / 4k steps", "Independent holdout: SAME checkpoints")):
        ax.set(title=title, xlabel="Step")
        ax.legend()
        ax.grid(alpha=.2)
    axes[0, 1].set_ylim(.35, 1.05)
    figures.savefig(root / "duration_comparison.png", dpi=160)
    plt.close(figures)
    work = WORK / "tabimpute-cell_query_20260909_checked-query2"
    states = {}
    for step in (12000, 36000):
        state = torch.load(work / f"checkpoint_{step}.pth", map_location="cpu", weights_only=True)["model"]
        states[step] = {k: v.float().clone() for k, v in state.items() if k.startswith("cell_query_decoder.")}
        del state
    for name, old in states[12000].items():
        new = states[36000][name]
        evidence["parameter_changes"][name] = {"norm_12k": float(old.norm()), "norm_36k": float(new.norm()),
                                              "relative_change": float((new - old).norm() / old.norm().clamp_min(1e-12))}
    evidence["bootstrap_caution"] = "Four matrices per generator/shape; coupled masks resampled together. Descriptive uncertainty, not a multi-seed training experiment."
    write_json(root / "duration_evidence.json", evidence)
    if (root / "matched_mteb.json").exists():
        matched = json.loads((root / "matched_mteb.json").read_text())
        fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        steps = [12000, 24000, 36000]
        for tag in ("control", "query2"):
            ax[0, 0].plot(steps, [matched[f"{tag}_{s}"]["scores"]["mi"] for s in steps], "o-", label=tag)
            work = WORK / f"tabimpute-cell_query_20260909_checked-{tag}"
            vals = [v for v in read_jsonl(work / "validation_stream.jsonl") if v["step"] > 0]
            ax[0, 1].plot([v["step"] for v in vals], [v["macro_nll"] for v in vals], label=tag)
        ax[0, 0].plot([12000, 36000], [matched[f"query2_{s}_no_query"]["scores"]["mi"] for s in (12000, 36000)],
                      "o--", label="Query-trained model, query branch OFF")
        ax[0, 0].set(title="MTEB MI R2@5: identical checkpoint steps", xlabel="Step", ylabel="R2")
        ax[0, 1].set(title="Synthetic validation NLL", xlabel="Step", ylabel="NLL")
        flow = [matched[f"query2_{s}_testrow_flow"]["flow"] for s in (12000, 36000)]
        ax[1, 0].plot([12000, 36000], [100 * x["residual"]["rms_ratio"] for x in flow], "o-")
        ax[1, 0].set(title="Query residual / base features (fold-0 test rows)", xlabel="Step", ylabel="RMS ratio (%)")
        labels, values = [], []
        for s, f in zip((12000, 36000), flow):
            for layer in (0, 1):
                labels.append(f"{s // 1000}k layer {layer + 1}")
                values.append(100 * f[f"layer_{layer}"]["row_mass"])
        ax[1, 1].bar(labels, values)
        ax[1, 1].axhline(100 * flow[0]["layer_0"]["uniform_row_mass"], linestyle="--", color="black", label="Uniform-attention expectation")
        ax[1, 1].set(title="Attention assigned to row observations (fold 0)", ylabel="Attention mass (%)", ylim=(0, 5))
        for panel in ax.flat:
            panel.grid(alpha=.2)
            if panel.get_legend_handles_labels()[0]:
                panel.legend(fontsize=8)
        fig.savefig(root / "generalization_and_query_flow.png", dpi=160)
        plt.close(fig)
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
