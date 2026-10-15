"""CPU-only horizon review; optional companion watches without changing training."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_rowcol_controlled_evidence import read_jsonl


def ema(steps, values, time_constant=500):
    result = [float(values[0])]
    for index in range(1, len(values)):
        alpha = -np.expm1(-(steps[index] - steps[index - 1]) / time_constant)
        result.append(result[-1] + alpha * (values[index] - result[-1]))
    return result


def analyze(work, out):
    train = read_jsonl(work / "metrics_stream.jsonl")
    val = read_jsonl(work / "validation_stream.jsonl")
    if not train or len(val) < 2:
        return None
    steps = np.array([r["step"] for r in train])
    groups = [g for g, v in val[0]["groups"].items() if not v["monitor_only"]]
    relative = [np.mean([v["groups"][g]["rmse"] / max(val[0]["groups"][g]["rmse"], 1e-12)
                         for g in groups]) for v in val]
    tail = val[-3:]
    a, b = tail[0], tail[-1]
    gains = {g: 100 * (a["groups"][g]["rmse"] - b["groups"][g]["rmse"])
                   / max(a["groups"][g]["rmse"], 1e-12) for g in groups}
    gain = 100 * (relative[-len(tail)] - relative[-1]) / max(relative[-len(tail)], 1e-12)
    lr_fraction = train[-1]["lr"] / max(r["lr"] for r in train)
    recent_best = min(val, key=lambda v: v["macro_rmse"])["step"] >= a["step"]
    count = sum(v > 0 for v in gains.values())
    # Screening rules, not statistical significance or proof of convergence.
    config = json.loads((work / "config.json").read_text())
    horizon = config.get("curriculum", [{"end_step": 12000}])[-1]["end_step"]
    previous_best = min(v["macro_rmse"] for v in val if v["step"] <= a["step"])
    unstable = max(v["macro_rmse"] for v in val) > 2 * val[0]["macro_rmse"]
    if steps[-1] < horizon:
        recommendation = "in progress; do not infer a stopping point yet"
    elif unstable or b["macro_rmse"] > previous_best * 1.01:
        recommendation = "instability/recovery warning: late improvement is not evidence for simply training longer"
    elif gain > .1 and count >= len(groups) / 2:
        recommendation = "longer-budget comparison warranted: late validation improvement is broad"
    elif recent_best or lr_fraction < .1:
        recommendation = "convergence unproven: compare a longer cosine horizon; floor may hide progress"
    else:
        recommendation = "no clear late improvement; inspect group regressions before extending"
    summary = {"work_dir": str(work), "step": int(steps[-1]),
               "validation_interval": [a["step"], b["step"]],
               "relative_macro_improvement_percent": float(gain), "groups_improved": count,
               "group_count": len(groups), "group_improvement_percent": gains,
               "last_lr_fraction_of_peak": lr_fraction,
               "best_raw_validation_step": min(val, key=lambda v: v["macro_rmse"])["step"],
               "recommendation": recommendation, "training_windows": {}}
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for ax, key, label in zip(axes.flat[:3],
            ("loss_missing", "nll_missing", "point_loss_missing"), ("Composite loss", "NLL", "Point loss")):
        values = np.array([r[key] for r in train])
        ax.plot(steps, values, alpha=.15, linewidth=.6, label="Sampled batches")
        ax.plot(steps, ema(steps, values), label="EMA (500-step time constant)")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(label + " (symlog)")
        ax.legend(fontsize=8)
        windows = []
        for start in range(max(0, int(steps[-1]) - 4000), int(steps[-1]), 1000):
            sample = values[(steps > start) & (steps <= start + 1000)]
            if len(sample):
                windows.append({"start": start, "n": len(sample), "mean": float(sample.mean()),
                                "median": float(np.median(sample)),
                                "q25": float(np.quantile(sample, .25)), "q75": float(np.quantile(sample, .75))})
        summary["training_windows"][key] = windows
    axes[1, 1].plot(steps, [r["lr"] for r in train])
    axes[1, 1].set_title("Learning rate")
    vsteps = [r["step"] for r in val]
    axes[2, 0].plot(vsteps, [r["macro_rmse"] for r in val], "o-", label="Original macro RMSE")
    axes[2, 0].set_title("Fixed validation (original selector)")
    for g in groups:
        axes[2, 1].plot(vsteps, [r["groups"][g]["rmse"] / max(val[0]["groups"][g]["rmse"], 1e-12)
                                for r in val], alpha=.4, linewidth=.8)
    axes[2, 1].plot(vsteps, relative, "ko-", label="Equal-group relative RMSE")
    axes[2, 1].set_title("Group trajectories / their step-0 errors")
    axes[2, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Step")
        ax.grid(alpha=.2)
    fig.suptitle(work.name, fontsize=10)
    fig.savefig(out / f"{work.name}.png", dpi=130)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="rowcol_output_remedies_paired_20260907")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / "runs/objective_queues" / args.queue
    out = root / "curve_analysis"
    out.mkdir(parents=True, exist_ok=True)
    previous_signature = None
    while True:
        work_root = repo / "src/tabimpute/workdir"
        works = sorted(work_root.glob(f"tabimpute-{args.queue}-*"))
        works += [work_root / f"tabimpute-rowcol_controlled_20260906-{tag}"
                  for tag in ("mse_broad_support", "lr_followup_2")]
        status = json.loads((root / "status.json").read_text())
        signature = [(str(w), (w / "validation_stream.jsonl").stat().st_mtime_ns)
                     for w in works if (w / "validation_stream.jsonl").exists()]
        signature.append(status["state"])
        if signature != previous_signature:
            summaries = [s for w in works if (s := analyze(w, out)) is not None]
            payload = {"updated_unix": time.time(), "queue_state": status, "runs": summaries}
            temp = out / "review.json.tmp"
            temp.write_text(json.dumps(payload, indent=2))
            temp.replace(out / "review.json")
            lines = ["# Training horizon review", "", "12k is a screening budget, not a demonstrated convergence limit.",
                "Training components are sampled every 25 steps; EMA/windows are descriptive and can be affected by changing batches.",
                "Validation groups use fixed queries. Late differences have no statistical confidence interval here.",
                "A cosine floor can create an apparent plateau. No training schedule or selector is changed by this report.", ""]
            for s in summaries:
                lines += [f"## {Path(s['work_dir']).name}",
                    f"Step {s['step']}; recent validation interval {s['validation_interval']}.",
                    f"Relative macro improvement: {s['relative_macro_improvement_percent']:.4f}%; "
                    f"groups improved: {s['groups_improved']}/{s['group_count']}; "
                    f"ending LR: {s['last_lr_fraction_of_peak']:.1%} of peak.",
                    s["recommendation"], ""]
            lines += ["## Follow-up design", "Compare 12k versus 36k cosine horizons from the same initialization, data seed and objective. "
                "Use fixed synthetic validation and saved holdout predictions, not MTEB to select the budget. "
                "A resumed optimizer with a gentle LR restart can be a cheaper exploratory test, but is not equivalent "
                "to training with the longer schedule from the start. Keep each run below 200k steps."]
            (out / "review.md").write_text("\n".join(lines) + "\n")
            print(json.dumps({Path(s["work_dir"]).name: s["recommendation"] for s in summaries}), flush=True)
            previous_signature = signature
        if not args.watch or status["state"] in {"complete", "error"}:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
