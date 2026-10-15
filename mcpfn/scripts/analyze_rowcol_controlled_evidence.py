"""CPU-only snapshots of existing controlled-run logs; never changes the queue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return result


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="rowcol_controlled_20260906")
    args = parser.parse_args()
    root = repo / "runs/objective_queues" / args.queue
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = repo / "runs/information_flow_diagnostics" / "controlled_evidence" / stamp
    out.mkdir(parents=True)
    completed = json.loads((root / "results.json").read_text()) if (root / "results.json").exists() else []
    records, curves = {}, {}
    for command_file in sorted(root.glob("*_command.json")):
        tag = command_file.name.removesuffix("_command.json")
        work = repo / "src/tabimpute/workdir" / f"tabimpute-{args.queue}-{tag}"
        training = read_jsonl(work / "metrics_stream.jsonl")
        validation = read_jsonl(work / "validation_stream.jsonl")
        if not training or not validation:
            continue
        baseline = validation[0]["groups"]
        groups = [g for g, v in baseline.items() if not v["monitor_only"]]
        gradient_rows = [r for r in training if "logit_gradient_norm/point" in r]
        for row in validation:
            row["relative_group_rmse"] = float(np.mean([
                row["groups"][g]["rmse"] / max(baseline[g]["rmse"], 1e-12) for g in groups]))
            row["groups_improved"] = sum(row["groups"][g]["rmse"] < baseline[g]["rmse"] for g in groups)
        update_groups = ("row_factor_head", "col_factor_head", "svt_threshold_unconstrained",
                         "row_input_proj", "message_layers.0", "row_attention_layers.0",
                         "output_decoder", "observed_attention_context_proj")
        updates = {}
        for group in update_groups:
            key = f"sampled_update_ratio/{group}"
            rows = [r for r in gradient_rows if key in r]
            if rows:
                updates[group] = {
                    "median_ratio": float(np.median([r[key] for r in rows])),
                    "zero_fraction": float(np.mean([r[key] == 0 for r in rows])),
                    "median_vs_lr_weight_decay": float(np.median([r[key] / (r["lr"] * .02) for r in rows])),
                }
        record = {"last_step": training[-1]["step"],
                  "completed": any(r["tag"] == tag for r in completed),
                  "clip_fraction_sampled": float(np.mean([r["grad_was_clipped"] for r in training])),
                  "point_to_nll_logit_gradient_median": float(np.median([
                      r["logit_gradient_norm/point"] / max(r["logit_gradient_norm/nll"], 1e-30)
                      for r in gradient_rows])) if gradient_rows else None,
                  "gradient_cosine_median": float(np.median([r["logit_gradient_cosine/nll_point"]
                                                              for r in gradient_rows])) if gradient_rows else None,
                  "best_raw_step": min(validation, key=lambda r: r["macro_rmse"])["step"],
                  "best_relative_step": min(validation, key=lambda r: r["relative_group_rmse"])["step"],
                  "raw_baseline_top_two_group_share": sum(sorted([baseline[g]["rmse"] for g in groups])[-2:])
                                                       / sum(baseline[g]["rmse"] for g in groups),
                  "sampled_updates": updates,
                  "validation": validation}
        records[tag] = record
        curves[tag] = (training, gradient_rows, validation)
    snapshot = {"utc": stamp, "queue_state": json.loads((root / "status.json").read_text()),
                "completed_evaluations": completed, "runs": records}
    if not records:
        raise RuntimeError("No training and validation records available yet")
    (out / "evidence.json").write_text(json.dumps(snapshot, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for tag, (training, gradients, validation) in curves.items():
        axes[0, 0].plot([r["step"] for r in validation], [r["macro_rmse"] for r in validation], marker="o", label=tag)
        axes[0, 1].plot([r["step"] for r in validation], [r["relative_group_rmse"] for r in validation], marker="o", label=tag)
        axes[1, 0].plot([r["step"] for r in gradients],
                        [r["logit_gradient_norm/point"] / max(r["logit_gradient_norm/nll"], 1e-30) for r in gradients], label=tag)
        axes[1, 1].plot([r["step"] for r in gradients], [r["logit_gradient_cosine/nll_point"] for r in gradients], label=tag)
    for ax, title, ylabel in zip(axes.flat,
        ("Current selection metric", "Diagnostic alternative (not used by queue)",
         "Weighted point/NLL gradient magnitude at logits", "Weighted point/NLL gradient alignment at logits"),
        ("Mean group RMSE", "Mean group RMSE / its step-0 RMSE", "Norm ratio", "Cosine")):
        ax.set(title=title, xlabel="Training step", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[1, 0].set_yscale("log")
    fig.savefig(out / "training_flow.png", dpi=160)
    plt.close(fig)
    lines = ["# Live controlled-run evidence", "", f"UTC snapshot: {stamp}. CPU-only log analysis; no queue changes.", "",
             "## Completed evaluations", "", "| Run | Selected step | MI R2@5 | Random R2@5 |",
             "| --- | ---: | ---: | ---: |"]
    for r in completed:
        lines.append(f"| {r['tag']} | {r['validation_step']} | {r['mi_r2_5']:.4f} | {r['random_r2_5']:.4f} |")
    lines += ["", "## Training signals", "",
              "| Run | Latest step | Median point/NLL logit gradient | Median cosine | Raw-best step | Relative-best step |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for tag, r in records.items():
        ratio, cosine = r["point_to_nll_logit_gradient_median"], r["gradient_cosine_median"]
        ratio_text = f"{ratio:.5f}" if ratio is not None else "pending"
        cosine_text = f"{cosine:.5f}" if cosine is not None else "pending"
        lines.append(f"| {tag} | {r['last_step']} | {ratio_text} | {cosine_text} | {r['best_raw_step']} | {r['best_relative_step']} |")
    share = next(iter(records.values()))["raw_baseline_top_two_group_share"]
    lines += ["", "## Interpretation and limits", "",
              f"The two largest step-0 group RMSEs contribute {share:.1%} of the raw macro-RMSE numerator. "
              "An equal arithmetic weight per group is therefore not equal sensitivity across groups. "
              "Inspect per-group trajectories before interpreting tiny changes in the aggregate.", "",
              "Relative group RMSE here divides each group's RMSE by that same group's step-0 RMSE. "
              "It is a diagnostic sensitivity check, not a proposed retroactive winner change. "
              "It also has limitations: small baseline errors can magnify changes. Larger independently seeded "
              "validation and per-group regression checks should precede a new selection rule.", "",
              "MSE versus Huber is matched on initialization, data stream, masks, and optimizer. "
              "The completed MTEB scores use different validation-selected epochs, so they establish a "
              "difference between these training-and-selection recipes, not a same-epoch effect or replicated result.", "",
              "Logit gradient norms measure output-space signals, not backbone gradient norms. "
              "Near-zero cosine is near orthogonality, not evidence of a strong negative conflict. "
              "Sampled parameter updates near LR times weight decay can be regularization rather than useful "
              "supervision; inspect gradients and the actual graph before assigning a cause.", "",
              "## Next discriminating measurements", "",
              "1. Finish both support-mask controls; compare per-prior and per-support results, not only aggregate RMSE.",
              "2. On a fixed probe, measure NLL and mean-MSE gradients at the decoder and early encoder, "
              "and separate optimizer updates from weight decay for the initialization heads.",
              "3. Compare same-step predictions and replicated recipes before attributing all the MTEB gap to the loss.",
              "4. Audit normalization/support counts in heavy-tail validation cases, then predeclare an expanded "
              "validation protocol that reports typical and tail errors separately without dropping difficult priors.",
              "5. Test a direct supervised initialization or explicit residual prediction route if useful solver "
              "parameters remain without task gradients; use causal branch ablations before widening the model.", "",
              "Detailed numbers and group trajectories: evidence.json. Figure: training_flow.png."]
    (out / "evidence.md").write_text("\n".join(lines) + "\n")
    print(out)
    print(json.dumps({tag: {k: v for k, v in r.items() if k not in {"validation", "sampled_updates"}}
                      for tag, r in records.items()}, indent=2))


if __name__ == "__main__":
    main()
