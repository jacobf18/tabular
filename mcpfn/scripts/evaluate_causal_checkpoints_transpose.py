"""Transposed and orientation-averaged causal runs for the MatrixFormer checkpoints.

Mirrors benchmark/eval_dwip_transpose.py: {method}-T runs the model on the
transposed panel (time periods as rows, its columns standardized) and maps the
completion back; {method}-avg averages the standard and transposed completions.
Observed cells are restored in both. Standard-orientation predictions come from
evaluate_causal_checkpoints.py. The 98k parent is rerun as a control against its
saved -T and -avg predictions; the benchmark folders are only read.
"""

import time

import numpy as np
import pandas as pd
import torch

from evaluate_causal_checkpoints import (CHECKPOINTS, DESIGNS, LABELS, OUTPUT, PARENT, causal, compute_metrics,
                                         draw_dirs, paper, write_json)
from eval_dwip_transpose import impute_transpose


STALE = "dwip-v2-2"
CHECKPOINT_LABELS = {**LABELS, STALE: "MatrixFormer (Stale)"}
ORIENTATIONS = {"": "standard", "-T": "transposed", "-avg": "average"}


def predictions(method, path):
    return np.load(OUTPUT / "predictions" / method / f"{path.replace('/', '__')}.npz")


def evaluate(method, impute, family, label, path):
    """Transposed run per draw (timed, synchronized) plus the average with the saved standard run."""
    draws = draw_dirs(path)
    standard = predictions(method, path)
    warm = np.load(draws[0] / "observed.npy")
    impute_transpose(impute, warm, np.isnan(warm), True)
    rows, out = [], {"-T": {}, "-avg": {}}
    parity = {suffix: {"identical": 0, "max_abs_diff": 0.0} for suffix in out}
    for draw in draws:
        observed = np.load(draw / "observed.npy")
        missing = np.isnan(observed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        completed_t = impute_transpose(impute, observed, missing, True)
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        completed_avg = (0.5 * (standard[draw.name].astype(np.float32) + completed_t.astype(np.float32))).astype(np.float32)
        completed_avg[~missing] = observed[~missing]
        true, mask = np.load(draw / "true.npy"), np.load(draw / "treatment_mask.npy")
        for suffix, completed, seconds in (("-T", completed_t, runtime), ("-avg", completed_avg, np.nan)):
            out[suffix][draw.name] = completed
            if method == PARENT:
                saved = np.load(draw / f"{PARENT}{suffix}.npy")
                parity[suffix]["identical"] += int(np.array_equal(completed, saved))
                parity[suffix]["max_abs_diff"] = max(parity[suffix]["max_abs_diff"],
                                                     float(np.nanmax(np.abs(completed - saved))))
            rows.append({"family": family, "design": label, "path": path, "draw_name": draw.name,
                         "method": f"{method}{suffix}", "runtime_seconds": seconds,
                         **compute_metrics(true, completed, mask)})
    for suffix, arrays in out.items():
        target = OUTPUT / "predictions" / f"{method}{suffix}" / f"{path.replace('/', '__')}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
    return rows, ({"draws": len(draws), **parity} if method == PARENT else None)


def all_rows(new):
    """Every MatrixFormer orientation plus the core baselines, one frame per design."""
    standard = pd.read_csv(OUTPUT / "per_draw_metrics.csv")
    frames = {}
    for family, label, path in DESIGNS:
        saved = pd.read_csv(paper.BENCHMARK / path / "results/per_draw_metrics.csv")
        keep = set(paper.CORE) | {f"{STALE}{s}" for s in ORIENTATIONS}
        ours = pd.concat([standard[(standard["path"] == path) & (standard["method"] != PARENT)],
                          new[(new["path"] == path) & ~new["method"].str.startswith(PARENT)]])
        frames[path] = pd.concat([saved[saved["method"].isin(keep | {f"{PARENT}{s}" for s in ORIENTATIONS})],
                                  ours], ignore_index=True)[["draw_name", "method", *paper.METRICS, "runtime_seconds"]]
    return frames


def anchored_scores(frames):
    """Paper normalization: per-draw min-max over the seven core methods; others placed on that scale."""
    out = []
    for family, designs in paper.FAMILIES.items():
        for metric in paper.METRICS:
            means, ses = [], []
            for _, path in designs:
                wide = frames[path].pivot_table(index="draw_name", columns="method", values=metric)
                if wide.isna().any().any():
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


def paired(frames):
    """Raw change of -T and -avg against the same checkpoint's standard orientation, per design."""
    out = []
    for family, label, path in DESIGNS:
        wide = {m: g.set_index("draw_name") for m, g in frames[path].groupby("method")}
        for base in CHECKPOINT_LABELS:
            for suffix in ("-T", "-avg"):
                row = {"family": family, "design": label, "checkpoint": base, "orientation": ORIENTATIONS[suffix]}
                for metric in paper.METRICS:
                    diff = wide[f"{base}{suffix}"][metric] - wide[base][metric]
                    row |= {f"{metric}_pct_change": 100 * diff.mean() / wide[base][metric].mean(),
                            f"{metric}_z": diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))),
                            f"{metric}_win_rate": float((diff < 0).mean())}
                out.append(row)
    return pd.DataFrame(out)


def report(scores, paired_df, new, parity):
    lines = ["# Causal panels: transposed and averaged MatrixFormer checkpoints", "",
             "Same 12 designs and 3,000 draws. -T runs the model on the transposed panel; -avg averages the",
             "standard and transposed completions. Scores use the paper's seven-method scale (lower is better).",
             "MatrixFormer and Stale -T/-avg are the saved benchmark predictions; Recent and RecSys are new.", "",
             "## Control", ""]
    for suffix in ("-T", "-avg"):
        same = sum(p[suffix]["identical"] for p in parity.values())
        worst = max(p[suffix]["max_abs_diff"] for p in parity.values())
        lines.append(f"- MatrixFormer{suffix} rerun: {same}/{sum(p['draws'] for p in parity.values())} draws "
                     f"bit-identical to the saved predictions (max abs diff {worst:.3g}).")
    for metric, name in (("treated_block_rmse", "Treated-block RMSE"), ("tau_abs_error", "Effect error")):
        lines += ["", f"## Normalized {name}", "",
                  "| Checkpoint | Orientation | " + " | ".join(paper.FAMILIES) + " |",
                  "|---|---|" + "---:|" * len(paper.FAMILIES)]
        for base, base_label in CHECKPOINT_LABELS.items():
            for suffix, orientation in ORIENTATIONS.items():
                cells = []
                for family in paper.FAMILIES:
                    r = scores[(scores["method"] == f"{base}{suffix}") & (scores["family"] == family)
                               & (scores["metric"] == metric)].iloc[0]
                    cells.append(f"{r['score']:.3f} ± {r['ci95']:.3f}")
                lines.append(f"| {base_label} | {orientation} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired raw change against the same checkpoint's standard orientation", "",
              "Mean over designs of the per-design % change (negative is better); designs with |z| > 2 in brackets.", "",
              "| Checkpoint | Orientation | RMSE change | RMSE better/worse designs | Effect-error change | "
              "Effect-error better/worse designs |", "|---|---|---:|---:|---:|---:|"]
    for (base, orientation), g in paired_df.groupby(["checkpoint", "orientation"], sort=False):
        cells = []
        for metric in paper.METRICS:
            z = g[f"{metric}_z"]
            cells += [f"{g[f'{metric}_pct_change'].mean():+.2f}%", f"[{(z < -2).sum()}/{(z > 2).sum()}]"]
        lines.append(f"| {CHECKPOINT_LABELS[base]} | {orientation} | " + " | ".join(cells) + " |")
    runtime = new[new["method"].str.endswith("-T")].groupby("method")["runtime_seconds"].median()
    lines += ["", "## Runtime", "",
              "Median warm, CUDA-synchronized seconds per draw for the transposed run (including its "
              "standardization); the average costs the standard plus the transposed run.", ""]
    lines += [f"- {CHECKPOINT_LABELS[m.removesuffix('-T')]} -T: {s:.4f} s" for m, s in runtime.items()]
    (OUTPUT / "TRANSPOSE_RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
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
            print(f"{method} -T/-avg {path}: {len(new_rows) // 2} draws", flush=True)
        del impute
        torch.cuda.empty_cache()
    new = pd.DataFrame(rows)
    new.to_csv(OUTPUT / "per_draw_metrics_transpose.csv", index=False)
    write_json(OUTPUT / "control_parity_transpose.json", parity)
    frames = all_rows(new)
    scores, paired_df = anchored_scores(frames), paired(frames)
    scores.to_csv(OUTPUT / "normalized_scores_transpose.csv", index=False)
    paired_df.to_csv(OUTPUT / "paired_transpose_vs_standard.csv", index=False)
    report(scores, paired_df, new, parity)
    print((OUTPUT / "TRANSPOSE_RESULTS.md").read_text())


if __name__ == "__main__":
    main()
