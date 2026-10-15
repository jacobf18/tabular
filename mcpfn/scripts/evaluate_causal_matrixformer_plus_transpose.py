"""Transposed and orientation-averaged MatrixFormer+ on the causal panels, and its best-orientation figure.

Mirrors evaluate_causal_checkpoints_transpose.py: matrixformer-plus-T runs the router on the transposed
panel (its columns standardized) and maps the completion back; matrixformer-plus-avg averages the
standard and transposed completions. Observed cells are restored in both. Both experts inside the
transposed router must match the saved -T predictions bit for bit. The orientation shown is picked by
the paper's rule (lowest mean normalized score over the six benchmark-metric panels), and the figure
shows it next to the six non-MatrixFormer core methods on the paper's unchanged seven-method scale.
"""

import time

import numpy as np
import pandas as pd
import torch

from evaluate_causal_matrixformer_plus import (DESIGNS, METHOD, OUTPUT, ROLES, SAVED, TwoExpertRouter,
                                               compute_metrics, draw_dirs, make_expert, paper, write_json)
from eval_dwip_transpose import impute_transpose


ORIENTATIONS = paper.ORIENTATIONS
VARIANTS = [f"{METHOD}{suffix}" for suffix in ORIENTATIONS]
FIGURE = OUTPUT / "causal_normalized_scores_matrixformer_plus_only"


def evaluate(router, family, label, path):
    """Transposed run per draw (timed, synchronized) plus the average with the saved standard run."""
    draws = draw_dirs(path)
    key = f"{path.replace('/', '__')}.npz"
    standard = np.load(OUTPUT / "predictions" / METHOD / key)
    saved = {name: np.load(SAVED / "predictions" / f"{name}-T" / key) for name in ROLES}
    warm = np.load(draws[0] / "observed.npy")
    impute_transpose(router.impute, warm, np.isnan(warm), True)
    rows, out, parity = [], {"-T": {}, "-avg": {}}, dict.fromkeys(ROLES, 0)
    for draw in draws:
        observed = np.load(draw / "observed.npy")
        missing = np.isnan(observed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        completed_t = impute_transpose(router.impute, observed, missing, True)
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        for name, role in ROLES.items():
            expert_t = impute_transpose(lambda _: router.last_result.predictions[role], observed, missing, True)
            parity[name] += int(np.array_equal(expert_t, saved[name][draw.name]))
        weight = router.last_result.decisions["observed"]["specialist_weight"]
        completed_avg = (0.5 * (standard[draw.name].astype(np.float32) + completed_t.astype(np.float32))).astype(np.float32)
        completed_avg[~missing] = observed[~missing]
        true, mask = np.load(draw / "true.npy"), np.load(draw / "treatment_mask.npy")
        for suffix, completed, seconds in (("-T", completed_t, runtime), ("-avg", completed_avg, np.nan)):
            out[suffix][draw.name] = completed
            rows.append({"family": family, "design": label, "path": path, "draw_name": draw.name,
                         "method": f"{METHOD}{suffix}", "runtime_seconds": seconds,
                         **compute_metrics(true, completed, mask),
                         "specialist_weight": weight if suffix == "-T" else np.nan})
    for suffix, arrays in out.items():
        target = OUTPUT / "predictions" / f"{METHOD}{suffix}" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
    return rows, {"draws": len(draws), **{f"{name}-T_identical": count for name, count in parity.items()}}


def scores(plus):
    """The paper's normalized_scores with the three MatrixFormer+ orientations as candidates."""
    paper.CANDIDATES.extend(VARIANTS)
    out = []
    for family, designs in paper.FAMILIES.items():
        frames = []
        for _, path in designs:
            saved, _ = paper.load(path)
            frames.append(pd.concat([saved, plus[plus["path"] == path]], ignore_index=True))
        for metric in paper.METRICS:
            mean, se, n = paper.normalized_scores(frames, metric)
            for m in [*paper.CORE, *VARIANTS]:
                out.append({"family": family, "metric": metric, "method": m, "score": mean[m], "se": se[m],
                            "ci95": 1.96 * se[m], "n_scenarios": len(frames), "n_draws": n})
    return pd.DataFrame(out)


def main():
    if (OUTPUT / "per_draw_metrics_transpose.csv").exists():
        raise SystemExit(f"{OUTPUT} already has transpose results")
    router = TwoExpertRouter(*(make_expert(name) for name in ROLES), normalization="column",
                             policy="observed_blend")
    rows, parity = [], {}
    for family, label, path in DESIGNS:
        new_rows, parity[path] = evaluate(router, family, label, path)
        rows += new_rows
        print(f"{path}: {len(new_rows) // 2} draws, parity {parity[path]}", flush=True)
    new = pd.DataFrame(rows)
    new.to_csv(OUTPUT / "per_draw_metrics_transpose.csv", index=False)
    write_json(OUTPUT / "control_parity_transpose.json", parity)
    broken = {p: c for p, c in parity.items() if any(c[f"{n}-T_identical"] != c["draws"] for n in ROLES)}
    if broken:
        raise SystemExit(f"Transposed expert outputs differ from the saved predictions, not scoring: {broken}")

    standard = pd.read_csv(OUTPUT / "per_draw_metrics.csv")
    score_df = scores(pd.concat([standard, new], ignore_index=True))
    means = score_df[score_df["method"].isin(VARIANTS)].groupby("method")["score"].mean()
    best = means.idxmin()
    suffix = best.removeprefix(METHOD)
    score_df["label"] = score_df["method"].map(
        {**paper.LABELS, **{f"{METHOD}{s}": "MatrixFormer+" + (f" ({ORIENTATIONS[s]})" if s else "")
                            for s in ORIENTATIONS}})
    score_df["shown"] = score_df["method"].isin([best, *paper.BASELINES])
    score_df.to_csv(OUTPUT / "normalized_scores_orientations.csv", index=False, float_format="%.6f")
    write_json(OUTPUT / "orientation_choice.json",
               {"rule": "lowest mean normalized score over the six benchmark-metric panels (paper rule)",
                "mean_scores": {m: float(v) for m, v in means.items()}, "chosen": best,
                "orientation": ORIENTATIONS[suffix]})

    paper.LABELS.update(dict(zip(score_df["method"], score_df["label"])))
    methods = [best, *paper.BASELINES]
    for ext in ("pdf", "png"):
        paper.plot(score_df, FIGURE.with_suffix(f".{ext}"), methods)

    table = score_df[score_df["method"].isin(VARIANTS)].pivot_table(
        index="label", columns=["metric", "family"], values="score", sort=False)
    print(table.round(3).to_string())
    print(f"Mean over panels: {means.round(4).to_dict()}; chosen {best}")
    print(f"Figure: {FIGURE.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
