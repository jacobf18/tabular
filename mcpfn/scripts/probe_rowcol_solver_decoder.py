"""Paired, fixed-query solver/decoder and branch ablations. No parameter updates."""
import argparse
from contextlib import ExitStack
import importlib.resources
import json
from pathlib import Path
from unittest.mock import patch

import torch

from rowcol_fixed_validation import evaluate_fixed_validation
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_rowcol_svt_hybrid import TabImputeModelRowColumnSVTHybrid


def run_probe(checkpoint, cache, output, device="cuda", model_overrides=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    cfg = json.loads((checkpoint.parent / "config.json").read_text())["model"]
    cfg = {**cfg, **(model_overrides or {})}
    model = TabImputeModelRowColumnSVTHybrid(**cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
    model.to(device).eval()
    criterion = FullSupportBarDistribution(torch.load(
        importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).to(device))
    cases = torch.load(cache, map_location="cpu", weights_only=True)["cases"]
    results = {}
    variants = ["full", "no_solver_features", "zero_initialization"]
    if getattr(model, "use_cell_query_decoder", False):
        variants.append("no_cell_query_decoder")
    if model.use_observed_attention_decoder:
        variants.append("no_observed_attention")
    if model.use_observed_neighbor_features:
        variants.append("no_neighbor_features")
    for variant in variants:
        path = output / f"{variant}.pth"
        print(f"[probe] {variant}", flush=True)
        with ExitStack() as stack:
            if variant == "no_solver_features":
                stack.enter_context(patch.object(model, "_svt_point_features",
                    side_effect=lambda point: point.new_zeros((*point.shape, 6), dtype=torch.float32)))
                if model.use_svt_magnitude_channel:
                    handle = model.svt_magnitude_proj.register_forward_hook(lambda m, i, o: torch.zeros_like(o))
                    stack.callback(handle.remove)
            elif variant == "zero_initialization":
                original = model._initial_grid
                stack.enter_context(patch.object(model, "_initial_grid",
                    side_effect=lambda *a, **kw: torch.zeros_like(original(*a, **kw))))
            elif variant == "no_observed_attention":
                stack.enter_context(patch.object(model, "use_observed_attention_decoder", False))
            elif variant == "no_cell_query_decoder":
                stack.enter_context(patch.object(model, "use_cell_query_decoder", False))
            elif variant == "no_neighbor_features":
                original_neighbor = model._observed_neighbor_point_features
                stack.enter_context(patch.object(model, "_observed_neighbor_point_features",
                    side_effect=lambda *a, **kw: torch.zeros_like(original_neighbor(*a, **kw))))
            metrics = evaluate_fixed_validation(model, criterion, cases, device, "amp_bfloat16", path)
        saved = torch.load(path, weights_only=True)["predictions"]
        sums = {}
        for case in cases:
            pred = saved[case["id"]]
            v = sums.setdefault(case["group"], [0., 0])
            v[0] += float((pred["svt_mean"].double() - pred["target"].double()).square().sum())
            v[1] += pred["target"].numel()
        metrics["svt_rmse_by_group"] = {g: (v[0] / v[1]) ** .5 for g, v in sums.items()}
        results[variant] = metrics
    (output / "results.json").write_text(json.dumps({"checkpoint": str(checkpoint),
        "cache": str(cache), "model_overrides": model_overrides, "ablations": results}, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = results["full"]
    names = list(full["groups"])
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(names, [full["groups"][g]["rmse"] for g in names], "o-", label="Decoded mean")
    ax.plot(names, [full["svt_rmse_by_group"][g] for g in names], "o-", label="Raw SVT")
    ax.set_yscale("log")
    ax.set_ylabel("RMSE (log scale)")
    ax.tick_params(axis="x", rotation=75)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "solver_vs_decoder.png", dpi=150)
    plt.close(fig)
    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_probe(args.checkpoint, args.cache, args.output, args.device)
