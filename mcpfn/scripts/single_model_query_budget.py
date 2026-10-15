"""Training-only query-budget averaging for the original column-NRMSE majorizer."""

import random

import torch

from single_model_column_objective import column_majorizer


SPECIFICATION = {"query_budgets": [2, 4, 8, 16], "full_query_weight": .5,
                 "subsampled_query_weight": .5, "sampling": "uniform without replacement within each queried column",
                 "source_unchanged": True, "validation_unchanged": True}


def budget_for_seed(seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("Require a nonnegative integer query-budget seed")
    return random.Random(seed).choice(SPECIFICATION["query_budgets"])


def subsample_query_mask(mask, *, seed, budget):
    if (mask.ndim != 3 or mask.shape[0] != 1 or mask.dtype != torch.bool or not mask.any()
            or type(seed) is not int or not 0 <= seed < 2**63
            or type(budget) is not int or budget not in SPECIFICATION["query_budgets"]):
        raise ValueError("Require one nonempty query grid, a bounded integer seed and a registered budget")
    cpu_mask = mask.detach().cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    priorities = torch.rand(cpu_mask.shape, generator=generator).masked_fill(~cpu_mask, float("inf"))
    chosen = priorities.topk(min(budget, mask.shape[1]), dim=1, largest=False).indices
    subset = torch.zeros_like(cpu_mask).scatter_(1, chosen, True) & cpu_mask
    if not torch.equal(subset.sum(dim=1), cpu_mask.sum(dim=1).clamp_max(budget)):
        raise RuntimeError("Query subsampling changed column coverage")
    return subset.to(mask.device)


@torch.no_grad()
def query_budget_majorizer(truth, mask, teacher, *, seed):
    budget = budget_for_seed(seed)
    subset = subsample_query_mask(mask, seed=seed, budget=budget)
    full_weights, full_offset, full_diagnostics = column_majorizer(truth, mask, teacher)
    subset_weights, subset_offset, subset_diagnostics = column_majorizer(truth, subset, teacher)
    count_ratio = int(mask.sum()) / int(subset.sum())
    combined = (SPECIFICATION["full_query_weight"] * full_weights
                + SPECIFICATION["subsampled_query_weight"] * count_ratio * subset_weights.masked_fill(~subset, 0.))
    offset = (SPECIFICATION["full_query_weight"] * full_offset
              + SPECIFICATION["subsampled_query_weight"] * subset_offset)
    if not torch.isfinite(combined).all() or not torch.isfinite(offset):
        raise FloatingPointError("Nonfinite query-budget majorizer")
    return combined, offset, {"seed": seed, "budget": budget, "full": full_diagnostics,
                              "subsampled": subset_diagnostics, "full_to_subsampled_query_ratio": count_ratio,
                              "maximum_cell_coefficient": float(combined.max()),
                              "full_query_weight": SPECIFICATION["full_query_weight"],
                              "subsampled_query_weight": SPECIFICATION["subsampled_query_weight"]}
