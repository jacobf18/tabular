"""Reproducible support masks and fixed synthetic validation for direct models."""
from __future__ import annotations

import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.prior.spectrum_training_set_generation import SpectrumMissingnessPrior
from tabimpute.model.prediction_distribution import prediction_distribution


@contextmanager
def preserved_rng(seed=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def apply_support_mask(source, target, mode="mixed", seed=None):
    if mode not in {"mixed", "broad_support", "sparse_query"}:
        raise ValueError(f"Unknown support mode: {mode}")
    if source.shape != target.shape or source.ndim != 3:
        raise ValueError("source and target must be matching B x R x C tensors")
    with preserved_rng(seed):
        if mode == "mixed" or (mode == "broad_support" and random.random() < 0.25):
            return source, target, torch.isnan(source), {"support_episode": 0.0}
        transpose = random.random() < 0.5
        src = source.transpose(1, 2) if transpose else source
        keep = ~torch.isnan(src)
        query_mask = torch.zeros_like(keep)
        for b in range(src.shape[0]):
            rows, cols = src.shape[1:]
            n_query = max(1, min(rows - 1, round(rows * random.uniform(0.10, 0.40))))
            query = torch.randperm(rows, device=src.device)[:n_query]
            fraction = math.exp(random.uniform(math.log(0.01), math.log(0.60)))
            k = max(1, min(cols - 1, round(cols * fraction)))
            allowed = torch.zeros((n_query, cols), device=src.device, dtype=torch.bool)
            if random.random() < 0.5:
                allowed[:, torch.randperm(cols, device=src.device)[:k]] = True
            else:
                for row in range(n_query):
                    allowed[row, torch.randperm(cols, device=src.device)[:k]] = True
            keep[b, query] &= allowed
            query_mask[b, query] = True
        if transpose:
            keep, query_mask = keep.transpose(1, 2), query_mask.transpose(1, 2)
        masked = source.float().masked_fill(~keep, float("nan"))
        counts = keep.sum(dim=1, keepdim=True)
        mean = masked.nan_to_num().sum(dim=1, keepdim=True) / counts.clamp_min(1)
        centered = torch.where(keep, masked - mean, 0.0)
        std = (centered.square().sum(dim=1, keepdim=True) / counts.clamp_min(1)).sqrt()
        std = torch.where((counts >= 2) & (std > 1e-3), std, 1.0)
        loss_mask = query_mask & ~keep
        if not loss_mask.any():
            loss_mask = ~keep
        stats = {
            "support_episode": 1.0,
            "support_transposed": float(transpose),
            "support_query_observed_rate": float(keep[query_mask].float().mean()),
            "support_observed_rate": float(keep.float().mean()),
        }
        return (masked - mean) / std, (target.float() - mean) / std, loss_mask, stats


def load_or_build_validation(path, base_prior_config, seed=20260906, cases_per_group=2):
    path = Path(path)
    manifest = {"version": 1, "seed": seed, "cases_per_group": cases_per_group,
                "prior": base_prior_config, "max_targets": 1024}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["fingerprint"] != fingerprint:
            raise ValueError(f"Validation cache configuration mismatch: {path}")
        return cached["cases"]
    cases = []
    specs = [(g, r, c, False) for g in ("latent_factor", "nonlinear_factor", "robust_pca")
             for r, c in ((80, 40), (240, 80))]
    specs += [("spectrum", 300, 300, True), ("spectrum", 500, 500, True)]
    spectral_specs = base_prior_config.get("validation_spectral_specs", {})
    if spectral_specs:
        specs = [(name, r, c, monitor) for name in spectral_specs
                 for r, c, monitor in ((80, 40, False), (240, 80, False), (500, 500, True))]
    with preserved_rng(seed):
        for generator, rows, cols, monitor in specs:
            for replicate in range(cases_per_group):
                cfg = {**base_prior_config, "num_rows_low": rows, "num_rows_high": rows,
                       "num_cols_low": cols, "num_cols_high": cols}
                if generator in spectral_specs:
                    cfg.update(spectral_specs[generator])
                    prior = SpectrumMissingnessPrior(config=cfg, batch_size=1, device="cpu",
                        missingness_type=cfg.pop("missingness_type", "mcar"), entry_wise_features=False)
                elif generator == "spectrum":
                    cfg.update(spectrum_rank_low=80, spectrum_rank_high=min(rows, cols),
                               spectrum_target_rank=min(rows, cols),
                               spectrum_noise_std_low=0.0, spectrum_noise_std_high=0.0,
                               spectrum_alpha_low=0.5, spectrum_alpha_high=1.5)
                    prior = SpectrumMissingnessPrior(config=cfg, batch_size=1, device="cpu",
                                                     missingness_type="mcar", entry_wise_features=False)
                else:
                    prior = MissingnessPrior(config=cfg, generator_type=generator,
                                             missingness_type="profile_mixed", batch_size=1,
                                             device="cpu", entry_wise_features=False)
                (src, target, _, _, _), _ = prior.get_batch()
                for mode in ("mixed", "sparse_query"):
                    x, y, mask, stats = apply_support_mask(src, target, mode, seed + len(cases))
                    indices = mask.flatten().nonzero().flatten()
                    if not indices.numel():
                        raise ValueError("Validation case has no targets")
                    if indices.numel() > manifest["max_targets"]:
                        indices = indices[torch.randperm(indices.numel())[:manifest["max_targets"]]]
                        mask = torch.zeros_like(mask).flatten()
                        mask[indices] = True
                        mask = mask.view_as(x)
                    group = f"{generator}_{rows}x{cols}_{mode}"
                    cases.append({"id": f"{group}_{replicate}", "group": group,
                                  "monitor_only": monitor, "source": x.cpu(), "target": y.cpu(),
                                  "mask": mask.cpu(), "stats": stats})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint": fingerprint, "manifest": manifest, "cases": cases}, path)
    path.with_suffix(".json").write_text(json.dumps({**manifest, "fingerprint": fingerprint,
                                                  "case_ids": [c["id"] for c in cases]}, indent=2))
    return cases


@torch.no_grad()
def evaluate_fixed_validation(model, criterion, cases, device, precision_mode, prediction_path=None):
    device = torch.device(device)
    dtype = torch.bfloat16 if precision_mode == "full_bfloat16" else torch.float32
    modes = [(module, module.training) for module in model.modules()]
    totals, predictions, slices = {}, {}, {}
    with preserved_rng():
        model.eval()
        try:
            for case in cases:
                src = case["source"].to(device=device, dtype=dtype)
                mask = case["mask"].to(device)
                target = case["target"].to(device)[mask].float()
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16 if precision_mode == "amp_float16" else torch.bfloat16,
                                    enabled=precision_mode.startswith("amp_")):
                    logits = model(src, decode_mask=mask)
                output_criterion = prediction_distribution(model, criterion)
                mean = output_criterion.mean(logits.float())
                nll = output_criterion(logits.float(), target)
                error = mean - target
                if not torch.isfinite(error).all() or not torch.isfinite(nll).all():
                    raise FloatingPointError(f"Nonfinite validation output: {case['id']}")
                bucket = totals.setdefault(case["group"], {"sse": 0.0, "nll_sum": 0.0,
                                                           "n": 0, "monitor_only": case["monitor_only"]})
                bucket["sse"] += float(error.double().square().sum())
                bucket["nll_sum"] += float(nll.double().sum())
                bucket["n"] += target.numel()
                centers = getattr(criterion, "_full_bucket_means", criterion.bucket_means).float()
                outside = (target < centers.min()) | (target > centers.max())
                floor_error = target - target.clamp(centers.min(), centers.max())
                bucket["fixed_range_floor_sse"] = bucket.get("fixed_range_floor_sse", 0.) + float(
                    floor_error.double().square().sum())
                for name, subset in (("central", ~outside), ("outside_fixed_mean_range", outside)):
                    entry = slices.setdefault(case["group"], {}).setdefault(name, {"sse": 0., "n": 0})
                    entry["sse"] += float(error[subset].double().square().sum())
                    entry["n"] += int(subset.sum())
                if prediction_path:
                    predictions[case["id"]] = {"mean": mean.cpu(), "target": target.cpu(),
                                                "flat_indices": mask.flatten().nonzero().flatten().cpu()}
                    solver = getattr(model, "last_svt_points", None)
                    if solver is not None:
                        predictions[case["id"]]["svt_mean"] = solver.cpu()
                del logits
        finally:
            for module, training in modes:
                module.training = training
    groups = {name: {"rmse": math.sqrt(v["sse"] / v["n"]),
                     "nll": v["nll_sum"] / v["n"], "n": v["n"],
                     "fixed_component_mean_oracle_rmse": math.sqrt(v["fixed_range_floor_sse"] / v["n"]),
                     "monitor_only": v["monitor_only"]} for name, v in totals.items()}
    selected = [v for v in groups.values() if not v["monitor_only"]]
    result = {"macro_rmse": float(np.mean([v["rmse"] for v in selected])),
              "macro_nll": float(np.mean([v["nll"] for v in selected])), "groups": groups}
    result["range_slices"] = {
        group: {name: {"n": v["n"], "rmse": math.sqrt(v["sse"] / v["n"]) if v["n"] else None}
                for name, v in values.items()} for group, values in slices.items()}
    if prediction_path:
        torch.save({"metrics": result, "predictions": predictions}, prediction_path)
    return result
