"""CPU-only coverage audit of unchanged mixed-geometry priors and df1 support masking."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from general_spectral_geometry import GeometrySpectrumGenerator, install_geometry_generator
from rowcol_fixed_validation import preserved_rng
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, write_json
from spectral_support_normalization import apply_shrunk_support_mask
from train_direct_pure_rabel_curriculum import CurriculumPhase, MixedMatrixPrior


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Summary requires nonempty finite values")
    return dict(zip(("min", "p50", "p95", "p99", "max"),
                    map(float, np.quantile(array, [0, .5, .95, .99, 1]))))


def support_statistics(source, target, loss_mask):
    observed = torch.isfinite(source)
    if source.ndim != 3 or source.shape != target.shape or source.shape != loss_mask.shape:
        raise ValueError("Expected matching B x R x C tensors")
    if not observed.any() or not loss_mask.any() or not torch.isfinite(target).all():
        raise ValueError("Need observed entries and finite held-out targets")
    row_count = observed.sum(2, keepdim=True).expand_as(observed)[loss_mask]
    column_count = observed.sum(1, keepdim=True).expand_as(observed)[loss_mask]
    residual = (source - target)[observed]
    return {"observed_fraction": float(observed.float().mean()),
            "observed_residual_rms": float(residual.square().mean().sqrt()),
            "observed_residual_max_abs": float(residual.abs().max()),
            "loss_target_rms": float(target[loss_mask].square().mean().sqrt()),
            "loss_target_max_abs": float(target[loss_mask].abs().max()),
            "loss_cells_zero_row_support_fraction": float((row_count == 0).float().mean()),
            "loss_cells_at_most_one_row_support_fraction": float((row_count <= 1).float().mean()),
            "loss_cells_zero_column_support_fraction": float((column_count == 0).float().mean()),
            "loss_cells_at_most_one_column_support_fraction": float((column_count <= 1).float().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=REPO / "runs/information_flow_diagnostics/dwip_spectral_coverage_20260916")
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    torch.set_num_threads(2)
    origin = REPO / "runs/objective_queues/dwip_spectral_20260915/manifest.json"
    registered = json.loads(origin.read_text())
    prior = registered["prior"]
    command = registered["commands"]["dwip1e5"]
    variance_df = float(command[command.index("--support-variance-df") + 1])
    support_mode = command[command.index("--support-mask-mode") + 1]
    if variance_df != 1 or support_mode != "broad_support":
        raise ValueError("Expected the unchanged DWIP df1/broad-support setup")
    if any(spec["config"].get("spectrum_basis_kind") != "mixed" for spec in prior["mixed_priors"]):
        raise ValueError("Expected current mixed-geometry training priors")
    sources = [Path(__file__), REPO / "scripts/general_spectral_geometry.py",
               REPO / "scripts/spectral_support_normalization.py", REPO / "scripts/rowcol_fixed_validation.py",
               REPO / "scripts/train_direct_pure_rabel_curriculum.py",
               REPO / "src/tabimpute/prior/spectrum_training_set_generation.py",
               REPO / "src/tabimpute/model/encoders.py"]
    manifest = {"origin_sha256": sha256(origin), "prior": prior, "samples_per_family": args.samples,
                "seed": args.seed, "support_mode": support_mode, "variance_df": variance_df,
                "shapes": {"rows": [80, 320], "columns": [35, 100], "batch_size": 2},
                "source_sha256": {str(path): sha256(path) for path in sources},
                "purpose": "Generic coverage diagnosis only; no benchmark data, training or checkpoint selection",
                "weighting": "Equal samples per family; quantiles are stratified diagnostics, not mixture-weighted population estimates"}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Audit inputs changed; use a new output directory")
    write_json(manifest_path, manifest)
    install_geometry_generator()
    phase = CurriculumPhase(name="coverage_audit", end_step=1, rows_low=80, rows_high=320,
                            cols_low=35, cols_high=100, batch_size=2)
    results = {}
    with preserved_rng(args.seed):
        mixed = MixedMatrixPrior(phase=phase, device="cpu", missingness_type="mcar", mixed_config=prior)
        for spec, name, child in zip(prior["mixed_priors"], mixed.names, mixed.children):
            write_json(args.output / "status.json", {"state": "auditing", "family": name})
            raw, episodes = [], []
            for episode in range(args.samples):
                rows, columns = int(np.random.randint(80, 321)), int(np.random.randint(35, 101))
                config = dict(spec["config"], num_rows_low=rows, num_rows_high=rows,
                              num_cols_low=columns, num_cols_high=columns)
                bundle = GeometrySpectrumGenerator(config).generate_matrix_bundle()
                clean = bundle["clean_signal"]
                torch.testing.assert_close(bundle["target_matrix"], clean, rtol=0, atol=0)
                noise = bundle["full_matrix"] - clean
                centered_scale = clean.std(0, unbiased=False).clamp_min(1e-12)
                column_ratios = noise.square().mean(0).sqrt() / centered_scale
                raw.append({"noise_to_signal_rms": float(noise.square().mean().sqrt() / clean.square().mean().sqrt()),
                            "column_noise_to_centered_signal_std_p50": float(column_ratios.median()),
                            "column_noise_to_centered_signal_std_p95": float(torch.quantile(column_ratios, .95)),
                            "column_noise_to_centered_signal_std_max": float(column_ratios.max()),
                            "max_noise_in_centered_signal_std": float((noise.abs() / centered_scale).max())})
                (source, target, _, _, _), _ = child.get_batch()
                masked, normalized, loss_mask, stats = apply_shrunk_support_mask(
                    source, target, support_mode, args.seed + 10000000 + episode, variance_df=variance_df)
                episodes.append({**support_statistics(masked, normalized, loss_mask),
                                 "support_episode": stats["support_episode"],
                                 "support_transposed": stats.get("support_transposed", 0.)})
            results[name] = {"raw": {key: quantiles([row[key] for row in raw]) for key in raw[0]},
                             "masked": {key: quantiles([row[key] for row in episodes]) for key in episodes[0]},
                             "raw_records": raw, "masked_records": episodes}
            write_json(args.output / "results.json", results)
            print(name, "noise/signal", results[name]["raw"]["noise_to_signal_rms"],
                  "zero-row-loss", results[name]["masked"]["loss_cells_zero_row_support_fraction"], flush=True)
    write_json(args.output / "status.json", {"state": "complete", "families": len(results)})


if __name__ == "__main__":
    main()
