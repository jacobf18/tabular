"""CPU-only audit of the generic spectral task and its fixed validation cases."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rowcol_fixed_validation import load_or_build_validation, preserved_rng, apply_support_mask
from run_spectral_query_experiments import spectral_config
from tabimpute.prior.spectrum_training_set_generation import DecayingSpectrumGenerator
from train_direct_pure_rabel_curriculum import CurriculumPhase, MixedMatrixPrior
from spectral_support_normalization import normalize_remaining_support


def summary(values):
    return dict(zip(("min", "p50", "p95", "max"),
                    map(float, np.quantile(values, [0, .5, .95, 1]))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prior = spectral_config()
    results = {}
    with preserved_rng(20260912):
        for spec in prior["mixed_priors"]:
            values = {k: [] for k in ("rank", "effective_rank", "noise_to_signal_rms",
                                      "tail_energy_after_16", "tail_energy_after_64")}
            for _ in range(64):
                rows, cols = int(np.random.randint(80, 321)), int(np.random.randint(35, 101))
                cfg = dict(spec["config"], num_rows_low=rows, num_rows_high=rows,
                           num_cols_low=cols, num_cols_high=cols)
                bundle = DecayingSpectrumGenerator(cfg).generate_matrix_bundle()
                torch.testing.assert_close(bundle["target_matrix"], bundle["clean_signal"], rtol=0, atol=0)
                s = np.asarray(bundle["spectrum"], dtype=np.float64)
                energy = s * s
                p = energy / energy.sum()
                values["rank"].append(bundle["rank"])
                values["effective_rank"].append(float(np.exp(-(p * np.log(p.clip(1e-300))).sum())))
                values["noise_to_signal_rms"].append(bundle["noise_std"] / float(bundle["clean_signal"].square().mean().sqrt()))
                for k in (16, 64):
                    values[f"tail_energy_after_{k}"].append(float(energy[k:].sum() / energy.sum()))
            results[spec["name"]] = {k: summary(v) for k, v in values.items()}
    cases = load_or_build_validation(args.output / "audit_validation.pth", prior,
                                     seed=20260910, cases_per_group=4)
    groups = {}
    for case in cases:
        y = case["target"][case["mask"]]
        entry = groups.setdefault(case["group"], {"rms": [], "max_abs": [], "observed_rate": [], "n": 0})
        entry["rms"].append(float(y.square().mean().sqrt()))
        entry["max_abs"].append(float(y.abs().max()))
        entry["observed_rate"].append(float(torch.isfinite(case["source"]).float().mean()))
        entry["n"] += y.numel()
    episodes = {}
    phase = CurriculumPhase(name="audit", end_step=1, rows_low=80, rows_high=320,
                            cols_low=35, cols_high=100, batch_size=2)
    with preserved_rng(20260913):
        mixed = MixedMatrixPrior(phase=phase, device="cpu", missingness_type="mcar", mixed_config=prior)
        for name, child in zip(mixed.names, mixed.children):
            values = {key: [] for key in ("target_rms", "normalization_rms_gain", "target_max_abs",
                                          "worst_column_observed_count", "shrunk_target_rms",
                                          "shrunk_target_max_abs", "legacy_outside_mean_range",
                                          "shrunk_outside_mean_range")}
            for episode in range(64):
                (source, target, _, _, _), _ = child.get_batch()
                masked, normalized, mask, _ = apply_support_mask(source, target, "broad_support", 20261000 + episode)
                y = normalized[mask]
                values["target_rms"].append(float(y.square().mean().sqrt()))
                values["normalization_rms_gain"].append(float(y.square().mean().sqrt() /
                    target[mask].square().mean().sqrt().clamp_min(1e-12)))
                values["target_max_abs"].append(float(y.abs().max()))
                if torch.isnan(masked).equal(torch.isnan(source)):
                    shrunk_y = normalized[mask]
                else:
                    _, shrunk, _, _ = normalize_remaining_support(source, target, torch.isfinite(masked), 4.)
                    shrunk_y = shrunk[mask]
                values["shrunk_target_rms"].append(float(shrunk_y.square().mean().sqrt()))
                values["shrunk_target_max_abs"].append(float(shrunk_y.abs().max()))
                values["legacy_outside_mean_range"].append(float(((y < -105.34) | (y > 97.88)).float().mean()))
                values["shrunk_outside_mean_range"].append(float(((shrunk_y < -105.34) | (shrunk_y > 97.88)).float().mean()))
                locations = mask.nonzero()
                b, _, c = locations[y.abs().argmax()].tolist()
                values["worst_column_observed_count"].append(int(torch.isfinite(masked[b, :, c]).sum()))
            episodes[name] = {key: summary(value) for key, value in values.items()}
    payload = {"samples_per_training_family": 64, "training": results, "masked_training": episodes,
               "validation": {g: {k: summary(v) if isinstance(v, list) else v for k, v in values.items()}
                              for g, values in groups.items()},
               "cautions": ["Entropy effective rank is not algebraic rank.",
                            "Haar singular vectors do not span coherent or nonlinear geometries.",
                            "These diagnostics cannot establish real-world transfer."]}
    (args.output / "data_audit.json").write_text(json.dumps(payload, indent=2))
    lines = ["# General spectral data audit", "", "64 fresh matrices per training family; no benchmark data.", "",
             "| Family | Median effective rank | P95 noise/signal RMS | Median energy beyond rank 16 |",
             "|---|---:|---:|---:|"]
    for name, row in results.items():
        lines.append(f"| {name} | {row['effective_rank']['p50']:.2f} | "
                     f"{row['noise_to_signal_rms']['p95']:.3f} | {row['tail_energy_after_16']['p50']:.4f} |")
    lines += ["", "All sampled targets exactly equal their full clean signal, without truncation.",
              "Validation uses 72 cases: four matrices per family/shape and paired support masks.",
              "The source generator uses random orthonormal singular vectors. Spectral diversity does",
              "not imply diversity of singular-vector coherence, offsets, or nonlinear structure.",
              "Retain broad synthetic regression evaluations to detect this limitation."]
    lines += ["", "## Masked training episodes", "",
              "| Family | P95 target RMS | Max RMS amplification | Max absolute target |",
              "|---|---:|---:|---:|"]
    for name, row in episodes.items():
        lines.append(f"| {name} | {row['target_rms']['p95']:.2f} | "
                     f"{row['normalization_rms_gain']['max']:.2f} | {row['target_max_abs']['max']:.2f} |")
    lines += ["", "## Paired variance shrinkage (pseudo-count 4)", "",
              "| Family | Legacy max target RMS | Shrunk max target RMS | Shrunk max absolute target |",
              "|---|---:|---:|---:|"]
    for name, row in episodes.items():
        lines.append(f"| {name} | {row['target_rms']['max']:.2f} | "
                     f"{row['shrunk_target_rms']['max']:.2f} | {row['shrunk_target_max_abs']['max']:.2f} |")
    lines += ["", "This comparison measures scaling stability, not prediction accuracy.",
              "The shrinkage function uses only the supplied remaining observed entries.",
              "The synthetic generator's upstream normalization is unchanged. Public evaluation",
              "normalization must remain fixed when comparing trained checkpoint accuracy."]
    (args.output / "DATA_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"training": results, "validation_groups": len(groups)}, indent=2))


if __name__ == "__main__":
    main()
