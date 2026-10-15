"""Registered model-free paired probe of synthetic imputation-mask coverage."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from dwip_teacher_mean_training import build_training_prior, control_recipe, verify_hashes
from general_spectral_geometry import GeometrySpectrumGenerator
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_csv, write_json
from rowcol_fixed_validation import preserved_rng
from spectral_support_normalization import normalize_remaining_support
from streamed_large_matrix_training import base
from tabimpute.model.encoders import normalize_data
from tabimpute.prior.mar_block_missing import mar_block_config
from tabimpute.prior.mar_missing import mar_config
from tabimpute.prior.training_set_generation import (
    LatentFactorPrior, MCARPattern, MissingnessPrior, ProfileMixedPattern,
)


FAMILIES = tuple(ProfileMixedPattern.pattern_classes) + (
    "mar_neural", "mar_block_neural", "mnar_latent_factor", "mnar_polarization", "mnar_soft_polarization",
)
SEED = 62100601


def audit_prior(prior):
    probabilities = dict.fromkeys(FAMILIES, 0.)
    children = []
    total = sum(prior.weights)
    if len(prior.children) != len(prior.weights) or total <= 0:
        raise ValueError("Require a complete positive prior mixture")
    for name, weight, child in zip(prior.names, prior.weights, prior.children):
        inducer = child.missingness_inducer
        if isinstance(inducer, ProfileMixedPattern):
            conditional = dict(inducer.weights)
        elif isinstance(inducer, MCARPattern):
            conditional = {"mcar": 1.}
        else:
            raise ValueError("Unexpected training mechanism; do not silently omit it")
        if not np.isclose(sum(conditional.values()), 1.):
            raise ValueError("Invalid conditional mechanism probabilities")
        for family, probability in conditional.items():
            probabilities[family] += weight / total * probability
        children.append({"name": name, "weight": weight, "config": deepcopy(child.config),
                         "generator": type(child.generator).__name__, "mechanisms": conditional})
    return {"children": children, "probabilities": probabilities,
            "absent_families": [family for family, probability in probabilities.items() if probability == 0]}


def pattern_configuration(generator_config, family, requested_missingness):
    if family not in FAMILIES or requested_missingness not in (.2, .6):
        raise ValueError("Require a registered family and requested missingness")
    result = ProfileMixedPattern(deepcopy(generator_config))._build_pattern_config(requested_missingness)
    result["mar_config"] = deepcopy(mar_config)
    result["mar_block_config"] = deepcopy(mar_block_config)
    return result


def generate_pair(specification):
    with preserved_rng(specification["seed"]):
        config = deepcopy(specification["generator_config"])
        if specification["kind"] == "spectrum_decay":
            bundle = GeometrySpectrumGenerator(config).generate_matrix_bundle()
            source, target = bundle["full_matrix"], bundle["target_matrix"]
        elif specification["kind"] == "latent_factor":
            source = torch.as_tensor(LatentFactorPrior(config).generate_complete_matrix().to_numpy(), dtype=torch.float32)
            target = source.clone()
        else:
            raise ValueError("Unknown registered generator")
    if source.shape != target.shape or not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("Require finite aligned clean/noisy arrays")
    mechanism_source = normalize_data(source.unsqueeze(1)).squeeze(1)
    return source, target, mechanism_source


def apply_family(mechanism_source, configuration, family, seed):
    if family not in FAMILIES or not torch.isfinite(mechanism_source).all():
        raise ValueError("Require a registered family and a complete finite matrix")
    constructor = ProfileMixedPattern.pattern_classes.get(family, MissingnessPrior.missingness_map[family])
    with preserved_rng(seed):
        masked = constructor(deepcopy(configuration))._induce_missingness(mechanism_source.clone())
    if masked.shape != mechanism_source.shape or torch.isinf(masked).any():
        raise ValueError("A masking mechanism returned invalid values")
    query = torch.isnan(masked)
    if not torch.equal(masked[~query], mechanism_source[~query]):
        raise ValueError("The mechanism changed an observed value")
    return query


def prepare_input(source, target, query):
    normalized, truth, mean, scale = normalize_remaining_support(
        source.unsqueeze(0), target.unsqueeze(0), (~query).unsqueeze(0), variance_df=1.,
    )
    if not torch.isfinite(truth).all() or not torch.equal(torch.isnan(normalized[0]), query):
        raise ValueError("Invalid observed-only normalization")
    return normalized[0], truth[0], mean[0], scale[0]


def average_defined(values):
    defined = [value for value in values if value is not None]
    if any(not np.isfinite(value) for value in defined):
        raise ValueError("Undefined values must be explicit nulls")
    return float(np.mean(defined)) if defined else None


def describe_mask(source, target, query):
    if source.ndim != 2 or source.shape != target.shape or source.shape != query.shape or query.dtype != torch.bool:
        raise ValueError("Require aligned source, clean target and Boolean mask")
    rows, columns = source.shape
    ratios, shifts = [], []
    for column in range(columns):
        hidden = target[:, column][query[:, column]].double()
        observed = target[:, column][~query[:, column]].double()
        ratio = shift = None
        if len(hidden) >= 2 and len(observed) >= 2 and observed.std(correction=0) > 0:
            ratio = float(hidden.std(correction=0) / observed.std(correction=0))
            shift = float((hidden.mean() - observed.mean()).abs() / observed.std(correction=0))
        ratios.append(ratio)
        shifts.append(shift)
    lower, upper = torch.quantile(source, .25, dim=0), torch.quantile(source, .75, dim=0)
    central = (source > lower) & (source < upper)
    tail = ~central
    counts = query.sum(0)
    return {"rows": rows, "columns": columns, "queries": int(query.sum()),
            "observations": int((~query).sum()), "effective_missingness": float(query.double().mean()),
            "empty_query_columns": int((counts == 0).sum()), "singleton_query_columns": int((counts == 1).sum()),
            "empty_observed_columns": int((counts == rows).sum()),
            "std_ratio_columns": sum(value is not None for value in ratios),
            "query_to_observed_target_std": average_defined(ratios),
            "absolute_target_mean_shift_observed_sd": average_defined(shifts),
            "central_missing_fraction": float(query[central].double().mean()) if central.any() else None,
            "tail_missing_fraction": float(query[tail].double().mean()) if tail.any() else None,
            "source_target_rmse": float((source.double() - target.double()).square().mean().sqrt())}


def source_hashes():
    paths = set()
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.suffix == ".py" and path.is_relative_to(REPO):
                paths.add(path)
    return {str(path): sha256(path) for path in sorted(paths)}


def register(output):
    if output.exists():
        raise FileExistsError("Never replace a registered probe")
    args = control_recipe()
    contexts, prior_audit = [], None
    for shape_index, (rows, columns) in enumerate(((24, 12), (48, 24))):
        phase = base.CurriculumPhase(name="coverage_probe", end_step=1, rows_low=rows, rows_high=rows,
                                     cols_low=columns, cols_high=columns, batch_size=1)
        prior = build_training_prior(args, phase=phase, device="cpu")
        audit = audit_prior(prior)
        if prior_audit is None:
            prior_audit = audit
        if audit["probabilities"] != prior_audit["probabilities"]:
            raise ValueError("Shape unexpectedly changed prior coverage")
        for component_index, child_index in enumerate((0, 2, 4, 6)):
            child = audit["children"][child_index]
            for replica in range(2):
                index = len(contexts)
                seed = SEED + 100 * index
                missingness = (.2, .6)[replica]
                contexts.append({"id": index, "seed": seed, "component": prior.names[child_index],
                                 "kind": "latent_factor" if component_index == 3 else "spectrum_decay",
                                 "requested_shape": [rows, columns], "replica": replica,
                                 "shape_group": shape_index, "generator_config": child["config"],
                                 "requested_missingness": missingness,
                                 "patterns": [{"family": family, "seed": seed + 1 + family_index,
                                               "config": pattern_configuration(child["config"], family, missingness)}
                                              for family_index, family in enumerate(FAMILIES)]})
    config = {"schema": 1, "purpose": "model-free paired synthetic missingness coverage; not model selection",
              "registered_unix": time.time(), "seed": SEED, "families": list(FAMILIES),
              "contexts": contexts, "expected_cases": len(contexts) * len(FAMILIES), "prior_audit": prior_audit,
              "mask_input": "existing normalize_data on complete synthetic noisy source, as in original prior",
              "model_normalization": "raw observed-source per-column mean and variance shrinkage DF1; no extra support masks",
              "target": "clean spectral target, or identical latent target; never noisy spectral input as truth",
              "undefined_statistics": "null with counts; retain every case; never redraw or drop",
              "limitations": ["compact imputation-only contexts, not a large-context or recommender prior audit",
                              "requested missingness is not calibrated; several original families ignore p_missing",
                              "neural configurations retain original seed42, including the block internal network",
                              "latent anchor intrinsic shape overrides retained; actual dimensions reported",
                              "DF1 normalization isolates masks; it does not reproduce full training support augmentation",
                              "descriptive synthetic evidence is not proof of accuracy, real causality, or improvement"],
              "neural_configuration_sources": [str(REPO / "src/tabimpute/prior/mar_missing.py"),
                                               str(REPO / "src/tabimpute/prior/mar_block_missing.py")],
              "source_hashes": source_hashes()}
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"state": "registered", "config_sha256": sha256(output / "config.json")})


def run(output):
    config = json.loads((output / "config.json").read_text())
    status = json.loads((output / "status.json").read_text())
    if status["state"] != "registered" or status["config_sha256"] != sha256(output / "config.json"):
        raise ValueError("Require an untouched registered probe; never overwrite attempts")
    verify_hashes(config["source_hashes"])
    if list(FAMILIES) != config["families"] or config["expected_cases"] != len(config["contexts"]) * len(FAMILIES):
        raise ValueError("Changed family coverage")
    write_json(output / "status.json", {"state": "running", "pid": os.getpid(), "updated_unix": time.time()})
    records, manifest = [], []
    try:
        for context in config["contexts"]:
            source, target, mechanism_source = generate_pair(context)
            folder = output / f"context_{context['id']:02d}"
            folder.mkdir()
            raw_path = folder / "raw_pair.npz"
            atomic_npz(raw_path, source=source.numpy(), target=target.numpy(), mechanism_source=mechanism_source.numpy())
            manifest.append({"path": str(raw_path), "sha256": sha256(raw_path)})
            for pattern in context["patterns"]:
                family = pattern["family"]
                query = apply_family(mechanism_source, pattern["config"], family, pattern["seed"])
                normalized, truth, mean, scale = prepare_input(source, target, query)
                case_path = folder / f"{family}.npz"
                atomic_npz(case_path, query=query.numpy(), model_source=normalized.numpy(), model_target=truth.numpy(),
                           observed_mean=mean.numpy(), observed_scale=scale.numpy())
                manifest.append({"path": str(case_path), "sha256": sha256(case_path)})
                record = {"context": context["id"], "component": context["component"], "family": family,
                          "requested_missingness": context["requested_missingness"],
                          "prior_probability": config["prior_audit"]["probabilities"][family],
                          **describe_mask(source, target, query)}
                records.append(record)
                print(json.dumps(record, allow_nan=False), flush=True)
                write_json(output / "status.json", {"state": "running", "pid": os.getpid(),
                                                     "completed_cases": len(records), "updated_unix": time.time()})
        if len(records) != config["expected_cases"]:
            raise ValueError("Incomplete paired probe")
        aggregates = []
        statistics = ("effective_missingness", "query_to_observed_target_std", "absolute_target_mean_shift_observed_sd",
                      "central_missing_fraction", "tail_missing_fraction")
        for family in FAMILIES:
            selected = [row for row in records if row["family"] == family]
            aggregate = {"family": family, "prior_probability": config["prior_audit"]["probabilities"][family],
                         "cases": len(selected)}
            for statistic in statistics:
                aggregate[statistic] = average_defined([row[statistic] for row in selected])
                aggregate[f"{statistic}_defined_cases"] = sum(row[statistic] is not None for row in selected)
            aggregates.append(aggregate)
        verify_hashes(config["source_hashes"])
        write_json(output / "per_case.json", records)
        write_csv(output / "per_case.csv", records)
        write_csv(output / "per_family.csv", aggregates)
        write_json(output / "manifest.json", {"config_sha256": sha256(output / "config.json"), "files": manifest})
        lines = ["# Synthetic missingness coverage probe", "", "No model inference, training, real data or checkpoint selection.", "",
                 "All 16 raw pairs × 12 families retained. Family statistics are macro means over defined cases; CSVs include coverage.", "",
                 "| Family | Base-prior probability | Cases | Missing fraction | Query/observed target SD | Central missing | Tail missing |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for row in aggregates:
            values = [f"{row[key]:.4f}" if row[key] is not None else "undefined" for key in
                      ("effective_missingness", "query_to_observed_target_std", "central_missing_fraction", "tail_missing_fraction")]
            lines.append(f"| {row['family']} | {row['prior_probability']:.3f} | {row['cases']} | " + " | ".join(values) + " |")
        lines += ["", "## Limitations", "", *[f"- {item}." for item in config["limitations"]], "",
                  "Full raw noisy/clean pairs, masks, observed-only normalization parameters and model inputs/targets are retained.",
                  "This is not evidence that changing the missingness mixture improves any model."]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
        write_json(output / "status.json", {"state": "complete", "cases": len(records), "contexts": len(config["contexts"]),
                                             "config_sha256": sha256(output / "config.json"), "updated_unix": time.time()})
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "completed_cases": len(records),
                                             "error": repr(error), "updated_unix": time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    (register if args.stage == "register" else run)(args.output.resolve())


if __name__ == "__main__":
    main()
