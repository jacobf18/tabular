"""Frozen MatrixFormer completion of subsampled noisy Pauli expectations."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time

import cvxpy as cp
import numpy as np
import pandas as pd
from qiskit.quantum_info import (
    DensityMatrix, Pauli, random_density_matrix, random_statevector, state_fidelity,
)
import torch

import evaluate_cylinder_completion as shared


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ["Haar pure", "Rank-two mixed", "Ising ground"]
METHODS = ["Zero-fill + PSD", "PSD least squares", "MatrixFormer", "MatrixFormer*"]
KEYS = {method: key for method, key in zip(METHODS, ["zero", "physical_ls", "single", "blend"])}


def pauli_basis(qubits):
    if qubits < 2 or qubits % 2:
        raise ValueError("Use a positive, even number of qubits (at least two)")
    labels = ["".join(parts) for parts in itertools.product("IXYZ", repeat=qubits)]
    operators = np.stack([Pauli(label).to_matrix() for label in labels])
    return labels, operators


def pauli_coefficients(density, operators):
    coefficients = np.einsum("pij,ji->p", operators, density)
    if np.max(np.abs(coefficients.imag)) > 1e-8:
        raise ValueError("Pauli expectations must be real")
    width = int(np.sqrt(len(operators)))
    return coefficients.real.reshape(width, width)


def density_from_coefficients(coefficients, operators):
    return np.einsum("p,pij->ij", coefficients.ravel(), operators) / operators.shape[-1]


def project_density(density):
    hermitian = (density + density.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    ordered = np.sort(eigenvalues)[::-1]
    thresholds = (np.cumsum(ordered) - 1) / np.arange(1, len(ordered) + 1)
    active = np.flatnonzero(ordered > thresholds)[-1]
    projected = np.maximum(eigenvalues - thresholds[active], 0)
    return (eigenvectors * projected) @ eigenvectors.conj().T


def generate_state(family, qubits, seed):
    dimension = 2 ** qubits
    if family == "Haar pure":
        return DensityMatrix(random_statevector(dimension, seed=1000 + seed)).data, {}
    if family == "Rank-two mixed":
        state = random_density_matrix(dimension, rank=2, method="Hilbert-Schmidt", seed=2000 + seed)
        return state.data, {"generation_rank": 2}
    if family != "Ising ground":
        raise ValueError(family)
    field = [.5, 1., 1.5][seed % 3]
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    for site in range(qubits):
        label = ["I"] * qubits
        label[site] = "X"
        hamiltonian -= field * Pauli("".join(label)).to_matrix()
    for site in range(qubits - 1):
        label = ["I"] * qubits
        label[site:site + 2] = ["Z", "Z"]
        hamiltonian -= Pauli("".join(label)).to_matrix()
    energy, vectors = np.linalg.eigh(hamiltonian)
    ground = vectors[:, 0]
    return np.outer(ground, ground.conj()), {"field": field, "ground_energy": float(energy[0])}


def measure_paulis(truth, rate, shots, seed):
    if not 0 < rate < 1 or shots < 1:
        raise ValueError("Require partial missingness and positive shots")
    if not np.isclose(truth.flat[0], 1.) or np.max(np.abs(truth)) > 1 + 1e-8:
        raise ValueError("Invalid Pauli coefficients")
    permutation = np.random.default_rng(np.random.SeedSequence([seed, 0])).permutation(truth.size - 1) + 1
    count = round((1 - rate) * (truth.size - 1))
    if not 0 < count < truth.size - 1:
        raise ValueError("Need measured and unmeasured nonidentity observables")
    selected = permutation[:count]
    probabilities = np.clip((1 + truth.ravel()[selected]) / 2, 0, 1)
    plus_counts = np.random.default_rng(np.random.SeedSequence([seed, 1])).binomial(shots, probabilities)
    counts = np.full(truth.size, -1, dtype=np.int64)
    counts[selected] = plus_counts
    matrix = np.full(truth.size, np.nan)
    matrix[0] = 1.
    matrix[selected] = 2 * plus_counts / shots - 1
    return matrix.reshape(truth.shape), counts.reshape(truth.shape)


def complete_coefficients(matrix, raw_prediction):
    if not np.isfinite(raw_prediction).all():
        raise ValueError("Invalid prediction")
    return np.where(np.isfinite(matrix), matrix, np.clip(raw_prediction, -1., 1.))


def physical_least_squares(matrix, operators):
    observed = np.flatnonzero(np.isfinite(matrix.ravel()))
    observed = observed[observed != 0]
    dimension = operators.shape[-1]
    density = cp.Variable((dimension, dimension), hermitian=True)
    design = operators[observed].transpose(0, 2, 1).reshape(len(observed), -1)
    expectation = cp.real(design @ cp.vec(density, order="C"))
    problem = cp.Problem(cp.Minimize(cp.sum_squares(expectation - matrix.ravel()[observed])),
                         [density >> 0, cp.trace(density) == 1])
    started = time.perf_counter()
    problem.solve(solver="SCS", eps=1e-6, max_iters=50000, verbose=False)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or density.value is None:
        raise RuntimeError(f"PSD least squares failed: {problem.status}")
    raw_density = np.asarray(density.value)
    solver_info = problem.solver_stats.extra_stats.get("info", {})
    diagnostics = {"status": problem.status, "objective": float(problem.value),
                   "seconds": time.perf_counter() - started,
                   "iterations": int(problem.solver_stats.num_iters),
                   "raw_min_eigenvalue": float(np.linalg.eigvalsh(raw_density).min()),
                   "raw_trace_real": float(np.trace(raw_density).real),
                   "residual_primal": float(solver_info["res_pri"]),
                   "residual_dual": float(solver_info["res_dual"]),
                   "duality_gap": float(solver_info["gap"])}
    return project_density(raw_density), raw_density, diagnostics


def reconstruction_metrics(truth_density, truth_coefficients, density, matrix, operators):
    eigenvalues = np.linalg.eigvalsh(density)
    if eigenvalues.min() < -1e-8 or not np.isclose(np.trace(density), 1, atol=1e-8):
        raise ValueError("Nonphysical reconstructed density")
    reconstructed = pauli_coefficients(density, operators)
    missing = ~np.isfinite(matrix)
    observed = np.isfinite(matrix)
    observed.flat[0] = False
    fidelity = float(np.clip(state_fidelity(DensityMatrix(truth_density), DensityMatrix(density)), 0, 1))
    return {"fidelity": fidelity, "infidelity": 1 - fidelity,
            "trace_distance": float(np.abs(np.linalg.eigvalsh(density - truth_density)).sum() / 2),
            "hidden_pauli_rmse": float(np.sqrt(np.mean((reconstructed[missing] - truth_coefficients[missing]) ** 2))),
            "observed_fit_rmse": float(np.sqrt(np.mean((reconstructed[observed] - matrix[observed]) ** 2))),
            "min_eigenvalue": float(eigenvalues.min()),
            "trace_error": float(abs(np.trace(density) - 1))}


def evaluate_case(truth_density, truth, matrix, counts, operators, models, distribution):
    started = time.perf_counter()
    predictions, features, decision, timings = shared.neural_predictions(matrix, models, distribution)
    hidden_indices = np.flatnonzero(~np.isfinite(matrix.ravel()))
    uncertainty, uncertainty_arrays = shared.neural_uncertainty(
        models, features, distribution, matrix, truth, hidden_indices, decision["specialist_weight"])
    physical, physical_raw, diagnostics = physical_least_squares(matrix, operators)
    predictions["Zero-fill + PSD"] = np.zeros_like(matrix)
    densities = {"PSD least squares": physical}
    completed = {"PSD least squares": pauli_coefficients(physical_raw, operators)}
    raw_densities = {"PSD least squares": physical_raw}
    for method, prediction in predictions.items():
        completed[method] = complete_coefficients(matrix, prediction)
        raw_densities[method] = density_from_coefficients(completed[method], operators)
        densities[method] = project_density(raw_densities[method])
    arrays = {"matrix": matrix, "counts": counts, "hidden_indices": hidden_indices}
    metrics = {}
    for method in METHODS:
        scores = reconstruction_metrics(truth_density, truth, densities[method], matrix, operators)
        scores["preprojection_hidden_rmse"] = float(np.sqrt(np.mean(
            (completed[method].ravel()[hidden_indices] - truth.ravel()[hidden_indices]) ** 2)))
        scores["projection_frobenius"] = float(np.linalg.norm(densities[method] - raw_densities[method]))
        key = KEYS[method]
        arrays[f"{key}_density"] = densities[method]
        arrays[f"{key}_raw_density"] = raw_densities[method]
        arrays[f"{key}_completed"] = completed[method]
        if method in predictions:
            arrays[f"{key}_raw_prediction"] = predictions[method]
        if method in uncertainty:
            scores.update(uncertainty[method])
            scores["raw_hidden_rmse"] = float(np.sqrt(np.mean(
                (predictions[method].ravel()[hidden_indices] - truth.ravel()[hidden_indices]) ** 2)))
            bounds = uncertainty_arrays[method]["quantiles"]
            scores["interval_90_outside_physical_range"] = float(np.mean((bounds[:, 1] < -1) | (bounds[:, 4] > 1)))
            for name, value in uncertainty_arrays[method].items():
                arrays[f"{key}_{name}"] = value
        metrics[method] = scores
    return {"metrics": metrics, "routing": decision, "neural_forward_seconds": timings,
            "physical_solver": diagnostics, "case_seconds": time.perf_counter() - started}, arrays


def build_manifest(args, checkpoints):
    files = [Path(__file__).resolve(), Path(shared.__file__).resolve(), ROOT / "PHYSICS_QUANTUM.md",
             ROOT / "src/tabimpute/routing.py", ROOT / "src/tabimpute/model/model_new_stable.py",
             ROOT / "src/tabimpute/model/bar_distribution.py", ROOT / "src/tabimpute/data/borders.pt"]
    for checkpoint in checkpoints.values():
        files.extend([checkpoint, checkpoint.with_name("config.json")])
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "protocol": "PHYSICS_QUANTUM.md; fixed before inspecting quantum scores",
            "qubits": args.qubits, "families": FAMILIES, "seeds": args.seeds,
            "missing_rates": args.rates, "shots_per_measured_pauli": args.shots,
            "state_seed_offsets": {"Haar pure": 1000, "Rank-two mixed": 2000},
            "measurement_seed_rule": "3000 + 100 * family_index + seed; shared across rates",
            "normalization": "observed-only global mean/population SD, including identity",
            "methods": METHODS, "checkpoints": {name: str(path) for name, path in checkpoints.items()},
            "file_sha256": {str(path): shared.sha256(path) for path in files},
            "packages": {name: metadata.version(name) for name in
                         ["qiskit", "cvxpy", "scs", "numpy", "scipy", "torch", "pandas"]},
            "python": platform.python_version(), "gpu": torch.cuda.get_device_name()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/physics_quantum_20261027/pilot")
    parser.add_argument("--qubits", type=int, default=4, choices=[2, 4, 6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rates", type=float, nargs="+", default=[.5, .8])
    parser.add_argument("--shots", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment: {args.output}")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.rates)) != len(args.rates):
        raise ValueError("Duplicate seeds or missingness rates")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    torch.backends.cuda.enable_cudnn_sdp(False)
    checkpoints = {"single": shared.SINGLE, "general": shared.GENERAL, "specialist": shared.SPECIALIST}
    manifest = build_manifest(args, checkpoints)
    if manifest["file_sha256"][str(shared.SINGLE)] != shared.SINGLE_SHA256:
        raise ValueError("Unexpected combined-model checkpoint")
    shared.write_json(args.output / "manifest.json", manifest)
    total = len(FAMILIES) * len(args.seeds) * len(args.rates)
    shared.write_json(args.output / "status.json", {"status": "running", "completed": 0, "total": total})
    models, distribution = shared.load_models(checkpoints)
    labels, operators = pauli_basis(args.qubits)
    (args.output / "states").mkdir()
    (args.output / "cases").mkdir()
    rows = []
    completed_cases = 0
    for family_index, family in enumerate(FAMILIES):
        for seed in args.seeds:
            state_id = f"family{family_index}_seed{seed}"
            truth_density, state_details = generate_state(family, args.qubits, seed)
            truth = pauli_coefficients(truth_density, operators)
            state_path = args.output / "states" / f"{state_id}.npz"
            np.savez_compressed(state_path, density=truth_density, coefficients=truth, labels=np.array(labels))
            for rate in args.rates:
                case_id = f"{state_id}_missing{rate:g}"
                matrix, counts = measure_paulis(truth, rate, args.shots, 3000 + 100 * family_index + seed)
                print(f"Running {case_id}: {family}", flush=True)
                result, arrays = evaluate_case(truth_density, truth, matrix, counts, operators, models, distribution)
                _, observed_mean, observed_std = shared.normalize_observed(matrix)
                result.update({"case_id": case_id, "state_id": state_id, "family": family, "seed": seed,
                               "rate": rate, "state_details": state_details,
                               "measured_nonidentity": int(np.isfinite(matrix).sum() - 1),
                               "actual_missing_fraction": float(np.isnan(matrix).sum() / (matrix.size - 1)),
                               "empty_rows": int(np.isnan(matrix).all(axis=1).sum()),
                               "empty_columns": int(np.isnan(matrix).all(axis=0).sum()),
                               "observed_mean": observed_mean, "observed_std": observed_std,
                               "state_sha256": shared.sha256(state_path)})
                artifact = args.output / "cases" / f"{case_id}.npz"
                np.savez_compressed(artifact, **arrays)
                result["arrays_sha256"] = shared.sha256(artifact)
                shared.write_json(artifact.with_suffix(".json"), result)
                rows.extend({**{key: result[key] for key in ["case_id", "state_id", "family", "seed", "rate"]},
                             "method": method, **metrics} for method, metrics in result["metrics"].items())
                frame = pd.DataFrame(rows)
                frame.to_csv(args.output / "per_case.csv", index=False)
                completed_cases += 1
                shared.write_json(args.output / "status.json",
                                  {"status": "running", "completed": completed_cases, "total": total})
                print({method: round(metrics["fidelity"], 4) for method, metrics in result["metrics"].items()}, flush=True)
    metrics = ["fidelity", "trace_distance", "hidden_pauli_rmse", "observed_fit_rmse", "crps", "coverage_90", "width_90"]
    summary = frame.groupby(["family", "rate", "method"])[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary.reset_index().to_csv(args.output / "summary.csv", index=False)
    shared.write_json(args.output / "status.json", {"status": "complete", "completed": completed_cases, "total": total})
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
