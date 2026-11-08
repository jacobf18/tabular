"""Audit saved quantum-tomography predictions and produce tables and a figure."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_quantum_tomography as evaluation
from evaluate_cylinder_completion import normalize_observed, sha256, write_json


def audit(output):
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    expected_cases = len(manifest["families"]) * len(manifest["seeds"]) * len(manifest["missing_rates"])
    assert status["status"] == "complete" and status["completed"] == expected_cases
    for path, digest in manifest["file_sha256"].items():
        assert sha256(path) == digest, f"Source/checkpoint changed: {path}"
    labels, operators = evaluation.pauli_basis(manifest["qubits"])
    rows = []
    checks, coverages = 0, 0
    records = sorted((output / "cases").glob("*.json"))
    expected_ids = {f"family{family_index}_seed{seed}_missing{rate:g}"
                    for family_index in range(len(manifest["families"]))
                    for seed in manifest["seeds"] for rate in manifest["missing_rates"]}
    assert {path.stem for path in records} == expected_ids
    for record_path in records:
        record = json.loads(record_path.read_text())
        state_path = output / "states" / f"{record['state_id']}.npz"
        artifact = record_path.with_suffix(".npz")
        assert sha256(state_path) == record["state_sha256"]
        assert sha256(artifact) == record["arrays_sha256"]
        with np.load(state_path) as state, np.load(artifact) as arrays:
            truth, truth_density = state["coefficients"], state["density"]
            np.testing.assert_array_equal(state["labels"], labels)
            regenerated, _ = evaluation.generate_state(record["family"], manifest["qubits"], record["seed"])
            np.testing.assert_allclose(truth_density, regenerated, atol=1e-14)
            np.testing.assert_allclose(truth, evaluation.pauli_coefficients(truth_density, operators), atol=1e-14)
            seed = 3000 + 100 * manifest["families"].index(record["family"]) + record["seed"]
            regenerated_matrix, regenerated_counts = evaluation.measure_paulis(
                truth, record["rate"], manifest["shots_per_measured_pauli"], seed)
            matrix = arrays["matrix"]
            np.testing.assert_array_equal(matrix, regenerated_matrix)
            np.testing.assert_array_equal(arrays["counts"], regenerated_counts)
            indices = arrays["hidden_indices"]
            np.testing.assert_array_equal(indices, np.flatnonzero(~np.isfinite(matrix)))
            _, observed_mean, observed_std = normalize_observed(matrix)
            np.testing.assert_allclose([observed_mean, observed_std], [record["observed_mean"], record["observed_std"]])
            assert set(record["metrics"]) == set(evaluation.METHODS)
            for method in evaluation.METHODS:
                key = evaluation.KEYS[method]
                density, raw_density = arrays[f"{key}_density"], arrays[f"{key}_raw_density"]
                completed = arrays[f"{key}_completed"]
                np.testing.assert_allclose(density, evaluation.project_density(raw_density), atol=1e-12)
                np.testing.assert_allclose(raw_density, evaluation.density_from_coefficients(completed, operators), atol=1e-12)
                if method != "PSD least squares":
                    np.testing.assert_array_equal(completed, evaluation.complete_coefficients(matrix, arrays[f"{key}_raw_prediction"]))
                scores = evaluation.reconstruction_metrics(truth_density, truth, density, matrix, operators)
                scores["preprojection_hidden_rmse"] = float(np.sqrt(np.mean(
                    (completed.ravel()[indices] - truth.ravel()[indices]) ** 2)))
                scores["projection_frobenius"] = float(np.linalg.norm(density - raw_density))
                if method in {"MatrixFormer", "MatrixFormer*"}:
                    bounds, target = arrays[f"{key}_quantiles"], truth.ravel()[indices]
                    assert bounds.shape == (len(indices), 6) and np.isfinite(bounds).all()
                    assert np.all(np.diff(bounds, axis=1) >= 0)
                    scores["raw_hidden_rmse"] = float(np.sqrt(np.mean((arrays[f"{key}_raw_prediction"].ravel()[indices] - target) ** 2)))
                    scores["crps"] = float(arrays[f"{key}_crps"].mean())
                    scores["normalized_crps"] = scores["crps"] / observed_std
                    scores["nll"] = float(arrays[f"{key}_nll"].mean())
                    for nominal, lower, upper in [(50, 2, 3), (90, 1, 4), (95, 0, 5)]:
                        scores[f"coverage_{nominal}"] = float(np.mean((target >= bounds[:, lower]) & (target <= bounds[:, upper])))
                        scores[f"width_{nominal}"] = float(np.mean(bounds[:, upper] - bounds[:, lower]))
                        coverages += 1
                    scores["interval_90_outside_physical_range"] = float(np.mean((bounds[:, 1] < -1) | (bounds[:, 4] > 1)))
                for metric, value in scores.items():
                    np.testing.assert_allclose(value, record["metrics"][method][metric], rtol=2e-6, atol=1e-9,
                                               err_msg=f"{record['case_id']}, {method}, {metric}")
                    checks += 1
                rows.append({**{name: record[name] for name in ["case_id", "state_id", "family", "seed", "rate"]},
                             "method": method, **record["metrics"][method]})
    frame = pd.DataFrame(rows)
    saved = pd.read_csv(output / "per_case.csv")
    numeric = frame.select_dtypes(include="number").columns
    left = frame.set_index(["case_id", "method"]).sort_index()
    right = saved.set_index(["case_id", "method"]).sort_index()
    assert list(left.index) == list(right.index)
    np.testing.assert_allclose(left[numeric], right[numeric], rtol=1e-10, atol=1e-12, equal_nan=True)
    result = {"status": "pass", "cases": len(records), "methods_per_case": len(evaluation.METHODS),
              "recomputed_metric_checks": checks, "recomputed_interval_coverages": coverages,
              "source_and_checkpoint_hashes_checked": len(manifest["file_sha256"]),
              "report_script_sha256": sha256(Path(__file__))}
    write_json(output / "audit.json", result)
    return frame, manifest, result


def markdown_table(frame, group_columns, metric, higher_better=False):
    table = frame.groupby(group_columns + ["method"])[metric].mean().unstack("method").reindex(columns=evaluation.METHODS)
    lines = ["| " + " | ".join(group_columns + evaluation.METHODS) + " |",
             "| " + " | ".join(["---"] * len(group_columns) + ["---:"] * len(evaluation.METHODS)) + " |"]
    for group, values in table.iterrows():
        group = group if isinstance(group, tuple) else (group,)
        best = values.max() if higher_better else values.min()
        formatted = [f"**{value:.4f}**" if value == best else f"{value:.4f}" for value in values]
        names = [f"{value:.0%}" if column == "rate" else str(value) for column, value in zip(group_columns, group)]
        lines.append("| " + " | ".join(names + formatted) + " |")
    return "\n".join(lines)


def make_plot(output, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=.95)
    palette = {"Zero-fill + PSD": "#a7a7a7", "PSD least squares": "#39986c",
               "MatrixFormer": "#3182bd", "MatrixFormer*": "#08519c"}
    rates = sorted(frame.rate.unique())
    figure, axes = plt.subplots(1, len(rates), figsize=(6.5 * len(rates), 4.6), sharey=True, squeeze=False)
    for axis, rate in zip(axes.ravel(), rates):
        sns.barplot(data=frame[frame.rate == rate], x="family", y="fidelity", hue="method",
                    order=evaluation.FAMILIES, hue_order=evaluation.METHODS, palette=palette,
                    errorbar="sd", capsize=.1, ax=axis)
        axis.set(title=f"{rate:.0%} missing Pauli observables", xlabel="", ylabel="State fidelity ↑", ylim=(0, 1.08))
        axis.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .025))
    figure.suptitle("Four-qubit tomography: frozen checkpoints, 2,000 shots per measured Pauli")
    figure.tight_layout(rect=(0, .14, 1, .96))
    figure.savefig(output / "quantum_fidelity.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / "quantum_fidelity.pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    frame, manifest, checks = audit(output)
    make_plot(output, frame)
    metrics = ["fidelity", "trace_distance", "hidden_pauli_rmse", "observed_fit_rmse", "crps", "coverage_90", "width_90"]
    frame.groupby(["rate", "method"])[metrics].mean().reset_index().to_csv(output / "overall_by_missingness.csv", index=False)
    frame.groupby("method")[metrics].mean().reset_index().to_csv(output / "overall.csv", index=False)
    paired = frame.pivot(index="case_id", columns="method", values="fidelity")
    combined_wins = int((paired["MatrixFormer"] > paired["PSD least squares"]).sum())
    ensemble_wins = int((paired["MatrixFormer*"] > paired["PSD least squares"]).sum())
    records = [json.loads(path.read_text()) for path in sorted((output / "cases").glob("*.json"))]
    solver_status = pd.Series([record["physical_solver"]["status"] for record in records]).value_counts().to_dict()
    report = ["# Quantum tomography: frozen-model pilot", "",
              f"Completed {checks['cases']} cases; {checks['methods_per_case']} methods per case. "
              f"Audit passed {checks['recomputed_metric_checks']} metric checks and "
              f"{checks['recomputed_interval_coverages']} interval-coverage checks.", "",
              "## What was tested", "",
              f"{manifest['qubits']} qubits; three Haar pure, three rank-two mixed, and three Ising ground states. "
              "Each state is tested with 50% and 80% missing nonidentity Pauli observables, "
              f"using {manifest['shots_per_measured_pauli']:,} independent shots per measured observable. "
              "The identity is known, not sampled. The same states are reused across missingness rates. "
              "This is simulated compressed-Pauli tomography using Qiskit, not quantum hardware, "
              "not masking complex density entries, and not a reproduction of a published result.", "",
              "MatrixFormer is the final combined checkpoint; MatrixFormer* is the existing expert blend. "
              "Neither is trained on quantum data. Both use full 16 × 16 coefficient context and "
              "observed-only global normalization. All four methods produce PSD trace-one density "
              "matrices. PSD least squares is a CVXPY/SCS physical reconstruction, not a neural "
              "method or Qiskit fitter. No nuclear-norm penalty or known-rank oracle is used.", "",
              "## Overall state fidelity (higher is better)", "",
              "Arithmetic means across nine states at each missingness rate. Fidelity uses the squared "
              "Uhlmann/Qiskit convention. Bold indicates the best method in each row.", "",
              markdown_table(frame, ["rate"], "fidelity", True), "",
              "## Fidelity by state family", "",
              "Each row averages three state instances; no cases or baseline failures are filtered out.", "",
              markdown_table(frame, ["family", "rate"], "fidelity", True), "",
              "## Hidden Pauli expectation RMSE (lower is better)", "",
              "RMSE against true expectations, not held-out noisy shot estimates, after physical projection. "
              "The natural expectation scale is [-1, 1]; these are RMSE, not the tabular NRMSE metric.", "",
              markdown_table(frame, ["family", "rate"], "hidden_pauli_rmse"), "",
              "## Trace distance (lower is better)", "",
              markdown_table(frame, ["rate"], "trace_distance"), "",
              "## Observed-measurement fit RMSE", "",
              "Physical projection may move measured coefficients. These values compare the reconstructed "
              "expectations with the actual measured, finite-shot observations, excluding identity.", "",
              markdown_table(frame, ["rate"], "observed_fit_rmse"), "",
              "## Native marginal uncertainty", "",
              "Intervals and CRPS concern **unprojected, unclipped Pauli coefficients**, not the physical "
              "density matrix or the post-projection point estimates. These are not joint quantum-state "
              "posteriors. No physics-label recalibration is applied. Coverage is assessed against the "
              "true expectation, not a future shot outcome. CRPS uses 199 midpoint quantiles and "
              "half-normal full-support tails.", "",
              "| Missing | Model | CRPS ↓ | 90% coverage | 90% width | Intervals outside [-1, 1] |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    neural = frame[frame.method.isin(["MatrixFormer", "MatrixFormer*"])]
    for (rate, method), rows in neural.groupby(["rate", "method"]):
        report.append(f"| {rate:.0%} | {method} | {rows.crps.mean():.4f} | "
                      f"{rows.coverage_90.mean():.1%} | {rows.width_90.mean():.4f} | "
                      f"{rows.interval_90_outside_physical_range.mean():.1%} |")
    report.extend(["", "## Interpretation and limits", "",
                   f"- MatrixFormer has higher fidelity than PSD least squares in **{combined_wins}/{len(paired)}** "
                   f"paired cases; MatrixFormer* does so in **{ensemble_wins}/{len(paired)}**.",
                   f"- Physics-baseline solver statuses: `{solver_status}`. Numerical cleanup is applied "
                   "to all reconstructed densities; original solver eigenvalues/residuals are archived.",
                   "- Low-rank quantum density matrices need not become low-rank folded Pauli matrices. "
                   "This adapter does not provide the networks with positivity, trace, or Pauli-algebra "
                   "constraints during imputation; it enforces physicality only afterward.",
                   "- Incomplete-measurement least squares can have multiple minimizers. This is one "
                   "standard physical control, not an exhaustive comparison with modern tomography methods.",
                   "- Nine states, one qubit count, one shot budget, two nested missingness levels, and "
                   "three seeds are only a pilot. Error bars in the plot show sample SD across three "
                   "state instances, not confidence intervals or independent hardware repetitions.",
                   "- No checkpoint selection, prior adaptation, transpose selection, or model training "
                   "used these targets. A failure here should not be hidden by selecting only easy states.",
                   "", "## Figure", "", "![State fidelity](quantum_fidelity.png)", "",
                   "## Reproduce and inspect", "",
                   f"- Protocol: `{evaluation.ROOT / 'PHYSICS_QUANTUM.md'}`.",
                   f"- Artifacts: `{output}`.",
                   "- `manifest.json`: protocol, package versions, source/checkpoint SHA256 values.",
                   "- `states/`: density matrices, true Pauli coefficients, and operator labels.",
                   "- `cases/`: masks, shot counts, predictions, physical densities, native quantiles, "
                   "routing weights, and solver diagnostics (NPZ/JSON).",
                   "- `per_case.csv`, `summary.csv`, `overall_by_missingness.csv`, `overall.csv`: scores.",
                   "- `audit.json`: regenerated data and independently recomputed saved-score checks.",
                   "- `quantum_fidelity.png` and `.pdf`: fidelity plot with state-instance SD bars.", "",
                   "## References", "",
                   "1. Gross et al. (2010), *Quantum State Tomography via Compressed Sensing*, "
                   "PRL 105, 150401. https://arxiv.org/abs/0909.3304 — motivates random Pauli sampling, "
                   "not a claim to reproduce their recovery algorithm.",
                   "2. Qiskit 1.4.5, https://github.com/Qiskit/qiskit — state generation, Pauli algebra, fidelity.",
                   "3. Qiskit Experiments state-tomography tutorial: "
                   "https://qiskit-community.github.io/qiskit-experiments/manuals/verification/state_tomography.html "
                   "— related standard tomography workflow; its fitter is not used here.", ""])
    (output / "RESULTS.md").write_text("\n".join(report))
    print(json.dumps(checks, indent=2))
    print(frame.groupby(["rate", "method"])[["fidelity", "hidden_pauli_rmse", "coverage_90"]].mean().to_string())
    print(output / "RESULTS.md")


if __name__ == "__main__":
    main()
