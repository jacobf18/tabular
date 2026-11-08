"""Audit phantom-MRI reconstructions and plot errors, images, and spectra."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_mri_completion as evaluation
from evaluate_cylinder_completion import sha256, write_json


def audit(output):
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    expected = {f"{pattern}_seed{seed}_missing{rate:g}" for pattern in manifest["patterns"]
                for seed in manifest["seeds"] for rate in manifest["rates"]}
    assert status["status"] == "complete" and status["completed"] == len(expected)
    for path, digest in manifest["file_sha256"].items():
        assert sha256(path) == digest, f"Changed source/checkpoint: {path}"
    lift = evaluation.SLift(manifest["size"], manifest["radius"])
    image = np.load(output / "image.npy")
    kspace = np.load(output / "raw_kspace.npy")
    truth_s = np.load(output / "lifted_s_matrix.npy")
    regenerated_image, regenerated_kspace = evaluation.make_phantom(manifest["size"])
    np.testing.assert_allclose(image, regenerated_image, atol=1e-14)
    np.testing.assert_allclose(kspace, regenerated_kspace, atol=1e-14)
    np.testing.assert_array_equal(truth_s, lift.forward(kspace))
    paths = sorted((output / "cases").glob("*.json"))
    assert {path.stem for path in paths} == expected
    rows, checks, records = [], 0, []
    for path in paths:
        record = json.loads(path.read_text())
        records.append(record)
        artifact = path.with_suffix(".npz")
        assert sha256(artifact) == record["arrays_sha256"]
        assert set(record["metrics"]) == set(evaluation.METHODS)
        with np.load(artifact) as arrays:
            acquired, central = evaluation.acquisition_mask(manifest["size"], record["rate"], record["pattern"],
                                                            record["seed"], manifest["calibration"])
            np.testing.assert_array_equal(arrays["acquired"], acquired)
            np.testing.assert_array_equal(arrays["central"], central)
            validation = evaluation.validation_mask(acquired, central, record["pattern"], record["seed"])
            np.testing.assert_array_equal(arrays["validation"], validation)
            assert not (validation & ~acquired).any() and not (validation & central).any()
            best_trial = min(record["tuning"]["trials"], key=lambda trial: trial["validation_rmse"])
            assert record["tuning"]["selected_rank"] == best_trial["rank"]
            assert [trial["rank"] for trial in record["tuning"]["trials"]] == manifest["rank_grid"]
            measured = np.where(acquired, kspace, complex(np.nan, np.nan))
            raw_input, lifted_input = evaluation.observed_inputs(measured, lift)
            np.testing.assert_array_equal(arrays["raw_input"], raw_input)
            np.testing.assert_array_equal(arrays["lifted_input"], lifted_input)
            raw_completed = np.where(np.isfinite(raw_input), raw_input, arrays["raw_prediction"])
            raw_kspace = raw_completed[:, :manifest["size"]] + 1j * raw_completed[:, manifest["size"]:]
            np.testing.assert_array_equal(arrays["raw_kspace"], evaluation.preserve_measurements(raw_kspace, measured))
            lifted_completed = np.where(np.isfinite(lifted_input), lifted_input, arrays["lifted_prediction"])
            np.testing.assert_allclose(arrays["lifted_kspace"],
                                       evaluation.preserve_measurements(lift.inverse(lifted_completed), measured), atol=1e-14)
            np.testing.assert_array_equal(arrays["zero_kspace"], np.nan_to_num(measured))
            for method in evaluation.METHODS:
                key = evaluation.KEYS[method]
                prediction = arrays[f"{key}_kspace"]
                scores = evaluation.score_reconstruction(image, kspace, prediction, acquired)
                for metric, value in scores.items():
                    np.testing.assert_allclose(value, record["metrics"][method][metric], rtol=1e-10, atol=1e-12)
                    checks += 1
                np.testing.assert_allclose(scores["complex_image_nrmse"],
                                           np.linalg.norm(prediction - kspace) / np.linalg.norm(kspace), atol=1e-12)
                np.testing.assert_allclose(arrays[f"{key}_s_singular_values"],
                                           np.linalg.svd(lift.forward(prediction), compute_uv=False), atol=1e-12)
                np.testing.assert_allclose(arrays[f"{key}_raw_singular_values"],
                                           np.linalg.svd(prediction, compute_uv=False), atol=1e-12)
                rows.append({**{name: record[name] for name in ["case_id", "pattern", "rate", "seed"]},
                             "method": method, **scores})
    frame = pd.DataFrame(rows)
    saved = pd.read_csv(output / "per_case.csv")
    left = frame.set_index(["case_id", "method"]).sort_index()
    right = saved.set_index(["case_id", "method"]).sort_index()
    assert list(left.index) == list(right.index)
    numeric = left.select_dtypes(include="number").columns
    np.testing.assert_allclose(left[numeric], right[numeric], rtol=1e-10, atol=1e-12)
    summary = {"status": "pass", "cases": len(paths), "methods_per_case": len(evaluation.METHODS),
               "recomputed_metric_checks": checks, "source_checkpoint_hashes_checked": len(manifest["file_sha256"]),
               "data_consistency": "all acquired complex frequencies preserved exactly",
               "report_script_sha256": sha256(Path(__file__))}
    write_json(output / "audit.json", summary)
    return frame, manifest, records, summary


def make_figures(output, frame, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=.95)
    palette = {"Zero-fill": "#999999", "Structured low-rank S": "#39986c",
               "MatrixFormer raw": "#9ecae1", "MatrixFormer S": "#2171b5"}
    patterns, rates = manifest["patterns"], sorted(manifest["rates"])
    figure, axes = plt.subplots(1, len(patterns), figsize=(6 * len(patterns), 4.5), sharey=True, squeeze=False)
    plot_frame = frame.copy()
    plot_frame["Missing k-space"] = plot_frame.rate.map(lambda rate: f"{rate:.0%}")
    for axis, pattern in zip(axes.ravel(), patterns):
        sns.barplot(data=plot_frame[plot_frame.pattern == pattern], x="Missing k-space", y="complex_image_nrmse",
                    hue="method", hue_order=evaluation.METHODS, palette=palette,
                    errorbar="sd", capsize=.1, ax=axis)
        axis.set(title="Random points" if pattern == "points" else "Cartesian lines", ylabel="Complex-image NRMSE ↓")
        axis.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Phantom MRI: raw versus lifted MatrixFormer input")
    figure.tight_layout(rect=(0, .12, 1, .95))
    for suffix in ["png", "pdf"]:
        figure.savefig(output / f"reconstruction_nrmse.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)

    truth_image = np.load(output / "image.npy")
    truth_amplitude = np.abs(truth_image)
    chosen_seed = min(manifest["seeds"])
    selections = [(pattern, rate) for pattern in patterns for rate in rates]
    figure, axes = plt.subplots(len(selections), 6, figsize=(17, 3 * len(selections)), squeeze=False)
    error_figure, error_axes = plt.subplots(len(selections), 4, figsize=(13, 3 * len(selections)), squeeze=False)
    for row_index, (pattern, rate) in enumerate(selections):
        case_id = f"{pattern}_seed{chosen_seed}_missing{rate:g}"
        with np.load(output / "cases" / f"{case_id}.npz") as arrays:
            axes[row_index, 0].imshow(truth_amplitude, cmap="gray", vmin=0, vmax=truth_amplitude.max())
            axes[row_index, 0].set_title(f"Truth: {pattern}, {rate:.0%} missing")
            axes[row_index, 1].imshow(arrays["acquired"], cmap="gray", vmin=0, vmax=1)
            axes[row_index, 1].set_title("Acquired k-space mask")
            for method_index, method in enumerate(evaluation.METHODS):
                key = evaluation.KEYS[method]
                image = evaluation.ifft_kspace(arrays[f"{key}_kspace"])
                score = frame[(frame.case_id == case_id) & (frame.method == method)].complex_image_nrmse.iloc[0]
                axes[row_index, method_index + 2].imshow(np.abs(image), cmap="gray", vmin=0, vmax=truth_amplitude.max())
                axes[row_index, method_index + 2].set_title(f"{method}\nNRMSE {score:.3f}")
                error_axes[row_index, method_index].imshow(np.abs(image - truth_image), cmap="magma", vmin=0, vmax=.3)
                error_axes[row_index, method_index].set_title(f"{pattern}, {rate:.0%}: {method}")
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis in error_axes.ravel():
        axis.set_axis_off()
    figure.suptitle(f"Magnitude reconstructions, fixed mask seed {chosen_seed}; common image intensity scale")
    error_figure.suptitle(f"Absolute complex-image error, fixed seed {chosen_seed}; common color range 0–0.3")
    figure.tight_layout(rect=(0, 0, 1, .97))
    error_figure.tight_layout(rect=(0, 0, 1, .97))
    figure.savefig(output / "reconstructions.png", dpi=160, bbox_inches="tight")
    error_figure.savefig(output / "reconstruction_errors.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    plt.close(error_figure)

    truth_singular = np.linalg.svd(np.load(output / "lifted_s_matrix.npy"), compute_uv=False)
    figure, axes = plt.subplots(len(patterns), len(rates), figsize=(6 * len(rates), 4 * len(patterns)), squeeze=False)
    for pattern_index, pattern in enumerate(patterns):
        for rate_index, rate in enumerate(rates):
            axis = axes[pattern_index, rate_index]
            axis.semilogy(np.arange(1, len(truth_singular) + 1), truth_singular / truth_singular[0],
                          color="black", linewidth=2, label="Truth")
            for method in evaluation.METHODS:
                spectra = []
                for seed in manifest["seeds"]:
                    with np.load(output / "cases" / f"{pattern}_seed{seed}_missing{rate:g}.npz") as arrays:
                        spectra.append(arrays[f"{evaluation.KEYS[method]}_s_singular_values"])
                mean = np.mean(spectra, axis=0) / truth_singular[0]
                axis.semilogy(np.arange(1, len(mean) + 1), mean, color=palette[method], label=method)
            axis.set(title=f"{pattern}: {rate:.0%} missing k-space", xlabel="Singular-value index",
                     ylabel="S singular value / largest true value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    figure.suptitle("S-matrix spectra after reconstruction; means across mask seeds")
    figure.tight_layout(rect=(0, .06, 1, .96))
    for suffix in ["png", "pdf"]:
        figure.savefig(output / f"reconstructed_spectra.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_scores(frame, metric, higher_better=False):
    table = frame.groupby(["pattern", "rate", "method"])[metric].mean().unstack("method").reindex(columns=evaluation.METHODS)
    lines = ["| Sampling | Missing | " + " | ".join(evaluation.METHODS) + " |",
             "| --- | ---: | " + " | ".join(["---:"] * len(evaluation.METHODS)) + " |"]
    for (pattern, rate), values in table.iterrows():
        best = values.max() if higher_better else values.min()
        formatted = [f"**{value:.4f}**" if value == best else f"{value:.4f}" for value in values]
        lines.append(f"| {pattern} | {rate:.0%} | " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    frame, manifest, records, checks = audit(output)
    make_figures(output, frame, manifest)
    spectra = json.loads((output / "spectra_truth/phantom_mri_spectra_summary.json").read_text())
    paired = frame.pivot(index="case_id", columns="method", values="complex_image_nrmse")
    lifted_raw_wins = int((paired["MatrixFormer S"] < paired["MatrixFormer raw"]).sum())
    lifted_baseline_wins = int((paired["MatrixFormer S"] < paired["Structured low-rank S"]).sum())
    report = ["# MatrixFormer on phantom MRI: raw versus lifted S-matrix", "",
              "## Setup", "",
              "The original MRI spectrum plotting script is reused, but its old data were not found. "
              "This experiment recreates a standard scikit-image Shepp–Logan phantom with fixed smooth "
              "phase. It is synthetic, noiseless, single-coil MRI, not clinical or fastMRI data.", "",
              f"Image: {manifest['size']} × {manifest['size']}; lifted real matrix: "
              f"{manifest['s_matrix_shape'][0]:,} × {manifest['s_matrix_shape'][1]}. "
              f"Completed {checks['cases']} cases with {checks['methods_per_case']} methods each. "
              "Masks remove k-space samples **before** lifting, so repeated appearances cannot leak "
              "unobserved measurements. Both MatrixFormer paths use the same frozen combined V57 "
              "checkpoint, observed-only global normalization, full context, and no retraining.", "",
              "## Complex-image NRMSE ↓", "",
              "NRMSE = `||image_hat - image||_F / ||image||_F`. This relative-L2 MRI metric is "
              "not the SD-normalized tabular metric. Values are means across three mask seeds "
              "on one phantom; bold is best in each row.", "",
              markdown_scores(frame, "complex_image_nrmse"), "",
              "## Magnitude-image NRMSE ↓", "", markdown_scores(frame, "magnitude_image_nrmse"), "",
              "## Magnitude SSIM ↑", "", markdown_scores(frame, "magnitude_ssim", True), "",
              "## Hidden k-space RMSE ↓", "", markdown_scores(frame, "hidden_kspace_rmse"), "",
              "## What the spectra show", "",
              "The complete-data spectra are diagnostic only; no rank or neural hyperparameter is "
              "chosen from them. Lifting changes the representation and introduces redundancy; "
              "it does not guarantee that every effective-rank summary decreases.", "",
              "| Matrix | Shape | Numerical rank | Effective rank | Modes for 95% energy | Modes for 99% energy |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for key, label in [("raw_mri", "Raw complex k-space"), ("lifted_s_matrix", "Real S-matrix")]:
        stats = spectra[key]
        report.append(f"| {label} | {stats['shape'][0]} × {stats['shape'][1]} | {stats['matrix_rank']} | "
                      f"{stats['effective_rank']:.3f} | {stats['power_rank_95']} | {stats['power_rank_99']} |")
    report.extend(["", "## Acquisition versus lifted missingness", "",
                   "Each S entry combines two k-space samples and is exposed only when both are "
                   "measured. The lifted input therefore has more missingness. Partially measured "
                   "pairs cannot be represented as known S entries; the iterative baseline can "
                   "still exploit those acquired values through its data-consistency step.", "",
                   "| Pattern | Missing k-space | Missing S entries | Selected ranks (seed order) | Final fits at iteration cap |",
                   "| --- | ---: | ---: | --- | ---: |"])
    for (pattern, rate), selected in pd.DataFrame(records).groupby(["pattern", "rate"]):
        selected = selected.sort_values("seed")
        ranks = [entry["selected_rank"] for entry in selected.tuning]
        capped = sum(not entry["final"]["converged"] for entry in selected.tuning)
        report.append(f"| {pattern} | {rate:.0%} | {selected.actual_s_missing_fraction.mean():.1%} | "
                      f"{ranks} | {capped}/{len(selected)} |")
    report.extend(["", "## Interpretation", "",
                   f"- The S-matrix adapter beats direct raw-k-space MatrixFormer in **{lifted_raw_wins}/{len(paired)}** cases.",
                   f"- It beats the structured low-rank control in **{lifted_baseline_wins}/{len(paired)}** cases.",
                   "- The low-rank control is a simple LORAKS-style alternating projection, selected "
                   "using held-out acquired frequencies/lines and refitted on all acquisitions. "
                   "It is not the full upstream LORAKS solver or a tuned state-of-the-art MRI baseline.",
                   "- All methods preserve acquired complex measurements exactly. The lift's "
                   "uncovered Nyquist row/column and corner samples remain zero when unmeasured. "
                   "Final metrics include those frequencies, not just the represented interior.",
                   "- The raw network receives no explicit frequency coordinates. The S adapter "
                   "supplies neighborhood structure but does not enforce it during neural inference; "
                   "overlap averaging and measured-data restoration are post-processing.",
                   "- All three seeds concern one phantom. Error bars are mask-seed SDs, not "
                   "subject-level confidence intervals. Results cannot establish clinical performance.",
                   "- This pilot evaluates predictive means only. It makes no claim that marginal "
                   "S-entry distributions constitute a valid joint MRI image posterior.", "",
                   "## Figures", "",
                   "![MRI reconstruction NRMSE](reconstruction_nrmse.png)", "",
                   "![Original spectrum script applied to recreated data](spectra_truth/phantom_mri_spectra.png)", "",
                   "![Reconstructed S-matrix spectra](reconstructed_spectra.png)", "",
                   "![Fixed-seed reconstructions](reconstructions.png)", "",
                   "![Fixed-seed complex-image errors](reconstruction_errors.png)", "",
                   "## Validation and artifacts", "",
                   f"Audit: **{checks['status']}**, {checks['recomputed_metric_checks']} independently recomputed metric values; "
                   "source/checkpoint hashes, regenerated inputs, masks, lifted inverse, singular values, "
                   "CSV consistency, Parseval identity, and measured-data consistency also checked.", "",
                   f"- Full protocol and citations: `{evaluation.ROOT / 'PHYSICS_MRI.md'}`.",
                   f"- Output directory: `{output}`.",
                   "- `image.npy`, `raw_kspace.npy`, `lifted_s_matrix.npy`: recreated reference matrices.",
                   "- `cases/*.npz`: acquisition masks, validation masks, model inputs/predictions, "
                   "reconstructed k-space and singular values for every method.",
                   "- `cases/*.json`: scores, rank-selection trials, convergence history, and neural timing/memory.",
                   "- `per_case.csv`, `summary.csv`: all metrics, including PSNR and hidden-k-space relative error.",
                   "- `spectra_truth/`: unchanged old spectrum script outputs for the recreated phantom.",
                   "- `manifest.json`, `status.json`, `audit.json`: provenance and completion checks.", ""])
    (output / "RESULTS.md").write_text("\n".join(report))
    print(json.dumps(checks, indent=2))
    print(markdown_scores(frame, "complex_image_nrmse"))
    print(output / "RESULTS.md")


if __name__ == "__main__":
    main()
