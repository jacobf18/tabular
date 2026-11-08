"""Audit and summarize the acquired USC phantom completion pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_real_mri_completion as evaluation
from evaluate_cylinder_completion import sha256, write_json


def audit(output):
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    expected = {f"{pattern}_seed{seed}_missing{rate:g}" for pattern in manifest["patterns"]
                for seed in manifest["seeds"] for rate in manifest["rates"]}
    assert status == {"status": "complete", "completed": len(expected), "failed": 0, "total": len(expected)}
    assert manifest["data_sha256"] == evaluation.DATA_SHA256
    assert manifest["rank_grid"] == evaluation.RANKS
    assert manifest["methods"] == evaluation.METHODS
    for path, digest in manifest["file_sha256"].items():
        assert sha256(path) == digest, f"Changed source/checkpoint: {path}"
    truth = evaluation.load_phantom(output / "phantom_data.mat")
    lift = evaluation.MultiCoilSLift(truth.shape[:2], truth.shape[2], manifest["radius"])
    assert list(lift.shape) == manifest["s_shape"]
    np.testing.assert_array_equal(evaluation.rss_image(truth), np.load(output / "reference_rss.npy"))
    assert {path.stem for path in (output / "cases").glob("*.json")} == expected
    rows, records, checks = [], [], 0
    for case_id in sorted(expected):
        print(f"Auditing {case_id}", flush=True)
        record = json.loads((output / "cases" / f"{case_id}.json").read_text())
        records.append(record)
        artifact = output / "cases" / f"{case_id}.npz"
        assert sha256(artifact) == record["arrays_sha256"]
        assert set(record["metrics"]) == set(evaluation.METHODS)
        acquired, central = evaluation.acquisition_mask(truth.shape[:2], record["rate"], record["pattern"], record["seed"])
        validation = evaluation.synthetic.validation_mask(acquired, central, record["pattern"], record["seed"])
        assert not (validation & ~acquired).any() and not (validation & central).any()
        assert [trial["rank"] for trial in record["tuning"]["trials"]] == manifest["rank_grid"]
        selected = min(record["tuning"]["trials"], key=lambda trial: trial["validation_rmse"])
        assert selected["rank"] == record["tuning"]["selected_rank"]
        assert record["tuning"]["final"]["rank"] == selected["rank"]
        np.testing.assert_allclose(record["actual_s_missing_fraction"], (~lift.observed(acquired)).mean())
        np.testing.assert_allclose(record["actual_kspace_missing_fraction"], (~acquired).mean())
        uncovered = lift.counts.reshape(acquired.shape) == 0
        assert record["uncovered_missing_spatial_frequencies"] == int((uncovered & ~acquired).sum())
        with np.load(artifact) as arrays:
            np.testing.assert_array_equal(arrays["acquired"], acquired)
            np.testing.assert_array_equal(arrays["central"], central)
            np.testing.assert_array_equal(arrays["validation"], validation)
            np.testing.assert_array_equal(arrays["zero_kspace"], np.where(acquired[..., None], truth, 0))
            for method in evaluation.METHODS:
                prediction = arrays[f"{evaluation.KEYS[method]}_kspace"]
                np.testing.assert_array_equal(prediction[acquired], truth[acquired])
                if method in ["Structured low-rank S", "MatrixFormer S"]:
                    assert not prediction[uncovered & ~acquired].any()
                scores = evaluation.score_reconstruction(truth, prediction, acquired)
                for metric, value in scores.items():
                    np.testing.assert_allclose(value, record["metrics"][method][metric], rtol=1e-10, atol=1e-14)
                    checks += 1
                np.testing.assert_allclose(scores["complex_image_nrmse"],
                                           np.linalg.norm(prediction - truth) / np.linalg.norm(truth), atol=1e-12)
                rows.append({"case_id": case_id, "pattern": record["pattern"], "rate": record["rate"],
                             "seed": record["seed"], "method": method, **scores})
    frame = pd.DataFrame(rows)
    keys = ["case_id", "method"]
    saved = pd.read_csv(output / "per_case.csv").sort_values(keys).reset_index(drop=True)
    rebuilt = frame.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved[rebuilt.columns], rebuilt, check_dtype=False, rtol=1e-10, atol=1e-14)
    aggregate = frame.groupby(["pattern", "rate", "method"])[evaluation.METRICS].agg(["mean", "std", "count"])
    aggregate.columns = ["_".join(parts) for parts in aggregate.columns]
    aggregate = aggregate.reset_index()
    pd.testing.assert_frame_equal(pd.read_csv(output / "summary.csv"), aggregate,
                                  check_dtype=False, rtol=1e-10, atol=1e-14)
    result = {"status": "passed", "cases": len(expected), "metric_values_checked": checks,
              "methods_per_case": len(evaluation.METHODS), "exact_measurement_preservation": True,
              "parseval_checks": len(frame), "input_source_and_checkpoint_hashes_verified": True,
              "aggregate_csv_verified": True}
    write_json(output / "audit.json", result)
    return frame, manifest, records, truth, lift


def singular_values(matrix):
    gram = matrix @ matrix.conj().T if matrix.shape[0] < matrix.shape[1] else matrix.conj().T @ matrix
    return np.sqrt(np.maximum(np.linalg.eigvalsh(gram).real[::-1], 0))


def spectral_statistics(values):
    energy = values ** 2
    probability = energy / energy.sum()
    positive = probability > 0
    return {"stable_rank": float(energy.sum() / energy[0]),
            "effective_rank": float(np.exp(-np.sum(probability[positive] * np.log(probability[positive])))),
            "energy_rank_99": int(np.searchsorted(np.cumsum(probability), .99) + 1),
            "energy_rank_999": int(np.searchsorted(np.cumsum(probability), .999) + 1)}


def figures(output, frame, manifest, truth, lift):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=.95)
    colors = {"Reference": "#222222", "Zero-fill": "#999999", "Structured low-rank S": "#39986c",
              "MatrixFormer raw": "#9ecae1", "MatrixFormer S": "#2171b5"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, pattern in zip(axes, manifest["patterns"]):
        selected = frame.loc[frame.pattern == pattern].copy()
        selected["Missing k-space"] = selected.rate.map(lambda rate: f"{rate:.0%}")
        sns.barplot(data=selected, x="Missing k-space", y="rss_image_nrmse", hue="method",
                    hue_order=evaluation.METHODS, palette=colors, errorbar="sd", capsize=.08, ax=axis)
        axis.set(title="Random points" if pattern == "points" else "Cartesian lines",
                 ylabel="RSS-image NRMSE (lower is better)")
        axis.legend(title=None, fontsize=8)
    figure.suptitle("Acquired USC phantom · native 256×340 · all 16 coils · bars: mask-seed SD")
    for extension in ["png", "pdf"]:
        figure.savefig(output / f"reconstruction_nrmse.{extension}", dpi=180)
    plt.close(figure)

    settings = [(pattern, rate) for pattern in manifest["patterns"] for rate in manifest["rates"]]
    reference = evaluation.rss_image(truth)
    images, image_axes = plt.subplots(len(settings), 5, figsize=(15, 12), layout="constrained")
    errors, error_axes = plt.subplots(len(settings), 4, figsize=(12, 12), layout="constrained")
    spectra, spectrum_axes = plt.subplots(len(settings), 2, figsize=(12, 14), layout="constrained")
    truth_spectra = {"raw": singular_values(truth.reshape(truth.shape[0], -1)),
                     "lifted": singular_values(lift.forward(truth))}
    saved_spectra = {f"reference_{key}": values for key, values in truth_spectra.items()}
    spectral_summary = {key: spectral_statistics(values) for key, values in truth_spectra.items()}
    for row, (pattern, rate) in enumerate(settings):
        case_id = f"{pattern}_seed{manifest['seeds'][0]}_missing{rate:g}"
        print(f"Plotting full-resolution spectra: {case_id}", flush=True)
        images_for_row = {"Reference": truth}
        with np.load(output / "cases" / f"{case_id}.npz") as arrays:
            images_for_row.update({method: arrays[f"{evaluation.KEYS[method]}_kspace"] for method in evaluation.METHODS})
        for column, (method, prediction) in enumerate(images_for_row.items()):
            reconstructed = evaluation.rss_image(prediction)
            image_axes[row, column].imshow(reconstructed, cmap="gray", vmin=0, vmax=reference.max())
            image_axes[row, column].set(title=method if row == 0 else None,
                                        ylabel=f"{pattern}, {rate:.0%} missing" if column == 0 else None,
                                        xticks=[], yticks=[])
            image_axes[row, column].grid(False)
            if column:
                error_axes[row, column - 1].imshow(np.abs(reconstructed - reference), cmap="magma",
                                                  vmin=0, vmax=.25 * reference.max())
                error_axes[row, column - 1].set(title=method if row == 0 else None,
                                              ylabel=f"{pattern}, {rate:.0%}" if column == 1 else None,
                                              xticks=[], yticks=[])
                error_axes[row, column - 1].grid(False)
            for spectrum_column, representation in enumerate(["raw", "lifted"]):
                if method == "Reference":
                    values = truth_spectra[representation]
                else:
                    matrix = prediction.reshape(prediction.shape[0], -1) if representation == "raw" else lift.forward(prediction)
                    values = singular_values(matrix)
                    saved_spectra[f"{case_id}_{evaluation.KEYS[method]}_{representation}"] = values
                    del matrix
                axis = spectrum_axes[row, spectrum_column]
                axis.semilogy(np.arange(1, len(values) + 1), np.maximum(values / truth_spectra[representation][0], 1e-12),
                              label=method, color=colors[method], linewidth=1.5)
                axis.set(xlabel="Singular-value index", ylabel="Singular value / reference leading value",
                         title=f"{pattern}, {rate:.0%} missing · {representation}", ylim=(1e-7, 2))
                axis.legend(fontsize=7)
    images.suptitle("Acquired phantom RSS reconstructions · fixed first mask seed · common intensity scale")
    errors.suptitle("Absolute RSS errors · fixed first mask seed · scale 0–25% of reference peak")
    spectra.suptitle("Native-resolution raw and joint S spectra · fixed first mask seed")
    for figure, name in [(images, "reconstructions"), (errors, "reconstruction_errors"), (spectra, "reconstructed_spectra")]:
        for extension in ["png", "pdf"]:
            figure.savefig(output / f"{name}.{extension}", dpi=160)
        plt.close(figure)
    np.savez_compressed(output / "singular_values.npz", **saved_spectra)
    write_json(output / "reference_spectral_statistics.json", spectral_summary)


def report(output, frame, manifest, records):
    lines = ["# Real acquired USC phantom: MatrixFormer pilot", "",
             "**Native 256 × 340 resolution, all 16 coils; no crop or coil compression.**",
             "The user-supplied MAT acquisition is the reference (not a noiseless target).",
             "", "## RSS-image NRMSE ↓", "",
             "Mean ± sample SD over three mask seeds; one physical phantom, not three subjects.", "",
             "| Acquisition | Missing | Zero-fill | Structured low-rank S | MatrixFormer raw | MatrixFormer S |",
             "|---|---:|---:|---:|---:|---:|"]
    for pattern in manifest["patterns"]:
        for rate in manifest["rates"]:
            selected = frame[(frame.pattern == pattern) & (frame.rate == rate)]
            entries = []
            for method in evaluation.METHODS:
                values = selected.loc[selected.method == method, "rss_image_nrmse"]
                entries.append(f"{values.mean():.4f} ± {values.std():.4f}")
            lines.append(f"| {pattern} | {rate:.0%} | " + " | ".join(entries) + " |")
    lines.extend(["", "## Complex multi-coil image relative L2 ↓", "",
                  "| Acquisition | Missing | Zero-fill | Structured low-rank S | MatrixFormer raw | MatrixFormer S |",
                  "|---|---:|---:|---:|---:|---:|"])
    for pattern in manifest["patterns"]:
        for rate in manifest["rates"]:
            selected = frame[(frame.pattern == pattern) & (frame.rate == rate)]
            entries = [f"{selected.loc[selected.method == method, 'complex_image_nrmse'].mean():.4f}"
                       for method in evaluation.METHODS]
            lines.append(f"| {pattern} | {rate:.0%} | " + " | ".join(entries) + " |")
    pivot = frame.pivot(index="case_id", columns="method", values="rss_image_nrmse")
    lines.extend(["", "## Interpretation and limits", "",
                  f"- S adapter beats raw MatrixFormer in {int((pivot['MatrixFormer S'] < pivot['MatrixFormer raw']).sum())}/{len(pivot)} cases.",
                  f"- S adapter beats zero-fill in {int((pivot['MatrixFormer S'] < pivot['Zero-fill']).sum())}/{len(pivot)} cases.",
                  f"- S adapter beats structured low-rank S in {int((pivot['MatrixFormer S'] < pivot['Structured low-rank S']).sum())}/{len(pivot)} cases.",
                  "- Exact preservation of every acquired sample and all saved metrics/aggregate CSVs passed audit.",
                  "- RSS NRMSE uses image relative L2, not tabular SD normalization. Full k-space and complex-image relative L2 agree by Parseval.",
                  "- Joint S shape: 168170 × 416. Neural tiles: at most 1024 × 416 (S), 256 × 1024 (raw), paired blocks with fixed permutations.",
                  "- Global normalization uses only observations. No model training, routing, transpose averaging, or hidden-target tuning.",
                  "- Baseline rank selection uses held-out acquired samples/lines, then refits all observations. It is a LORAKS-style projection baseline, not the original LORAKS optimizer.",
                  "- Neural S masking discards partially observed sums/differences; the physics baseline retains the corresponding measured-data constraints.",
                  "- No predictive-interval calibration evaluated. This is one phantom, not a clinical study.", "",
                  "## Runtime and convergence", ""])
    for name, field, key in [("Structured baseline incl. rank selection", "tuning", "seconds_including_selection"),
                             ("MatrixFormer raw", "raw_inference", "seconds"),
                             ("MatrixFormer S", "lifted_inference", "seconds")]:
        lines.append(f"- {name}: {np.mean([record[field][key] for record in records]):.2f} seconds/case (mean).")
    converged = sum(record["tuning"]["final"]["converged"] for record in records)
    lines.append(f"- Baseline refits reaching the 1e-5 tolerance: {converged}/{len(records)}; others stop at the fixed 100-iteration budget.")
    peak = max(record[field]["peak_cuda_allocated_gib"] for record in records for field in ["raw_inference", "lifted_inference"])
    lines.extend([f"- Peak neural CUDA allocation: {peak:.2f} GiB.", "", "## Artifacts", "",
                  "`per_case.csv`, `summary.csv`, `manifest.json`, `audit.json`, `cases/*.npz`, `cases/*.json`.",
                  "Figures: `reconstruction_nrmse.png`, `reconstructions.png`, `reconstruction_errors.png`, `reconstructed_spectra.png` (also PDF).",
                  "Spectra: `singular_values.npz`, `reference_spectral_statistics.json`.", "",
                  "Source: https://mr.usc.edu/download/data/ (user-supplied file; preserve USC license/citation obligations).",
                  "Haldar (2014), LORAKS, DOI: https://doi.org/10.1109/TMI.2013.2293974.",
                  f"Protocol: `{evaluation.ROOT / 'PHYSICS_MRI_REAL.md'}`.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame, manifest, records, truth, lift = audit(args.output)
    report(args.output, frame, manifest, records)
    figures(args.output, frame, manifest, truth, lift)
    print(args.output / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
