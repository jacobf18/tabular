"""Audit the saved cylinder pilot and write a readable results report."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.io import loadmat

from evaluate_cylinder_completion import METHODS, gaussian_metrics, point_metrics, sha256, write_json


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def save_bar_panels(output, panels, rows, columns, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font="STIXGeneral", font_scale=.85)
    plt.rcParams.update({"pdf.fonttype": 42, "mathtext.fontset": "stix"})
    figure, axes = plt.subplots(rows, columns, figsize=(3.9 * columns, 3.1 * rows), squeeze=False)
    for axis, (title, data) in zip(axes.ravel(), panels):
        data = data.sort_values("mean")
        error = data["std"].fillna(0).to_numpy()
        colors = ["#2A6FBB" if label.startswith("MatrixFormer") or label == "Uncertainty" else "#b9bec8" for label in data.label]
        axis.barh(data.label, data["mean"], color=colors, xerr=error, capsize=2)
        largest = float(np.max(data["mean"].to_numpy() + error))
        for position, (value, deviation) in enumerate(zip(data["mean"], error)):
            axis.text(value + deviation + .015 * largest, position, f"{value:.3f}", va="center", fontsize=8,
                      fontweight="bold" if position == 0 else "normal")
        axis.set_xlim(0, largest * 1.2)
        axis.invert_yaxis()
        axis.set(title=title, xlabel="NRMSE ↓ (original observed global SD)")
        axis.grid(axis="y", visible=False)
    sns.despine(fig=figure)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output / f"{filename}.{extension}", dpi=180)
    plt.close(figure)


def audit(output):
    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "status.json").read_text())
    records = [json.loads(path.read_text()) for path in sorted((output / "cases").glob("*.json"))]
    expected = {(pattern, rate, seed) for pattern in manifest["patterns"] for rate in manifest["rates"] for seed in manifest["mask_seeds"]}
    assert status["state"] == "complete" and len(records) == manifest["expected_cases"]
    assert {(record["pattern"], record["rate"], record["seed"]) for record in records} == expected
    for group in ("checkpoint_sha256", "config_sha256", "source_sha256"):
        for filename, digest in manifest[group].items():
            assert sha256(filename) == digest, filename
    data_path = output.parent / "data/CYLINDER_ALL_LOW_RES.mat"
    assert sha256(data_path) == manifest["data_sha256"]
    with np.load(output / "dataset.npz") as dataset:
        truth = dataset["truth"]
        np.testing.assert_array_equal(truth, loadmat(data_path)["VORTALL"].real[dataset["spatial_indices"]])
    for record in records:
        path = output / "predictions" / f"{record['case_id']}.npz"
        assert sha256(path) == record["prediction_sha256"]
        with np.load(path) as saved:
            missing, validation = saved["missing"], saved["validation"]
            assert not np.any(missing & validation)
            assert not missing.all(axis=0).any() and not missing.all(axis=1).any()
            assert missing.sum() == record["test_cells"]
            np.testing.assert_allclose(np.std(truth[~missing]), record["observed_std"], rtol=1e-12)
            for method in METHODS:
                values = point_metrics(truth, saved[f"prediction_{METHODS.index(method)}"], missing, record["observed_std"])
                for key, value in values.items():
                    np.testing.assert_allclose(value, record["metrics"][method][key], rtol=2e-5, atol=1e-7)
            indices = saved["uncertainty_indices"]
            assert np.all(missing.ravel()[indices])
            assert len(indices) == len(np.unique(indices))
            for method in ("MatrixFormer", "MatrixFormer*"):
                prefix = f"uncertainty_{METHODS.index(method)}_"
                quantiles = saved[prefix + "quantiles"]
                assert np.all(np.diff(quantiles, axis=1) >= -1e-6)
                for coverage, lower, upper in [(50, 2, 3), (90, 1, 4), (95, 0, 5)]:
                    target = truth.ravel()[indices]
                    covered = (target >= quantiles[:, lower]) & (target <= quantiles[:, upper])
                    np.testing.assert_allclose(covered.mean(), record["metrics"][method][f"coverage_{coverage}"])
                for key in ("crps", "nll"):
                    np.testing.assert_allclose(saved[prefix + key].mean(), record["metrics"][method][key])
            for method, details in record["baseline_tuning"].items():
                assert details["selected"] == min(details["trials"], key=lambda trial: trial["validation_rmse"])
                values = gaussian_metrics(truth.ravel()[indices], saved[f"prediction_{METHODS.index(method)}"].ravel()[indices],
                                          details["selected"]["validation_rmse"], record["observed_std"])
                for key in ("crps", "nll", "coverage_90"):
                    np.testing.assert_allclose(values[key], record["metrics"][method][key], rtol=2e-5, atol=1e-7)
    report = {"passed": True, "cases": len(records), "point_scores": len(records) * len(METHODS) * 4,
              "neural_coverage_checks": len(records) * 2 * 3, "observed_only_validation_masks": True,
              "checkpoint_and_source_hashes_verified": True}
    write_json(output / "audit.json", report)
    for filename in manifest["source_sha256"]:
        source = Path(filename)
        destination = output / "source_snapshot" / source.relative_to("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return manifest, records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output.resolve()
    manifest, records, checks = audit(output)
    summary = pd.read_csv(output / "summary.csv")
    rates = sorted(summary.rate.unique())
    seed_count = len(manifest["mask_seeds"])
    lines = ["# Cylinder-wake completion: zero-shot physics pilot", "",
             f"**Complete:** {checks['cases']} masks, {manifest['matrix_shape'][0]:,} spatial locations × {manifest['matrix_shape'][1]} snapshots.", "",
             "PyDMD Reynolds-number-100 cylinder-flow vorticity; no added noise, no new training, no row-wise train/test split.",
             "Spatial locations are subsampled from the 9,834-location public dataset. All snapshots are retained.", "",
             "## Accuracy", "",
             "NRMSE = hidden-entry RMSE / SD of observed entries (global population SD). Lower is better.",
             f"Values are mean ± sample SD over {seed_count} mask seeds, not confidence intervals across independent flows.", ""]
    for pattern in manifest["patterns"]:
        lines += [f"### {pattern.replace('_', ' ').upper()}", ""]
        current = summary[summary.pattern == pattern].set_index(["method", "rate"])
        rows = []
        for method in METHODS:
            cells = []
            for rate in rates:
                item = current.loc[method, rate]
                text = f"{item.nrmse_mean:.4f} ± {item.nrmse_std:.4f}"
                if item.nrmse_mean == current.xs(rate, level="rate").nrmse_mean.min():
                    text = f"**{text}**"
                cells.append(text)
            rows.append([method, *cells])
        lines += [markdown_table(["Method", *[f"{rate:.0%} missing" for rate in rates]], rows), ""]
    lines += ["## Native predictive distributions", "",
              f"Scores use up to {manifest['uncertainty_max_cells']:,} uniformly sampled hidden entries per mask, sampled independently of errors.",
              "Normalized CRPS uses the same observed SD as NRMSE; lower is better. CRPS uses 199-quantile numerical integration.",
              "Coverage should be close to 90%; width is in original vorticity units. No physics-label recalibration is applied.", ""]
    rows = []
    for pattern in manifest["patterns"]:
        for rate in rates:
            for method in ("MatrixFormer", "MatrixFormer*"):
                item = summary[(summary.pattern == pattern) & (summary.rate == rate) & (summary.method == method)].iloc[0]
                rows.append([pattern, f"{rate:.0%}", method, f"{item.normalized_crps_mean:.4f}",
                             f"{item.coverage_90_mean:.1%}", f"{item.width_90_mean:.4f}", f"{item.nll_mean:.3f}"])
    lines += [markdown_table(["Pattern", "Missing", "Method", "Normalized CRPS ↓", "90% coverage", "90% width", "NLL ↓"], rows), "",
              "NLL is a continuous-density score and can legitimately be negative. Baseline validation-Gaussian uncertainty,",
              "50%/95% coverage, RMSE/MAE/R², individual seeds and tuning decisions are also saved in the CSV/JSON files.", ""]
    acquisition_path = output / "acquisition/summary.csv"
    if acquisition_path.exists():
        acquisition = pd.read_csv(acquisition_path)
        acquisition_manifest = json.loads((output / "acquisition/manifest.json").read_text())
        assert json.loads((output / "acquisition/audit.json").read_text())["passed"]
        rows, panels = [], []
        for pattern in manifest["patterns"]:
            current = acquisition[acquisition.pattern == pattern]
            panel = [{"label": "No acquisition", "mean": current.iloc[0].nrmse_before_mean,
                      "std": current.iloc[0].nrmse_before_std}]
            for item in current.itertuples():
                rows.append([pattern, item.policy, f"{item.nrmse_before_mean:.4f}",
                             f"{item.nrmse_after_mean:.4f} ± {item.nrmse_after_std:.4f}", f"{item.relative_reduction_mean:.1%}"])
                panel.append({"label": item.policy.capitalize(), "mean": item.nrmse_after_mean, "std": item.nrmse_after_std})
            panels.append((pattern.replace("_", " ").upper() + " · 80% initially missing", pd.DataFrame(panel)))
        save_bar_panels(output, panels, 1, len(panels), "acquisition_comparison")
        lines += ["## Exploratory measurement acquisition", "",
                  f"Single MatrixFormer, 80% initial missingness, {acquisition_manifest['budget']:,} additional measurements from the same 10,000-cell pool.",
                  "Scores use a fixed test set excluding the ENTIRE acquisition pool, and the original observed SD denominator.",
                  "Selections use original predictions/uncertainties only; values are revealed afterward, and the frozen model is rerun.", "",
                  markdown_table(["Pattern", "Acquisition policy", "Before NRMSE", "After NRMSE", "Mean per-seed reduction"], rows), "",
                  "Uncertainty-guided acquisition improves over random acquisition on average in this pilot, but the magnitude-only",
                  "control is slightly better on average. Therefore this experiment does **not** establish a unique advantage of",
                  "distributional uncertainty over a simpler signal-strength heuristic. Results have substantial mask-seed variability.",
                  "This follow-up was designed after the completion pilot and is exploratory. All indices, masks, predictions,",
                  "protocols and scores are saved under `acquisition/`. Figure: `acquisition_comparison.png/pdf`.", ""]
    lines += ["## Interpretation and limits", "",
              "- SoftImpute is much stronger at 20% missingness and at 50% MCAR; MatrixFormer is not a universal winner.",
              "- MatrixFormer performs well relative to these baselines at 80% MCAR and for long sensor outages.",
              "- Nominal 90% MatrixFormer intervals cover about 92–96% under MCAR, but only about 63% under 80% outages: severe-block uncertainty is overconfident.",
              "- This is a pilot on one noiseless, coherent flow trajectory, not evidence across fluids or physics broadly.",
              "- The block mask wraps time; it represents synthetic circular sensor outages, not an exact physical recording model.",
              "- The 10% inner holdout is random even for outer outages. This can mismatch the outer task, particularly for baseline uncertainty.",
              "- MatrixFormer* is the existing observed-error router with both experts globally normalized for physics. Its evidence is in-sample reconstruction, not independent model-selection evidence.",
              "- Native intervals are marginal cellwise distributions; no joint conservation or physical-consistency guarantee is asserted.",
              "- Baseline rank/shrinkage is selected only on held-out observations; no hidden test values enter normalization or tuning.",
              "- The iterative gappy POD baseline is fancyimpute IterativeSVD, not an exact reproduction of the 2004 weighted method.", "",
              "## Reproducibility", "",
              f"- Output directory: `{output}`", f"- Dataset revision: `{manifest['data_revision']}`",
              f"- Data SHA256: `{manifest['data_sha256']}`", "- Protocol, environments and model/source hashes: `manifest.json`.",
              "- Audited saved predictions and masks: `audit.json`.", "- All outcomes: `per_case.csv`, `summary.csv`, `cases/`, `predictions/`.",
              "- Figures: `accuracy.png/pdf`, `calibration.png/pdf`.",
              "- Protocol and citations: `/root/tabular/mcpfn/PHYSICS_CYLINDER.md`.",
              "- Source snapshots: `source_snapshot/`.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    panels = []
    for pattern in manifest["patterns"]:
        for rate in rates:
            data = summary[(summary.pattern == pattern) & (summary.rate == rate)]
            panel = data.rename(columns={"method": "label", "nrmse_mean": "mean", "nrmse_std": "std"})
            panels.append((f"{pattern.replace('_', ' ').upper()} · {rate:.0%} missing", panel))
    save_bar_panels(output, panels, len(manifest["patterns"]), len(rates), "accuracy")
    write_json(output / "report_manifest.json", {"report_source_sha256": sha256(__file__),
                "summary_sha256": sha256(output / "summary.csv"),
                "acquisition_summary_sha256": sha256(acquisition_path) if acquisition_path.exists() else None,
                "artifacts_sha256": {path.name: sha256(path) for path in output.iterdir() if path.suffix in {".pdf", ".png", ".md"}}})
    print(json.dumps(checks))
    print(output / "RESULTS.md")


if __name__ == "__main__":
    main()
