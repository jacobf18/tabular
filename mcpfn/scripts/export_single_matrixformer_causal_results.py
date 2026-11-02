"""Verify and export the combined-model causal evaluation to the paper repository."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_single_matrixformer_causal import CANDIDATES, METRICS, OUTPUT, REFERENCES, summarize
from recsys_evaluation_common import sha256, write_json


LABELS = {"matrixformer": "MatrixFormer (standard)", "matrixformer-T": "MatrixFormer (transpose)",
          "matrixformer-avg": "MatrixFormer (average)", "matrixformer-plus": "MatrixFormer*",
          "sdid": "SDiD", "sc": "SC", "did": "DiD", "softimpute": "SoftImpute",
          "knn": "K-NN", "hyperimpute": "HyperImpute"}
PAPER_METHOD = "matrixformer"


def paper_scores(scores):
    if not set(CANDIDATES).issubset(set(scores.method)):
        raise ValueError("Expected all single-model orientations")
    selected = scores.copy()
    selected["shown"] = selected.method.isin((*REFERENCES, PAPER_METHOD))
    return selected


def verify_archive(scores, historical):
    keys = ["family", "metric", "method"]
    archived = historical[historical.shown].set_index(keys).sort_index()
    current = scores[scores.method.isin(REFERENCES)].set_index(keys).sort_index()
    if not current.index.equals(archived.index):
        raise ValueError("Historical reference coverage mismatch")
    for column in ("score", "se", "ci95"):
        np.testing.assert_allclose(current[column], archived[column], rtol=0, atol=5.01e-7)
    for column in ("n_scenarios", "n_draws"):
        np.testing.assert_array_equal(current[column], archived[column])
    return float(np.max(np.abs(current[["score", "se", "ci95"]].to_numpy() -
                               archived[["score", "se", "ci95"]].to_numpy())))


def report(scores):
    lines = ["# Combined MatrixFormer: causal-panel evaluation", "",
             "Fixed checkpoint, 3,000 frozen placebo draws in 12 designs. The paper shows the",
             "standard-orientation predictions for MatrixFormer (one forward pass), restoring",
             "the original evaluation's primary orientation and matching the other main comparisons.",
             "An intermediate manuscript used averaging after reviewing all orientations; that",
             "post-evaluation reporting choice is not retained in the primary comparison.",
             "All orientations remain archived and plotted in the appendix. MatrixFormer* retains",
             "its archived standard orientation. No new inference is needed for this update.",
             "This evaluation performs no training or checkpoint reselection.", "",
             "Scores are per-draw min-max normalized on the **fixed seven-reference scale**",
             "(MatrixFormer*, SDiD, SC, DiD, SoftImpute, K-NN, HyperImpute), then averaged",
             "over draws and equally over designs within each benchmark. This is **not**",
             "standard-deviation-normalized imputation NRMSE. New scores may fall outside [0, 1].",
             "Intervals are 95% normal intervals over draws, treating designs as fixed.",
             "Draws share source data, so these are not independent-dataset uncertainty estimates.",
             "Orientation ablations are exploratory; intervals do not adjust for selection among them.", ""]
    for metric, title in zip(METRICS, ("Treated-block RMSE", "Absolute treatment-effect error")):
        lines += [f"## Normalized {title}", "", "| Method | California smoking | CPS | Kansas |",
                  "|---|---:|---:|---:|"]
        for method in (*CANDIDATES, *REFERENCES):
            cells = []
            for family in ("California smoking", "CPS", "Kansas"):
                selected = scores[(scores.family == family) & (scores.metric == metric) &
                                  (scores.method == method)].iloc[0]
                cells.append(f"{selected.score:.3f} ± {selected.ci95:.3f}")
            lines.append(f"| {LABELS[method]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Artifacts", "", "- `per_draw_metrics.csv`: all three single-model orientations.",
              "- `reference_per_draw_metrics.csv`: frozen seven-method reference errors.",
              "- `normalized_scores.csv`: plot data, full precision and 95% intervals.",
              "- `per_design_scores.csv`: raw and normalized means and standard errors by design.",
              "- `evaluation_manifest.json`: checkpoint, input, code, environment and result hashes.",
              "- `control_parity.json`: first draw of every design, all three parent orientations.",
              "- Full reconstructed panels remain in the research run's `predictions/` directory.", "",
              "## Reproduce", "", "From `/root/tabular`:", "", "```bash",
              "PYTHONPATH=mcpfn/scripts:mcpfn/src python -m pytest -q mcpfn/tests/test_single_matrixformer_causal.py",
              "python mcpfn/scripts/evaluate_single_matrixformer_causal.py",
              "python mcpfn/scripts/export_single_matrixformer_causal_results.py --paper-root /root/MatrixFormerPaper",
              "cd /root/MatrixFormerPaper && python scripts/plot_causal_ensemble_results.py", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUTPUT)
    parser.add_argument("--paper-root", type=Path, default=Path("/root/MatrixFormerPaper"))
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if "completed_utc" not in manifest:
        raise ValueError("Evaluation is incomplete")
    for filename, expected in manifest["outputs"].items():
        if sha256(args.run / filename) != expected:
            raise ValueError(f"Evaluation output changed: {filename}")
    candidate = pd.read_csv(args.run / "per_draw_metrics.csv")
    reference = pd.read_csv(args.run / "reference_per_draw_metrics.csv")
    scores, designs = summarize(pd.concat([reference, candidate], ignore_index=True))
    saved_scores = pd.read_csv(args.run / "normalized_scores.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(scores, saved_scores, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(designs, pd.read_csv(args.run / "per_design_scores.csv"), rtol=1e-12, atol=1e-12)
    archived = args.paper_root / "results/single_model/causal_ensemble_scores.csv"
    difference = verify_archive(scores, pd.read_csv(archived))
    destination = args.paper_root / "results/single_model/causal"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("per_draw_metrics.csv", "reference_per_draw_metrics.csv",
                     "per_design_scores.csv", "control_parity.json"):
        shutil.copy2(args.run / filename, destination / filename)
    scores = paper_scores(saved_scores)
    scores.to_csv(destination / "normalized_scores.csv", index=False)
    shutil.copy2(args.run / "manifest.json", destination / "evaluation_manifest.json")
    text = report(scores)
    (args.run / "RESULTS.md").write_text(text)
    (destination / "README.md").write_text(text)
    write_json(destination / "export_manifest.json", {
        "evaluation_directory": str(args.run), "exporter_sha256": sha256(Path(__file__)),
        "historical_scores_sha256": sha256(archived), "historical_max_rounding_difference": difference,
        "reference_scores_unchanged": True,
        "original_evaluation_primary_orientation": "standard",
        "paper_single_model_method": PAPER_METHOD,
        "paper_single_model_orientation": "standard; one forward pass",
        "orientation_selection": "restored original standard-primary protocol; no best-orientation selection",
        "reporting_history": "an intermediate manuscript showed averaging after reviewing all orientations",
        "original_evaluation_scores_sha256": sha256(args.run / "normalized_scores.csv"),
        "paper_score_changes": "shown flags only; all estimates and confidence intervals unchanged",
        "files": {source.name: sha256(source) for source in destination.iterdir()
                  if source.name != "export_manifest.json"},
    })
    print(text)
    print(f"Exported to {destination}; old score/CI differences <= {difference:.3g} (six-decimal rounding)")


if __name__ == "__main__":
    main()
