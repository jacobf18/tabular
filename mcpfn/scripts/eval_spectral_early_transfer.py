"""Report-only paired synthetic transfer checks at a prespecified early snapshot."""
import argparse
import importlib.resources
import json
import shutil
from pathlib import Path

import torch

from rowcol_fixed_validation import evaluate_fixed_validation
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_rowcol_svt_hybrid import TabImputeModelRowColumnSVTHybrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="spectral_query_20260910")
    parser.add_argument("--trial", default="joint_low")
    parser.add_argument("--step", type=int, default=4000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    torch.backends.cuda.enable_cudnn_sdp(False)
    root = REPO / "runs/objective_queues" / args.queue
    manifest = json.loads((root / "manifest.json").read_text())
    work = WORK / f"tabimpute-{args.queue}-{args.trial}"
    config = json.loads((work / "config.json").read_text())["model"]
    criterion = FullSupportBarDistribution(torch.load(
        importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).cuda())
    caches = {"spectral": root / "holdout.pth",
              "general": REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth"}
    output = root / "early_transfer"
    output.mkdir(exist_ok=True)
    results = {}
    for label, checkpoint in (("anchor", Path(manifest["initial_checkpoint"])),
                               (str(args.step), work / f"checkpoint_{args.step}.pth")):
        model = None
        for domain, cache in caches.items():
            metric_path = output / f"{args.trial}_{label}_{domain}.json"
            if not metric_path.exists():
                if model is None:
                    model = TabImputeModelRowColumnSVTHybrid(**config).cuda().eval()
                    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
                cases = torch.load(cache, map_location="cpu", weights_only=True)["cases"]
                metrics = evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16",
                    output / f"{args.trial}_{label}_{domain}_predictions.pth")
                write_json(metric_path, metrics)
            results[f"{label}_{domain}"] = json.loads(metric_path.read_text())
            if label != "anchor" and domain == "spectral":
                # Populate the queued evaluator's cache so it does not repeat this work.
                destination = root / f"{args.trial}_{label}_holdout.json"
                if destination.exists() and json.loads(destination.read_text()) != results[f"{label}_{domain}"]:
                    raise ValueError("Existing queued metrics differ from early evaluation")
                if not destination.exists():
                    shutil.copy2(output / f"{args.trial}_{label}_{domain}_predictions.pth",
                                 root / f"{args.trial}_{label}_predictions.pth")
                    write_json(destination, results[f"{label}_{domain}"])
        del model
        torch.cuda.empty_cache()
    comparison = {}
    for domain in caches:
        before, after = results[f"anchor_{domain}"], results[f"{args.step}_{domain}"]
        groups = {name: {"before": group["rmse"], "after": after["groups"][name]["rmse"],
            "ratio": after["groups"][name]["rmse"] / group["rmse"], "monitor_only": group["monitor_only"]}
            for name, group in before["groups"].items()}
        comparison[domain] = dict(before_rmse=before["macro_rmse"], after_rmse=after["macro_rmse"],
            before_nll=before["macro_nll"], after_nll=after["macro_nll"], groups=groups,
            selection_groups_improved=sum(v["ratio"] < 1 for v in groups.values() if not v["monitor_only"]))
    write_json(output / f"{args.trial}_{args.step}_comparison.json", comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
