"""Registered inner-only context screen with unchanged synthetic weights."""
import argparse
import importlib.resources
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from dwip_recsys_experiment import DEFAULT_ROOT, rating_validation, selection_score
from dwip_recsys_provenance import require_synthetic_checkpoint
from evaluate_dwip_recsys import CHECKPOINT, DATASETS, OUTPUT, sha256, write_json
from evaluate_dwip_recsys_candidate import verify_selection
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", choices=("items", "users"), default="items")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    provenance = require_synthetic_checkpoint(args.checkpoint)
    selected = verify_selection(args.checkpoint)
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    assert config["model"] == json.loads((CHECKPOINT.parent / "config.json").read_text())["model"]
    normalization = config["arguments"]["normalization"]
    assert normalization == "two_way_user_scale"
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    original = payload["model"]
    model = TabImputeModelStable(**config["model"]).cuda().float().eval()
    model.load_state_dict(original, strict=True)
    assert len(original) == 254 and sum(tensor.numel() for tensor in original.values()) == 77998688
    borders = importlib.resources.files("tabimpute") / "data/borders.pt"
    criterion = FullSupportBarDistribution(torch.load(borders, map_location="cuda", weights_only=True))
    baseline = json.loads((DEFAULT_ROOT / "mf_baseline.json").read_text())["validation"]
    expected = json.loads((args.checkpoint.parent / f"validation_{selected['step']}.json").read_text())["datasets"]
    report = {"purpose": "Inner-only inference geometry screen; no optimizer, neural updates, or outer evaluation",
              "synthetic_provenance": provenance, "selection": selected, "normalization": normalization,
              "axis": args.axis,
              "registered_windows": [[1000, 1000], [1000, 2000] if args.axis == "items" else [2000, 1000]], "candidates": []}
    for rows, cols in report["registered_windows"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        values = rating_validation(model, criterion, normalization=normalization, block_rows=rows, block_cols=cols)
        torch.cuda.synchronize()
        if (rows, cols) == (1000, 1000):
            for name in values:
                for metric in ("MAE", "RMSE"):
                    assert np.isclose(values[name][metric], expected[name][metric], rtol=0, atol=1e-12)
        candidate = {"block_shape": [rows, cols], "datasets": values,
                     "worst_inner_rating_ratio": selection_score(values, baseline, "worst_rating"),
                     "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                     "seconds": time.monotonic() - started}
        report["candidates"].append(candidate)
        print(json.dumps(candidate), flush=True)
        assert all(torch.equal(tensor.cpu(), original[name]) for name, tensor in model.state_dict().items())
        assert all(parameter.grad is None for parameter in model.parameters())
    report.update(weights_bitwise_unchanged=True, gradients_absent=True, legacy_scores_reproduced=True,
                  checkpoint_sha256=sha256(args.checkpoint), baseline_sha256=sha256(DEFAULT_ROOT / "mf_baseline.json"),
                  split_sha256={spec["folder"]: sha256(OUTPUT / spec["folder"] / "split.npz") for spec in DATASETS.values()},
                  best_inner_window=min(report["candidates"], key=lambda value: value["worst_inner_rating_ratio"])["block_shape"],
                  outer_evaluated=False, goal_achieved=False)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("dwip_recsys_experiment.py").resolve(),
               root / "src/tabimpute/prior/recsys_rating_episode.py", root / "src/tabimpute/model/model_new_stable.py",
               root / "src/tabimpute/model/bar_distribution.py", root / "src/tabimpute/interface.py"]
    snapshot = args.output.with_suffix(".source_snapshot")
    snapshot.mkdir()
    report["source_sha256"] = {str(source): sha256(source) for source in sources}
    for source in sources:
        shutil.copyfile(source, snapshot / source.name)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
