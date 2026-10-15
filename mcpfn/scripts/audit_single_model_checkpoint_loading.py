"""CPU-only exact checkpoint loading through both frozen benchmark inference adapters."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from tabimpute.interface import ImputePFN
from tabimpute.streamed_inference import StreamedImputePFN


def verify_loaded(model, expected):
    actual = model.state_dict()
    if tuple(actual) != tuple(expected):
        raise ValueError("Inference adapter changed the checkpoint inventory")
    for name, value in actual.items():
        reference = expected[name]
        if (value.device.type != "cpu" or value.shape != reference.shape or value.dtype != reference.dtype
                or not torch.isfinite(value).all() or not torch.equal(value, reference)):
            raise ValueError(f"Inference adapter changed a saved tensor: {name}")
    if model.training:
        raise ValueError("Require evaluation mode")
    return {"model_class": type(model).__name__, "state_tensors": len(actual),
            "state_elements": sum(value.numel() for value in actual.values()),
            "parameter_elements": sum(value.numel() for value in model.parameters()),
            "all_tensors_bitwise_equal": True, "all_tensors_on_cpu": True}


def audit(checkpoint, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("This check must not use the training GPU")
    if output.exists() or output.is_relative_to(checkpoint.parent):
        raise ValueError("Use a new, separate artifact directory")
    config_path = checkpoint.parent / "config.json"
    inputs = {str(path): sha256(path) for path in (checkpoint, config_path)}
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    fingerprints = {str(path): sha256(path) for path in sources}
    config = json.loads(config_path.read_text())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" not in config or "model" not in payload:
        raise ValueError("Require one explicitly configured ordinary checkpoint")
    output.mkdir(parents=True)
    write_json(output / "registration.json", {"input_sha256": inputs, "source_sha256": fingerprints,
        "scope": "Exact CPU state loading only; no predictions, benchmark data, synthetic bank or model selection"})
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    records = []
    for factory in (ImputePFN, StreamedImputePFN):
        adapter = factory(device="cpu", checkpoint_path=str(checkpoint), entry_wise_features=False,
            json_config={"config": config["model"]}, model_family="new_stable", precision_mode="amp_bfloat16",
            prediction_stat="mean", max_num_rows=None, max_num_cols=None, max_num_chunks=1, verbose=False)
        records.append({"adapter": factory.__name__, **verify_loaded(adapter.model, payload["model"]),
                        "prediction_stat": adapter.prediction_stat, "precision_mode": adapter.precision_mode})
        del adapter
        gc.collect()
    verify_hashes(inputs)
    verify_hashes(fingerprints)
    if torch.cuda.is_initialized():
        raise ValueError("Loading unexpectedly initialized CUDA")
    result = {"complete": True, "checkpoint": str(checkpoint), "checkpoint_sha256": inputs[str(checkpoint)],
        "adapters": records, "cuda_initialized": False, "forward_passes": 0, "predictions_created": False,
        "candidate_qualified": False, "goal_success_not_assessed": True, "original_artifacts_unchanged": True}
    write_json(output / "audit.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.checkpoint.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
