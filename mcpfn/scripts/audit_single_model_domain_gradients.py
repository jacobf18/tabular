"""CPU remeasurement of every saved paired domain gradient, without CUDA replay."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from audit_single_matrixformer_structured_training import close_values
from dwip_teacher_mean_training import DwipTrainingAdapter, verify_hashes
import probe_single_model_domain_gradients as probe
from recsys_evaluation_common import sha256, write_json


def ready(run):
    status = probe.read(run / "status.json")
    if status["state"] == "complete":
        return True
    if status["state"] in ("failed", "interrupted", "registered"):
        raise ValueError("No complete live diagnostic to audit; never restart it here")
    try:
        os.kill(status["pid"], 0)
    except ProcessLookupError:
        if probe.read(run / "status.json")["state"] == "complete":
            return True
        raise RuntimeError("Diagnostic handle disappeared; inspect the original failure") from None
    return False


def verify_payload(payload, record, specification, elements):
    expected = {f"{domain}{suffix}" for domain in ("imputation", "recsys") for suffix in ("", "_episode", "_teacher")}
    if set(payload) != expected or any(record[key] != value for key, value in specification.items()):
        raise ValueError("Original paired domains or registered case changed")
    for domain, expected_teacher in (("imputation", "initialization"), ("recsys", "specialist")):
        vector, episode, teacher = payload[domain], payload[f"{domain}_episode"], payload[f"{domain}_teacher"]
        if vector.dtype != torch.float32 or vector.shape != (elements,) or not torch.isfinite(vector).all():
            raise ValueError("Require every original finite FP32 gradient coordinate")
        if episode["metadata"] != record["metadata"][domain] or record["teachers"][domain]["target"] != expected_teacher:
            raise ValueError("Recorded source or frozen training teacher changed")
        requested = specification["imputation" if domain == "imputation" else "rating"]
        expected_seed = requested["structured"]["seed"] if "structured" in requested else requested["seed"]
        metadata = episode["metadata"]
        if (metadata["seed"] != expected_seed or metadata["gradient_data"] != "synthetic"
                or metadata["domain"] != domain or not math.isfinite(record["losses"][domain])):
            raise ValueError("Require the original finite-loss synthetic pair and its seeds")
        source, target, mask = episode["source"], episode["target"], episode["mask"]
        if (source.shape != target.shape or source.shape != mask.shape or source.shape != teacher.shape
                or mask.dtype != torch.bool or not mask.any() or not torch.isfinite(target[mask]).all()
                or not torch.isfinite(teacher).all()):
            raise ValueError("Saved source, target, query or frozen teacher grid is incomplete")
        if domain == "recsys" and list(source.shape[1:]) != requested["shape"]:
            raise ValueError("Rating contexts must not shrink")
        if domain == "imputation" and "structured" in requested:
            structured = requested["structured"]
            if metadata["requested_shape"] != structured["shape"] or any(metadata[name] != structured[name]
                    for name in ("family", "component", "regime", "requested_missingness")):
                raise ValueError("Structured mechanism or requested geometry changed")


def verify_inventory(inventory, model):
    expected = [[name, list(parameter.shape), parameter.numel()] for name, parameter in model.named_parameters()]
    if inventory != expected:
        raise ValueError("Require the complete ordered original parameter inventory, not a subset")


def markdown(groups, limitations):
    lines = ["# Paired synthetic domain-gradient diagnosis", "", "64 pairs at the unchanged V8 checkpoint; no optimizer updates or checkpoint selection.",
             "The48 structured cases cover all twelve mechanisms and four generators. Sixteen legacy cases span all eight original components.",
             "All64 original rating episodes use1000x1000 or2000x1000 contexts.",
             "", "The CPU audit reloads every saved FP32 gradient coordinate and recomputes the statistics. It does not repeat autograd or predict AdamW/generalization outcomes.",
             "Positive gradient-direction dot products imply descent only under the reported first-order SGD approximation.",
             "The all-case row describes this stratified probe; it is not an estimate of the50/50 imputation-prior training mixture.",
             "", "| Group | Pairs | Median imputation/rating norm | Median cross cosine | Conflicting pairs | Rating adverse direction: joint clip | Rating adverse direction: domain clip |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group, values in groups.items():
        ratio = "undefined" if values["median_imputation_to_rating_norm"] is None else f'{values["median_imputation_to_rating_norm"]:.6f}'
        cosine = "undefined" if values["median_cross_cosine"] is None else f'{values["median_cross_cosine"]:.6f}'
        lines.append(f'| {group} | {values["cases"]} | {ratio} | {cosine} | {values["negative_cross_dot_cases"]} | '
                     f'{values["methods"]["joint_clip"]["rating_negative_descent_cases"]} | '
                     f'{values["methods"]["domain_clip_then_mean"]["rating_negative_descent_cases"]} |')
    return "\n".join([*lines, "", "Limitations: " + limitations + ".", "",
        "Both methods retain the original scalar clipping threshold0.5 and domain weights0.5. The proposed comparison clips isolated domain gradients before averaging instead of clipping their mean.",
        "Raw paired vectors, input/target/query grids and teacher grids remain in the source diagnostic. No training recipe is selected by this report.", ""])


def audit(run, output):
    config = probe.verify_registration(run)
    summary = probe.read(run / "summary.json")
    if (not summary["complete"] or summary["cases"] != 64 or not summary["model_state_unchanged"]
            or summary["training_performed"] or summary["checkpoint_selection_performed"]):
        raise ValueError("Require the completed read-only64-pair diagnostic")
    inventory = probe.read(run / "parameter_inventory.json")
    original = torch.load(config["base"]["parent_checkpoint"], map_location="cpu", weights_only=True)["model"]
    reference_model = DwipTrainingAdapter(**config["base"]["model"])
    reference_model.load_state_dict(original, strict=True)
    verify_inventory(inventory, reference_model)
    del reference_model
    if len({name for name, shape, count in inventory}) != len(inventory):
        raise ValueError("Parameter coordinates must have distinct original names")
    for name, shape, count in inventory:
        if name not in original or list(original[name].shape) != shape or original[name].numel() != count:
            raise ValueError("Gradient inventory does not match the original checkpoint")
    elements = sum(count for name, shape, count in inventory)
    del original
    hashes = {str(run / name): sha256(run / name) for name in
              ("config.json", "registration.json", "status.json", "summary.json", "parameter_inventory.json")}
    records = []
    for specification in config["cases"]:
        record_path = run / f'pair_{specification["index"]:03d}.json'
        record = probe.read(record_path)
        hashes[str(record_path)] = sha256(record_path)
        path = Path(record["raw_gradients"])
        if path.resolve() != run / "pairs" / f'pair_{specification["index"]:03d}.pt':
            raise ValueError("The paired raw gradients must belong to the original diagnostic")
        if summary["raw_gradient_sha256"].get(str(path)) != record["raw_sha256"]:
            raise ValueError("Summary no longer binds the paired gradients")
        hashes[str(path)] = record["raw_sha256"]
        verify_hashes({str(path): record["raw_sha256"]})
        payload = torch.load(path, map_location="cpu", weights_only=True)
        verify_payload(payload, record, specification, elements)
        recalculated = probe.geometry(payload["imputation"], payload["recsys"])
        close_values(recalculated, record["geometry"])
        records.append({**record, "geometry": recalculated})
        del payload
        write_json(output / "status.json", {"state": "auditing", "completed": len(records), "expected": 64,
                                           "pid": os.getpid(), "updated_unix": time.time()})
    groups = probe.summarize(records)
    close_values(groups, summary["groups"])
    if len(summary["raw_gradient_sha256"]) != 64:
        raise ValueError("Every pair must remain represented exactly once")
    verify_hashes(hashes)
    probe.verify_registration(run)
    result = {"complete": True, "pairs": 64, "gradient_vectors": 128, "elements_per_vector": elements,
        "all_original_coordinates_remeasured": True, "original_artifacts_unchanged": True,
        "cuda_initialized": torch.cuda.is_initialized(), "autograd_replayed": False, "optimizer_trajectory_replayed": False,
        "checkpoint_selection_performed": False, "goal_success_not_assessed": True,
        "runtime_reported_model_unchanged": summary["model_state_unchanged"], "groups": groups,
        "original_artifact_sha256": hashes, "source_sha256": sha256(Path(__file__))}
    write_json(output / "audit.json", result)
    (output / "RESULTS.md").write_text(markdown(groups, summary["limitations"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Require CUDA_VISIBLE_DEVICES='' for the independent CPU audit")
    run, output = args.run.resolve(), args.output.resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Preserve the original diagnostic in a separate directory")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    try:
        while not ready(run):
            if not args.wait:
                raise RuntimeError("Diagnostic is still live")
            write_json(output / "status.json", {"state": "waiting_for_diagnostic", "pid": os.getpid(), "updated_unix": time.time()})
            time.sleep(30)
        audit(run, output)
        write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
