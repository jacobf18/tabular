"""Preserve the terminal V12 failure without restarting or weakening its audit."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
import train_single_matrixformer_family_projection as training


def read(path):
    return json.loads(Path(path).read_text())


def prefix(path, length):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if [record["step"] for record in records] != list(range(1, length + 1)):
        raise ValueError("Require the exact contiguous terminal log prefix")
    return records


def report(output):
    if output.exists():
        raise FileExistsError("Never overwrite terminal-failure evidence")
    training.verify_registration(training.PAIR)
    pair = training.PAIR
    queue = training.ROOT / "family_projection_v12_pipeline_recovery1"
    statuses = {str(directory): read(directory / "status.json")
                for directory in (pair, queue, *training.ARMS.values())}
    if any(status["state"] != "failed" for status in statuses.values()):
        raise ValueError("Require terminal failed training, arms and queue")
    if any(Path(f'/proc/{status["pid"]}').exists() for status in statuses.values()):
        raise ValueError("The recorded workers must no longer exist")
    if "Do not accept an adverse or zero family-protected update" not in statuses[str(pair)]["error"]:
        raise ValueError("Unexpected terminal failure")
    inputs = [Path(__file__).resolve(), pair / "config.json", pair / "registration.json",
              pair / "worker.log", pair / "training.jsonl", queue / "worker.log",
              *[Path(directory) / "status.json" for directory in statuses]]
    pair_rows = prefix(pair / "training.jsonl", 86)
    arms = {}
    for name, directory in training.ARMS.items():
        count = 87 if name == "control" else 86
        rows = prefix(directory / "training.jsonl", count)
        saved = torch.load(directory / "latest_checkpoint.pth", map_location="cpu", weights_only=False)
        metadata = {key: saved[key] for key in ("step", "best_step", "best_score", "gradient_updates_performed",
                                               "independent_confirmation_passed")}
        if (metadata["step"] != 80 or metadata["gradient_updates_performed"] != 80 or metadata["best_step"] != 0
                or metadata["independent_confirmation_passed"]):
            raise ValueError("Unexpected latest checkpoint or promotion")
        del saved
        if any((directory / filename).exists() for filename in
               ("selected_checkpoint.pth", "best_validation_checkpoint.pth", "raw_update_100.pt", "validation_100.json")):
            raise ValueError("Failed trial unexpectedly contains a selected or completed endpoint artifact")
        boundaries = [read(directory / f"validation_{step}.json") for step in (0, 25, 50, 75)]
        if any(boundary["best_step"] != 0 for boundary in boundaries):
            raise ValueError("Unexpected selected trained boundary")
        for record in rows:
            if record["task_manifest_sha256"] != sha256(record["task_manifest"]):
                raise ValueError("Logged task manifest changed")
            if name == "projected" and (record["adverse_tasks_after"] != 0
                    or np.any(np.asarray(record["actual_products_after"]) < 0)
                    or not np.any(np.asarray(record["actual_products_after"]) > 0)):
                raise ValueError("A completed projected update lacks its logged protection certificate")
        arms[name] = {"completed_logged_updates": count, "latest_checkpoint": metadata,
                      "logged_adverse_directions_before": sum(row["adverse_tasks_before"] for row in rows),
                      "logged_adverse_directions_after": sum(row["adverse_tasks_after"] for row in rows),
                      "scheduled_boundaries": [{"step": row["step"], "score": row["decision"]["score"],
                          "failed_guards": sum(not item["retained"] for item in row["decision"]["comparisons"])}
                                               for row in boundaries]}
        inputs.extend(directory / filename for filename in
                      ("training.jsonl", "latest_checkpoint.pth", "config.json", "raw_update_1.pt"))
        inputs.extend(directory / f"validation_{step}.json" for step in (0, 25, 50, 75))
    task_hashes = {}
    for step in range(1, 88):
        path = pair / "training_batches" / f"step_{step:03d}" / "manifest.json"
        manifest = read(path)
        if manifest["step"] != step or len(manifest["tasks"]) != 14:
            raise ValueError("Incomplete cached task manifest")
        task_hashes[str(path)] = sha256(path)
        task_hashes.update({record["artifact"]: record["sha256"] for record in manifest["tasks"]})
        if step <= 86 and pair_rows[step - 1]["manifest_sha256"] != task_hashes[str(path)]:
            raise ValueError("Paired log manifest binding changed")
    verify_hashes(task_hashes)
    prior_audit = training.ROOT / "family_projection_v12_boundary_075_audit" / "audit.json"
    inputs.append(prior_audit)
    hashes = {**task_hashes, **{str(path): sha256(path) for path in inputs}}
    result = {"complete": True, "report_unix": time.time(), "statuses": statuses, "arms": arms,
              "paired_completed_updates": 86, "cached_task_batches": 87, "cached_tasks": 1218,
              "failed_projected_update": 87, "offending_update_raw_state_saved": False,
              "exact_failure_mechanism_reproduced": False, "training_restarted": False,
              "full_100_update_audit_complete": False, "confirmation_performed": False,
              "real_benchmarks_performed": False, "checkpoint_promoted": False,
              "gradient_certificates_are_logged_not_replayed": True, "cuda_initialized": torch.cuda.is_initialized(),
              "input_sha256": hashes, "goal_achieved": False}
    output.mkdir(parents=True)
    write_json(output / "audit.json", result)
    (output / "RESULTS.md").write_text("\n".join([
        "# V12 terminal failure", "",
        "The projected arm fails its realized-parameter direction guard at update87. The trial and queue are terminal failed.",
        "Control logs87 completed updates; projected logs86; paired log ends86. Both latest model/optimizer checkpoints are update80.",
        "All1218 cached tasks from87 complete manifests retain their bound hashes. No selected/best checkpoint or raw100 exists.",
        "All evaluated trained boundaries25/50/75 fail retention. No confirmation or real benchmark runs occur.",
        "The offending update87 raw gradients, optimizer state and proposed/realized vectors were not saved.",
        "FP32 rounding is a hypothesis, not an established cause; the exact failure is not replayable from saved raw endpoints.",
        "Logged protection certificates through86 are not independent autograd replays or finite-step accuracy guarantees.",
        "The registered100-update audit remains incomplete. This additive terminal report does not replace it or restart training.", ""]))
    verify_hashes(hashes)
    print(json.dumps({key: value for key, value in result.items() if key not in ("input_sha256", "statuses", "arms")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report(parser.parse_args().output.resolve())
