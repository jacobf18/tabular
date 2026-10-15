"""Seed only audited unchanged-parent predictions for a future frozen campaign."""

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np

import evaluate_single_matrixformer_recsys as evaluation
from recsys_evaluation_common import sha256, write_json
from run_single_matrixformer_benchmarks import require_full_audit


def read(path):
    return json.loads(Path(path).read_text())


def validated_parent_controls(source, protocol_path):
    protocol = evaluation.load_protocol(protocol_path)
    if read(source / "status.json")["state"] != "complete":
        raise ValueError("Require a completed source recommender evaluation")
    manifest = read(source / "manifest.json")
    manifest_hash = sha256(source / "manifest.json")
    audit = read(source / "audit.json")
    require_full_audit("recsys", audit)
    if audit["manifest_sha256"] != manifest_hash:
        raise ValueError("Source audit does not bind the source manifest")
    evaluation.verify_hashes(manifest["candidate_sha256"])
    bound_protocol = Path(manifest["protocol"]) / "protocol.json"
    if (sha256(bound_protocol) != manifest["protocol_sha256"]
            or manifest["protocol_sha256"] != sha256(protocol_path / "protocol.json")
            or read(bound_protocol) != protocol):
        raise ValueError("Parent controls use a different frozen protocol")
    parent_hash = sha256(evaluation.GENERAL)
    specialist_hash = sha256(evaluation.SPECIALIST)
    controls = []
    for case in protocol["cases"]:
        directory = source / case["dataset"]
        prediction_path = directory / "parent_same_recipe.npz"
        record_path = prediction_path.with_suffix(".json")
        record = read(record_path)
        parity = read(directory / "specialist_parity.json")
        if not parity["passed"] or parity["checkpoint_sha256"] != specialist_hash:
            raise ValueError("Source lacks specialist recipe parity")
        if (record["dataset"] != case["dataset"] or record["role"] != "parent_same_recipe"
                or record["manifest_sha256"] != manifest_hash or record["checkpoint_sha256"] != parent_hash
                or record["prediction_sha256"] != sha256(prediction_path)):
            raise ValueError("Source parent checkpoint, identity or prediction bytes changed")
        split = evaluation.load_split(case["split"])
        with np.load(prediction_path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["counts"], evaluation.expected_coverage(
                case["shape"], split["test_users"], split["test_items"]))
            if (saved["prediction"].shape != split["test_ratings"].shape
                    or not np.isfinite(saved["prediction"]).all()
                    or (saved["prediction"] < case["rating_range"][0]).any()
                    or (saved["prediction"] > case["rating_range"][1]).any()):
                raise ValueError("Parent predictions violate finite frozen rating bounds")
            metrics = evaluation.audit_scores(split, saved["prediction"], record["metrics"])
        audited = [row for row in audit["cases"]
                   if row["dataset"] == case["dataset"] and row["role"] == "parent_same_recipe"]
        if (len(audited) != 1 or audited[0]["queries"] != len(split["test_ratings"])
                or any(audited[0][key] != value for key, value in metrics.items())):
            raise ValueError("Parent metrics do not match the complete source audit")
        controls.append({"dataset": case["dataset"], "prediction": prediction_path, "record": record,
                         "record_sha256": sha256(record_path), "metrics": metrics,
                         "queries": len(split["test_ratings"])})
    return protocol, controls, {"source": str(source), "manifest_sha256": manifest_hash,
                                "audit_sha256": sha256(source / "audit.json"), "parent_sha256": parent_hash}


def candidate_manifest(protocol_path, checkpoint, confirmation):
    gate = read(confirmation / "gate.json")
    if (read(confirmation / "status.json")["state"] != "complete" or not gate["passed"]
            or gate["selected_step"] <= 0 or gate["checkpoint_sha256"] != sha256(checkpoint)):
        raise ValueError("Require the frozen synthetic-confirmed candidate before cache import")
    return {"candidate_sha256": {str(checkpoint): sha256(checkpoint),
                                  str(checkpoint.parent / "config.json"): sha256(checkpoint.parent / "config.json")},
            "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path / "protocol.json"),
            "confirmation": str(confirmation), "confirmation_sha256": sha256(confirmation / "gate.json"),
            "selected_step": gate["selected_step"], "student_has_one_checkpoint": True}


@contextmanager
def locked_new_output(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / "run.lock").open("a") as campaign_lock:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        yield


def prepare(source, protocol_path, checkpoint, confirmation, output):
    source, protocol_path, checkpoint, confirmation, output = [Path(path).resolve()
                                                               for path in (source, protocol_path, checkpoint, confirmation, output)]
    if output.exists():
        raise FileExistsError("Cache import never modifies an existing evaluation directory")
    manifest = candidate_manifest(protocol_path, checkpoint, confirmation)
    _, controls, provenance = validated_parent_controls(source, protocol_path)
    with locked_new_output(output), (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(output / "manifest.json", manifest)
        target_manifest_hash = sha256(output / "manifest.json")
        copied = []
        for control in controls:
            destination = output / control["dataset"]
            destination.mkdir()
            path = destination / "parent_same_recipe.npz"
            shutil.copy2(control["prediction"], path)
            if sha256(path) != control["record"]["prediction_sha256"]:
                raise ValueError("Copied parent prediction bytes changed")
            record = {**control["record"], "manifest_sha256": target_manifest_hash, "seconds": 0.,
                      "original_inference_seconds": control["record"]["seconds"],
                      "inference_performed_in_this_campaign": False,
                      "cache_source": {**provenance, "prediction": str(control["prediction"]),
                                       "record_sha256": control["record_sha256"]}}
            write_json(path.with_suffix(".json"), record)
            copied.append({"dataset": control["dataset"], "queries": control["queries"],
                           "prediction_sha256": sha256(path), "metrics": control["metrics"]})
        report = {"complete": True, "prepared_unix": time.time(), "controls": copied, **provenance,
                  "source_script": str(Path(__file__).resolve()), "source_script_sha256": sha256(Path(__file__)),
                  "target_manifest_sha256": target_manifest_hash, "student_predictions_copied": False,
                  "specialist_parity_copied": False, "gpu_inference_performed": False,
                  "full_candidate_audit_complete": False,
                  "policy": "Reuse only unchanged general-parent predictions under byte-identical frozen protocols; all student inference, fresh specialist parity and complete all-role audits still run"}
        write_json(output / "parent_cache_import.json", report)
        shutil.copy2(Path(__file__), output / Path(__file__).name)
        write_json(output / "status.json", {"state": "parent_controls_prepared_not_evaluated", "controls": len(controls)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = prepare(arguments.source, arguments.protocol, arguments.checkpoint, arguments.confirmation, arguments.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
