"""Audit unchanged top-block rate evidence with correct logged-versus-remeasured identities."""

import ast
import hashlib
import os
from pathlib import Path

import torch

from dwip_teacher_mean_training import verify_hashes
import probe_single_model_top_block_rates as probe
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


SOURCE = probe.OUTPUT
OUTPUT = SOURCE.parent / "top_block_rate_probe_v1_audit_recovery1"
LOG = SOURCE.parent / "top_block_rate_probe_chain.log"
FAILURE = "Complete gradient/metric coverage and the registered margin are mandatory"
POLICY = ("Preserve the failed original CPU audit, complete four-rate neural panel, registered sources and all "
    "optimizer evidence. Correct only exact logged-record identity versus independently remeasured CPU reduction "
    "rounding. Bind exact records to exact logged scores; independently rescore all280 full grids under the existing "
    "scalar comparator. Preserve exact retention/improvement signs for every paired AND individual-draw metric, "
    "both-repeat decisions, all84-gradient/28-composite endpoint checks, one-ULP projection allowance, unchanged "
    "CUDA Adam audit, original rejected candidate identity and storage hashes. This is a separately declared CPU-only "
    "audit recovery, not a rerun of neural evaluation, a relaxed gate, a retry of training, a new optimizer step, "
    "validation, confirmation, checkpoint nomination, benchmark or inference ensemble.")


def compare_repeated_decisions(actual, expected):
    probe.compare_nested(actual, expected)

    def check_signs(observed, reference):
        if isinstance(reference, dict):
            for name, value in reference.items():
                if name == "changes":
                    for metric, change in value.items():
                        current = observed[name][metric]
                        if change is not None and ((current < 0) != (change < 0) or (current <= 0) != (change <= 0)):
                            raise ValueError("Preserve exact paired and individual metric improvement/retention signs")
                else:
                    check_signs(observed[name], value)
        elif isinstance(reference, (list, tuple)):
            for current, value in zip(observed, reference):
                check_signs(current, value)

    check_signs(actual, expected)


def adapted_audit_source():
    nodes = [node for node in ast.parse(probe.ADAPTED_SOURCE).body if isinstance(node, ast.FunctionDef) and node.name == "audit"]
    if len(nodes) != 1:
        raise ValueError("Require the single exact registered audit implementation")
    text = ast.get_source_segment(probe.ADAPTED_SOURCE, nodes[0])
    for before, after in {
        'def audit(output):': 'def recovered_audit(output, recovery):',
        'read(output / "status.json")["state"] != "awaiting_independent_audit"':
            'read(output / "status.json")["state"] != "failed"',
        'paired.check_record(record, [task["weight"] for task in config["batch"]["tasks"]], before["0"], 84)':
            'paired.check_record(record, [task["weight"] for task in config["batch"]["tasks"]], result["before"]["0"], 84)',
        'if repeated != row["repeated"] or row["finite_feasible"] != bool(record["accepted"] and repeated and repeated["finite_feasible"]):':
            'compare_repeated_decisions(repeated, row["repeated"])\n'
            '        if row["finite_feasible"] != bool(record["accepted"] and repeated and repeated["finite_feasible"]):',
        'write_json(output / "audit.json",': 'write_json(recovery / "audit.json",',
        '(output / "RESULTS.md").write_text': '(recovery / "RESULTS.md").write_text',
        'status(output, "complete")': 'status(recovery, "complete")',
    }.items():
        text = replace_once(text, before, after)
    return text


def recover():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Recovery requires a separate CPU-only process")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Preserve every completed or failed recovery attempt")
    probe.verify(SOURCE)
    failed = probe.read(SOURCE / "status.json")
    result = probe.read(SOURCE / "result.json")
    if (failed["state"] != "failed" or FAILURE not in failed.get("error", "") or FAILURE not in LOG.read_text()
            or Path(f'/proc/{failed["pid"]}').exists() or result["complete"] is not True
            or [row["rate"] for row in result["rates"]] != list(probe.RATES) or (SOURCE / "audit.json").exists()):
        raise ValueError("Recover only the stopped exact-log versus CPU-reduction audit failure")
    inputs = {str(SOURCE / name): sha256(SOURCE / name) for name in
              ("config.json", "registration.json", "status.json", "result.json")}
    inputs[str(LOG)] = sha256(LOG)
    for row in result["rates"]:
        inputs[row["endpoint"]] = row["endpoint_sha256"]
        path = SOURCE / f'rate_{row["index"]}' / "result.json"
        inputs[str(path)] = sha256(path)
    text = adapted_audit_source()
    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_top_block_rate_audit_recovery.py"]
    config = {"policy": POLICY, "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sources},
              "adapted_audit_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "original_registration_sha256": sha256(SOURCE / "registration.json"),
              "neural_evaluation_performed": False, "research_selection_tolerance": 0}
    verify_hashes(inputs)
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    probe.status(OUTPUT, "auditing_preserved_evidence")
    try:
        namespace = {**probe.__dict__, "compare_repeated_decisions": compare_repeated_decisions}
        exec(compile(text, __file__, "exec"), namespace)
        namespace["recovered_audit"](SOURCE, OUTPUT)
        audit = probe.read(OUTPUT / "audit.json")
        if not audit["complete"] or audit["prediction_grids"] != 280 or audit["training_tasks"] != 28:
            raise ValueError("Require all four endpoints and all280 complete grids")
        verify_hashes(inputs)
        verify_hashes(config["source_sha256"])
        write_json(OUTPUT / "recovery.json", {"complete": True, "original_artifacts_unchanged": True,
                   "neural_evaluation_repeated": False, "prediction_grids": 280, "research_selection_tolerance": 0,
                   "config_sha256": sha256(OUTPUT / "config.json"), "audit_sha256": sha256(OUTPUT / "audit.json"),
                   "goal_success_not_assessed": True})
        probe.status(OUTPUT, "complete", original_failed_status_preserved=True)
    except BaseException as error:
        probe.status(OUTPUT, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    recover()
