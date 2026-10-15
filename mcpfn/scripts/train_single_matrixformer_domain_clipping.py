"""Paired V9 optimizer sibling, admitted only by the complete synthetic gradient probe."""

import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import time

import diagnose_single_model_missingness_coverage as coverage
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_domain_clipping as clipping
import train_single_matrixformer_structured_mixture as previous
from train_streamed_large_matrices import wait_for_gpu


ROOT = previous.ROOT
CONTROL = ROOT / "structured_mixture_v9"
EVIDENCE = ROOT / "domain_gradient_probe_v1_audit"
SOURCES = ("single_model_domain_clipping.py", "train_single_matrixformer_domain_clipping.py",
           "confirm_single_matrixformer_domain_clipping.py", "audit_single_matrixformer_domain_clipping.py")
ALLOWED_DIFFERENCES = {"version", "source_sha256", "input_sha256", "experiment"}
EXTRA_FIELDS = {"domain_clipping", "gradient_admission", "gradient_evidence", "paired_control"}


def validate_pair(config, base):
    if set(config) != set(base) | EXTRA_FIELDS or config.get("version") != 10:
        raise ValueError("Require the exact paired optimizer experiment fields")
    for name in set(base) - ALLOWED_DIFFERENCES:
        if config[name] != base[name]:
            raise ValueError(f"Original paired training or evaluation recipe changed: {name}")
    for name in ("source_sha256", "input_sha256"):
        if any(config[name].get(path) != digest for path, digest in base[name].items()):
            raise ValueError("Preserve every original source and input fingerprint")
    if (config["domain_clipping"] != clipping.SPECIFICATION or config["gradient_admission"] != clipping.ADMISSION
            or config["gradient_evidence"] != str(EVIDENCE) or config["paired_control"] != str(CONTROL)):
        raise ValueError("Domain clipping or its fixed exploratory admission evidence changed")


def validate_recipe(config):
    base = previous.read(CONTROL / "config.json")
    previous.validate_recipe(base)
    validate_pair(config, base)
    original_form = deepcopy(config)
    original_form["version"] = 9
    previous.validate_recipe(original_form)


def make_config(base):
    config = deepcopy(base)
    config.update(version=10, domain_clipping=deepcopy(clipping.SPECIFICATION), gradient_admission=deepcopy(clipping.ADMISSION),
        gradient_evidence=str(EVIDENCE), paired_control=str(CONTROL),
        experiment="Paired V9 optimizer sibling: same V8 initialization, every training/selection seed, prior, scalar loss, teacher, context, learning rate and gate; clip isolated domain gradients before their equal-weight sum")
    return config


def register(output):
    if output.exists():
        raise FileExistsError("Never overwrite a registered optimizer experiment")
    base = previous.read(CONTROL / "config.json")
    previous.validate_recipe(base)
    previous.backend.verify_run(base, CONTROL)
    status = previous.read(CONTROL / "status.json")
    if status["state"] != "training_complete" or status["best_step"] != 0:
        raise ValueError("Require the completed rejected V9 control")
    evidence = previous.read(EVIDENCE / "audit.json")
    if not clipping.admitted(evidence):
        raise ValueError("Complete synthetic gradient evidence does not admit this optimizer hypothesis")
    verify_hashes(evidence["original_artifact_sha256"])
    old_confirmation = ROOT / "structured_mixture_v9_confirmation"
    gate = previous.read(old_confirmation / "gate.json")
    if gate["passed"] or gate["selected_step"] != 0 or any((old_confirmation / name).exists() for name in
            ("synthetic_bank.pt", "reference.json", "initialization.json", "student.json")):
        raise ValueError("The paired independent confirmation reservation must remain unevaluated")
    config = make_config(base)
    artifacts = [CONTROL / "config.json", CONTROL / "status.json", old_confirmation / "gate.json",
                 EVIDENCE / "audit.json", ROOT / "domain_gradient_probe_v1/config.json", ROOT / "domain_gradient_probe_v1/summary.json"]
    config["input_sha256"].update({str(path): sha256(path) for path in artifacts})
    config["source_sha256"].update(coverage.source_hashes())
    config["source_sha256"].update({str(REPO / "scripts" / name): sha256(REPO / "scripts" / name) for name in SOURCES})
    validate_recipe(config)
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    previous.storage_guard(output)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
        "training_started": False, "selected_from_synthetic_evidence_only": True,
        "paired_selection_bank_is_development_data": True, "independent_confirmation_not_previously_evaluated": True,
        "admission_is_exploratory_not_blind": True, "real_benchmark_protocols_unchanged": True})
    for name in SOURCES:
        shutil.copy2(REPO / "scripts" / name, output / name)
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if previous.read(output / "status.json")["state"] != "registered":
            raise ValueError("Never automatically restart a started experiment")
        try:
            write_json(output / "status.json", {"state": "starting_training", "pid": os.getpid(), "updated_unix": time.time()})
            config = previous.read(output / "config.json")
            validate_recipe(config)
            previous.backend.verify_run(config, output)
            wait_for_gpu(output, True, False)
            with previous.mixture.runtime(config, output), clipping.runtime(previous.backend):
                write_json(output / "status.json", {"state": "preflighting", "pid": os.getpid(), "updated_unix": time.time()})
                previous.preflight(config, output)
                previous.backend.train(config, output, resume=False)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
