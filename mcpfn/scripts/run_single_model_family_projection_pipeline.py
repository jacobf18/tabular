"""Audit both paired arms, independently confirm selections, then serialize frozen campaigns."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

import audit_single_model_family_projection_training as training_audit
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import train_single_matrixformer_family_projection as training
from train_streamed_large_matrices import wait_for_gpu


ROOT = training.ROOT
OUTPUT = ROOT / "family_projection_v12_pipeline"
AUDIT = ROOT / "family_projection_v12_audit"
ORDER = ("projected", "control")
CONFIRMATIONS = {name: ROOT / f"family_projection_v12_{name}_confirmation" for name in ORDER}
CAMPAIGNS = {name: ROOT / f"family_projection_v12_{name}_benchmarks" for name in ORDER}
REVIEWS = {name: ROOT / f"family_projection_v12_{name}_review" for name in ORDER}
PARENT_CACHE = ROOT / "query_budget_v8_benchmarks_defined1/recsys"


def status(output, state, **fields):
    write_json(output / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def verify(output):
    config = training.read(output / "config.json")
    if training.read(output / "registration.json")["config_sha256"] != sha256(output / "config.json"):
        raise ValueError("Pipeline registration changed")
    if config["qualification"] != training.QUALIFICATION or config["campaign_order"] != list(ORDER):
        raise ValueError("Never change selection, confirmation, or campaign order")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    training.verify_registration(training.PAIR)
    return config


def register(output):
    training.verify_registration(training.PAIR)
    destinations = [output, AUDIT, *CONFIRMATIONS.values(), *CAMPAIGNS.values(), *REVIEWS.values()]
    if output != OUTPUT or any(path.exists() for path in destinations):
        raise FileExistsError("Require the entire fresh paired qualification pipeline")
    sources = [Path(__file__).resolve(), Path(training_audit.__file__).resolve(),
               REPO / "scripts/run_single_matrixformer_cached_benchmarks.py", REPO / "scripts/run_single_matrixformer_benchmarks.py",
               REPO / "scripts/review_single_matrixformer_campaign.py"]
    inputs = [training.PAIR / "config.json", training.PAIR / "registration.json", PARENT_CACHE / "manifest.json", PARENT_CACHE / "audit.json"]
    inputs += [ROOT / f"{suite}_protocol_v1/protocol.json" for suite in ("tabular", "recsys", "llm")]
    inputs += [arm / "config.json" for arm in training.ARMS.values()]
    output.mkdir(parents=True)
    write_json(output / "config.json", {"pair": str(training.PAIR), "qualification": training.QUALIFICATION,
        "campaign_order": list(ORDER), "source_sha256": {str(path): sha256(path) for path in sources},
        "input_sha256": {str(path): sha256(path) for path in inputs}, "checkpoint_reselection_performed": False,
        "audit": str(AUDIT), "confirmations": {name: str(path) for name, path in CONFIRMATIONS.items()},
        "campaigns": {name: str(path) for name, path in CAMPAIGNS.items()}})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
        "confirmation_evaluated": False, "real_benchmarks_evaluated": False})
    for name, directory in CONFIRMATIONS.items():
        directory.mkdir()
        write_json(directory / "config.json", {"arm": name, "run": str(training.ARMS[name]),
            "run_config_sha256": sha256(training.ARMS[name] / "config.json"),
            "pipeline_config_sha256": sha256(output / "config.json"),
            "cases": training.read(training.ARMS[name] / "config.json")["confirmation_cases"],
            "qualification": training.QUALIFICATION})
        write_json(directory / "registration.json", {"config_sha256": sha256(directory / "config.json"), "student_evaluated": False})
        status(directory, "registered")
    status(output, "registered")


def confirmation_registration(output, name):
    config = verify(output)
    directory = CONFIRMATIONS[name]
    registration = training.read(directory / "config.json")
    expected = {"arm": name, "run": str(training.ARMS[name]), "run_config_sha256": sha256(training.ARMS[name] / "config.json"),
        "pipeline_config_sha256": sha256(output / "config.json"), "cases": training.read(training.ARMS[name] / "config.json")["confirmation_cases"],
        "qualification": training.QUALIFICATION}
    if registration != expected or training.read(directory / "registration.json")["config_sha256"] != sha256(directory / "config.json"):
        raise ValueError("Fresh independent confirmation registration changed")
    report = training.read(AUDIT / "audit.json")
    if not report["complete"] or report["optimizer_updates_per_arm"] != 100 or report["research_selection_tolerance"] != 0:
        raise ValueError("Require completed paired-training audit before confirmation")
    verify_hashes(report["artifact_sha256"])
    return config, registration, report["arms"][name]["best"]


@torch.no_grad()
def confirm(output, name):
    _, registration, best = confirmation_registration(output, name)
    directory, arm = CONFIRMATIONS[name], training.ARMS[name]
    if training.read(directory / "status.json")["state"] != "registered":
        raise ValueError("Never repeat or reselect after a started confirmation")
    if best["step"] == 0:
        write_json(directory / "gate.json", {"passed": False, "selected_step": 0, "confirmation_evaluated": False,
            "reason": "No trained boundary passed every original retention guard and strict minimax improvement",
            "real_benchmark_success": False, "training_audit_sha256": sha256(AUDIT / "audit.json")})
        status(directory, "complete", confirmation_evaluated=False)
        return
    wait_for_gpu(directory, True, False)
    status(directory, "confirming", selected_step=best["step"])
    base = training.read(training.PAIR / "config.json")["base"]
    path = arm / "selected_checkpoint.pth"
    fingerprint = sha256(path)
    selected = torch.load(path, map_location="cpu", weights_only=True)
    if selected["step"] != best["step"] or selected["config_sha256"] != registration["run_config_sha256"]:
        raise ValueError("Only the audited selection may consume confirmation")
    write_json(directory / "manifest.json", {"checkpoint": str(path), "checkpoint_sha256": fingerprint,
        "selected_step": best["step"], "cases": registration["cases"], "training_audit_sha256": sha256(AUDIT / "audit.json")})
    with training.mixture.runtime(base, directory):
        student, optimizer, criterion, teachers, sampler = training.mixture.backend.setup(base)
        del optimizer
        bank, reference = training.mixture.backend.prepare_bank(sampler, teachers, criterion, registration["cases"], directory)
        initial_metrics, initial_cases = training.mixture.backend.evaluate_bank(student, criterion, bank)
        student.load_state_dict(selected["model"], strict=True)
        student_metrics, student_cases = training.mixture.backend.evaluate_bank(student, criterion, bank)
    if len(bank) != 284 or len(initial_cases) != 284 or len(student_cases) != 284:
        raise ValueError("Require all reserved independent confirmation cases")
    write_json(directory / "initialization.json", {"student": initial_metrics, "cases": initial_cases})
    write_json(directory / "student.json", {"student": student_metrics, "cases": student_cases})
    decision = training.mixture.confirmation_gate(student_metrics, initial_metrics, reference["general"], reference["references"])
    write_json(directory / "decision.json", decision)
    verify_hashes({str(path): fingerprint})
    status(directory, "awaiting_independent_audit", selected_step=best["step"])


def audit_confirmation(output, name):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Independent confirmation audit must be CPU-only")
    _, registration, best = confirmation_registration(output, name)
    directory = CONFIRMATIONS[name]
    if training.read(directory / "status.json")["state"] != "awaiting_independent_audit" or (directory / "audit.json").exists():
        raise ValueError("Require a fresh completed independent confirmation to audit")
    paths = [directory / filename for filename in ("config.json", "manifest.json", "synthetic_bank.pt", "reference.json", "initialization.json", "student.json", "decision.json")]
    hashes = {str(path): sha256(path) for path in paths}
    reference, initial, student = [training.read(directory / filename) for filename in ("reference.json", "initialization.json", "student.json")]
    bank = torch.load(directory / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    if len(bank) != len(registration["cases"]) or len(bank) != 284:
        raise ValueError("Require every reserved fresh case")
    verify_hashes({str(directory / "synthetic_bank.pt"): reference["bank_sha256"]})
    for cached, specification in zip(bank, registration["cases"]):
        if cached["metadata"]["seed"] != specification.get("structured", specification)["seed"]:
            raise ValueError("Fresh case order or identity changed")
    for report, roles in ((reference, ("general", "teacher")), (initial, ("student",)), (student, ("student",))):
        if len(report["cases"]) != 284:
            raise ValueError("Never drop confirmation cases")
        for role in roles:
            if training.mixture.aggregate(report["cases"], role) != report[role]:
                raise ValueError("Fresh metric aggregation changed")
    if training.fixed.reference_values(reference["general"], reference["teacher"]) != reference["references"]:
        raise ValueError("Fresh reference floors changed")
    count, prediction_hashes = training.fixed.raw_audit(bank, [(reference, ("general", "teacher")), (initial, ("student",)), (student, ("student",))])
    if count != 1136:
        raise ValueError("Require all fresh full prediction grids")
    hashes.update(prediction_hashes)
    decision = training.mixture.confirmation_gate(student["student"], initial["student"], reference["general"], reference["references"])
    if decision != training.read(directory / "decision.json"):
        raise ValueError("Unchanged confirmation gate did not reproduce")
    manifest = training.read(directory / "manifest.json")
    if manifest["selected_step"] != best["step"]:
        raise ValueError("Confirmation evaluated a different selected boundary")
    hashes[manifest["checkpoint"]] = manifest["checkpoint_sha256"]
    verify_hashes(hashes)
    verify(output)
    write_json(directory / "audit.json", {"complete": True, "prediction_grids": count, "cases": 284,
        "research_selection_tolerance": 0, "artifact_sha256": hashes, "cuda_initialized": torch.cuda.is_initialized(), "goal_success_not_assessed": True})
    gate = {**decision, "selected_step": best["step"], "checkpoint_sha256": manifest["checkpoint_sha256"],
        "confirmation_evaluated": True, "cases": 284, "legacy_cases": 92, "structured_cases": 192,
        "bank_sha256": reference["bank_sha256"], "audit_sha256": sha256(directory / "audit.json"),
        "training_audit_sha256": sha256(AUDIT / "audit.json")}
    if gate["passed"]:
        alias = training.ARMS[name] / "best_validation_checkpoint.pth"
        if alias.exists():
            raise FileExistsError("Never replace an already published alias")
        shutil.copy2(manifest["checkpoint"], alias)
        verify_hashes({str(alias): manifest["checkpoint_sha256"]})
    write_json(directory / "gate.json", gate)
    status(directory, "complete", passed=gate["passed"])


def run_child(output, label, script, arguments, *, cpu=False):
    command = [sys.executable, "-u", str(script), *arguments]
    environment = {**os.environ, "PYTHONPATH": f"{REPO / 'scripts'}:{REPO / 'src'}", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    with (output / f"{label}.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, cwd=REPO.parent)
        status(output, label, child_pid=process.pid, command=command)
        if process.wait():
            raise RuntimeError(f"{label} failed; preserve its log and artifacts")


def run(output):
    verify(output)
    while True:
        current = training.read(training.PAIR / "status.json")
        if current["state"] == "training_complete":
            break
        if current["state"] in {"failed", "interrupted"}:
            raise RuntimeError(f"Paired training stopped: {current}")
        os.kill(current["pid"], 0)
        status(output, "waiting_for_paired_training", training_pid=current["pid"], completed_step=current.get("step", 0))
        time.sleep(30)
    run_child(output, "training_audit", Path(training_audit.__file__).resolve(), ["--pair", str(training.PAIR), "--output", str(AUDIT)], cpu=True)
    for name in ORDER:
        run_child(output, f"{name}_confirmation", Path(__file__).resolve(), ["--stage", "confirm", "--output", str(output), "--arm", name])
        if training.read(CONFIRMATIONS[name] / "status.json")["state"] == "awaiting_independent_audit":
            run_child(output, f"{name}_confirmation_audit", Path(__file__).resolve(), ["--stage", "audit-confirmation", "--output", str(output), "--arm", name], cpu=True)
    gates = {name: training.read(directory / "gate.json") for name, directory in CONFIRMATIONS.items()}
    for name in ORDER:
        if not gates[name]["passed"]:
            continue
        run_child(output, f"{name}_benchmarks", REPO / "scripts/run_single_matrixformer_cached_benchmarks.py",
            ["--run", str(training.ARMS[name]), "--confirmation", str(CONFIRMATIONS[name]), "--source", str(PARENT_CACHE), "--output", str(CAMPAIGNS[name])])
        run_child(output, f"{name}_review", REPO / "scripts/review_single_matrixformer_campaign.py",
            ["--campaign", str(CAMPAIGNS[name]), "--output", str(REVIEWS[name])], cpu=True)
    verify(output)
    write_json(output / "result.json", {"complete": True, "gates": gates, "campaigns": {name: str(CAMPAIGNS[name]) for name in ORDER if gates[name]["passed"]},
        "training_audit_sha256": sha256(AUDIT / "audit.json"), "goal_success_not_assessed": True, "no_inference_ensemble": True})
    lines = ["# Paired family-protected training", "", training.POLICY, "", "Both arms use one ordinary inference-time network. All original selection and confirmation guards remain unchanged.",
        "", "| Arm | Selected update | Fresh confirmation run | Synthetic gate passed |", "|---|---:|:---:|:---:|"]
    for name in ORDER:
        lines.append(f'| {name} | {gates[name]["selected_step"]} | {gates[name]["confirmation_evaluated"]} | {gates[name]["passed"]} |')
    lines.extend(["", "Synthetic eligibility is not blend parity. Read the full frozen benchmark review for each qualifying arm before making performance claims.", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    status(output, "complete", confirmed_arms=[name for name in ORDER if gates[name]["passed"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "confirm", "audit-confirmation"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--arm", choices=ORDER)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    if args.stage in {"confirm", "audit-confirmation"}:
        if args.arm is None:
            parser.error("Confirmation requires --arm")
        with (CONFIRMATIONS[args.arm] / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                (confirm if args.stage == "confirm" else audit_confirmation)(output, args.arm)
            except BaseException as error:
                status(CONFIRMATIONS[args.arm], "failed", error=repr(error))
                raise
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if training.read(output / "status.json")["state"] != "registered":
            raise ValueError("Never silently restart a started pipeline")
        try:
            run(output)
        except BaseException as error:
            status(output, "failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
