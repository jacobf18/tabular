"""A prospectively bounded synthetic-only search over two trained weight directions."""

import argparse
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from confirm_single_model_weight_average import inventory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_fixed_contraction as fixed
import single_model_structured_mixture as mixture
from streamed_large_matrix_training import atomic_torch_save
from train_single_matrixformer_column_aligned import repeated_bank
from train_streamed_large_matrices import wait_for_gpu


ROOT = fixed.ROOT
PARENT = fixed.PARENT
PRODUCER = fixed.PRODUCER
TAILS = (fixed.TAIL, ROOT / "family_rehearsal_v11/latest_checkpoint.pth")
GRID = [{"id": f"candidate_{index:02d}", "alpha": alpha, "beta": beta}
        for index, (alpha, beta) in enumerate((alpha, beta) for alpha in (.10, .25, .40)
                                              for beta in (0., .025, .05, .10))]
SEEDS = {"legacy": 111100701, "structured": 112100701}
SEMANTICS = "Derived V8 plus alpha times unselected V10 displacement plus beta times unselected V11 displacement; step400 denotes source updates, not new training or promotion of either tail"
POLICY = "Evaluate all12 in registered order on the immutable152-case V10 bank; select lowest16-ratio minimax only among all22-guard-retained strict improvements; exact ties use registered order; confirm only that one winner on fresh284 cases, without fallback"


def read(path):
    return json.loads(Path(path).read_text())


def confirmation_cases():
    return repeated_bank(SEEDS["legacy"], confirmation=True) + mixture.structured_bank(SEEDS["structured"], confirmation=True)


def validate_spec(specification):
    if specification not in GRID:
        raise ValueError("Candidate is outside the registered grid")


def combine(parent, first, second, specification):
    validate_spec(specification)
    inventory(first, parent)
    inventory(second, parent)
    inventory(parent, parent)
    alpha, beta = specification["alpha"], specification["beta"]
    result = {}
    for name, value in parent.items():
        if value.is_floating_point():
            origin = value.double()
            result[name] = (origin + alpha * (first[name].double() - origin)
                            + beta * (second[name].double() - origin)).to(value.dtype)
        elif torch.equal(value, first[name]) and torch.equal(value, second[name]):
            result[name] = value.clone()
        else:
            raise ValueError(f"Nonfloating state changed: {name}")
    inventory(result, parent)
    return result


def check_arithmetic(parent, first, second, candidate, specification):
    validate_spec(specification)
    for state in (parent, first, second):
        inventory(state, parent)
    details = inventory(candidate, parent)
    alpha, beta = specification["alpha"], specification["beta"]
    different, maximum = 0, 0.
    for name, value in candidate.items():
        if value.is_floating_point():
            expected = ((1 - alpha - beta) * parent[name].double() + alpha * first[name].double()
                        + beta * second[name].double()).to(value.dtype)
            lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
            upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
            if not torch.isfinite(expected).all() or not ((value >= lower) & (value <= upper)).all():
                raise ValueError(f"Independent weighted-state reconstruction differs: {name}")
            different += int((value != expected).sum())
            maximum = max(maximum, float((value.double() - expected.double()).abs().max()))
        elif any(not torch.equal(value, state[name]) for state in (parent, first, second)):
            raise ValueError("Nonfloating candidate state changed")
    return {**details, "arithmetic_independently_reconstructed": True, "allowed_output_ulps": 1,
            "non_bitwise_elements": different, "maximum_absolute_difference": maximum}


def metadata(specification, config_hash):
    return {"step": 400, "epoch": 400, "config_sha256": config_hash, "derived_checkpoint": True,
            "gradient_updates_performed_here": 0, "candidate": specification, "step_semantics": SEMANTICS}


def load_candidate(run, specification):
    path = run / specification["id"] / "checkpoint.pth"
    build = read(run / "build.json")
    if not build["complete"] or list(build["candidates"]) != [candidate["id"] for candidate in GRID]:
        raise ValueError("Require the entire built grid")
    receipt = build["candidates"][specification["id"]]
    verify_hashes({str(path): receipt["checkpoint_sha256"]})
    state = torch.load(path, map_location="cpu", weights_only=True)
    if any(state.get(key) != value for key, value in metadata(specification, sha256(run / "config.json")).items()):
        raise ValueError("Candidate metadata changed")
    return state


def verify_registration(run):
    config = read(run / "config.json")
    if read(run / "registration.json")["config_sha256"] != sha256(run / "config.json"):
        raise ValueError("Grid registration changed")
    if (config["grid"] != GRID or config["parent"] != str(PARENT) or config["tails"] != list(map(str, TAILS))
            or config["producer"] != str(PRODUCER) or config["selection"] != POLICY
            or config["confirmation_cases"] != json.loads(json.dumps(confirmation_cases()))
            or config["step_semantics"] != SEMANTICS or config["confirmation_seeds"] != SEEDS):
        raise ValueError("Never change registered candidates, ordering, selection or confirmation")
    fixed.validate_recipe(config["base"])
    if config["base"] != read(PRODUCER / "config.json") or config["model"] != config["base"]["model"]:
        raise ValueError("Original model or evaluation recipe changed")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def register(run):
    if run.exists():
        raise FileExistsError("Never replace a registered experiment")
    base = read(PRODUCER / "config.json")
    fixed.validate_recipe(base)
    inputs = {**base["input_sha256"]}
    for tail, audit_name in zip(TAILS, ("domain_clipping_v10_audit_400", "family_rehearsal_v11_audit_400")):
        producer = tail.parent
        recipe = read(producer / "config.json")
        mixture.backend.verify_run(recipe, producer)
        status = read(producer / "status.json")
        if status["state"] != "training_complete" or status["step"] != 400 or status["best_step"] != 0:
            raise ValueError("Require completed, unselected tails")
        audit_path = ROOT / audit_name / "audit.json"
        audit = read(audit_path)
        if not audit["complete"] or not audit["original_artifacts_unchanged"] or audit["through_step"] != 400:
            raise ValueError("Require full completed source-run audits")
        inputs.update(audit["original_artifact_sha256"])
        for path in (tail, producer / "config.json", producer / "status.json", audit_path):
            inputs[str(path)] = sha256(path)
    known = ROOT / "fixed_contraction_v1_qualification"
    if read(known / "status.json")["state"] != "complete" or read(known / "gate.json")["passed"]:
        raise ValueError("Known quarter-step control must be terminal and rejected")
    for path in (PARENT, known / "gate.json", known / "selection/audit.json", known / "selection/decision.json"):
        inputs[str(path)] = sha256(path)
    old_configs = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    prior_seeds = set().union(*(fixed.seeds_in(read(path)) for path in old_configs))
    cases = confirmation_cases()
    new_seeds = [case.get("structured", case)["seed"] for case in cases]
    if len(cases) != 284 or len(set(new_seeds)) != 284 or prior_seeds.intersection(new_seeds):
        raise ValueError("Reserve unused confirmation seeds before evaluation")
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"grid": GRID, "parent": str(PARENT), "tails": list(map(str, TAILS)), "producer": str(PRODUCER),
              "base": base, "model": base["model"], "confirmation_cases": cases, "confirmation_seeds": SEEDS,
              "selection": POLICY, "step_semantics": SEMANTICS, "gradient_updates_performed_here": 0,
              "known_control": "alpha=.25,beta=0 was already rejected; rerun on saved V10 bank, no blinded-search claim",
              "inference": "One original network per candidate; no output averaging or runtime router",
              "input_sha256": inputs, "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "confirmation_seed_inventory": {str(path): sha256(path) for path in old_configs}}
    verify_hashes(inputs)
    verify_hashes(config["source_sha256"])
    run.mkdir(parents=True)
    write_json(run / "config.json", config)
    write_json(run / "registration.json", {"config_sha256": sha256(run / "config.json"), "registered_unix": time.time(),
               "candidates_built": False, "candidates_evaluated": False, "known_control_previously_evaluated": True,
               "confirmation_cases_disjoint_from_recorded_experiments": True})
    shutil.copy2(Path(__file__), run / Path(__file__).name)
    write_json(run / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def build(run):
    verify_registration(run)
    if read(run / "status.json")["state"] != "registered":
        raise ValueError("Never rebuild a started grid")
    write_json(run / "status.json", {"state": "building", "pid": os.getpid(), "updated_unix": time.time()})
    parent = torch.load(PARENT, map_location="cpu", weights_only=True)
    tails = [torch.load(path, map_location="cpu", weights_only=True) for path in TAILS]
    if parent["step"] != 250 or any(tail["step"] != 400 or tail["best_step"] != 0
            or tail["config_sha256"] != sha256(path.parent / "config.json") for path, tail in zip(TAILS, tails)):
        raise ValueError("Source provenance changed")
    receipts = {}
    for specification in GRID:
        directory = run / specification["id"]
        directory.mkdir()
        candidate = combine(parent["model"], tails[0]["model"], tails[1]["model"], specification)
        arithmetic = check_arithmetic(parent["model"], tails[0]["model"], tails[1]["model"], candidate, specification)
        if not arithmetic["changed_tensors"]:
            raise ValueError("Do not present unchanged initialization as a candidate")
        path = directory / "checkpoint.pth"
        atomic_torch_save({"model": candidate, **metadata(specification, sha256(run / "config.json"))}, path)
        receipts[specification["id"]] = {"checkpoint_sha256": sha256(path), **arithmetic}
    verify_registration(run)
    write_json(run / "build.json", {"complete": True, "candidates": receipts, "single_parameter_set_per_candidate": True,
                                   "cuda_initialized": torch.cuda.is_initialized()})
    write_json(run / "status.json", {"state": "built_not_qualified", "pid": os.getpid(), "updated_unix": time.time()})


def choose(decisions):
    if list(decisions) != [specification["id"] for specification in GRID]:
        raise ValueError("Must evaluate the complete grid in registered order")
    eligible = [specification for specification in GRID if decisions[specification["id"]]["passed"]]
    selected = min(eligible, key=lambda specification: decisions[specification["id"]]["score"]) if eligible else None
    return {"selected": selected, "eligible_candidates": [specification["id"] for specification in eligible],
            "candidates_evaluated": len(GRID), "passed": selected is not None, "policy": POLICY}


def audited_metrics(bank, report, roles):
    if len(report["cases"]) != len(bank):
        raise ValueError("Prediction report must contain exactly the registered cases")
    count, hashes = fixed.raw_audit(bank, [(report, roles)])
    for role in roles:
        if mixture.aggregate(report["cases"], role) != report[role]:
            raise ValueError("Original aggregation differs from reported metrics")
    return count, hashes


def audit(run, output, phase):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Independent audit must be CPU-only")
    config = verify_registration(run)
    directory = output / phase
    if (directory / "audit.json").exists():
        raise FileExistsError("Never overwrite a completed audit")
    development = phase == "selection"
    bank_directory = PRODUCER if development else directory
    bank_path = bank_directory / "synthetic_bank.pt"
    reference_path = bank_directory / "reference.json"
    initial_path = bank_directory / ("validation_0.json" if development else "initialization.json")
    reference, initial = read(reference_path), read(initial_path)
    verify_hashes({str(bank_path): reference["bank_sha256"]})
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    expected = config["base"]["validation_cases"] if development else config["confirmation_cases"]
    if len(bank) != len(expected) or len(bank) != (152 if development else 284):
        raise ValueError("Never omit a registered case")
    for cached, specification in zip(bank, expected):
        if cached["metadata"]["seed"] != specification.get("structured", specification)["seed"]:
            raise ValueError("Bank does not bind registered seeds")
    paths = (bank_path, reference_path, initial_path, run / "build.json", directory / "decision.json")
    artifacts = {str(path): sha256(path) for path in paths}
    count = 0
    for report, roles in ((reference, ("general", "teacher")), (initial, ("student",))):
        rescored, hashes = audited_metrics(bank, report, roles)
        count += rescored
        artifacts.update(hashes)
    if fixed.reference_values(reference["general"], reference["teacher"]) != reference["references"]:
        raise ValueError("Original reference floors changed")
    parent = torch.load(PARENT, map_location="cpu", weights_only=True)["model"]
    tails = [torch.load(path, map_location="cpu", weights_only=True)["model"] for path in TAILS]
    specifications = GRID if development else [read(output / "selection/decision.json")["selected"]]
    decisions, arithmetic = {}, {}
    for specification in specifications:
        candidate = load_candidate(run, specification)
        identifier = specification["id"]
        arithmetic[identifier] = check_arithmetic(parent, tails[0], tails[1], candidate["model"], specification)
        candidate_directory = directory / identifier if development else directory
        student_path = candidate_directory / "student.json"
        student = read(student_path)
        rescored, hashes = audited_metrics(bank, student, ("student",))
        count += rescored
        artifacts.update(hashes)
        decision = (fixed.selection_decision(student["student"], initial["student"], reference["references"], config["base"])
                    if development else mixture.confirmation_gate(student["student"], initial["student"], reference["general"], reference["references"]))
        if decision != read(candidate_directory / "decision.json"):
            raise ValueError("Strict candidate decision does not reproduce")
        decisions[identifier] = decision
        for path in (student_path, candidate_directory / "decision.json", run / identifier / "checkpoint.pth"):
            artifacts[str(path)] = sha256(path)
    if development and choose(decisions) != read(directory / "decision.json"):
        raise ValueError("Full-grid selection does not reproduce")
    verify_registration(run)
    verify_hashes(artifacts)
    write_json(directory / "audit.json", {"complete": True, "phase": phase, "cases": len(bank),
        "prediction_grids_rescored": count, "candidates_audited": len(specifications), "arithmetic": arithmetic,
        "decision_sha256": sha256(directory / "decision.json"), "artifact_sha256": artifacts,
        "research_selection_tolerance": 0, "cuda_initialized": torch.cuda.is_initialized(), "goal_success_not_assessed": True})


def audit_subprocess(run, output, phase):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "audit", "--run", str(run),
               "--output", str(output), "--phase", phase]
    with (output / f"{phase}_audit.log").open("x") as stream:
        subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=stream, stderr=subprocess.STDOUT, check=True)


def require_audit(output, phase):
    record = read(output / phase / "audit.json")
    cases, candidates = (152, len(GRID)) if phase == "selection" else (284, 1)
    if (not record["complete"] or record["cuda_initialized"] or record["cases"] != cases
            or record["candidates_audited"] != candidates or record["prediction_grids_rescored"] != cases * (3 + candidates)
            or record["decision_sha256"] != sha256(output / phase / "decision.json")):
        raise ValueError("Require a complete, CPU-only, unchanged phase audit")
    verify_hashes(record["artifact_sha256"])


def report(output, gate):
    selection = read(output / "selection/decision.json")
    lines = ["# Single-network displacement grid", "", SEMANTICS + ".", "", POLICY + ".", "",
        "Synthetic development selection only; both tail models were rejected. The quarter-step is a known control, not a blinded test.",
        "Each candidate is a single original network. No new gradient updates, output averaging or inference-time router.", "",
        "| Candidate | V10 alpha | V11 beta | Minimax | Failed guards /22 | Eligible |",
        "|---|---:|---:|---:|---:|:---:|"]
    for specification in GRID:
        decision = read(output / "selection" / specification["id"] / "decision.json")
        failures = sum(not row["retained"] for row in decision["comparisons"])
        lines.append(f'| {specification["id"]} | {specification["alpha"]} | {specification["beta"]} | {decision["score"]:.10f} | {failures} | {decision["passed"]} |')
    lines += ["", f'Selected: {selection["selected"]}.', f'Fresh confirmation evaluated: {gate["confirmation_evaluated"]}.',
              f'Final synthetic gate passed: {gate["passed"]}.', "",
              "All candidates, prediction grids, arithmetic and strict gates are independently CPU-audited before promotion.",
              "Real-benchmark success is not assessed by these synthetic results. No failed guard is waived.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))


def start_qualification(run, output):
    if read(run / "status.json")["state"] != "built_not_qualified":
        raise ValueError("Qualify a built grid exactly once")
    receipt = {"run": str(run), "output": str(output), "config_sha256": sha256(run / "config.json"),
               "registered_unix": time.time(), "pid": os.getpid(), "candidates_evaluated_here": False}
    with (run / "qualification.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    write_json(output / "registration.json", receipt)
    write_json(run / "status.json", {"state": "qualifying", "pid": os.getpid(), "updated_unix": time.time()})


@torch.no_grad()
def qualify(run, output):
    config = verify_registration(run)
    base = deepcopy(config["base"])
    selection = output / "selection"
    selection.mkdir()
    bank = torch.load(PRODUCER / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    initial, reference = read(PRODUCER / "validation_0.json"), read(PRODUCER / "reference.json")
    decisions = {}
    with mixture.runtime(base, selection):
        student, optimizer, criterion, teachers, sampler = mixture.backend.setup(base)
        del optimizer
        for specification in GRID:
            write_json(output / "status.json", {"state": "evaluating_grid", "candidate": specification,
                       "completed": len(decisions), "total": len(GRID), "pid": os.getpid(), "updated_unix": time.time()})
            candidate = load_candidate(run, specification)
            student.load_state_dict(candidate["model"], strict=True)
            metrics, cases = mixture.backend.evaluate_bank(student, criterion, bank)
            directory = selection / specification["id"]
            directory.mkdir()
            decision = fixed.selection_decision(metrics, initial["student"], reference["references"], base)
            write_json(directory / "student.json", {"student": metrics, "cases": cases})
            write_json(directory / "decision.json", decision)
            decisions[specification["id"]] = decision
            print(json.dumps({"candidate": specification, "score": decision["score"], "passed": decision["passed"],
                              "failed_guards": sum(not row["retained"] for row in decision["comparisons"])}), flush=True)
    chosen = choose(decisions)
    write_json(selection / "decision.json", chosen)
    audit_subprocess(run, output, "selection")
    require_audit(output, "selection")
    del student, criterion, teachers, sampler, bank, candidate
    gc.collect()
    torch.cuda.empty_cache()
    selected = chosen["selected"]
    decision = {"passed": False, "reason": "No candidate passes every original development guard and strict minimax improvement"}
    if selected is not None:
        confirmation = output / "confirmation"
        confirmation.mkdir()
        write_json(output / "status.json", {"state": "confirming_selected_candidate", "candidate": selected,
                   "pid": os.getpid(), "updated_unix": time.time()})
        with mixture.runtime(base, confirmation):
            student, optimizer, criterion, teachers, sampler = mixture.backend.setup(base)
            del optimizer
            bank, reference = mixture.backend.prepare_bank(sampler, teachers, criterion, config["confirmation_cases"], confirmation)
            initial_metrics, initial_cases = mixture.backend.evaluate_bank(student, criterion, bank)
            candidate = load_candidate(run, selected)
            student.load_state_dict(candidate["model"], strict=True)
            metrics, cases = mixture.backend.evaluate_bank(student, criterion, bank)
        write_json(confirmation / "initialization.json", {"student": initial_metrics, "cases": initial_cases})
        write_json(confirmation / "student.json", {"student": metrics, "cases": cases})
        decision = mixture.confirmation_gate(metrics, initial_metrics, reference["general"], reference["references"])
        write_json(confirmation / "decision.json", decision)
        audit_subprocess(run, output, "confirmation")
        require_audit(output, "confirmation")
    fingerprint = None if selected is None else read(run / "build.json")["candidates"][selected["id"]]["checkpoint_sha256"]
    gate = {**decision, "checkpoint_sha256": fingerprint, "selected_step": 400 if selected else 0, "selected_candidate": selected,
            "derived_checkpoint": True, "step_semantics": SEMANTICS, "gradient_updates_performed_here": 0,
            "real_benchmark_success": False, "confirmation_evaluated": selected is not None, "single_parameter_set": True}
    verify_registration(run)
    require_audit(output, "selection")
    if gate["passed"]:
        load_candidate(run, selected)
        (run / "best_validation_checkpoint.pth").symlink_to(Path(selected["id"]) / "checkpoint.pth")
    write_json(output / "gate.json", gate)
    report(output, gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "build", "qualify", "audit"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("selection", "confirmation"))
    args = parser.parse_args()
    run = args.run.resolve()
    if args.stage == "register":
        register(run)
        return
    if args.stage == "build":
        with (run / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                build(run)
            except BaseException as error:
                write_json(run / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
                raise
        return
    if args.output is None or (args.stage == "audit" and args.phase is None):
        parser.error("Qualification/audit needs --output; audit also needs --phase")
    output = args.output.resolve()
    if args.stage == "audit":
        audit(run, output, args.phase)
        return
    if run.is_relative_to(output) or output.is_relative_to(run):
        raise ValueError("Keep evaluation separate from immutable candidates")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock, (run / "qualification.lock").open("a") as run_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify_registration(run)
            start_qualification(run, output)
            write_json(output / "status.json", {"state": "waiting_for_gpu", "pid": os.getpid(), "updated_unix": time.time()})
            wait_for_gpu(output, True, False)
            qualify(run, output)
            write_json(run / "status.json", {"state": "qualified" if read(output / "gate.json")["passed"] else "rejected",
                       "pid": os.getpid(), "updated_unix": time.time()})
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
