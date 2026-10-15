"""A fixed quarter-step in V10 weight space, with full unchanged synthetic guards."""

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

import numpy as np
import torch

from audit_single_matrixformer_structured_training import close_values
from confirm_single_model_retained_average import prerequisite_ready
from confirm_single_model_weight_average import inventory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from report_single_model_structured_run import retention_values
import single_model_structured_mixture as mixture
from streamed_large_matrix_training import atomic_torch_save
from train_single_matrixformer_column_aligned import repeated_bank
from train_single_matrixformer_domain_clipping import validate_recipe
from train_streamed_large_matrices import wait_for_gpu


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
PRODUCER = ROOT / "domain_clipping_v10"
PARENT = ROOT / "query_budget_v8/best_validation_checkpoint.pth"
TAIL = PRODUCER / "latest_checkpoint.pth"
AFTER = ROOT / "family_rehearsal_v11_benchmarks"
ALPHA = .25
SEEDS = {"legacy": 101100601, "structured": 102100601}
SEMANTICS = "Derived quarter-step parameter contraction from selected V8 toward unselected V10 update400; step400 denotes the source update, not new training or promotion of that tail"


def read(path):
    return json.loads(Path(path).read_text())


def interpolate(parent, tail, alpha=ALPHA):
    if alpha != ALPHA or tuple(parent) != tuple(tail):
        raise ValueError("Only the fixed quarter-step and identical single-network inventories are allowed")
    result = {}
    for name, value in parent.items():
        other = tail[name]
        if value.shape != other.shape or value.dtype != other.dtype or not torch.isfinite(value).all() or not torch.isfinite(other).all():
            raise ValueError(f"Invalid parameter pair: {name}")
        if value.is_floating_point():
            result[name] = ((1 - alpha) * value.double() + alpha * other.double()).to(value.dtype)
        elif torch.equal(value, other):
            result[name] = value.clone()
        else:
            raise ValueError(f"Nonfloating state must remain identical: {name}")
    return result


def check_arithmetic(parent, tail, candidate):
    details = inventory(candidate, parent)
    different, maximum = 0, 0.
    for name, value in candidate.items():
        if value.is_floating_point():
            expected = (parent[name].double() + ALPHA * (tail[name].double() - parent[name].double())).to(value.dtype)
            lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
            upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
            if not torch.isfinite(expected).all() or not ((value >= lower) & (value <= upper)).all():
                raise ValueError(f"Independent parameter contraction does not reproduce: {name}")
            different += int((value != expected).sum())
            maximum = max(maximum, float((value.double() - expected.double()).abs().max()))
        elif not torch.equal(value, parent[name]) or not torch.equal(value, tail[name]):
            raise ValueError("Nonfloating candidate state changed")
    return {**details, "arithmetic_independently_reconstructed": True, "allowed_output_ulps": 1,
            "non_bitwise_elements": different, "maximum_absolute_difference": maximum}


def check_candidate_metadata(state, config_hash):
    expected = {"step": 400, "epoch": 400, "config_sha256": config_hash, "derived_checkpoint": True,
                "gradient_updates_performed_here": 0, "alpha": ALPHA, "step_semantics": SEMANTICS}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Derived candidate provenance or source-step semantics changed")


def confirmation_cases():
    return repeated_bank(SEEDS["legacy"], confirmation=True) + mixture.structured_bank(SEEDS["structured"], confirmation=True)


def seeds_in(value):
    result = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if "seed" in key and type(nested) is int:
                result.add(nested)
            result.update(seeds_in(nested))
    elif isinstance(value, list):
        for nested in value:
            result.update(seeds_in(nested))
    return result


def verify_registration(run):
    config = read(run / "config.json")
    if read(run / "registration.json")["config_sha256"] != sha256(run / "config.json"):
        raise ValueError("Contraction registration changed")
    if (config["alpha"] != ALPHA or config["parent"] != str(PARENT) or config["tail"] != str(TAIL)
            or config["producer"] != str(PRODUCER) or config["after"] != str(AFTER)
            or config["confirmation_cases"] != json.loads(json.dumps(confirmation_cases()))
            or config["step_semantics"] != SEMANTICS):
        raise ValueError("Never change the fixed candidate, bank or execution order")
    validate_recipe(config["base"])
    if config["base"] != read(PRODUCER / "config.json"):
        raise ValueError("The original training recipe changed")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    return config


def register(run):
    if run.exists():
        raise FileExistsError("Never replace an experiment")
    base = read(PRODUCER / "config.json")
    validate_recipe(base)
    mixture.backend.verify_run(base, PRODUCER)
    status = read(PRODUCER / "status.json")
    if status["state"] != "training_complete" or status["step"] != 400 or status["best_step"] != 0:
        raise ValueError("Require the terminal rejected V10, not an active tail")
    audit_path = ROOT / "domain_clipping_v10_audit_400/audit.json"
    audit = read(audit_path)
    if not audit["complete"] or not audit["original_artifacts_unchanged"] or audit["through_step"] != 400:
        raise ValueError("Require the complete raw-prediction audit")
    verify_hashes(audit["original_artifact_sha256"])
    old_configs = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    prior_seeds = set().union(*(seeds_in(read(path)) for path in old_configs))
    cases = confirmation_cases()
    new_seeds = [case["structured"]["seed"] if "structured" in case else case["seed"] for case in cases]
    if len(cases) != 284 or len(set(new_seeds)) != 284 or prior_seeds.intersection(new_seeds):
        raise ValueError("Reserve disjoint fresh confirmation seeds before any candidate evaluation")
    inputs = {**base["input_sha256"], **audit["original_artifact_sha256"]}
    paths = [PARENT, TAIL, audit_path, PRODUCER / "status.json", ROOT / "domain_clipping_v10_family_diagnosis/columns.csv",
             ROOT / "domain_clipping_v10_family_diagnosis/summary.json", AFTER / "cache_launcher_config.json"]
    inputs.update({str(path): sha256(path) for path in paths})
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"alpha": ALPHA, "parent": str(PARENT), "tail": str(TAIL), "producer": str(PRODUCER), "after": str(AFTER),
        "base": base, "model": base["model"], "confirmation_cases": cases, "confirmation_seeds": SEEDS,
        "input_sha256": inputs, "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
        "step_semantics": SEMANTICS, "selection": "One fixed candidate; every original22 retention guard and strict16-ratio minimax improvement on the original152 development cases",
        "confirmation": "Fresh92 legacy and192 structured cases; unchanged original confirmation gate, including all initialization guards and original general-model aggregate guards",
        "motivation": "Exploratory saved-output displacement diagnosis, not a claim that parameter interpolation equals prediction averaging",
        "inference": "Exactly one ordinary network/checkpoint; no output blending or router", "gradient_updates_performed_here": 0,
        "confirmation_seed_inventory": {str(path): sha256(path) for path in old_configs}}
    verify_hashes(inputs)
    verify_hashes(config["source_sha256"])
    run.mkdir(parents=True)
    write_json(run / "config.json", config)
    write_json(run / "registration.json", {"config_sha256": sha256(run / "config.json"), "registered_unix": time.time(),
        "candidate_built": False, "candidate_evaluated": False, "coefficient_search_performed": False,
        "confirmation_cases_disjoint_from_recorded_experiments": True, "unselected_tail_promoted": False})
    shutil.copy2(Path(__file__), run / Path(__file__).name)
    write_json(run / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


def build(run):
    config = verify_registration(run)
    if read(run / "status.json")["state"] != "registered":
        raise ValueError("Never rebuild a started candidate")
    write_json(run / "status.json", {"state": "building", "pid": os.getpid(), "updated_unix": time.time()})
    parent = torch.load(PARENT, map_location="cpu", weights_only=True)
    tail = torch.load(TAIL, map_location="cpu", weights_only=True)
    if parent["step"] != 250 or tail["step"] != 400 or tail["best_step"] != 0 or tail["config_sha256"] != sha256(PRODUCER / "config.json"):
        raise ValueError("Source checkpoint provenance changed")
    candidate = interpolate(parent["model"], tail["model"])
    arithmetic = check_arithmetic(parent["model"], tail["model"], candidate)
    if not arithmetic["changed_tensors"]:
        raise ValueError("Do not present unchanged initialization as a new candidate")
    checkpoint = run / "contracted_checkpoint.pth"
    if checkpoint.exists():
        raise FileExistsError("Never overwrite a candidate")
    atomic_torch_save({"model": candidate, "step": 400, "epoch": 400, "config_sha256": sha256(run / "config.json"),
        "derived_checkpoint": True, "gradient_updates_performed_here": 0, "alpha": ALPHA, "step_semantics": SEMANTICS}, checkpoint)
    verify_registration(run)
    write_json(run / "build.json", {"complete": True, "checkpoint_sha256": sha256(checkpoint), **arithmetic,
        "input_sha256": {str(PARENT): config["input_sha256"][str(PARENT)], str(TAIL): config["input_sha256"][str(TAIL)]},
        "single_parameter_set": True, "step_semantics": SEMANTICS, "cuda_initialized": torch.cuda.is_initialized()})
    write_json(run / "status.json", {"state": "built_not_qualified", "pid": os.getpid(), "updated_unix": time.time()})


def raw_audit(bank, fields):
    count = 0
    hashes = {}
    for index, cached in enumerate(bank):
        episode = mixture.backend.restore_episode(cached, "cpu")
        for report, roles in fields:
            record = report["cases"][index]
            if record["index"] != index or record["seed"] != episode.metadata["seed"]:
                raise ValueError("Prediction identity changed")
            path = Path(record["prediction_file"])
            hashes[str(path)] = record["prediction_sha256"]
            verify_hashes({str(path): record["prediction_sha256"]})
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(roles):
                    raise ValueError("Prediction roles changed")
                for role in roles:
                    prediction = torch.from_numpy(saved[role])
                    if prediction.shape != episode.target.shape or not torch.isfinite(prediction).all():
                        raise ValueError("Require the entire finite prediction grid")
                    close_values(mixture.defined.episode_metrics(prediction, episode), record[role])
                    count += 1
        del episode
    verify_hashes(hashes)
    return count, hashes


def reference_values(general, teacher):
    return {"imputation": {name: min(general["imputation"][name], teacher["imputation"][name]) for name in mixture.STRUCTURED_METRICS},
        "recsys": teacher["recsys"], "structured": {
            "overall": {name: min(general["structured"]["overall"][name], teacher["structured"]["overall"][name]) for name in mixture.STRUCTURED_METRICS},
            "families": {family: {"common_nrmse": min(general["structured"]["families"][family]["common_nrmse"],
                teacher["structured"]["families"][family]["common_nrmse"])} for family in mixture.FAMILIES}}}


def selection_decision(student, baseline, reference, base):
    score, ratios, retained = mixture.selection_score(student, baseline, reference)
    initial_score, _, _ = mixture.selection_score(baseline, baseline, reference)
    initial = {(group, metric): value for group, metric, value in retention_values(baseline, base)}
    comparisons = [{"group": group, "metric": metric, "initialization": initial[group, metric], "candidate": value,
                    "retained": value <= initial[group, metric]} for group, metric, value in retention_values(student, base)]
    return {"passed": retained and score < initial_score, "score": score, "initialization_score": initial_score,
            "ratios": ratios, "retained": retained, "comparisons": comparisons, "research_selection_tolerance": 0}


def audit(run, output, phase):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Independent audit must be CPU-only")
    config = verify_registration(run)
    checkpoint = run / "contracted_checkpoint.pth"
    verify_hashes({str(checkpoint): read(run / "build.json")["checkpoint_sha256"]})
    candidate = torch.load(checkpoint, map_location="cpu", weights_only=True)
    check_candidate_metadata(candidate, sha256(run / "config.json"))
    parent = torch.load(PARENT, map_location="cpu", weights_only=True)["model"]
    tail = torch.load(TAIL, map_location="cpu", weights_only=True)["model"]
    arithmetic = check_arithmetic(parent, tail, candidate["model"])
    del parent, tail, candidate
    directory = output / phase
    if (directory / "audit.json").exists():
        raise FileExistsError("Never overwrite a completed audit")
    if phase == "selection":
        bank_path = PRODUCER / "synthetic_bank.pt"
        reference_path, initial_path = PRODUCER / "reference.json", PRODUCER / "validation_0.json"
    else:
        bank_path = directory / "synthetic_bank.pt"
        reference_path, initial_path = directory / "reference.json", directory / "initialization.json"
    reference, initial = read(reference_path), read(initial_path)
    artifacts = {str(path): sha256(path) for path in (reference_path, initial_path, bank_path, checkpoint,
                 directory / "student.json", directory / "decision.json")}
    verify_hashes({str(bank_path): reference["bank_sha256"]})
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    student = read(directory / "student.json")
    expected = config["base"]["validation_cases"] if phase == "selection" else config["confirmation_cases"]
    if len(bank) != len(expected) or len(bank) != (152 if phase == "selection" else 284):
        raise ValueError("Never omit a development or confirmation case")
    for cached, specification in zip(bank, expected):
        seed = specification["structured"]["seed"] if "structured" in specification else specification["seed"]
        if cached["metadata"]["seed"] != seed:
            raise ValueError("Bank does not bind the registered seeds")
    count, prediction_hashes = raw_audit(bank, [(reference, ("general", "teacher")), (initial, ("student",)), (student, ("student",))])
    artifacts.update(prediction_hashes)
    for role in ("general", "teacher"):
        if mixture.aggregate(reference["cases"], role) != reference[role]:
            raise ValueError("Frozen reference aggregation changed")
    if reference_values(reference["general"], reference["teacher"]) != reference["references"]:
        raise ValueError("Original reference floors changed")
    student_metrics = mixture.aggregate(student["cases"], "student")
    initial_metrics = mixture.aggregate(initial["cases"], "student")
    if student_metrics != student["student"] or initial_metrics != initial["student"]:
        raise ValueError("Original aggregation differs from reported metrics")
    if phase == "selection":
        decision = selection_decision(student_metrics, initial_metrics, reference["references"], config["base"])
    else:
        decision = mixture.confirmation_gate(student_metrics, initial_metrics, reference["general"], reference["references"])
    if decision != read(directory / "decision.json"):
        raise ValueError("Strict decision does not reproduce")
    verify_registration(run)
    verify_hashes(artifacts)
    write_json(directory / "audit.json", {"complete": True, "phase": phase, "cases": len(bank), "prediction_grids_rescored": count,
        "decision_sha256": sha256(directory / "decision.json"), "bank_sha256": sha256(bank_path), "checkpoint_sha256": sha256(checkpoint),
        **arithmetic, "artifact_sha256": artifacts, "research_selection_tolerance": 0,
        "cuda_initialized": torch.cuda.is_initialized(), "goal_success_not_assessed": True})


def audit_subprocess(run, output, phase):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "audit", "--run", str(run), "--output", str(output), "--phase", phase]
    with (output / f"{phase}_audit.log").open("x") as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def report_qualification(output, gate, base):
    selection = read(output / "selection/decision.json")
    lines = ["# Fixed single-checkpoint contraction", "", SEMANTICS + ".", "",
             "One original network: 75% V8 parameters plus25% unselected V10 parameters, computed once in weight space.",
             "No coefficient search, new gradient updates, inference-time routing or prediction averaging.", "",
             f'Original152-case development gate passed: **{selection["passed"]}**.',
             f'Fresh284-case confirmation evaluated: **{gate["confirmation_evaluated"]}**.',
             f'Final synthetic gate passed: **{gate["passed"]}**. Real-benchmark success is not assessed.', ""]
    for phase in ("selection", "confirmation"):
        if phase == "confirmation" and not gate["confirmation_evaluated"]:
            continue
        if phase == "selection":
            comparisons = selection["comparisons"]
        else:
            baseline = read(output / phase / "initialization.json")["student"]
            candidate = read(output / phase / "student.json")["student"]
            initial = {(group, metric): value for group, metric, value in retention_values(baseline, base)}
            comparisons = [{"group": group, "metric": metric, "initialization": initial[group, metric],
                            "candidate": value, "retained": value <= initial[group, metric]}
                           for group, metric, value in retention_values(candidate, base)]
        lines += [f"## {phase}: all22 initialization retention guards", "",
                  "| Group | Metric | Initial | Candidate | Retained |", "|---|---|---:|---:|:---:|"]
        for row in comparisons:
            lines.append(f'| {row["group"]} | {row["metric"]} | {row["initialization"]:.10f} | {row["candidate"]:.10f} | {row["retained"]} |')
        decision = read(output / phase / "decision.json")
        lines += ["", f'Extended minimax: {decision["score"]:.10f}; initialization: {decision["initialization_score"]:.10f}.', ""]
    if gate["confirmation_evaluated"]:
        lines += [f'Fresh-bank legacy general-control retention: {gate["legacy_general_retained"]}.',
                  f'Fresh-bank structured general-control retention: {gate["structured_general_retained"]}.', ""]
    lines += ["Independent CPU audits rescore every saved full prediction grid and reconstruct the parameter arithmetic.",
              "Synthetic eligibility does not establish parity on the frozen real benchmarks; no failed guard is waived.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))


@torch.no_grad()
def qualify(run, output):
    config = verify_registration(run)
    checkpoint = run / "contracted_checkpoint.pth"
    fingerprint = read(run / "build.json")["checkpoint_sha256"]
    verify_hashes({str(checkpoint): fingerprint})
    candidate = torch.load(checkpoint, map_location="cpu", weights_only=True)
    check_candidate_metadata(candidate, sha256(run / "config.json"))
    base = deepcopy(config["base"])
    selection = output / "selection"
    selection.mkdir()
    bank = torch.load(PRODUCER / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    initial, reference = read(PRODUCER / "validation_0.json"), read(PRODUCER / "reference.json")
    with mixture.runtime(base, selection):
        student, optimizer, criterion, teachers, sampler = mixture.backend.setup(base)
        del optimizer
        student.load_state_dict(candidate["model"], strict=True)
        metrics, cases = mixture.backend.evaluate_bank(student, criterion, bank)
    decision = selection_decision(metrics, initial["student"], reference["references"], base)
    write_json(selection / "student.json", {"student": metrics, "cases": cases})
    write_json(selection / "decision.json", decision)
    audit_subprocess(run, output, "selection")
    if decision["passed"]:
        confirmation = output / "confirmation"
        confirmation.mkdir()
        del student, criterion, teachers, sampler, bank
        gc.collect()
        torch.cuda.empty_cache()
        with mixture.runtime(base, confirmation):
            student, optimizer, criterion, teachers, sampler = mixture.backend.setup(base)
            del optimizer
            bank, reference = mixture.backend.prepare_bank(sampler, teachers, criterion, config["confirmation_cases"], confirmation)
            initial_metrics, initial_cases = mixture.backend.evaluate_bank(student, criterion, bank)
            student.load_state_dict(candidate["model"], strict=True)
            metrics, cases = mixture.backend.evaluate_bank(student, criterion, bank)
        write_json(confirmation / "initialization.json", {"student": initial_metrics, "cases": initial_cases})
        write_json(confirmation / "student.json", {"student": metrics, "cases": cases})
        decision = mixture.confirmation_gate(metrics, initial_metrics, reference["general"], reference["references"])
        write_json(confirmation / "decision.json", decision)
        audit_subprocess(run, output, "confirmation")
    gate = {**decision, "checkpoint_sha256": fingerprint, "selected_step": 400, "derived_checkpoint": True,
            "step_semantics": SEMANTICS, "gradient_updates_performed_here": 0, "real_benchmark_success": False,
            "confirmation_evaluated": (output / "confirmation/decision.json").exists(), "single_parameter_set": True}
    verify_registration(run)
    verify_hashes({str(checkpoint): fingerprint})
    if gate["passed"]:
        (run / "best_validation_checkpoint.pth").symlink_to(checkpoint.name)
    write_json(output / "gate.json", gate)
    report_qualification(output, gate, base)
    print(json.dumps({key: gate[key] for key in ("passed", "selected_step", "confirmation_evaluated", "checkpoint_sha256", "step_semantics")}), flush=True)


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
        parser.error("Qualification/audit requires an output; audit also requires a phase")
    output = args.output.resolve()
    if args.stage == "audit":
        audit(run, output, args.phase)
        return
    if run.is_relative_to(output) or output.is_relative_to(run):
        raise ValueError("Keep qualification separate from immutable candidate artifacts")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            config = verify_registration(run)
            write_json(output / "registration.json", {"run": str(run), "config_sha256": sha256(run / "config.json"),
                "registered_unix": time.time(), "candidate_evaluated": False})
            while not prerequisite_ready(Path(config["after"]), campaign=True):
                write_json(output / "status.json", {"state": "waiting_for_v11_campaign", "pid": os.getpid(), "updated_unix": time.time()})
                time.sleep(30)
            write_json(output / "status.json", {"state": "waiting_for_gpu", "pid": os.getpid(), "updated_unix": time.time()})
            wait_for_gpu(output, True, False)
            write_json(output / "status.json", {"state": "qualifying", "pid": os.getpid(), "updated_unix": time.time()})
            qualify(run, output)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
