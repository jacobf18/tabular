"""One CPU-controlled teacher-mean trial with preregistered full follow-ups."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.resources
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as xml

import psutil
import torch

from audit_dwip_continuous_completion import full_panel, save_new
from audit_dwip_teacher_mean_horizon import DATASETS, load_arrays
from diagnose_dwip_matched_objective_gradients import require_completed_stages
from dwip_teacher_mean_horizon import HORIZON_STEPS, PREFLIGHT, PREVIOUS_QUEUE, PROPOSAL, QUEUE, horizon_budget, horizon_recipe
from dwip_teacher_mean_training import DESIGN, base, read_json, verify_hashes
from rowcol_fixed_validation import evaluate_fixed_validation
from run_dwip_budgeted_queue_v1 import check_storage, regular_bytes
from run_dwip_continuous_spectrum import case_digest
from run_dwip_transfer_evaluation import compare_fold_metrics, evaluation_environment, gpu_processes, sha256
from run_rowcol_controlled_experiments import BENCH, REPO, WORK, write_json
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable


ROOT = REPO / "runs/objective_queues" / QUEUE
CONTROL_ROOT = ROOT.parent / PREVIOUS_QUEUE
RETENTION = REPO / "runs/information_flow_diagnostics/dwip_latent_retention_20260916"
SOURCES = (
    "scripts/run_dwip_teacher_mean_horizon.py", "scripts/audit_dwip_teacher_mean_horizon.py",
    "tests/test_dwip_teacher_mean_horizon_pipeline.py", "tests/test_dwip_teacher_mean_horizon_completion.py",
    "scripts/run_dwip_transfer_evaluation.py", "scripts/evaluate_dwip_retention_followup.py",
    "scripts/evaluate_dwip_runtime_followup.py", "scripts/benchmark_dwip_geometry_runtime.py",
    "scripts/diagnose_dwip_latent_retention.py", "scripts/audit_dwip_continuous_completion.py",
    "scripts/analyze_dwip_spectral_evidence.py",
)
TEST_COMMAND = [sys.executable, "-m", "pytest", "-q"]


def known_hashes(*mappings):
    result = {}
    for mapping in mappings:
        for name, digest in mapping.items():
            for alias in (str(Path(name).absolute()), str(Path(name).resolve(strict=True))):
                if alias in result and result[alias] != digest:
                    raise ValueError(f"Conflicting previously verified input: {alias}")
                result[alias] = digest
    return result


def environment(gpu=False):
    result = {key: value for key, value in os.environ.items() if not key.startswith("BENCHSELECT_")}
    result.update({"CUDA_VISIBLE_DEVICES": "0" if gpu else "", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1",
                   "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}", "OMP_NUM_THREADS": "4",
                   "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    return result


def test_paths():
    return [str(path.relative_to(REPO)) for path in sorted((REPO / "tests").glob("test_dwip*.py"))] + [
        "tests/test_spectral_query_experiments.py", "tests/test_general_spectral_geometry.py", "tests/test_spectral_support_normalization.py"]


def check_test_report(path):
    suites = xml.parse(path).getroot().iter("testsuite")
    totals = {name: 0 for name in ("tests", "errors", "failures", "skipped")}
    for suite in suites:
        for name in totals:
            totals[name] += int(suite.attrib.get(name, 0))
    if totals["tests"] < 500 or any(totals[name] for name in ("errors", "failures", "skipped")):
        raise ValueError("Require the complete baseline plus integration suite without failures/skips")
    return totals


def checked_link(destination, source):
    destination, source = Path(destination), Path(source).resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Preserve unexpected existing output: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def storage_guard(root, budget, launch=False):
    work = WORK / f"tabimpute-{root.name}-dwip1e5"
    external = list((BENCH / "logs").glob(f"tabimpute_{root.name}_*.npz"))
    allocated = sum(regular_bytes(path) for path in (root, work, *external))
    status = check_storage(budget, shutil.disk_usage(WORK).free, allocated, launch=launch)
    write_json(root / "storage_status.json", {**status, "updated_unix": time.time()})
    return status


def commands(root):
    return {
        "training": [sys.executable, "scripts/train_dwip_teacher_mean_horizon.py", "--registration", str(root / "training_registration.json")],
        "training_audit": [sys.executable, "scripts/audit_dwip_teacher_mean_horizon.py", "--queue", root.name, "--training-only"],
        "synthetic": [sys.executable, "scripts/run_dwip_teacher_mean_horizon.py", "synthetic"],
        "mteb": [sys.executable, "code/eval_tabimpute.py"],
        "transfer": [sys.executable, "scripts/run_dwip_transfer_evaluation.py", "--queue", root.name],
        "retention": [sys.executable, "scripts/evaluate_dwip_retention_followup.py", "--queue", root.name],
        "runtime": [sys.executable, "scripts/evaluate_dwip_runtime_followup.py", "--queue", root.name],
        "completion_audit": [sys.executable, "scripts/audit_dwip_teacher_mean_horizon.py", "--queue", root.name],
    }


def register():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "admission.json").exists() or (WORK / f"tabimpute-{QUEUE}-dwip1e5").exists():
        raise ValueError("Preserve existing admission/training; do not register a retry")
    attempts = list(ROOT.glob("registration_attempt_*"))
    attempt = ROOT / f"registration_attempt_{len(attempts) + 1}"
    attempt.mkdir()
    with (attempt / "tests.log").open("x") as log:
        subprocess.run([*TEST_COMMAND, *test_paths(), f"--junitxml={attempt / 'tests.xml'}"], cwd=REPO,
                       env=environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    tests = check_test_report(attempt / "tests.xml")
    preflight = read_json(PREFLIGHT / "results.json")
    if not preflight["core_preflight_passed"] or preflight["training_ready"] is not False:
        raise ValueError("Require the completed immutable core preflight")
    verify_hashes(preflight["input_sha256"])
    verify_hashes(read_json(PREFLIGHT / "completion_audit.json")["input_sha256"])
    proposal = read_json(PROPOSAL)
    verify_hashes(proposal["input_sha256"])
    if proposal["proposed_queue"] != QUEUE:
        raise ValueError("Only the existing single-model proposal is admitted")
    require_completed_stages(CONTROL_ROOT)
    args = horizon_recipe()
    previous_inventory = CONTROL_ROOT / "final_saved_model_inventory.json"
    inventory = read_json(previous_inventory)
    verify_hashes(inventory["input_sha256"])
    if inventory["winner"]["queue"] != PREVIOUS_QUEUE:
        raise ValueError("The fixed winner differs from the proposed control")
    reference_path = CONTROL_ROOT / "teacher_full_comparison.json"
    previous_manifest_path = CONTROL_ROOT / "manifest.json"
    reference = read_json(reference_path)
    previous_manifest = read_json(previous_manifest_path)
    independent_path = DESIGN / "prespecification.json"
    known = known_hashes(reference["input_sha256"], read_json(CONTROL_ROOT / "training_stage_audit.json")["input_sha256"],
                         read_json(independent_path)["input_sha256"], preflight["input_sha256"])
    inputs = dict(preflight["input_sha256"])

    def record(path, require_known=False):
        path = Path(path).absolute()
        expected = known.get(str(path), known.get(str(path.resolve())))
        if require_known and expected is None:
            raise ValueError(f"No previously verified hash for reused evidence: {path}")
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Previously verified input changed: {path}")
        inputs[str(path)] = actual
        return str(path)

    for path in (*[REPO / source for source in SOURCES], reference_path, previous_manifest_path, independent_path, PROPOSAL,
                 previous_inventory, PREFLIGHT / "results.json", PREFLIGHT / "completion_audit.json",
                 attempt / "tests.log", attempt / "tests.xml"):
        record(path)
    for name, expected in inventory["input_sha256"].items():
        if sha256(name) != expected:
            raise ValueError("Existing inventory changed")
        inputs[name] = expected
    initial = {"checkpoint": proposal["initial_checkpoint"], "sha256": previous_manifest["initial_checkpoint_sha256"]}
    if record(initial["checkpoint"], True) not in inputs or inputs[initial["checkpoint"]] != initial["sha256"]:
        raise ValueError("Original checkpoint identity changed")
    for member in proposal["teachers"]:
        path = WORK / f"tabimpute-{member['queue']}-dwip1e5/best_validation_checkpoint.pth"
        if sha256(path) != member["sha256"]:
            raise ValueError("Frozen teacher checkpoint changed")
        record(path, True)
    references = {"original": {}, "primary_mean64": {}}
    for dataset in DATASETS:
        references["original"][dataset] = record(previous_manifest["reference_panels"]["original"][dataset], True)
        references["primary_mean64"][dataset] = record(previous_manifest["selected_panels"][dataset], True)
        observed, _ = full_panel(load_arrays(references["original"][dataset]), load_arrays(references["primary_mean64"][dataset]))
        if observed != reference["reports"]["original"][dataset]:
            raise ValueError("Completed reference panel does not reproduce")
        record(BENCH / "data" / f"{dataset}.matrix.npz", True)
    if set(references["original"]) != set(DATASETS):
        raise ValueError("Require all four canonical benchmark matrices")
    record(BENCH / "code/eval_tabimpute.py", True)
    record(importlib.resources.files("tabimpute") / "data/borders.pt")
    retention_manifest = read_json(RETENTION / "manifest.json")
    verify_hashes(retention_manifest["source_sha256"])
    for path, expected in retention_manifest["source_sha256"].items():
        inputs[path] = expected
    for name in ("manifest.json", "cases.pth", "cases_sha256.json", "results.json"):
        record(RETENTION / name)
    case_paths = {"fixed": record(args.fixed_validation_cache, True),
                  "spectral": record(CONTROL_ROOT / "holdout.pth", True),
                  "general": record(ROOT.parent / "rowcol_output_remedies_paired_20260907/independent_validation.pth", True)}
    digests, counts = {}, {}
    for name, path in case_paths.items():
        cases = torch.load(path, map_location="cpu", weights_only=True)["cases"]
        digests[name], counts[name] = case_digest(cases), len(cases)
    if digests["fixed"] != inventory["winner"]["case_digest"] or counts != {"fixed": 72, "spectral": 72, "general": 64}:
        raise ValueError("Cached synthetic case populations differ")
    synthetic_references = {"original": {"spectral": record(CONTROL_ROOT / "baseline_spectral_predictions.pth", True),
                                          "general": record(CONTROL_ROOT / "baseline_general_predictions.pth", True)},
                            "primary_mean64": {"spectral": record(CONTROL_ROOT / "dwip1e5_selected_predictions.pth", True),
                                               "general": record(CONTROL_ROOT / "dwip1e5_general_predictions.pth", True)}}
    record(CONTROL_ROOT / "baseline/checkpoint.pth", True)
    record(CONTROL_ROOT / "baseline/config.json")
    if read_json(CONTROL_ROOT / "baseline/config.json")["model"] != json.loads(args.model_overrides_json):
        raise ValueError("Original baseline architecture differs from the student")
    primary_retention = record(CONTROL_ROOT / "retention_evaluation/results.json", True)
    original_retention = record(RETENTION / "results.json")
    checked_link(ROOT / "baseline", CONTROL_ROOT / "baseline")
    checked_link(ROOT / "validation.pth", case_paths["fixed"])
    checked_link(ROOT / "holdout.pth", case_paths["spectral"])
    for family, source in synthetic_references["original"].items():
        checked_link(ROOT / f"baseline_{family}_predictions.pth", source)
    for dataset in DATASETS[1:]:
        checked_link(ROOT / f"transfer_evaluation/logs/tabimpute_dwip_transfer_baseline_{dataset}.npz", references["original"][dataset])
    selected_panels = {dataset: str(ROOT / f"transfer_evaluation/logs/tabimpute_dwip_transfer_selected_{dataset}.npz") for dataset in DATASETS[1:]}
    selected_panels["mteb"] = str(ROOT / "mteb_evaluation/logs/tabimpute_dwip_teacher_selected_mteb.npz")
    budget = horizon_budget()
    storage = storage_guard(ROOT, budget, launch=True)
    save_new(ROOT / "artifact_budget.json", budget)
    manifest = {"queue": QUEUE, "initial_checkpoint": initial["checkpoint"], "initial_checkpoint_sha256": initial["sha256"],
                "model_config": json.loads(args.model_overrides_json), "prior": base.build_base_prior_config(args),
                "source_sha256": {path: digest for path, digest in inputs.items() if Path(path).suffix == ".py"},
                "case_paths": case_paths, "case_digests": digests, "fixed_case_digest": digests["fixed"],
                "reference_panels": references, "selected_panels": selected_panels, "synthetic_references": synthetic_references,
                "original_retention": original_retention, "primary_retention": primary_retention,
                "previous_inventory": str(previous_inventory), "stage_commands": commands(ROOT), "storage_budget": budget,
                "selection": "Unchanged fixed72-case macro RMSE only, earliest strict minimum. Public results never select checkpoints.",
                "learning_factor": "Only extend the phase endpoint64k to128k and cosine decay63400 to127400; preserve the completed mean-Huber objective, teachers, seeds, priors and architecture.",
                "runtime_allowance_note": "Frozen runtime helper prose mentions2GiB; actual registered cap is the measured allowance in storage_budget under the enclosing two-file compact guard.",
                "limits": "One trial; each owned stage and controller have24h caps. No retries, pair/weight/seed grid, old job restart or artifact deletion."}
    save_new(ROOT / "manifest.json", manifest)
    record(ROOT / "artifact_budget.json")
    record(ROOT / "manifest.json")
    verify_hashes(inputs)
    admission = {"utc": datetime.now(timezone.utc).isoformat(), "training_ready": True, "core_preflight_passed": True,
                 "integration_tests": tests, "reference_panels_verified": True, "full_pipeline_and_completion_audit_registered": True,
                 "input_sha256": inputs, "storage_at_admission": storage, "scope": "Combined admission, not a replacement of the immutable core preflight or objective completion."}
    save_new(ROOT / "admission.json", admission)
    training_registration = {"kind": "dwip_teacher_mean_horizon_v1", "queue": QUEUE, "budget": budget,
                             "input_sha256": {**inputs, str(ROOT / "admission.json"): sha256(ROOT / "admission.json")},
                             "preflight": str(ROOT / "admission.json"), "preflight_sha256": sha256(ROOT / "admission.json")}
    save_new(ROOT / "training_registration.json", training_registration)
    from train_dwip_teacher_mean_horizon import validate_registration
    validate_registration(training_registration)
    write_json(ROOT / "status.json", {"state": "registered", "updated_unix": time.time()})
    print(json.dumps({"registered": QUEUE, "training_ready": True, "tests": tests, "input_count": len(inputs),
                      "required_gib": budget["required_free_bytes_at_launch"] / 1024**3}, indent=2))


def checked_handle(process, command):
    return {"pid": process.pid, "created": psutil.Process(process.pid).create_time(), "command": command}


def original_live(handle):
    try:
        process = psutil.Process(handle["pid"])
        return process.create_time() == handle["created"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def launch():
    if (ROOT / "pipeline_launch.json").exists() or (WORK / f"tabimpute-{QUEUE}-dwip1e5").exists():
        raise ValueError("Preserve an existing launch or partial work; no restart")
    admission = read_json(ROOT / "admission.json")
    verify_hashes(admission["input_sha256"])
    storage_guard(ROOT, read_json(ROOT / "artifact_budget.json"), launch=True)
    if gpu_processes().strip():
        raise RuntimeError("GPU occupied; preserve jobs and wait without launching")
    command = ["timeout", "--signal=TERM", "--kill-after=120s", "24h", sys.executable,
               "scripts/run_dwip_teacher_mean_horizon.py", "run"]
    with (ROOT / "controller.log").open("x") as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment(), stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    handle = checked_handle(process, command)
    save_new(ROOT / "pipeline_launch.json", handle)
    print(json.dumps(handle, indent=2))


def synthetic():
    admission, manifest = read_json(ROOT / "admission.json"), read_json(ROOT / "manifest.json")
    verify_hashes(admission["input_sha256"])
    if (ROOT / "synthetic_evaluation.json").exists():
        raise ValueError("Preserve existing synthetic results")
    training = read_json(ROOT / "training_stage_audit.json")
    work = WORK / f"tabimpute-{QUEUE}-dwip1e5"
    checkpoint = work / "best_validation_checkpoint.pth"
    expected = training["checkpoint_sha256"]["best_validation_checkpoint.pth"]
    if sha256(checkpoint) != expected or gpu_processes().strip():
        raise ValueError("Require frozen selected weights and an idle GPU")
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = TabImputeModelStable(**manifest["model_config"]).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
    criterion = FullSupportBarDistribution(torch.load(importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).cuda())
    results = {}
    for family in ("spectral", "general"):
        storage_guard(ROOT, manifest["storage_budget"])
        cases = torch.load(manifest["case_paths"][family], map_location="cpu", weights_only=True)["cases"]
        if case_digest(cases) != manifest["case_digests"][family]:
            raise ValueError("Independent case contents changed")
        label = "selected" if family == "spectral" else "general"
        results[family] = evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16", ROOT / f"dwip1e5_{label}_predictions.pth")
        save_new(ROOT / f"dwip1e5_{label}_holdout.json", results[family])
    verify_hashes(admission["input_sha256"])
    if sha256(checkpoint) != expected:
        raise ValueError("Selected checkpoint changed during evaluation")
    save_new(ROOT / "synthetic_evaluation.json", {"checkpoint_sha256": expected, "scores": results})


def run():
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / "controller_identity.json").exists():
            raise ValueError("Preserve an already-started controller; never restart")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Controller must remain CPU-only")
        admission, manifest = read_json(ROOT / "admission.json"), read_json(ROOT / "manifest.json")
        verify_hashes(admission["input_sha256"])
        budget, stages = manifest["storage_budget"], manifest["stage_commands"]
        identity = {"pid": os.getpid(), "created": psutil.Process().create_time()}
        save_new(ROOT / "controller_identity.json", identity)
        (ROOT / "launcher.pid").write_text(str(identity["pid"]) + "\n")
        processes = {}

        def start(name, gpu=False, cwd=REPO, env=None):
            command = ["timeout", "--signal=TERM", "--kill-after=120s", "24h", *stages[name]]
            storage_guard(ROOT, budget)
            with (ROOT / f"{name}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=cwd, env=environment(gpu) if env is None else env,
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            handle = checked_handle(process, command)
            processes[name] = (process, handle)
            folder = ROOT / "stage_handles"
            folder.mkdir(exist_ok=True)
            save_new(folder / f"{name}.json", handle)
            return handle

        def wait(names):
            pending = set(names)
            while pending:
                storage_guard(ROOT, budget)
                for name in tuple(pending):
                    process, handle = processes[name]
                    result = process.poll()
                    if result is None:
                        if not original_live(handle):
                            raise RuntimeError(f"Original {name} identity disappeared; no restart")
                    elif result:
                        raise subprocess.CalledProcessError(result, handle["command"])
                    else:
                        pending.remove(name)
                if pending:
                    time.sleep(15)

        def state(value, **extra):
            write_json(ROOT / "pipeline_status.json", {"state": value, "updated_unix": time.time(), **extra})

        def interrupted(signum, frame):
            raise TimeoutError(f"Controller received signal{signum}; terminate only its verified owned stages")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            storage_guard(ROOT, budget, launch=True)
            write_json(ROOT / "status.json", {"state": "training", "updated_unix": time.time()})
            state("training")
            start("training", gpu=True)
            wait(["training"])
            state("auditing_terminal_training")
            start("training_audit")
            wait(["training_audit"])
            state("evaluating_selected_synthetic")
            start("synthetic", gpu=True)
            wait(["synthetic"])
            state("evaluating_selected_mteb")
            checkpoint = WORK / f"tabimpute-{QUEUE}-dwip1e5/best_validation_checkpoint.pth"
            expected_sha = read_json(ROOT / "training_stage_audit.json")["checkpoint_sha256"]["best_validation_checkpoint.pth"]
            if sha256(checkpoint) != expected_sha:
                raise ValueError("Selected weights changed before MTEB evaluation")
            mteb_output = ROOT / "mteb_evaluation"
            mteb_output.mkdir(exist_ok=True)
            env = evaluation_environment(mteb_output, checkpoint, "dwip_teacher_selected", "mteb")
            env["CUDA_VISIBLE_DEVICES"] = "0"
            start("mteb", gpu=True, cwd=BENCH, env=env)
            wait(["mteb"])
            if sha256(checkpoint) != expected_sha:
                raise ValueError("Selected weights changed during MTEB evaluation")
            paired = compare_fold_metrics(load_arrays(manifest["reference_panels"]["original"]["mteb"]),
                                          load_arrays(manifest["selected_panels"]["mteb"]))
            mteb = {f"{method}_r2_{budget_step}": paired[f"{method}_r2_{budget_step}"]["selected"]
                    for method in ("entropy", "mi", "random") for budget_step in (5, 10, 15)}
            save_new(ROOT / "dwip1e5_mteb.json", mteb)
            result = {"tag": "dwip1e5", "holdout": read_json(ROOT / "synthetic_evaluation.json")["scores"],
                      "mteb": mteb, "selection": read_json(ROOT / "training_stage_audit.json")["selected"]}
            save_new(ROOT / "dwip1e5_result.json", result)
            save_new(ROOT / "results.json", [result])
            write_json(ROOT / "status.json", {"state": "complete", "updated_unix": time.time()})
            for name in ("transfer", "retention", "runtime"):
                output = ROOT / f"{name}_evaluation"
                output.mkdir(exist_ok=True)
                write_json(output / "status.json", {"state": "queued", "updated_unix": time.time()})
            state("full_transfer_retention_runtime")
            transfer = start("transfer", gpu=True)
            save_new(ROOT / "launch.json", {"controller": identity,
                     "training_launcher_pid": processes["training"][1]["pid"], "training_launcher_created": processes["training"][1]["created"],
                     "transfer_launcher_pid": transfer["pid"], "transfer_launcher_created": transfer["created"]})
            retention = start("retention", gpu=True)
            save_new(ROOT / "retention_evaluation/launch.json", retention)
            start("runtime", gpu=True)
            wait(["transfer", "retention", "runtime"])
            state("auditing_completed_trial")
            start("completion_audit")
            wait(["completion_audit"])
            verify_hashes(admission["input_sha256"])
            state("complete", goal_status="active", audit_sha256=sha256(ROOT / "completed_stage_audit.json"))
        except BaseException as error:
            state("error", error=repr(error))
            if read_json(ROOT / "status.json")["state"] != "complete":
                write_json(ROOT / "status.json", {"state": "error", "error": repr(error), "updated_unix": time.time()})
            for process, handle in processes.values():
                if process.poll() is None and original_live(handle):
                    os.killpg(process.pid, signal.SIGTERM)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("register", "launch", "run", "synthetic"))
    args = parser.parse_args()
    {"register": register, "launch": launch, "run": run, "synthetic": synthetic}[args.mode]()


if __name__ == "__main__":
    main()
