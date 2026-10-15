"""Audit V15, replay its decoder endpoint, confirm once, then conditionally run frozen campaigns."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import audit_single_model_targeted_decoder as training_audit
import single_model_decoder_training as decoder
from recover_single_model_decoder_finite_audit import compare_nested
from audit_single_matrixformer_structured_training import close_values
from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_json
from single_model_point_proposal_grid import direction_products, displacement_decision
import train_single_matrixformer_targeted_decoder as training
from train_streamed_large_matrices import wait_for_gpu


ROOT = training.ROOT
OUTPUT = ROOT / "targeted_decoder_v15_pipeline"
AUDIT = ROOT / "targeted_decoder_v15_audit"
REPLAY = ROOT / "targeted_decoder_v15_endpoint_replay"
CONFIRMATION = ROOT / "targeted_decoder_v15_confirmation"
CAMPAIGN = ROOT / "targeted_decoder_v15_benchmarks"
REVIEW = ROOT / "targeted_decoder_v15_review"
PARENT_CACHE = ROOT / "query_budget_v8_benchmarks_defined1/recsys"
POLICY = ("Wait for all one hundred registered attempts; independently audit every cached training task, scheduled raw "
          "validation and selection, and the final decoder-only raw optimizer/projection endpoint and frozen V14 backbone. Reconstruct the last candidate "
          "with the unchanged GPU kernel and replay all defined point scores before/after on all fourteen tasks; "
          "independently CPU-rescore those grids. Reevaluate any selected checkpoint on all152 development cases "
          "without reselection, then consume exactly its reserved284 fresh confirmation cases under the unchanged "
          "gate. Publish the best alias only after the independent1136-grid confirmation audit passes. Only then "
          "run every frozen real suite and the complete per-comparison review. No fallback, guard relaxation, "
          "training restart, or inference ensemble. Retention and confirmation remain anchored to V8, never the V14 training input.")


def read(path):
    return json.loads(Path(path).read_text())


def status(directory, state, **fields):
    write_json(directory / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def verify_reservation(current, pipeline_digest, run_digest, confirmation):
    if (current["pipeline_config_sha256"] != pipeline_digest or current["run_config_sha256"] != run_digest
            or current["policy"] != POLICY
            or current["confirmation_cases"] != (training.confirmation_cases() if confirmation else None)):
        raise ValueError("Endpoint/confirmation reservation changed")


def verify(output):
    config = read(output / "config.json")
    if (read(output / "registration.json")["config_sha256"] != sha256(output / "config.json")
            or config["policy"] != POLICY or config["qualification"] != training.QUALIFICATION
            or config["confirmation_cases"] != training.confirmation_cases()):
        raise ValueError("Never change the registered qualification or confirmation policy")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    training.verify_registration(training.OUTPUT)
    for directory in (REPLAY, CONFIRMATION):
        current = read(directory / "config.json")
        verify_reservation(current, sha256(output / "config.json"), sha256(training.OUTPUT / "config.json"), directory == CONFIRMATION)
        if read(directory / "registration.json")["config_sha256"] != sha256(directory / "config.json"):
            raise ValueError("Endpoint/confirmation reservation changed")
    return config


def register(output):
    if output != OUTPUT or any(path.exists() for path in (output, AUDIT, REPLAY, CONFIRMATION, CAMPAIGN, REVIEW)):
        raise FileExistsError("Use a completely new qualification pipeline")
    training.verify_registration(training.OUTPUT)
    inputs = [training.OUTPUT / "config.json", training.OUTPUT / "registration.json",
              PARENT_CACHE / "manifest.json", PARENT_CACHE / "audit.json"]
    inputs.extend(ROOT / f"{suite}_protocol_v1/protocol.json" for suite in ("tabular", "recsys", "llm"))
    sources = {Path(__file__).resolve(), REPO / "scripts/run_single_matrixformer_cached_benchmarks.py",
               REPO / "scripts/run_single_matrixformer_benchmarks.py", REPO / "scripts/review_single_matrixformer_campaign.py",
               REPO / "tests/test_single_model_finite_guarded_audit.py",
               REPO / "tests/test_single_model_finite_guarded_training_audit.py",
               REPO / "tests/test_single_model_finite_guarded_pipeline.py",
               REPO / "tests/test_single_model_targeted_decoder_audit.py",
               REPO / "tests/test_single_model_targeted_decoder_pipeline.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    output.mkdir(parents=True)
    write_json(output / "config.json", {"run": str(training.OUTPUT), "qualification": training.QUALIFICATION, "policy": POLICY,
               "confirmation_cases": training.confirmation_cases(), "input_sha256": {str(path): sha256(path) for path in inputs},
               "source_sha256": {str(path): sha256(path) for path in sources}, "training_audit": str(AUDIT),
               "endpoint_replay": str(REPLAY), "confirmation": str(CONFIRMATION), "campaign": str(CAMPAIGN), "review": str(REVIEW)})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "confirmation_evaluated": False, "real_benchmarks_evaluated": False})
    for directory in (REPLAY, CONFIRMATION):
        directory.mkdir()
        write_json(directory / "config.json", {"pipeline_config_sha256": sha256(output / "config.json"),
                   "run_config_sha256": sha256(training.OUTPUT / "config.json"), "policy": POLICY,
                   "confirmation_cases": training.confirmation_cases() if directory == CONFIRMATION else None})
        write_json(directory / "registration.json", {"config_sha256": sha256(directory / "config.json"), "model_evaluated": False})
        status(directory, "registered")
    status(output, "registered")


def prerequisite(output):
    verify(output)
    audit = read(AUDIT / "audit.json")
    if (not audit["complete"] or not audit["complete_training_audit"] or audit["through_step"] != 100
            or audit["attempted_batches"] != 100 or audit["training_episodes"] != 1400
            or audit["research_selection_tolerance"] != 0 or audit["raw_endpoint"] is None
            or not audit["decoder_only_training"] or audit["prediction_grids"] != 1368
            or audit["retention_baseline"] != str(training.original.alignment.fixed.PARENT)
            or audit["training_initialization"] != str(training.PARENT)):
        raise ValueError("Require the complete declared training audit, never a prefix")
    verify_hashes(audit["artifact_sha256"])
    return audit


def no_confirmation_reason(best, replay_gate):
    if best["step"] == 0:
        return "No trained boundary retains every original guard and strictly improves minimax"
    if not replay_gate["selected_validation_passed"]:
        return "Audited selected checkpoint fails the unchanged development guards on independent neural replay; no fallback"
    return None


def endpoint_setup(config):
    model, optimizer, criterion, teachers, sampler = training.original.alignment.mixture.original_setup(config["base"])
    del optimizer, teachers, sampler
    parent = torch.load(training.PARENT, map_location="cpu", weights_only=False, mmap=True)["model"]
    model.load_state_dict(parent, strict=True)
    model.train()
    view = decoder.DecoderParameterView(model)
    parameters = read(training.OUTPUT / "parameter_inventory.json")
    selected_parameters = read(training.OUTPUT / "decoder_inventory.json")
    current_inventory = [[name, list(parameter.shape), parameter.numel()] for name, parameter in model.named_parameters()]
    training_audit.check_inventories(current_inventory, selected_parameters, parameters)
    return model, view, criterion, parent, {row[0] for row in selected_parameters}


@torch.no_grad()
def replay(output):
    audit = prerequisite(output)
    if read(REPLAY / "status.json")["state"] != "registered":
        raise ValueError("Never repeat a started endpoint replay")
    wait_for_gpu(REPLAY, True, False)
    status(REPLAY, "replaying_endpoint")
    config = read(training.OUTPUT / "config.json")
    path = training.OUTPUT / "raw_final_update.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    snapshot, record = payload["snapshot"], payload["record"]
    manifest = read(payload["task_manifest"])
    model, view, criterion, parent, mutable_names = endpoint_setup(config)
    roles = {"before": snapshot["before"]}
    candidate_digest = None
    if "projection" in record:
        gradients = payload["gradients"].to("cuda")
        candidate = training.original.alignment.projection.projected_parameters(snapshot["tentative"].to("cuda"), gradients,
                                        record["projection"]["raw_gradient_coefficients"])
        products, norm = direction_products(gradients, snapshot["before"].to("cuda").double() - candidate.double())
        np.testing.assert_allclose(products, record["candidate_products"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(norm, record["candidate_norm"], rtol=1e-12, atol=1e-12)
        if displacement_decision(products, norm, record["adam_norm"]) != record["displacement"]:
            raise ValueError("The reconstructed actual rounded candidate decision differs")
        candidate_cpu = candidate.cpu()
        if record["accepted"] and not torch.equal(candidate_cpu, snapshot["after"]):
            raise ValueError("Identical GPU projection inputs must reproduce the actually accepted FP32 endpoint")
        candidate_digest = training.original.save_tensor({"parameters": candidate_cpu, "products": products.tolist(), "norm": norm},
                                                REPLAY / "reconstructed_candidate.pt")
        if record["point_after"] is not None:
            roles["candidate"] = candidate_cpu
        del candidate, gradients
    rows, measured, hashes = [], {}, {}
    for role, values in roles.items():
        training.original.alignment.preflight.assign(view, values)
        metrics_for_role = {}
        for task in manifest["tasks"]:
            cached = torch.load(task["artifact"], map_location="cpu", weights_only=False)
            episode = training.original.alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
            torch.manual_seed(task["gradient_seed"])
            with compute_context(episode.source.device):
                prediction = training.original.streamed_point_predictions(model, criterion, episode.source, episode.mask, **training.original.TRAIN_OPTIONS)
            metrics = training.original.alignment.metric_summary(prediction, episode, training.original.point_metric_terms(prediction, episode))
            target = REPLAY / f'{role}_task_{task["index"]:02d}.npz'
            atomic_npz(target, prediction=prediction.cpu().numpy())
            hashes[str(target)] = sha256(target)
            rows.append({"role": role, "task": task["name"], "artifact": task["artifact"], "artifact_sha256": task["sha256"],
                         "prediction": str(target), "prediction_sha256": hashes[str(target)], "metrics": metrics})
            metrics_for_role.update({f'{task["name"]}/{name}': None if value is None else value["value"] for name, value in metrics.items()})
            del cached, episode, prediction
        expected = record["point_before"] if role == "before" else record["point_after"]
        compare_nested(metrics_for_role, expected)
        measured[role] = metrics_for_role
    if "candidate" in measured:
        finite = training.original.guarded.metric_decision(measured["before"], measured["candidate"])
        compare_nested(finite, record["finite"])
    frozen = decoder.frozen_state_equal(parent, model.state_dict(), mutable_names)
    selected = None
    if audit["best"]["step"]:
        path = training.OUTPUT / "selected_checkpoint.pth"
        digest = sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["step"] != audit["best"]["step"] or checkpoint["gradient_updates_performed"] < 1:
            raise ValueError("Only the audited trained selection may be replayed")
        model.load_state_dict(checkpoint["model"], strict=True)
        decoder.frozen_state_equal(parent, model.state_dict(), mutable_names)
        bank = torch.load(training.original.alignment.fixed.PRODUCER / "synthetic_bank.pt", map_location="cpu", weights_only=True)
        with training.original.alignment.mixture.runtime(config["base"], REPLAY):
            metrics, cases = training.original.alignment.mixture.backend.evaluate_bank(model, criterion, bank)
        original = read(training.OUTPUT / f'validation_{audit["best"]["step"]}.json')
        close_values(metrics, original["student"])
        reference = read(training.original.alignment.fixed.PRODUCER / "reference.json")
        initial = read(training.original.alignment.fixed.PRODUCER / "validation_0.json")["student"]
        decision = training.original.alignment.fixed.selection_decision(metrics, initial, reference["references"], config["base"])
        selected = {"checkpoint": str(path), "checkpoint_sha256": digest, "student": metrics, "cases": cases, "decision": decision}
        verify_hashes({str(path): digest})
    write_json(REPLAY / "result.json", {"endpoint_step": record["step"], "accepted": record["accepted"], "cases": rows,
               "measured": measured, "candidate_sha256": candidate_digest, "selected": selected, "frozen": frozen,
               "decoder_inventory_sha256": sha256(training.OUTPUT / "decoder_inventory.json"),
               "training_initialization_sha256": sha256(training.PARENT),
               "training_audit_sha256": sha256(AUDIT / "audit.json"), "raw_endpoint_sha256": sha256(training.OUTPUT / "raw_final_update.pt"),
               "goal_success_not_assessed": True})
    status(REPLAY, "awaiting_independent_audit")


def audit_replay(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Replay audit must be CPU only")
    audit = prerequisite(output)
    if read(REPLAY / "status.json")["state"] != "awaiting_independent_audit" or (REPLAY / "audit.json").exists():
        raise ValueError("Require a fresh complete endpoint replay")
    result = read(REPLAY / "result.json")
    raw_path = training.OUTPUT / "raw_final_update.pt"
    hashes = {str(REPLAY / "result.json"): sha256(REPLAY / "result.json"), str(raw_path): result["raw_endpoint_sha256"]}
    hashes[str(training.PARENT)] = result["training_initialization_sha256"]
    hashes[str(training.OUTPUT / "decoder_inventory.json")] = result["decoder_inventory_sha256"]
    if result["frozen"]["frozen_tensors"] != 248 or not result["frozen"]["backbone_and_buffers_bitwise_unchanged"]:
        raise ValueError("Replayed nondecoder state must remain frozen")
    payload = torch.load(raw_path, map_location="cpu", weights_only=False, mmap=True)
    expected_count = 14 if payload["record"]["point_after"] is None else 28
    if len(result["cases"]) != expected_count or result["training_audit_sha256"] != sha256(AUDIT / "audit.json"):
        raise ValueError("Every endpoint case must bind the complete training audit")
    measured = {}
    for row in result["cases"]:
        hashes[row["artifact"]] = row["artifact_sha256"]
        hashes[row["prediction"]] = row["prediction_sha256"]
        cached = torch.load(row["artifact"], map_location="cpu", weights_only=False)
        episode = training.original.alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
        with np.load(row["prediction"], allow_pickle=False) as saved:
            prediction = torch.from_numpy(saved["prediction"])
        metrics = training.original.alignment.metric_summary(prediction, episode, training.original.point_metric_terms(prediction, episode))
        compare_nested(metrics, row["metrics"])
        values = measured.setdefault(row["role"], {})
        for name, metric in metrics.items():
            key = f'{row["task"]}/{name}'
            if key in values:
                raise ValueError("Never duplicate or omit an endpoint task/metric slot")
            values[key] = None if metric is None else metric["value"]
    compare_nested(measured, result["measured"])
    compare_nested(measured["before"], payload["record"]["point_before"])
    if "candidate" in measured:
        compare_nested(measured["candidate"], payload["record"]["point_after"])
        compare_nested(training.original.guarded.metric_decision(measured["before"], measured["candidate"]),
                                                 payload["record"]["finite"])
    if result["candidate_sha256"]:
        path = REPLAY / "reconstructed_candidate.pt"
        hashes[str(path)] = result["candidate_sha256"]
        candidate = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        products, norm = direction_products(payload["gradients"], payload["snapshot"]["before"].double() - candidate["parameters"].double())
        np.testing.assert_allclose(products, payload["record"]["candidate_products"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(norm, payload["record"]["candidate_norm"], rtol=1e-12, atol=1e-12)
        if payload["record"]["accepted"] and not torch.equal(candidate["parameters"], payload["snapshot"]["after"]):
            raise ValueError("Replayed accepted weights must exactly equal the actual endpoint")
    selected_passed, selected_grids = False, 0
    if result["selected"] is not None:
        report = result["selected"]
        if not audit["best"]["step"] or len(report["cases"]) != 152:
            raise ValueError("Require every selected-development replay case")
        hashes[report["checkpoint"]] = report["checkpoint_sha256"]
        bank = torch.load(training.original.alignment.fixed.PRODUCER / "synthetic_bank.pt", map_location="cpu", weights_only=True)
        selected_grids, predicted_hashes = training.original.alignment.fixed.raw_audit(bank, [(report, ("student",))])
        hashes.update(predicted_hashes)
        if training.original.alignment.mixture.aggregate(report["cases"], "student") != report["student"]:
            raise ValueError("Selected replay aggregation changed")
        baseline = read(training.original.alignment.fixed.PRODUCER / "validation_0.json")["student"]
        reference = read(training.original.alignment.fixed.PRODUCER / "reference.json")["references"]
        decision = training.original.alignment.fixed.selection_decision(report["student"], baseline, reference,
                                                               read(training.OUTPUT / "config.json")["base"])
        if decision != report["decision"] or len(decision["comparisons"]) != 22:
            raise ValueError("Selected replay must use the unchanged22-guard decision")
        selected_passed = decision["passed"]
    verify_hashes(hashes)
    verify(output)
    write_json(REPLAY / "audit.json", {"complete": True, "endpoint_grids": expected_count, "selected_grids": selected_grids,
               "artifact_sha256": hashes, "cuda_initialized": False, "autograd_replayed": False,
               "research_selection_tolerance": 0, "goal_success_not_assessed": True})
    write_json(REPLAY / "gate.json", {"endpoint_reproduced": True, "selected_validation_passed": selected_passed,
               "selected_step": audit["best"]["step"], "audit_sha256": sha256(REPLAY / "audit.json")})
    status(REPLAY, "complete", selected_validation_passed=selected_passed)


def confirmation_setup(config):
    result = training.original.alignment.mixture.backend.setup(config["base"])
    baseline = torch.load(training.original.alignment.fixed.PARENT, map_location="cpu", weights_only=False, mmap=True)["model"]
    current = result[0].state_dict()
    if set(current) != set(baseline) or any(not torch.equal(value.detach().cpu(), baseline[name]) for name, value in current.items()):
        raise ValueError("Independent confirmation must initialize from V8, never the V14 training input")
    return result


@torch.no_grad()
def confirm(output):
    audit = prerequisite(output)
    if read(REPLAY / "status.json")["state"] != "complete" or not read(REPLAY / "gate.json")["endpoint_reproduced"]:
        raise ValueError("Require the independently audited endpoint replay")
    if read(CONFIRMATION / "status.json")["state"] != "registered":
        raise ValueError("Never repeat a started confirmation")
    reason = no_confirmation_reason(audit["best"], read(REPLAY / "gate.json"))
    if reason is not None:
        write_json(CONFIRMATION / "gate.json", {"passed": False, "selected_step": audit["best"]["step"],
                   "confirmation_evaluated": False, "reason": reason, "real_benchmark_success": False})
        status(CONFIRMATION, "complete", confirmation_evaluated=False)
        return
    wait_for_gpu(CONFIRMATION, True, False)
    status(CONFIRMATION, "confirming", selected_step=audit["best"]["step"])
    config = read(training.OUTPUT / "config.json")
    path = training.OUTPUT / "selected_checkpoint.pth"
    fingerprint = sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != audit["best"]["step"] or checkpoint["config_sha256"] != sha256(training.OUTPUT / "config.json"):
        raise ValueError("Only the audited single selected checkpoint may consume confirmation")
    write_json(CONFIRMATION / "manifest.json", {"checkpoint": str(path), "checkpoint_sha256": fingerprint,
               "selected_step": audit["best"]["step"], "cases": config["confirmation_cases"],
               "training_audit_sha256": sha256(AUDIT / "audit.json"), "replay_audit_sha256": sha256(REPLAY / "audit.json"),
               "retention_baseline": str(training.original.alignment.fixed.PARENT),
               "retention_baseline_sha256": sha256(training.original.alignment.fixed.PARENT)})
    with training.original.alignment.mixture.runtime(config["base"], CONFIRMATION):
        student, optimizer, criterion, teachers, sampler = confirmation_setup(config)
        del optimizer
        bank, reference = training.original.alignment.mixture.backend.prepare_bank(sampler, teachers, criterion, config["confirmation_cases"], CONFIRMATION)
        initial_metrics, initial_cases = training.original.alignment.mixture.backend.evaluate_bank(student, criterion, bank)
        student.load_state_dict(checkpoint["model"], strict=True)
        metrics, cases = training.original.alignment.mixture.backend.evaluate_bank(student, criterion, bank)
    if len(bank) != 284 or len(initial_cases) != 284 or len(cases) != 284:
        raise ValueError("Never omit any reserved confirmation case")
    write_json(CONFIRMATION / "initialization.json", {"student": initial_metrics, "cases": initial_cases})
    write_json(CONFIRMATION / "student.json", {"student": metrics, "cases": cases})
    decision = training.original.alignment.mixture.confirmation_gate(metrics, initial_metrics, reference["general"], reference["references"])
    write_json(CONFIRMATION / "decision.json", decision)
    verify_hashes({str(path): fingerprint})
    status(CONFIRMATION, "awaiting_independent_audit")


def audit_confirmation(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Independent confirmation audit must use CPU only")
    audit = prerequisite(output)
    if read(CONFIRMATION / "status.json")["state"] != "awaiting_independent_audit" or (CONFIRMATION / "audit.json").exists():
        raise ValueError("Require a fresh complete independent confirmation")
    reference, initial, student = [read(CONFIRMATION / name) for name in ("reference.json", "initialization.json", "student.json")]
    bank = torch.load(CONFIRMATION / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    config = read(training.OUTPUT / "config.json")
    if len(bank) != len(config["confirmation_cases"]) or len(bank) != 284:
        raise ValueError("Every reserved independent case is mandatory")
    verify_hashes({str(CONFIRMATION / "synthetic_bank.pt"): reference["bank_sha256"]})
    for cached, specification in zip(bank, config["confirmation_cases"]):
        if cached["metadata"]["seed"] != specification.get("structured", specification)["seed"]:
            raise ValueError("Confirmation identity or order changed")
    fields = [(reference, ("general", "teacher")), (initial, ("student",)), (student, ("student",))]
    for report, roles in fields:
        if len(report["cases"]) != 284:
            raise ValueError("Never drop confirmation cases")
        for role in roles:
            if training.original.alignment.mixture.aggregate(report["cases"], role) != report[role]:
                raise ValueError("Confirmation aggregation changed")
    if training.original.alignment.fixed.reference_values(reference["general"], reference["teacher"]) != reference["references"]:
        raise ValueError("Original confirmation reference floors changed")
    count, hashes = training.original.alignment.fixed.raw_audit(bank, fields)
    if count != 1136:
        raise ValueError("Require all1136 independent full prediction grids")
    hashes.update({str(CONFIRMATION / name): sha256(CONFIRMATION / name)
                   for name in ("config.json", "manifest.json", "synthetic_bank.pt", "reference.json", "initialization.json", "student.json", "decision.json")})
    decision = training.original.alignment.mixture.confirmation_gate(student["student"], initial["student"], reference["general"], reference["references"])
    if decision != read(CONFIRMATION / "decision.json"):
        raise ValueError("Unchanged independent confirmation gate does not reproduce")
    manifest = read(CONFIRMATION / "manifest.json")
    if manifest["selected_step"] != audit["best"]["step"] or manifest["cases"] != config["confirmation_cases"]:
        raise ValueError("Confirmation must bind the sole audited selection and reservation")
    if (manifest["retention_baseline"] != str(training.original.alignment.fixed.PARENT)
            or manifest["training_audit_sha256"] != sha256(AUDIT / "audit.json")
            or manifest["replay_audit_sha256"] != sha256(REPLAY / "audit.json")
            or manifest["checkpoint"] != str(training.OUTPUT / "selected_checkpoint.pth")):
        raise ValueError("Confirmation must bind the V8 baseline and audited selected checkpoint")
    hashes[manifest["retention_baseline"]] = manifest["retention_baseline_sha256"]
    hashes[manifest["checkpoint"]] = manifest["checkpoint_sha256"]
    verify_hashes(hashes)
    verify(output)
    write_json(CONFIRMATION / "audit.json", {"complete": True, "prediction_grids": count, "cases": 284,
               "research_selection_tolerance": 0, "artifact_sha256": hashes, "cuda_initialized": False, "goal_success_not_assessed": True})
    gate = {**decision, "selected_step": manifest["selected_step"], "checkpoint_sha256": manifest["checkpoint_sha256"],
            "confirmation_evaluated": True, "cases": 284, "legacy_cases": 92, "structured_cases": 192,
            "bank_sha256": reference["bank_sha256"], "audit_sha256": sha256(CONFIRMATION / "audit.json"),
            "training_audit_sha256": sha256(AUDIT / "audit.json"), "replay_audit_sha256": sha256(REPLAY / "audit.json")}
    if gate["passed"]:
        alias = training.OUTPUT / "best_validation_checkpoint.pth"
        if alias.exists():
            raise FileExistsError("Never replace a published best alias")
        shutil.copy2(manifest["checkpoint"], alias)
        verify_hashes({str(alias): manifest["checkpoint_sha256"]})
    write_json(CONFIRMATION / "gate.json", gate)
    status(CONFIRMATION, "complete", passed=gate["passed"])


def run_child(output, label, script, arguments, cpu=False):
    command = [sys.executable, "-u", str(script), *arguments]
    environment = {**os.environ, "PYTHONPATH": f"{REPO / 'scripts'}:{REPO / 'src'}", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    with (output / f"{label}.log").open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, cwd=REPO.parent)
        status(output, label, child_pid=process.pid, command=command)
        if process.wait():
            raise RuntimeError(f"{label} failed; preserve its artifacts without restarting")


def run(output):
    verify(output)
    while True:
        current = read(training.OUTPUT / "status.json")
        if current["state"] == "training_complete":
            break
        if current["state"] in {"failed", "interrupted"}:
            raise RuntimeError(f"Training stopped: {current}")
        os.kill(current["pid"], 0)
        status(output, "waiting_for_training", training_pid=current["pid"], active_step=current.get("active_step", 0))
        time.sleep(30)
    run_child(output, "training_audit", Path(training_audit.__file__).resolve(), ["--output", str(AUDIT)], cpu=True)
    for stage, cpu in (("replay", False), ("audit-replay", True), ("confirm", False)):
        run_child(output, stage, Path(__file__).resolve(), ["--stage", stage, "--output", str(output)], cpu=cpu)
    if read(CONFIRMATION / "status.json")["state"] == "awaiting_independent_audit":
        run_child(output, "audit-confirmation", Path(__file__).resolve(), ["--stage", "audit-confirmation", "--output", str(output)], cpu=True)
    gate = read(CONFIRMATION / "gate.json")
    if gate["passed"]:
        run_child(output, "benchmarks", REPO / "scripts/run_single_matrixformer_cached_benchmarks.py",
                  ["--run", str(training.OUTPUT), "--confirmation", str(CONFIRMATION), "--source", str(PARENT_CACHE), "--output", str(CAMPAIGN)])
        run_child(output, "review", REPO / "scripts/review_single_matrixformer_campaign.py", ["--campaign", str(CAMPAIGN), "--output", str(REVIEW)], cpu=True)
    verify(output)
    write_json(output / "result.json", {"complete": True, "gate": gate, "training_audit_sha256": sha256(AUDIT / "audit.json"),
               "campaign": str(CAMPAIGN) if gate["passed"] else None, "review": str(REVIEW) if gate["passed"] else None,
               "no_inference_ensemble": True, "goal_success_not_assessed": True})
    status(output, "complete", confirmed=gate["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit-replay", "confirm", "audit-confirmation"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "register":
        register(output)
        return
    directory = output if args.stage == "run" else REPLAY if "replay" in args.stage else CONFIRMATION
    functions = {"run": run, "replay": replay, "audit-replay": audit_replay, "confirm": confirm, "audit-confirmation": audit_confirmation}
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "run" and read(output / "status.json")["state"] != "registered":
            raise ValueError("Never restart a started pipeline")
        try:
            functions[args.stage](output)
        except BaseException as error:
            status(directory, "failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
