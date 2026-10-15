"""Run the fixed two-method/two-rate matched training diagnostic without reselection."""

import argparse
from copy import deepcopy
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from dwip_teacher_mean_training import verify_hashes
import preflight_single_model_point_inclusive as preflight
from recsys_evaluation_common import REPO, sha256, write_json
from recover_single_model_decoder_finite_audit import compare_nested
import single_model_matched_proposal_evidence as evidence
import single_model_paired_protection as paired
import single_model_point_inclusive_proposal as inclusive
import single_model_shared_tensor_storage as storage
from single_model_finite_guarded_audit import optimizer_equal
from single_model_point_proposal_grid import direction_products


ROOT = preflight.ROOT
SOURCE = preflight.OUTPUT
OUTPUT = ROOT / "point_proposal_matched_v1"
CASES = (
    {"name": "composite_2e-5", "method": "composite", "rate": 2e-5, "reused": False},
    {"name": "composite_1e-5", "method": "composite", "rate": 1e-5, "reused": False},
    {"name": "point_1e-5", "method": "point_inclusive", "rate": 1e-5, "reused": False},
    {"name": "point_2e-5", "method": "point_inclusive", "rate": 2e-5, "reused": True},
)
POLICY = ("Compare exactly the original composite-only and half-point-inclusive Adam proposals at2e-5 and1e-5 "
          "on the identical28 cached synthetic training tasks and all84 original V8 gradients. Reuse the audited "
          "point-inclusive2e-5 failure without replacing it. Independently initialize each of the other three "
          "controls from original V8 and a fresh empty-moment optimizer at the original preceding rate2e-5. "
          "Preserve paired protection,0.1 margin,2x bound, finite paired acceptance, individual regressions, "
          "all evaluation seeds and all teachers/priors/normalization. Verify every newly replayed before grid "
          "against the original bitwise, and restore full parameters and optimizer after every control. "
          "Save actual proposals, moments, candidate vectors and complete prediction grids; reference existing "
          "identical gradients and initial/final vectors losslessly instead of duplicating them. Independently "
          "CPU-audit all outcomes. No fresh data, autograd replay, held-out validation, selection, confirmation, "
          "model publication, real benchmark or future initialization from a diagnostic state.")


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def prerequisite():
    config = preflight.verify(SOURCE)
    current, audit, gate = [read(SOURCE / name) for name in ("status.json", "audit.json", "gate.json")]
    if (current["state"] != "complete" or Path(f'/proc/{current["pid"]}').exists()
            or not audit["complete"] or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
            or audit["raw_endpoint"]["gradient_vectors"] != 84 or audit["raw_endpoint"]["accepted_updates"] != 0
            or not audit["teacher_replay"]["complete"] or audit["teacher_replay"]["full_grids"] != 28
            or not audit["teacher_replay"]["bitwise_equal"]
            or gate["passed"] or gate["reason"] != "finite_metrics_rejected"
            or gate["audit_sha256"] != sha256(SOURCE / "audit.json")):
        raise ValueError("Require the completed independently audited rejected point-inclusive preflight")
    verify_hashes(audit["artifact_sha256"])
    return config, audit


def registered_cases():
    return deepcopy(list(CASES))


def register():
    if OUTPUT.exists():
        raise FileExistsError("Never repeat or overwrite a matched diagnostic")
    config, audit = prerequisite()
    original = read(SOURCE / "before_predictions.json")
    prediction_bytes = 0
    for row in original["rows"]:
        with np.load(row["prediction"], allow_pickle=False) as saved:
            prediction_bytes += saved["prediction"].nbytes
    inventory = read(SOURCE / "decoder_inventory.json")
    allowance = evidence.storage_allowance(sum(row[2] for row in inventory), prediction_bytes)
    if shutil.disk_usage(ROOT).free - allowance < storage.MINIMUM_FREE:
        raise OSError("Reserve the entire bounded matched diagnostic above18 GiB before starting")
    paths = [SOURCE / name for name in ("config.json", "registration.json", "status.json", "audit.json", "gate.json",
             "raw_update.pt", "result.json", "before_predictions.json", "candidate_predictions.json",
             "candidate_parameters.pt", "training_batch/manifest.json", "decoder_inventory.json")]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_matched_proposal_evidence.py",
               REPO / "tests/test_single_model_matched_proposals.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"policy": POLICY, "cases": registered_cases(), "source": str(SOURCE),
               "parent": str(preflight.PARENT), "batch": config["batch"], "base": config["base"],
               "storage_allowance_bytes": allowance, "prediction_bytes_per_role": prediction_bytes,
               "minimum_free_bytes": storage.MINIMUM_FREE,
               "reference": {"path": str(SOURCE / "raw_update.pt"), "sha256": sha256(SOURCE / "raw_update.pt")},
               "input_sha256": {**config["input_sha256"], **audit["artifact_sha256"],
                                **{str(path): sha256(path) for path in paths}},
               "source_sha256": {**config["source_sha256"], **{str(path): sha256(path) for path in sources}}})
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
               "registered_unix": time.time(), "new_control_predictions_performed": False, "selection_performed": False})
    status("registered")


def verify():
    config = read(OUTPUT / "config.json")
    if (read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["cases"] != registered_cases() or config["source"] != str(SOURCE)
            or config["parent"] != str(preflight.PARENT) or config["minimum_free_bytes"] != storage.MINIMUM_FREE
            or config["batch"] != read(SOURCE / "config.json")["batch"]
            or config["base"] != read(SOURCE / "config.json")["base"]
            or config["reference"] != {"path": str(SOURCE / "raw_update.pt"), "sha256": config["input_sha256"][str(SOURCE / "raw_update.pt")]}
            or config["storage_allowance_bytes"] != evidence.storage_allowance(
                sum(row[2] for row in read(SOURCE / "decoder_inventory.json")), config["prediction_bytes_per_role"])):
        raise ValueError("Never alter registered matched controls, source identity, coverage or storage bounds")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def check_before(directory, expected):
    report = read(directory / "before_predictions.json")
    if len(report["rows"]) != 28 or report["values"] != expected["values"]:
        raise ValueError("Every control must reproduce the same original28-task initial metrics")
    for actual, original in zip(report["rows"], expected["rows"]):
        if actual["task"] != original["task"] or actual["index"] != original["index"]:
            raise ValueError("Preserve original before-grid task order")
        with np.load(actual["prediction"], allow_pickle=False) as current, np.load(original["prediction"], allow_pickle=False) as baseline:
            if set(current.files) != {"prediction"} or not np.array_equal(current["prediction"], baseline["prediction"]):
                raise ValueError("Every independently replayed original before grid must be bitwise equal")


def run():
    config = verify()
    if read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never restart a started matched diagnostic")
    if shutil.disk_usage(ROOT).free - config["storage_allowance_bytes"] < storage.MINIMUM_FREE:
        raise OSError("Retain the full matched diagnostic write allowance")
    preflight.wait_for_gpu(OUTPUT, True, False)
    status("initializing")
    reference = torch.load(SOURCE / "raw_update.pt", map_location="cpu", weights_only=False, mmap=True)
    manifest, original_before = read(SOURCE / "training_batch/manifest.json"), read(SOURCE / "before_predictions.json")
    weights = [task["weight"] for task in config["batch"]["tasks"]]
    paired.validate_layout(reference["entries"], config["batch"]["tasks"], original_before["values"])
    parent = torch.load(preflight.PARENT, map_location="cpu", weights_only=False, mmap=True)
    model, old_optimizer, criterion, teachers, sampler = preflight.training.alignment.mixture.original_setup(config["base"])
    del old_optimizer, teachers, sampler
    model.load_state_dict(parent["model"], strict=True)
    model.train()
    view = preflight.decoder.DecoderParameterView(model)
    selected = [[name, list(parameter.shape), parameter.numel()] for name, parameter in view.named_parameters()]
    if selected != read(SOURCE / "decoder_inventory.json"):
        raise ValueError("Preserve the original decoder coordinates")
    gradients = reference["gradients"].to("cuda")
    results = []
    for case in CASES:
        if case["reused"]:
            results.append({**case, "source": str(SOURCE), "audit_sha256": sha256(SOURCE / "audit.json"),
                            "record": read(SOURCE / "result.json")["record"]})
            continue
        directory = OUTPUT / case["name"]
        directory.mkdir()
        status("running_control", case=case["name"])
        model.load_state_dict(parent["model"], strict=True)
        if not torch.equal(preflight.training.alignment.preflight.flatten(view).cpu(), reference["snapshot"]["before"]):
            raise ValueError("Each control must initialize exactly from original V8, never a preceding candidate")
        optimizer = torch.optim.AdamW([{"params": list(view.parameters()), "lr": preflight.RATE}], lr=preflight.RATE,
            weight_decay=config["base"]["weight_decay"], betas=(.9, .999), eps=1e-8, fused=True)
        initial_optimizer = deepcopy(optimizer.state_dict())
        if initial_optimizer["state"]:
            raise ValueError("Each control requires fresh empty optimizer moments")
        before = preflight.predictions(directory, model, view, criterion, manifest["tasks"], "before")
        check_before(directory, original_before)
        method = paired if case["method"] == "composite" else inclusive
        try:
            record, snapshot = method.protected_update(view, optimizer, gradients, weights, case["rate"], before,
                lambda current: preflight.predictions(directory, model, current, criterion, manifest["tasks"], "candidate"))
        except method.StepEvaluationError as error:
            snapshot = error.snapshot
            snapshot["protected_record"] = deepcopy(error.record)
            record = {**error.record, "step": 1, "accepted_updates": 0}
            actual = preflight.training.original.cpu_copy(optimizer.state_dict())
            failed = evidence.compact(snapshot, record, reference, config["reference"], case["method"], None, actual)
            failed.update(case=case, exception=repr(error), protected_record_copied_from_error=True)
            storage.save(failed, directory / "failed_endpoint.pt")
            raise
        record.update(step=1, accepted_updates=int(record["accepted"]))
        candidate_path = directory / "candidate_parameters.pt"
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=True) if candidate_path.exists() else None
        actual_optimizer = preflight.training.original.cpu_copy(optimizer.state_dict())
        payload = evidence.compact(snapshot, record, reference, config["reference"], case["method"], candidate, actual_optimizer)
        frozen = preflight.decoder.frozen_state_equal(parent["model"], model.state_dict(), {row[0] for row in selected})
        payload.update(case=case, frozen=frozen)
        digest = storage.save(payload, directory / "endpoint.pt")
        model.load_state_dict(parent["model"], strict=True)
        optimizer.load_state_dict(initial_optimizer)
        optimizer.zero_grad(set_to_none=True)
        restored = preflight.decoder.frozen_state_equal(parent["model"], model.state_dict(), set())
        optimizer_equal(preflight.training.original.cpu_copy(optimizer.state_dict()), initial_optimizer)
        row = {**case, "record": record, "endpoint_sha256": digest, "frozen": frozen,
               "restored": restored, "optimizer_restored": True, "before_grids_bitwise_reproduced": 28}
        write_json(directory / "result.json", row)
        results.append(row)
        del optimizer, snapshot, payload, actual_optimizer, candidate
        gc.collect()
    write_json(OUTPUT / "result.json", {"complete": True, "cases": results, "run_pid": os.getpid(),
               "config_sha256": sha256(OUTPUT / "config.json"), "selection_performed": False,
               "autograd_replayed": False, "real_benchmarks_evaluated": False, "goal_success_not_assessed": True})
    verify()
    status("awaiting_independent_audit")


def audit_predictions(directory, manifest, record):
    hashes, measured, grids = {}, {}, 0
    for role in (("before", "candidate") if record["point_after"] is not None else ("before",)):
        path = directory / f"{role}_predictions.json"
        report = read(path)
        if report["role"] != role or len(report["rows"]) != 28:
            raise ValueError("Never omit a registered before/candidate prediction grid")
        hashes[str(path)] = sha256(path)
        values = {}
        for row, task in zip(report["rows"], manifest["tasks"]):
            if (row["task"] != task["name"] or row["index"] != task["index"]
                    or row["prediction"] != str(directory / f'{role}_task_{task["index"]:02d}.npz')):
                raise ValueError("Retain every original cached task identity")
            cached = torch.load(task["artifact"], map_location="cpu", weights_only=False, mmap=True)
            episode = preflight.training.alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
            with np.load(row["prediction"], allow_pickle=False) as saved:
                if set(saved.files) != {"prediction"}:
                    raise ValueError("Preserve the complete original prediction array")
                prediction = torch.from_numpy(saved["prediction"])
            metrics = preflight.training.alignment.metric_summary(prediction, episode, preflight.training.point_metric_terms(prediction, episode))
            compare_nested(metrics, row["metrics"])
            values.update({f'{task["name"]}/{name}': None if metric is None else metric["value"] for name, metric in metrics.items()})
            hashes[row["prediction"]] = row["sha256"]
            grids += 1
        compare_nested(values, report["values"])
        compare_nested(values, record["point_before"] if role == "before" else record["point_after"])
        measured[role] = values
    if "candidate" in measured:
        compare_nested(paired.metric_decision(measured["before"], measured["candidate"]), record["finite"])
    return hashes, grids


def audit():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Independent audit must use CPU only")
    config = verify()
    if read(OUTPUT / "status.json")["state"] != "awaiting_independent_audit" or (OUTPUT / "audit.json").exists():
        raise ValueError("Audit the complete original matched diagnostic once")
    result = read(OUTPUT / "result.json")
    if (not result["complete"] or result["run_pid"] == os.getpid() or len(result["cases"]) != 4
            or Path(f'/proc/{result["run_pid"]}').exists() or result["selection_performed"]
            or result["config_sha256"] != sha256(OUTPUT / "config.json")):
        raise ValueError("Require every declared matched cell from a finished separate process")
    reference = torch.load(SOURCE / "raw_update.pt", map_location="cpu", weights_only=False, mmap=True)
    manifest, original_before = read(SOURCE / "training_batch/manifest.json"), read(SOURCE / "before_predictions.json")
    selected = read(SOURCE / "decoder_inventory.json")
    weights = [task["weight"] for task in config["batch"]["tasks"]]
    hashes = {str(OUTPUT / name): sha256(OUTPUT / name) for name in ("config.json", "registration.json", "result.json")}
    grids, rows = 0, []
    for specification, row in zip(CASES, result["cases"]):
        if any(row[key] != value for key, value in specification.items()):
            raise ValueError("Never omit, substitute or reorder a registered matched cell")
        record = row["record"]
        if specification["reused"]:
            if row["source"] != str(SOURCE) or row["audit_sha256"] != sha256(SOURCE / "audit.json") or record != read(SOURCE / "result.json")["record"]:
                raise ValueError("Reuse the original audited point-inclusive2e-5 outcome unchanged")
            fields, count = audit_predictions(SOURCE, manifest, record)
            hashes.update(fields)
            grids += count
        else:
            directory = OUTPUT / specification["name"]
            if row != read(directory / "result.json"):
                raise ValueError("Original control evidence and summary differ")
            hashes[str(directory / "result.json")] = sha256(directory / "result.json")
            path = directory / "endpoint.pt"
            hashes[str(path)] = row["endpoint_sha256"]
            compact = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            hashes.update(storage.audit(compact, path))
            candidate_path = directory / "candidate_parameters.pt"
            candidate = torch.load(candidate_path, map_location="cpu", weights_only=True) if candidate_path.exists() else None
            if candidate is not None:
                hashes[str(candidate_path)] = sha256(candidate_path)
            payload = evidence.expand(compact, reference, config["reference"], candidate)
            if (compact["case"] != specification or compact["method"] != specification["method"]
                    or payload["record"] != record or compact["frozen"] != row["frozen"]):
                raise ValueError("Bind actual optimizer evidence to the registered method and rate")
            if (record["rate"] != specification["rate"] or record["point_before"] != original_before["values"]
                    or row["before_grids_bitwise_reproduced"] != 28 or not row["optimizer_restored"]
                    or row["restored"] != {"frozen_tensors": 254, "changed_decoder_tensors": [], "backbone_and_buffers_bitwise_unchanged": True}
                    or row["frozen"]["frozen_tensors"] != 248 or not row["frozen"]["backbone_and_buffers_bitwise_unchanged"]):
                raise ValueError("Preserve initialization, all frozen tensors and exact independent-control restoration")
            method = paired if specification["method"] == "composite" else inclusive
            method.check_record(record, weights, original_before["values"], len(reference["entries"]))
            method.check_endpoint(payload, record, [entry[2] for entry in selected], config["base"]["weight_decay"], preflight.RATE,
                                  evidence.expected_optimizer(payload["snapshot"], record))
            if candidate is not None:
                expected = preflight.training.alignment.projection.projected_parameters(payload["snapshot"]["tentative"],
                    reference["gradients"], record["projection"]["raw_gradient_coefficients"])
                lower, upper = torch.nextafter(expected, torch.full_like(expected, -torch.inf)), torch.nextafter(expected, torch.full_like(expected, torch.inf))
                if not ((candidate >= lower) & (candidate <= upper)).all():
                    raise ValueError("Scored candidate exceeds the original one-ULP reconstruction allowance")
                products, norm = direction_products(reference["gradients"], reference["snapshot"]["before"].double() - candidate.double())
                np.testing.assert_allclose(products, record["candidate_products"], rtol=1e-9, atol=1e-12)
                np.testing.assert_allclose(norm, record["candidate_norm"], rtol=1e-12, atol=1e-12)
            check_before(directory, original_before)
            fields, count = audit_predictions(directory, manifest, record)
            hashes.update(fields)
            grids += count
        changes = record.get("finite", {}).get("changes", {})
        rows.append({**specification, "accepted": record["accepted"], "reason": record["reason"],
                     "paired_improved": sum(value is not None and value < 0 for value in changes.values()),
                     "paired_worsened": sum(value is not None and value > 0 for value in changes.values()),
                     "finite": record.get("finite")})
    verify_hashes(hashes)
    verify()
    write_json(OUTPUT / "audit.json", {"complete": True, "matched_cells": 4, "new_controls": 3,
               "prediction_grids": grids, "rows": rows, "artifact_sha256": hashes, "cuda_initialized": False,
               "selection_performed": False, "autograd_replayed": False, "goal_success_not_assessed": True})
    lines = ["# Matched proposal diagnostic", "", POLICY, "",
             "| Proposal | Rate | Accepted | Improved paired metrics | Worsened paired metrics |",
             "|---|---:|---|---:|---:|"]
    lines.extend(f'| {row["method"]} | {row["rate"]:g} | {row["accepted"]} | {row["paired_improved"]} | {row["paired_worsened"]} |' for row in rows)
    lines.extend(["", "Training-batch diagnostics only. No held-out improvement, selection or model promotion.", ""])
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines))
    status("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register()
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "audit": audit}[arguments.stage]()
            except BaseException as error:
                status("failed", error=repr(error))
                raise
