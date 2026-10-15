"""One fresh actual decoder-only CUDA AdamW update from the unqualified V13 training tail."""

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

import audit_single_model_finite_guarded_training as task_audit
from dwip_teacher_mean_training import compute_context, verify_hashes
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_json
import single_model_decoder_training as decoder
import single_model_finite_guarded_audit as checks
from single_model_point_proposal_grid import direction_products
import train_single_matrixformer_finite_guarded as training
from train_streamed_large_matrices import wait_for_gpu


ROOT = training.ROOT
OUTPUT = ROOT / "decoder_training_preflight_v1"
PARENT = training.OUTPUT / "latest_checkpoint.pth"
EVIDENCE = ROOT / "decoder_finite_steps_v1_audit_recovery1"
RATE = 2e-6
DATA_SEED, GRADIENT_SEED = 191100901, 192100901
POLICY = ("One fresh fourteen-task actual fused-CUDA AdamW decoder-only proposal from the unqualified V13 final "
          "training state. Preserve every encoder/attention parameter and every buffer bit-for-bit; same ordinary "
          "network, original teachers, priors, normalizations and Huber/NLL/CDF/query objectives. Recompute clipped "
          "gradients and Adam moments in decoder coordinates, rather than restricting an old full-model step. "
          "Use the unchanged0.1 joint composite/point protection,2x norm bound and strict finite-point rollback. "
          "Save every raw decoder gradient and optimizer state, before/candidate predictions and the actual final "
          "model. Independently audit all saved data and scores on CPU. This is only a training-feasibility test; "
          "no validation, confirmation, benchmark, checkpoint selection or promotion. V8 remains the required "
          "retention baseline for any future trial, not this unqualified training input.")


def read(path):
    return json.loads(Path(path).read_text())


def status(output, state, **fields):
    write_json(output / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def batch():
    tasks = deepcopy(training.alignment.preflight.specifications())
    for task in tasks:
        task["gradient_seed"] = GRADIENT_SEED + task["index"] * 100
        task["specification"].get("structured", task["specification"])["seed"] = DATA_SEED + task["index"] * 100
    return {"step": 1, "tasks": tasks}


def verify(output):
    config = read(output / "config.json")
    if (sha256(output / "config.json") != read(output / "registration.json")["config_sha256"]
            or config["policy"] != POLICY or config["batch"] != batch() or config["rate"] != RATE
            or config["parent"] != str(PARENT) or config["mutable_prefix"] != "decoder."
            or config["base"] != read(training.alignment.fixed.PRODUCER / "config.json")):
        raise ValueError("Never change the declared fresh-data decoder preflight")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    return config


def register(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Use only the new declared preflight directory")
    if shutil.disk_usage(ROOT).free < 24 * 1024**3:
        raise OSError("Reserve all raw decoder evidence and18 GiB of free space")
    terminal = read(ROOT / "finite_guarded_v13_pipeline/result.json")
    audit = read(ROOT / "finite_guarded_v13_audit/audit.json")
    recovery, finite = read(EVIDENCE / "recovery.json"), read(EVIDENCE / "audit.json")
    if (not terminal["complete"] or terminal["gate"]["selected_step"] != 0 or terminal["gate"]["confirmation_evaluated"]
            or not audit["complete_training_audit"] or audit["through_step"] != 50 or audit["accepted_updates"] != 46
            or not recovery["complete"] or not recovery["original_artifacts_unchanged"] or not finite["complete"]
            or finite["prediction_grids"] != 84 or not finite["finite_decisions"]["decoder"]["finite_feasible"]):
        raise ValueError("Require the unqualified audited V13 tail and recovered finite decoder evidence")
    if sha256(PARENT) != audit["artifact_sha256"][str(PARENT)]:
        raise ValueError("The unqualified training input must be the actual audited V13 endpoint")
    verify_hashes(finite["artifact_sha256"])
    base = read(training.alignment.fixed.PRODUCER / "config.json")
    previous = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    used = set().union(*(training.alignment.fixed.seeds_in(read(path)) for path in previous))
    fresh = training.alignment.fixed.seeds_in(batch())
    if len(fresh) != 28 or fresh.intersection(used):
        raise ValueError("All fourteen data and backward seed pairs must be fresh")
    paths = [PARENT, training.OUTPUT / "config.json", training.alignment.fixed.PRODUCER / "config.json",
             ROOT / "finite_guarded_v13_pipeline/result.json", ROOT / "finite_guarded_v13_audit/audit.json",
             EVIDENCE / "audit.json", EVIDENCE / "recovery.json", EVIDENCE / "recovery_config.json",
             ROOT / "domain_gradient_probe_v1/parameter_inventory.json"]
    sources = {Path(__file__).resolve(), Path(decoder.__file__).resolve(), REPO / "tests/test_single_model_decoder_training.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "base": base, "parent": str(PARENT), "parent_selection_qualified": False,
              "retention_baseline_for_future_training": str(training.alignment.fixed.PARENT), "mutable_prefix": "decoder.",
              "batch": batch(), "rate": RATE, "input_sha256": {**base["input_sha256"], **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in previous}}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "optimizer_update_performed": False, "confirmation_or_real_benchmarks_evaluated": False})
    status(output, "registered")


@torch.no_grad()
def predictions(output, model, view, criterion, tasks, role):
    if role == "candidate":
        training.save_tensor(training.alignment.preflight.flatten(view).cpu(), output / "candidate_parameters.pt")
    rows, values = [], {}
    for task in tasks:
        cached = torch.load(task["artifact"], map_location="cpu", weights_only=False)
        episode = training.alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
        torch.manual_seed(task["gradient_seed"])
        with compute_context(episode.source.device):
            prediction = training.streamed_point_predictions(model, criterion, episode.source, episode.mask, **training.TRAIN_OPTIONS)
        metrics = training.alignment.metric_summary(prediction, episode, training.point_metric_terms(prediction, episode))
        path = output / f'{role}_task_{task["index"]:02d}.npz'
        atomic_npz(path, prediction=prediction.cpu().numpy())
        rows.append({"task": task["name"], "index": task["index"], "metrics": metrics, "prediction": str(path), "sha256": sha256(path)})
        values.update({f'{task["name"]}/{name}': None if metric is None else metric["value"] for name, metric in metrics.items()})
    write_json(output / f"{role}_predictions.json", {"role": role, "rows": rows, "values": values})
    return values


def run(output):
    config = verify(output)
    if read(output / "status.json")["state"] != "registered":
        raise ValueError("Never restart a started preflight")
    wait_for_gpu(output, True, False)
    status(output, "initializing_decoder_training")
    parent = torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)
    with training.alignment.mixture.runtime(config["base"], output):
        model, old_optimizer, criterion, teachers, sampler = training.alignment.mixture.backend.setup(config["base"])
        del old_optimizer
        model.load_state_dict(parent["model"], strict=True)
        model.train()
        view = decoder.DecoderParameterView(model)
        inventory = [[name, list(parameter.shape), parameter.numel()] for name, parameter in model.named_parameters()]
        selected = [[name, list(parameter.shape), parameter.numel()] for name, parameter in view.named_parameters()]
        if inventory != read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json") or selected != [row for row in inventory if row[0].startswith("decoder.")]:
            raise ValueError("Preserve the original full model and its exact decoder coordinates")
        write_json(output / "parameter_inventory.json", inventory)
        write_json(output / "decoder_inventory.json", selected)
        optimizer = torch.optim.AdamW([{"params": list(view.parameters()), "lr": RATE}], lr=RATE,
            weight_decay=config["base"]["weight_decay"], betas=(.9, .999), eps=1e-8, fused=True)
        gradients = torch.empty((42, sum(row[2] for row in selected)), device="cuda", dtype=torch.float32)
        directory = output / "training_batch"
        directory.mkdir()
        torch.cuda.reset_peak_memory_stats()
        with decoder.decoder_capture_coordinates(model, view):
            entries, tasks, before = training.capture_batch(output, directory, config["batch"], model, optimizer, criterion, teachers, sampler, gradients)
        measured = predictions(output, model, view, criterion, tasks, "before")
        training.alignment.compare_metric_records(measured, before)
        status(output, "protecting_decoder_update")
        try:
            record, snapshot = training.guarded.protected_update(view, optimizer, gradients[:len(entries)],
                [task["weight"] for task in tasks], RATE, before,
                lambda current: predictions(output, model, current, criterion, tasks, "candidate"))
        except training.guarded.StepEvaluationError as error:
            training.save_tensor({"snapshot": error.snapshot, "record": error.record, "gradients": gradients[:len(entries)].cpu(),
                                  "entries": entries, "batch": config["batch"]}, output / "failed_update.pt")
            raise
        record.update(step=1, accepted_updates=int(record["accepted"]))
        frozen = decoder.frozen_state_equal(parent["model"], model.state_dict(), {row[0] for row in selected})
        digest = training.save_tensor({"snapshot": snapshot, "record": record, "gradients": gradients[:len(entries)].cpu(),
                    "entries": entries, "batch": config["batch"], "config_sha256": sha256(output / "config.json")}, output / "raw_update.pt")
        checkpoint_digest = training.save_tensor({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 1,
                    "preflight_only": True, "selection_qualified": False, "gradient_updates_performed": int(record["accepted"]),
                    "parent_sha256": config["input_sha256"][str(PARENT)], "config_sha256": sha256(output / "config.json")}, output / "preflight_checkpoint.pth")
        write_json(output / "result.json", {"record": record, "frozen": frozen, "raw_sha256": digest,
                    "checkpoint_sha256": checkpoint_digest, "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "trainable_parameters": sum(row[2] for row in selected), "backbone_parameters": sum(row[2] for row in inventory) - sum(row[2] for row in selected),
                    "preflight_only": True, "goal_success_not_assessed": True})
    verify(output)
    status(output, "awaiting_independent_audit")


def audit(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Audit must be CPU only")
    config = verify(output)
    if read(output / "status.json")["state"] != "awaiting_independent_audit":
        raise ValueError("Require a fresh completed preflight")
    result, manifest = read(output / "result.json"), read(output / "training_batch/manifest.json")
    inventory, selected = read(output / "parameter_inventory.json"), read(output / "decoder_inventory.json")
    if (inventory != read(ROOT / "domain_gradient_probe_v1/parameter_inventory.json")
            or selected != [row for row in inventory if row[0].startswith("decoder.")]
            or manifest["step"] != 1 or len(manifest["tasks"]) != 14):
        raise ValueError("Preserve all original model coordinates and all fourteen fresh tasks")
    hashes = {str(output / name): sha256(output / name) for name in ("config.json", "registration.json", "result.json", "parameter_inventory.json",
               "decoder_inventory.json", "training_batch/manifest.json", "before_predictions.json")}
    before = {}
    entries = [{"task": task["name"], "kind": "composite", "index": task["index"]} for task in config["batch"]["tasks"]]
    for task, logged in zip(config["batch"]["tasks"], manifest["tasks"]):
        if logged["artifact"] != str(output / "training_batch" / f'task_{task["index"]:02d}.pt'):
            raise ValueError("Every cached task must occupy its declared location")
        hashes[logged["artifact"]] = logged["sha256"]
        cached = torch.load(logged["artifact"], map_location="cpu", weights_only=False)
        before.update(task_audit.check_task(cached, task, logged))
        for metric, values in logged["metrics_before"].items():
            if values is not None:
                entries.append({"task": task["name"], "kind": metric, "index": len(entries)})
    if entries != manifest["entries"]:
        raise ValueError("Preserve the complete original composite and defined-point gradient order")
    record = result["record"]
    checks.check_record(record, [task["weight"] for task in config["batch"]["tasks"]], before, len(manifest["entries"]))
    if record["step"] != 1 or record["accepted_updates"] != int(record["accepted"]) or record["rate"] != RATE:
        raise ValueError("Exactly one declared restricted optimizer attempt is allowed")
    raw_path, checkpoint_path = output / "raw_update.pt", output / "preflight_checkpoint.pth"
    hashes.update({str(raw_path): result["raw_sha256"], str(checkpoint_path): result["checkpoint_sha256"]})
    verify_hashes(hashes)
    payload = torch.load(raw_path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    parent = torch.load(PARENT, map_location="cpu", weights_only=False, mmap=True)
    if (payload["batch"] != config["batch"] or payload["entries"] != manifest["entries"]
            or payload["config_sha256"] != sha256(output / "config.json") or not checkpoint["preflight_only"]
            or checkpoint["selection_qualified"] or checkpoint["gradient_updates_performed"] != int(record["accepted"])
            or checkpoint["parent_sha256"] != config["input_sha256"][str(PARENT)] or checkpoint["step"] != 1
            or checkpoint["config_sha256"] != sha256(output / "config.json")
            or result["trainable_parameters"] != sum(row[2] for row in selected)):
        raise ValueError("Raw state and non-promotable checkpoint must bind this exact preflight")
    endpoint = checks.check_endpoint(payload, record, [row[2] for row in selected], config["base"]["weight_decay"], RATE, checkpoint["optimizer"])
    for role, source in (("before", parent["model"]), ("after", checkpoint["model"])):
        vector = torch.cat([source[name].flatten() for name, shape, size in selected])
        if not torch.equal(vector, payload["snapshot"][role]):
            raise ValueError("Decoder vectors must bind the exact parent and actual final full model")
    frozen = decoder.frozen_state_equal(parent["model"], checkpoint["model"], {row[0] for row in selected})
    if frozen != result["frozen"] or bool(frozen["changed_decoder_tensors"]) != record["accepted"]:
        raise ValueError("Only an accepted decoder update may change any model state")
    measured, grids = {}, 0
    roles = ("before", "candidate") if record["point_after"] is not None else ("before",)
    for role in roles:
        path = output / f"{role}_predictions.json"
        hashes[str(path)] = sha256(path)
        report = read(path)
        if report["role"] != role or len(report["rows"]) != 14:
            raise ValueError("Every full before/candidate grid is mandatory")
        values = {}
        for row, task in zip(report["rows"], manifest["tasks"]):
            if row["task"] != task["name"] or row["index"] != task["index"]:
                raise ValueError("Prediction identities must retain their original task order")
            hashes[row["prediction"]] = row["sha256"]
            cached = torch.load(task["artifact"], map_location="cpu", weights_only=False)
            episode = training.alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
            with np.load(row["prediction"], allow_pickle=False) as saved:
                prediction = torch.from_numpy(saved["prediction"])
            metrics = training.alignment.metric_summary(prediction, episode, training.point_metric_terms(prediction, episode))
            training.alignment.compare_metric_records(metrics, row["metrics"])
            values.update({f'{task["name"]}/{name}': None if metric is None else metric["value"] for name, metric in metrics.items()})
            grids += 1
        training.alignment.compare_metric_records(values, report["values"])
        training.alignment.compare_metric_records(values, record["point_before"] if role == "before" else record["point_after"])
        measured[role] = values
    if "candidate" in measured:
        decision = training.guarded.metric_decision(measured["before"], measured["candidate"])
        training.alignment.compare_metric_records(decision, record["finite"])
        candidate_path = output / "candidate_parameters.pt"
        hashes[str(candidate_path)] = sha256(candidate_path)
        candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
        expected = training.alignment.projection.projected_parameters(payload["snapshot"]["tentative"], payload["gradients"],
                                                                      record["projection"]["raw_gradient_coefficients"])
        lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
        upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
        if not ((candidate >= lower) & (candidate <= upper)).all():
            raise ValueError("Scored candidate exceeds the original one-ULP projection arithmetic allowance")
        products, norm = direction_products(payload["gradients"], payload["snapshot"]["before"].double() - candidate.double())
        np.testing.assert_allclose(products, record["candidate_products"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(norm, record["candidate_norm"], rtol=1e-12, atol=1e-12)
        if record["accepted"] and not torch.equal(candidate, payload["snapshot"]["after"]):
            raise ValueError("Scored accepted decoder parameters must equal the actual final model")
    verify_hashes(hashes)
    verify(output)
    write_json(output / "audit.json", {"complete": True, "training_tasks": 14, "prediction_grids": grids,
               "raw_endpoint": endpoint, "frozen": frozen, "artifact_sha256": hashes, "cuda_initialized": False,
               "research_selection_tolerance": 0, "autograd_replayed": False, "goal_success_not_assessed": True})
    write_json(output / "gate.json", {"passed": record["accepted"], "preflight_only": True, "reason": record["reason"],
               "audit_sha256": sha256(output / "audit.json"), "real_benchmark_success": False})
    lines = ["# Fresh decoder-only optimizer preflight", "", POLICY, "",
             f'Accepted: {record["accepted"]}; reason: {record["reason"]}; audited prediction grids: {grids}.',
             f'Trainable parameters: {result["trainable_parameters"]}; frozen backbone parameters: {result["backbone_parameters"]}.',
             f'Peak allocated GPU memory: {result["peak_gib"]:.3f} GiB.',
             "", "No held-out performance improvement or successful new model is established.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    status(output, "complete", accepted=record["accepted"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if arguments.stage == "register":
        register(output)
    else:
        with (output / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "audit": audit}[arguments.stage](output)
            except BaseException as error:
                status(output, "failed", error=repr(error))
                raise
