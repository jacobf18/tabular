"""Diagnose one rejected synthetic proposal without new data or optimizer updates."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
import preflight_single_model_shrunk_column_anchor_retention as probe
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_json
from recover_single_model_decoder_finite_audit import compare_nested
import recover_single_model_shrunk_retention_cpu_audit as recovery
from single_model_point_proposal_grid import direction_products, displacement_decision


OUTPUT = probe.OUTPUT.parent / "shrunk_retention_displacement_panel_v1"
SCRATCH = Path("/mnt/matrixformer_scratch/single_model_goal_20261005_owned") / OUTPUT.name
FRACTIONS = (0., 1., .5, .25, .125, .0625)
REPEATS = (0, 1)
BUDGETS = {"checkpoint": 64 * 1024**2, "root": 16 * 1024**2, "scratch": 4 * 1024**3}
RECOVERY_SHA256 = "a37b726e5d46dad92de64d860b46f0661ac5b8f9ccbcfa13cb4a97902df6ba8e"
POLICY = ("Bounded saved-proposal displacement contraction, not Adam learning-rate recomputation. Reuse only "
    "the28 consumed synthetic training tasks and84 saved gradients of the independently audited rejected "
    "count-shrunk retention probe. Before is raw snapshot.before; projected is the separate saved candidate, "
    "never snapshot.after, which contains rollback. Test fractions0,1,.5,.25,.125,.0625 in that order, with "
    "two full original-seed prediction repetitions each. Use FP64 before+fraction*(projected-before), cast "
    "to FP32, with exact0/1 copies. Independently audit weighted-state arithmetic within one output ULP "
    "only for arithmetic, not scientific metrics. Reproduce both original control prediction grids bitwise. "
    "Retain all30 unchanged point guards, all individual metrics and all84 gradient products. Both repeats "
    "must retain every protected finite metric and improve at least one; geometry must retain all30 point "
    "directions and the original2x Adam norm bound. CPU-audit all336 complete grids, six full parameter "
    "vectors and16 frozen buffers per fraction. No autograd, new seeds/data, optimizer updates, validation, "
    "checkpoint nomination, warm start, benchmark, inference ensemble or changed guards. Feasibility is "
    "local to this consumed training batch, not generalization. Preserve18/6/50 GiB reserves with64/16 MiB "
    "checkpoint/root and4 GiB scratch allowances. Large outputs use machine-local scratch, not backup. "
    "Preserve every original failed probe and completed negative recovery artifact; no scientific retry.")


def read(path):
    return json.loads(Path(path).read_text())


def require_cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU audit requires CUDA explicitly hidden and uninitialized")


def capacity():
    roots = {"checkpoint": OUTPUT.parent, "root": REPO, "scratch": SCRATCH.parent}
    for name, root in roots.items():
        if shutil.disk_usage(root).free < probe.RESERVES[name] + BUDGETS[name]:
            raise OSError(f"Preserve {name} reserve and full panel write allowance")


def validate_predecessor(result, audit, gate):
    if (result.get("complete") is not True or result.get("accepted") is not False
            or result.get("original_rejection_preserved") is not True or result.get("gate_changed") is not False
            or audit.get("complete") is not True or audit.get("training_tasks") != 28
            or audit.get("prediction_grids") != 56 or audit.get("cuda_initialized") is not False
            or gate.get("passed") is not False or gate.get("reason") != "finite_metrics_rejected"
            or gate.get("preflight_only") is not True or gate.get("real_benchmark_success") is not False):
        raise ValueError("Require the complete independently audited rejected original probe")


def register():
    require_cpu()
    if OUTPUT.exists() or OUTPUT.is_symlink() or SCRATCH.exists() or SCRATCH.is_symlink():
        raise FileExistsError("Never replace or restart this fixed panel")
    capacity()
    if sha256(recovery.OUTPUT / "recovery_result.json") != RECOVERY_SHA256:
        raise ValueError("Require the exact completed negative CPU recovery")
    validate_predecessor(*(read(recovery.OUTPUT / name) for name in
        ("recovery_result.json", "audit.json", "gate.json")))
    if read(recovery.OUTPUT / "status.json")["state"] != "complete":
        raise ValueError("The original audit must be terminal")
    recovery_config = read(recovery.OUTPUT / "recovery_config.json")
    verify_hashes(recovery_config["source_sha256"])
    verify_hashes(read(probe.OUTPUT / "config.json")["source_sha256"])
    inputs = dict(recovery_config["input_sha256"])
    inputs.update({str(path): sha256(path) for path in recovery.OUTPUT.iterdir() if path.is_file()})
    inputs[str(probe.PARENT)] = probe.PARENT_SHA256
    verify_hashes(inputs)
    sources = {Path(module.__file__).resolve() for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
        and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file()}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.add(REPO / "tests/test_single_model_shrunk_retention_displacements.py")
    OUTPUT.mkdir()
    SCRATCH.mkdir()
    config = {"policy": POLICY, "fractions": list(FRACTIONS), "repeats": list(REPEATS),
        "source": str(probe.OUTPUT), "scratch": str(SCRATCH), "parent": str(probe.PARENT),
        "input_sha256": inputs, "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
        "reserves": probe.RESERVES, "write_budgets": BUDGETS, "new_seeds": [],
        "checkpoint_selected": False, "scientific_metric_tolerance": 0, "arithmetic_output_ulps": 1}
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
        "registered_unix": time.time(), "model_inference_performed": False})
    probe.status(OUTPUT, "registered")


def verify():
    config = read(OUTPUT / "config.json")
    if (read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["fractions"] != list(FRACTIONS)
            or config["repeats"] != list(REPEATS) or config["scratch"] != str(SCRATCH)
            or config["source"] != str(probe.OUTPUT) or config["write_budgets"] != BUDGETS
            or config["reserves"] != probe.RESERVES):
        raise ValueError("Keep the prospectively registered panel unchanged")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    return config


def vectors(raw, candidate):
    before = raw["snapshot"]["before"]
    if (before.ndim != 1 or before.shape != candidate.shape or before.dtype != torch.float32
            or candidate.dtype != torch.float32 or before.device.type != "cpu" or candidate.device.type != "cpu"
            or not torch.isfinite(before).all() or not torch.isfinite(candidate).all()
            or not torch.equal(before, raw["snapshot"]["after"]) or torch.equal(before, candidate)
            or raw["record"]["accepted"] or raw["record"]["reason"] != "finite_metrics_rejected"):
        raise ValueError("Require the original rolled-back before and distinct separately saved projection")
    return before, candidate


def contract(before, projected, fraction):
    if fraction not in FRACTIONS:
        raise ValueError("Never expand the registered displacement grid")
    if fraction in (0., 1.):
        return (before if fraction == 0. else projected).clone()
    return (before.double() + fraction * (projected.double() - before.double())).float()


def audit_arithmetic(before, projected, candidate, fraction):
    if fraction not in FRACTIONS or candidate.shape != before.shape or candidate.dtype != before.dtype:
        raise ValueError("Require the registered FP32 full vector")
    expected = ((1. - fraction) * before.double() + fraction * projected.double()).float()
    if fraction in (0., 1.):
        expected = before if fraction == 0. else projected
        if not torch.equal(candidate, expected):
            raise ValueError("Control parameter vectors must reproduce bitwise")
    lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
    upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
    if not torch.isfinite(candidate).all() or not ((candidate >= lower) & (candidate <= upper)).all():
        raise ValueError("Independent weighted-state arithmetic exceeded one output ULP")
    return {"parameters": candidate.numel(), "non_bitwise_elements": int((candidate != expected).sum()),
        "allowed_output_ulps": 0 if fraction in (0., 1.) else 1}


def geometry(raw, candidate):
    labels, names, reduction = probe.paired.gradient_layout(raw["record"]["point_before"], 28)
    if len(labels) != 84 or len(names) != 30 or raw["gradients"].shape != (84, 77998496):
        raise ValueError("Preserve all84 raw directions and30 protected directions")
    products, norm = direction_products(raw["gradients"], raw["snapshot"]["before"].double() - candidate.double())
    protected = reduction @ products
    return {"raw_products": products.tolist(), "protected_products": dict(zip(names, protected.tolist())),
        "norm": norm, "decision": displacement_decision(protected, norm, raw["record"]["adam_norm"])}


def finite(before, candidate):
    if set(before) != {str(repeat) for repeat in REPEATS} or set(candidate) != set(before):
        raise ValueError("Both original-seed repetitions remain mandatory")
    decisions = {repeat: probe.paired.metric_decision(before[repeat], candidate[repeat]) for repeat in before}
    if any(row["defined_slots"] != 30 for row in decisions.values()):
        raise ValueError("Retain all30 unchanged point guards")
    return {"repetitions": decisions, "finite_feasible": all(row["accepted"] for row in decisions.values())}


def values_from_rows(rows):
    return {f"{row['task']}/{name}": None if metric is None else metric["value"]
        for row in rows for name, metric in row["metrics"].items()}


def check_prediction(current, reference):
    if current.dtype != reference.dtype or current.shape != reference.shape or not np.array_equal(current, reference):
        raise ValueError("Original controls and same-seed repetitions must reproduce complete grids bitwise")


@torch.no_grad()
def evaluate(index, fraction, repeat, model, criterion, tasks):
    rows = []
    for task in tasks:
        cached = torch.load(task["artifact"], map_location="cpu", weights_only=False)
        episode = probe.training.alignment.mixture.backend.restore_episode(cached["episode"], "cuda")
        torch.manual_seed(task["gradient_seed"])
        with compute_context(episode.source.device):
            prediction = probe.training.streamed_point_predictions(model, criterion, episode.source, episode.mask,
                **probe.training.TRAIN_OPTIONS)
        metrics = probe.training.alignment.metric_summary(prediction, episode, probe.training.point_metric_terms(prediction, episode))
        path = SCRATCH / f"fraction_{index}_repeat_{repeat}_task_{task['index']:02d}.npz"
        array = prediction.cpu().numpy()
        atomic_npz(path, prediction=array)
        if fraction in (0., 1.):
            role = "before" if fraction == 0. else "candidate"
            with np.load(probe.OUTPUT / f"{role}_task_{task['index']:02d}.npz", allow_pickle=False) as original:
                check_prediction(array, original["prediction"])
        if repeat == 1:
            with np.load(SCRATCH / f"fraction_{index}_repeat_0_task_{task['index']:02d}.npz", allow_pickle=False) as first:
                check_prediction(array, first["prediction"])
        rows.append({"task": task["name"], "index": task["index"], "prediction": str(path),
            "sha256": sha256(path), "metrics": metrics})
    report = {"index": index, "fraction": fraction, "repeat": repeat, "rows": rows, "values": values_from_rows(rows)}
    write_json(OUTPUT / f"fraction_{index}_repeat_{repeat}.json", report)
    return report["values"]


@torch.no_grad()
def run():
    verify()
    if read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never restart a started panel")
    capacity()
    probe.wait_for_gpu(OUTPUT, True, False)
    probe.status(OUTPUT, "loading_saved_proposal")
    torch.set_num_threads(4)
    raw = torch.load(probe.OUTPUT / "raw_update.pt", map_location="cpu", weights_only=False, mmap=True)
    before, projected = vectors(raw, torch.load(probe.OUTPUT / "candidate_parameters.pt", map_location="cpu", weights_only=False, mmap=True))
    parent = torch.load(probe.PARENT, map_location="cpu", weights_only=False, mmap=True)["model"]
    tasks = read(probe.OUTPUT / "training_batch/manifest.json")["tasks"]
    if len(tasks) != 28 or before.numel() != 77998496:
        raise ValueError("Never change task or parameter coverage")
    base = read(probe.OUTPUT / "config.json")["base"]
    with probe.training.alignment.mixture.runtime(base, OUTPUT):
        model, optimizer, criterion, teachers, sampler = probe.training.alignment.mixture.backend.setup(base)
        del optimizer, teachers, sampler
        model.load_state_dict(parent, strict=True)
        model.train()
        view = probe.full_network.FullNetworkParameterView(model)
        if ([[name, list(parameter.shape), parameter.numel()] for name, parameter in view.named_parameters()]
                != read(probe.OUTPUT / "decoder_inventory.json")
                or not torch.equal(probe.training.alignment.preflight.flatten(view).cpu(), before)):
            raise ValueError("Require the original full-network parameter inventory and initialization")
        rows, initial = [], None
        for index, fraction in enumerate(FRACTIONS):
            probe.status(OUTPUT, "evaluating_displacement", index=index, fraction=fraction)
            endpoint = contract(before, projected, fraction)
            products = geometry(raw, endpoint)
            probe.training.alignment.preflight.assign(view, endpoint.to("cuda"))
            measured = {str(repeat): evaluate(index, fraction, repeat, model, criterion, tasks) for repeat in REPEATS}
            if not torch.equal(probe.training.alignment.preflight.flatten(view).cpu(), endpoint):
                raise ValueError("Prediction must not mutate parameters")
            path = SCRATCH / f"fraction_{index}_parameters.pt"
            frozen_path = SCRATCH / f"fraction_{index}_buffers.pt"
            torch.save(endpoint, path)
            torch.save({name: value.cpu() for name, value in model.named_buffers()}, frozen_path)
            frozen = probe.full_network.frozen_state_equal(parent, model, view)
            if fraction == 0.:
                initial = measured
            decision = finite(initial, measured)
            if fraction in (0., 1.):
                role = "point_before" if fraction == 0. else "point_after"
                for values in measured.values():
                    compare_nested(values, raw["record"][role])
                if fraction == 1.:
                    compare_nested(decision["repetitions"]["0"], raw["record"]["finite"])
            rows.append({"index": index, "fraction": fraction, "parameters": str(path), "sha256": sha256(path),
                "buffers": str(frozen_path), "buffers_sha256": sha256(frozen_path), "frozen": frozen,
                "geometry": products, "finite": decision,
                "locally_feasible": bool(products["decision"]["admissible"] and decision["finite_feasible"])})
            print(json.dumps({"fraction": fraction, "locally_feasible": rows[-1]["locally_feasible"],
                "finite": decision, "geometry": products["decision"]}), flush=True)
        write_json(OUTPUT / "result.json", {"complete": True, "fractions": rows, "prediction_grids": 336,
            "config_sha256": sha256(OUTPUT / "config.json"), "checkpoint_selected": False,
            "optimizer_updates": 0, "new_data": False, "real_benchmark_success": False})
    probe.status(OUTPUT, "awaiting_independent_audit")


def audit_grids(index, fraction, repeat, tasks, hashes):
    path = OUTPUT / f"fraction_{index}_repeat_{repeat}.json"
    hashes[str(path)] = sha256(path)
    report = read(path)
    if (report["index"], report["fraction"], report["repeat"], len(report["rows"])) != (index, fraction, repeat, 28):
        raise ValueError("Preserve all task/role/repetition slots")
    rows = []
    for row, task in zip(report["rows"], tasks):
        prediction_path = SCRATCH / f"fraction_{index}_repeat_{repeat}_task_{task['index']:02d}.npz"
        if row["task"] != task["name"] or row["index"] != task["index"] or row["prediction"] != str(prediction_path):
            raise ValueError("Prediction identity changed")
        hashes[str(prediction_path)] = row["sha256"]
        cached = torch.load(task["artifact"], map_location="cpu", weights_only=False)
        episode = probe.training.alignment.mixture.backend.restore_episode(cached["episode"], "cpu")
        with np.load(prediction_path, allow_pickle=False) as saved:
            array = saved["prediction"]
        if tuple(array.shape) != tuple(episode.target.shape) or not np.isfinite(array).all():
            raise ValueError("Require a complete finite prediction grid")
        if fraction in (0., 1.):
            role = "before" if fraction == 0. else "candidate"
            with np.load(probe.OUTPUT / f"{role}_task_{task['index']:02d}.npz", allow_pickle=False) as original:
                check_prediction(array, original["prediction"])
        if repeat == 1:
            with np.load(SCRATCH / f"fraction_{index}_repeat_0_task_{task['index']:02d}.npz", allow_pickle=False) as first:
                check_prediction(array, first["prediction"])
        prediction = torch.from_numpy(array)
        metrics = probe.training.alignment.metric_summary(prediction, episode, probe.training.point_metric_terms(prediction, episode))
        compare_nested(metrics, row["metrics"])
        rows.append({**row, "metrics": metrics})
    if values_from_rows(report["rows"]) != report["values"]:
        raise ValueError("Logged metrics must retain exact report identities")
    measured = values_from_rows(rows)
    compare_nested(measured, report["values"])
    return measured, report["values"]


def audit():
    require_cpu()
    verify()
    if read(OUTPUT / "status.json")["state"] != "awaiting_independent_audit" or (OUTPUT / "audit.json").exists():
        raise ValueError("Audit the complete panel only once")
    torch.set_num_threads(4)
    result = read(OUTPUT / "result.json")
    if (result["complete"] is not True or [row["fraction"] for row in result["fractions"]] != list(FRACTIONS)
            or result["checkpoint_selected"] is not False or result["optimizer_updates"] != 0
            or result["prediction_grids"] != 336 or result["config_sha256"] != sha256(OUTPUT / "config.json")):
        raise ValueError("Retain the whole diagnostic and every failure without checkpoint selection")
    raw = torch.load(probe.OUTPUT / "raw_update.pt", map_location="cpu", weights_only=False, mmap=True)
    before, projected = vectors(raw, torch.load(probe.OUTPUT / "candidate_parameters.pt", map_location="cpu", weights_only=False, mmap=True))
    parent = torch.load(probe.PARENT, map_location="cpu", weights_only=False, mmap=True)["model"]
    mutable = {item[0] for item in read(probe.OUTPUT / "decoder_inventory.json")}
    tasks = read(probe.OUTPUT / "training_batch/manifest.json")["tasks"]
    hashes = {str(OUTPUT / "result.json"): sha256(OUTPUT / "result.json")}
    initial, logged_initial, rows = None, None, []
    for index, row in enumerate(result["fractions"]):
        fraction = FRACTIONS[index]
        path, buffer_path = SCRATCH / f"fraction_{index}_parameters.pt", SCRATCH / f"fraction_{index}_buffers.pt"
        if row["index"] != index or row["parameters"] != str(path) or row["buffers"] != str(buffer_path):
            raise ValueError("Keep every registered endpoint identity")
        hashes[str(path)], hashes[str(buffer_path)] = row["sha256"], row["buffers_sha256"]
        endpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        arithmetic = audit_arithmetic(before, projected, endpoint, fraction)
        frozen = torch.load(buffer_path, map_location="cpu", weights_only=False)
        if len(frozen) != 16 or set(frozen) != set(parent) - mutable or row["frozen"]["frozen_tensors"] != 16:
            raise ValueError("Preserve all16 original frozen buffers")
        for name, value in frozen.items():
            if value.dtype != parent[name].dtype or not torch.equal(value, parent[name]):
                raise ValueError("A frozen buffer changed")
        products = geometry(raw, endpoint)
        compare_nested(products, row["geometry"])
        measured, logged = {}, {}
        for repeat in REPEATS:
            measured[str(repeat)], logged[str(repeat)] = audit_grids(index, fraction, repeat, tasks, hashes)
        if index == 0:
            initial, logged_initial = measured, logged
        decision = finite(initial, measured)
        if finite(logged_initial, logged) != row["finite"]:
            raise ValueError("Do not round away exact logged acceptance decisions")
        compare_nested(decision, row["finite"])
        feasible = bool(products["decision"]["admissible"] and decision["finite_feasible"])
        if feasible != row["locally_feasible"]:
            raise ValueError("Independent CPU finite/geometry decision changed")
        rows.append({"fraction": fraction, "arithmetic": arithmetic, "finite": decision, "locally_feasible": feasible})
    verify_hashes(hashes)
    require_cpu()
    write_json(OUTPUT / "audit.json", {"complete": True, "prediction_grids": 336, "training_tasks": 28,
        "parameter_vectors": 6, "gradient_vectors_per_fraction": 84, "protected_directions": 30,
        "frozen_buffers_per_fraction": 16, "cuda_initialized": False, "artifact_sha256": hashes,
        "fractions": rows, "research_selection_tolerance": 0, "full_autograd_replay": False,
        "checkpoint_selected": False, "real_benchmark_success": False})
    lines = ["# Saved-proposal displacement diagnostic", "", POLICY, "",
        "| Displacement fraction | Local feasibility | Worsened protected slots (repeat0) |",
        "|---:|:---:|---:|"]
    for row in rows:
        changes = row["finite"]["repetitions"]["0"]["changes"]
        lines.append(f"| {row['fraction']:g} | {row['locally_feasible']} | {sum(value is not None and value > 0 for value in changes.values())} |")
    lines.extend(["", "Synthetic training-batch diagnosis only; no checkpoint selected or benchmark improvement established."])
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    probe.status(OUTPUT, "complete", locally_feasible_fractions=[row["fraction"] for row in rows if row["locally_feasible"]],
        checkpoint_selected=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "run", "audit"))
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register()
        return
    with (OUTPUT / ".panel.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            {"run": run, "audit": audit}[arguments.stage]()
        except BaseException as error:
            probe.status(OUTPUT, "failed", failed_stage=arguments.stage, error=repr(error))
            raise


if __name__ == "__main__":
    main()
