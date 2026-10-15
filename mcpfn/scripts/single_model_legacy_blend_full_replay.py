"""Preserve and independently reconstruct every full-trial legacy blend target."""

from copy import deepcopy
import json
from math import prod
import os
from pathlib import Path

import numpy as np
import torch

from dwip_teacher_mean_training import compute_context, verify_hashes
from probe_single_model_regenerated_training import content_digest
from recsys_evaluation_common import atomic_npz, sha256, write_json
from train_streamed_large_matrices import append_json
from train_single_matrixformer_parent_teacher import streamed_point_predictions as teacher_predictions
import single_model_legacy_blend_replay as replay
import single_model_legacy_blend_teacher as teacher_policy
import single_model_compact_training_cache as compact_cache


POLICY = ("During the separate complete training audit, check every admitted task's original role and full "
    "typed payload digest. For every legacy task, independently run both original frozen teachers again, "
    "refit the original observed-only convex weight and require the exact cached full target and routing. "
    "Archive source, target and both raw component grids without query truth. Reopen every archive and "
    "independently refit/reconstruct using CPU arithmetic; repeat that saved-grid check during qualification. "
    "Preserve all legacy components, queries, roles, original checkpoint identities and adverse outcomes. "
    "This additional CPU computation occurs within the independent GPU audit, not a separate CPU-only "
    "process. The original complete task/teacher, rating-context, reference and endpoint audits remain "
    "mandatory. No hidden-truth routing, inference ensemble, training, selection or full-autograd claim.")


def layout(scratch):
    return Path(scratch) / "legacy_blend_teacher_components"


def storage_accounting(batches):
    areas = []
    if len(batches) != 64:
        raise ValueError("Budget the original64 attempts before training")
    for batch in batches:
        if len(batch["tasks"]) != 28:
            raise ValueError("Preserve every original28-task batch")
        selected = [task for task in batch["tasks"] if replay.expected_role(task) == "observed_blend"]
        if len(selected) != 2:
            raise ValueError("Budget both legacy draws in every attempt")
        for task in selected:
            shape = task["specification"]["shape"]
            if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
                raise ValueError("Account for complete fixed positive source and component dimensions")
            areas.append(prod(shape))
    return {"legacy_tasks": len(areas), "raw_component_grids": 2 * len(areas), "cells": sum(areas),
        "array_bytes": 16 * sum(areas), "atomic_archive_bytes": 16 * max(areas),
        "serialization_and_manifest_allowance": 8 * 1024**2}


def storage_bytes(batches):
    accounting = storage_accounting(batches)
    return sum(accounting[name] for name in ("array_bytes", "atomic_archive_bytes", "serialization_and_manifest_allowance"))


def specification(scratch, batches):
    return {"policy": POLICY, "directory": str(layout(scratch)), "storage": storage_accounting(batches),
        "provider_persistent": False}


def validate_config(training, config):
    if (config["legacy_blend_replay"] != specification(training.SCRATCH, config["batches"])
            or config["teacher_policy"] != teacher_policy.POLICY):
        raise ValueError("Require the separately registered legacy-blend policy and full archive budget")


def tasks_for(training, config, counts):
    steps = counts["admitted_steps"]
    if (steps != sorted(set(steps)) or any(type(step) is not int or step not in range(1, 65) for step in steps)
            or counts["training_tasks"] != 28 * len(steps)):
        raise ValueError("Retain every admitted task and exclude only independently verified data-only attempts")
    tasks = {}
    for step in steps:
        batch = config["batches"][step - 1]
        if batch["step"] != step or [task["index"] for task in batch["tasks"]] != list(range(28)):
            raise ValueError("Preserve the original complete task order for every admitted step")
        for task in batch["tasks"]:
            path = training.OUTPUT / "training_batches" / f"step_{step:03d}" / f'task_{task["index"]:02d}.pt'
            tasks[str(path)] = (step, task)
    return tasks


def verify_records(training, config, counts, rows):
    validate_config(training, config)
    tasks = tasks_for(training, config, counts)
    if len(rows) != len(tasks) or [row["task"] for row in rows] != list(tasks):
        raise ValueError("Reopen all admitted tasks in original order, never only favorable legacy targets")
    hashes, legacy_tasks = {}, 0
    for row in rows:
        path = Path(row["task"])
        step, task = tasks[str(path)]
        capsule = torch.load(path, map_location="cpu", weights_only=True)
        compact_cache.validate_capsule(capsule)
        role = replay.expected_role(task)
        if (row["task_sha256"] != sha256(path) or capsule["task"] != task
                or row["content_sha256"] != capsule["content_sha256"] or row["role"] != role
                or row["complete_payload_checked"] is not True):
            raise ValueError("Bind the complete original capsule and teacher role before checking raw components")
        hashes[str(path)] = row["task_sha256"]
        extra_fields = {"archive", "archive_sha256", "metadata", "routing", "source_content_sha256", "teacher_content_sha256"}
        if role != "observed_blend":
            if extra_fields.intersection(row):
                raise ValueError("Keep unchanged structured and rating teachers outside blend routing")
            continue
        if not extra_fields.issubset(row):
            raise ValueError("Every legacy task must retain both raw components, source, target and routing")
        archive = layout(training.SCRATCH) / f"step_{step:03d}_task_{task['index']:02d}.npz"
        if row["archive"] != str(archive) or row["archive_sha256"] != sha256(archive):
            raise ValueError("Preserve each complete independent component archive")
        with np.load(archive, allow_pickle=False) as saved:
            if set(saved.files) != {"source", "teacher", "general", "specialist"}:
                raise ValueError("Archive exactly the observed source and all three full prediction grids")
            source, target = torch.from_numpy(saved["source"]), torch.from_numpy(saved["teacher"])
            if (content_digest(source) != row["source_content_sha256"]
                    or content_digest(target) != row["teacher_content_sha256"]):
                raise ValueError("Preserve the actual regenerated source and target, including every query location")
            cached = {"episode": {"source": source, "metadata": row["metadata"]},
                "teacher": target, "routing": row["routing"]}
            components = replay.verify_archive({name: saved[name] for name in ("teacher", "general", "specialist")},
                cached, task, config)
            if components != 2:
                raise ValueError("Independently check both original legacy components")
        hashes[str(archive)] = row["archive_sha256"]
        legacy_tasks += 1
    if legacy_tasks != 2 * len(counts["admitted_steps"]):
        raise ValueError("Never omit a legacy draw from an admitted batch")
    return {"training_tasks_checked": len(tasks), "legacy_tasks": legacy_tasks,
        "raw_component_grids": 2 * legacy_tasks, "observed_weights_refitted_on_cpu": True,
        "artifact_sha256": hashes}


class ComponentReplay:
    def __init__(self, training, config, counts, teachers, criterion):
        validate_config(training, config)
        self.training, self.config, self.counts, self.criterion = training, config, counts, criterion
        self.tasks, self.seen = tasks_for(training, config, counts), set()
        self.models = {"general": teachers[0], "specialist": teachers[1]}
        self.states = {role: teacher_policy.verify_model(model, config["teacher_identities"][role])
            for role, model in self.models.items()}
        self.output = layout(training.SCRATCH)
        self.output.mkdir()
        self.rows = []

    @torch.no_grad()
    def check(self, path, payload):
        key = str(path)
        if key not in self.tasks or key in self.seen:
            raise ValueError("Check every original admitted task once inside its independent audit")
        step, task = self.tasks[key]
        capsule = torch.load(path, map_location="cpu", weights_only=True)
        compact_cache.validate_capsule(capsule)
        if (payload["task"] != task or capsule["task"] != task
                or capsule["content_sha256"] != content_digest(payload)):
            raise ValueError("The regenerated complete payload must match the actual training capsule")
        role = replay.check_routing(payload, task, self.config)
        record = {"task": key, "task_sha256": sha256(path), "content_sha256": capsule["content_sha256"],
            "role": role, "complete_payload_checked": True}
        if role == "observed_blend":
            source = payload["episode"]["source"]
            device = next(self.models["general"].parameters()).device
            with compute_context(device):
                target, arrays, decision = replay.replay_prediction(teacher_predictions, self.models, role,
                    source.to(device), self.criterion, teacher_policy.STREAM_OPTIONS)
            if (not torch.equal(target, payload["teacher"])
                    or replay.routing_record(role, self.config, decision) != payload["routing"]):
                raise ValueError("Independent original-checkpoint replay must reproduce the full actual blend")
            replay.verify_archive(arrays, payload, task, self.config)
            path = self.output / f"step_{step:03d}_task_{task['index']:02d}.npz"
            atomic_npz(path, source=source.cpu().numpy(), **arrays)
            record.update(archive=str(path), archive_sha256=sha256(path), metadata=deepcopy(payload["episode"]["metadata"]),
                routing=deepcopy(payload["routing"]), source_content_sha256=content_digest(source),
                teacher_content_sha256=content_digest(payload["teacher"]))
        self.rows.append(record)
        append_json(self.output / "tasks.jsonl", record)
        self.seen.add(key)

    def finish(self):
        if self.seen != set(self.tasks) or (self.output / "summary.json").exists():
            raise ValueError("Finish exactly one complete admitted-task/component audit")
        for role, model in self.models.items():
            if teacher_policy.verify_model(model, self.config["teacher_identities"][role]) != self.states[role]:
                raise ValueError("Both independent original teacher checkpoints must remain frozen")
        if not self.rows:
            (self.output / "tasks.jsonl").touch(exist_ok=False)
        rows = [json.loads(line) for line in (self.output / "tasks.jsonl").read_text().splitlines()]
        evidence = verify_records(self.training, self.config, self.counts, rows)
        evidence["artifact_sha256"].update({str(self.output / "tasks.jsonl"): sha256(self.output / "tasks.jsonl"),
            str(self.training.OUTPUT / "config.json"): sha256(self.training.OUTPUT / "config.json"),
            **{identity["checkpoint"]: identity["sha256"] for role, identity in self.config["teacher_identities"].items()
                if role in self.models}})
        verify_hashes(evidence["artifact_sha256"])
        write_json(self.output / "summary.json", {**evidence, "complete": True, "policy": POLICY,
            "training_config_sha256": sha256(self.training.OUTPUT / "config.json"), "teacher_states": self.states,
            "pid": os.getpid(), "cuda_initialized": torch.cuda.is_initialized(), "separate_cpu_only_process": False,
            "query_truth_used_for_routing": False, "training_performed": False, "inference_ensemble": False,
            "historical_full_autograd_replayed": False, "goal_success_not_assessed": True})
        return {str(self.output / name): sha256(self.output / name) for name in ("summary.json", "tasks.jsonl")}


def require_report(training, audit):
    config = training.read(training.OUTPUT / "config.json")
    output = layout(training.SCRATCH)
    report = training.read(output / "summary.json")
    rows = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
    if (report["complete"] is not True or report["policy"] != POLICY
            or report["training_config_sha256"] != sha256(training.OUTPUT / "config.json")
            or report["observed_weights_refitted_on_cpu"] is not True
            or any(report[name] is not False for name in ("separate_cpu_only_process", "query_truth_used_for_routing",
                "training_performed", "inference_ensemble", "historical_full_autograd_replayed"))
            or set(report["teacher_states"]) != {"general", "specialist"}
            or any(audit["artifact_sha256"].get(str(output / name)) != sha256(output / name)
                for name in ("summary.json", "tasks.jsonl"))):
        raise ValueError("Require the complete independently hash-bound original-component replay before qualification")
    for role, state in report["teacher_states"].items():
        identity = config["teacher_identities"][role]
        if any(state[name] != identity[name] for name in ("checkpoint", "sha256")) or not state["bitwise_equal"] or state["state_tensors"] < 1:
            raise ValueError("Retain both original verified frozen teacher identities")
    evidence = verify_records(training, config, audit["admission_attempts"], rows)
    for name in ("training_tasks_checked", "legacy_tasks", "raw_component_grids", "observed_weights_refitted_on_cpu"):
        if report[name] != evidence[name]:
            raise ValueError("Derive full component coverage from the actual independently admitted attempts")
    if any(report["artifact_sha256"].get(path) != digest for path, digest in evidence["artifact_sha256"].items()):
        raise ValueError("Bind every reopened source and raw component to the independent replay report")
    verify_hashes(report["artifact_sha256"])
    return report
