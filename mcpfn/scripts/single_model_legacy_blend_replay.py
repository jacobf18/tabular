"""Independently verify legacy blend components, observed weights and full targets."""

import inspect
import math

import numpy as np
import torch

import preflight_single_model_structured_teacher as template
import single_model_legacy_blend_teacher as teacher_policy
from single_model_registered_adaptation import replace_once
from tabimpute.routing import fit_routing, observed_scales


POLICY = ("Replay all original training teacher targets in a separate process. For legacy imputation, "
    "independently run both original frozen checkpoints, recompute the observed-source-only convex weight "
    "and preserve both full component grids alongside the final target. The CPU audit independently refits "
    "the weight from those grids and the original observed source, checks all routing fields and both "
    "checkpoint identities, and exactly reconstructs every FP32 target cell including queries. Preserve "
    "the existing single-general structured and single-specialist rating replay. No hidden truth, "
    "inference ensemble, new seed, selection, changed evaluation reference or historical-autograd claim.")
ROUTING_FIELDS = {"evidence_cells", "general_mse", "specialist_mse", "selected", "specialist_weight", "fallback"}
ALLOWANCE = 4 * 1024**2


def expected_role(task):
    specification = task["specification"]
    if specification["domain"] == "recsys":
        if "structured" in specification:
            raise ValueError("Structured teachers apply only to imputation")
        return "specialist"
    if specification["domain"] != "imputation":
        raise ValueError("Unsupported registered training domain")
    return "general" if "structured" in specification else "observed_blend"


def teacher_identity(role, config):
    if role == "observed_blend":
        return {name: config["teacher_identities"][name] for name in ("general", "specialist")}
    return config["teacher_identities"][role]


def routing_record(role, config, decision=None):
    if role == "observed_blend":
        if decision is None or set(decision) != ROUTING_FIELDS:
            raise ValueError("Require every original observed routing field")
        result = dict(decision)
    else:
        if decision is not None:
            raise ValueError("Do not route the unchanged single-teacher tasks")
        result = {"specialist_weight": float(role == "specialist")}
    return {**result, "target": role, "policy": teacher_policy.POLICY,
        "teacher_identity": teacher_identity(role, config)}


def check_routing(cached, task, config):
    role = expected_role(task)
    if teacher_policy.role_for(cached["episode"]["metadata"]) != role:
        raise ValueError("Actual synthetic metadata must match the registered teacher cohort")
    routing = cached["routing"]
    if role != "observed_blend":
        if routing != routing_record(role, config):
            raise ValueError("Preserve the exact original single-teacher callback and identity")
        return role
    if set(routing) != ROUTING_FIELDS | {"target", "policy", "teacher_identity"}:
        raise ValueError("Retain all routing evidence and both original checkpoint identities")
    decision = {name: routing[name] for name in ROUTING_FIELDS}
    source = cached["episode"]["source"]
    observed = int(torch.isfinite(source).sum())
    if (source.ndim != 3 or source.shape[0] != 1 or torch.isinf(source).any()
            or routing != routing_record(role, config, decision)
            or type(decision["evidence_cells"]) is not int or decision["evidence_cells"] != observed
            or not math.isfinite(decision["specialist_weight"]) or not 0 <= decision["specialist_weight"] <= 1):
        raise ValueError("Fit only the actual observed cells and preserve a finite convex weight")
    if observed:
        errors = [decision[name] for name in ("general_mse", "specialist_mse")]
        if (any(value is None or not math.isfinite(value) or value < 0 for value in errors)
                or decision["fallback"] is not None
                or decision["selected"] != ("specialist" if errors[1] < errors[0] else "general")):
            raise ValueError("Preserve finite observed losses and the original tie rule")
    elif decision != {"evidence_cells": 0, "general_mse": None, "specialist_mse": None,
            "selected": "general", "specialist_weight": 0., "fallback": "no evidence"}:
        raise ValueError("Preserve the original no-evidence fallback")
    return role


@torch.no_grad()
def blend_components(source, general, specialist):
    if (source.ndim != 3 or source.shape[0] != 1 or source.dtype != torch.float32
            or any(value.dtype != torch.float32 or value.shape != source.shape
                or not torch.isfinite(value).all() for value in (general, specialist))
            or general.device != specialist.device or torch.isinf(source).any()):
        raise ValueError("Require complete finite FP32 component grids on the same original matrix")
    matrix = source.detach().cpu().numpy()[0]
    decision = fit_routing(matrix, general.detach().cpu().numpy()[0], specialist.detach().cpu().numpy()[0],
        np.isfinite(matrix), observed_scales(matrix))
    weight = decision["specialist_weight"]
    return general * (1. - weight) + specialist * weight, decision


@torch.no_grad()
def replay_prediction(predict, models, role, source, criterion, options):
    if role == "observed_blend":
        specialist = predict(models["specialist"], source, criterion, **options).detach().float()
        general = predict(models["general"], source, criterion, **options).detach().float()
        prediction, decision = blend_components(source, general, specialist)
        arrays = {"general": general.cpu().numpy(), "specialist": specialist.cpu().numpy()}
    else:
        prediction = predict(models[role], source, criterion, **options).detach().float()
        arrays, decision = {}, None
    target = prediction.cpu()
    return target, {"teacher": target.numpy(), **arrays}, decision


def verify_archive(saved, cached, task, config):
    role = check_routing(cached, task, config)
    expected = {"teacher", "general", "specialist"} if role == "observed_blend" else {"teacher"}
    if (set(saved) != expected or any(saved[name].dtype != np.float32
            or saved[name].shape != tuple(cached["teacher"].shape) or not np.isfinite(saved[name]).all()
            for name in expected) or not np.array_equal(saved["teacher"], cached["teacher"].numpy())):
        raise ValueError("Preserve every full FP32 target and all required independent component grids")
    if role == "observed_blend":
        prediction, decision = blend_components(cached["episode"]["source"],
            torch.from_numpy(saved["general"]), torch.from_numpy(saved["specialist"]))
        if (routing_record(role, config, decision) != cached["routing"]
                or not torch.equal(prediction, cached["teacher"])):
            raise ValueError("Independently refitted observed weights must exactly reconstruct the cached target")
    return 2 if role == "observed_blend" else 0


def storage_accounting(batch):
    tasks = [task for task in batch["tasks"] if expected_role(task) == "observed_blend"]
    if len(tasks) != 2:
        raise ValueError("Keep both original legacy draws in the28-task probe")
    elements = 0
    for task in tasks:
        shape = task["specification"]["shape"]
        if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("Budget both full component grids from fixed positive dimensions")
        elements += math.prod(shape)
    arrays = 2 * 4 * elements
    if arrays + 1024**2 > ALLOWANCE:
        raise ValueError("The complete raw components and serialization must fit the registered extra allowance")
    return {"legacy_tasks": 2, "component_grids": 4, "array_bytes": arrays,
        "serialization_allowance_bytes": 1024**2, "reserved_checkpoint_bytes": ALLOWANCE}


GPU_SOURCE = inspect.getsource(template.replay_targets)
GPU_SOURCE = replace_once(GPU_SOURCE, "def replay_targets(output):", "def replay_admitted(output):")
GPU_SOURCE = replace_once(GPU_SOURCE,
    '            prediction = teacher_predictions(models[role], episode.source, criterion, **teacher_policy.STREAM_OPTIONS)\n'
    '        target = prediction.detach().float().cpu()',
    '            target, arrays, decision = blend_replay.replay_prediction(teacher_predictions, models, role,\n'
    '                episode.source, criterion, teacher_policy.STREAM_OPTIONS)\n'
    '        if blend_replay.routing_record(role, config, decision) != cached["routing"]:\n'
    '            raise ValueError("Independent observed blend routing differs from the actual training target")')
GPU_SOURCE = replace_once(GPU_SOURCE, 'atomic_npz(path, teacher=target.numpy())', 'atomic_npz(path, **arrays)')
GPU_SOURCE = replace_once(GPU_SOURCE, '"identity": config["teacher_identities"][role]',
    '"identity": blend_replay.teacher_identity(role, config)')
GPU_SOURCE = replace_once(GPU_SOURCE, '"student_or_validation_evaluated": False}',
    '"student_or_validation_evaluated": False, "blend_replay_policy": blend_replay.POLICY,\n'
    '               "blend_tasks": sum(row["role"] == "observed_blend" for row in rows),\n'
    '               "component_grids": 2 * sum(row["role"] == "observed_blend" for row in rows)}')

AUDIT_SOURCE = inspect.getsource(template.audit_teacher_replay)
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    '    hashes = {str(path): sha256(path),',
    '    blend_tasks = sum(expected_role(task) == "observed_blend" for task in config["batch"]["tasks"])\n'
    '    if (report["blend_replay_policy"] != blend_replay.POLICY or report["blend_tasks"] != blend_tasks\n'
    '            or report["component_grids"] != 2 * blend_tasks):\n'
    '        raise ValueError("Require every independently replayed legacy component grid")\n'
    '    components = 0\n'
    '    hashes = {str(path): sha256(path),')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE, 'row["identity"] != config["teacher_identities"][role]',
    'row["identity"] != blend_replay.teacher_identity(role, config)')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    '            if set(saved.files) != {"teacher"} or not np.array_equal(saved["teacher"], cached["teacher"].numpy()):\n'
    '                raise ValueError("Stored independent teacher grids differ from actual cached training targets")',
    '            components += blend_replay.verify_archive({name: saved[name] for name in saved.files}, cached, task, config)')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    '    return {"complete": True, "full_grids": 28, "bitwise_equal": True, "artifact_sha256": hashes}',
    '    if components != 2 * blend_tasks:\n'
    '        raise ValueError("Never omit an independently checked component grid")\n'
    '    return {"complete": True, "full_grids": 28, "bitwise_equal": True, "artifact_sha256": hashes,\n'
    '        "blend_tasks": blend_tasks, "component_grids": components, "observed_weights_refitted": True}')
