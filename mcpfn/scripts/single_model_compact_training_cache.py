"""Retain exact current-batch payloads in memory and small durable replay capsules."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

import torch

from probe_single_model_regenerated_training import content_digest
from recsys_evaluation_common import sha256
import single_model_compact_rating_replay as compact
import single_model_replicated_decoder_batch as replicated
from single_model_registered_adaptation import replace_once


POLICY = ("Keep only the current batch's complete original CPU episode/teacher payloads in memory for "
          "unchanged finite-point evaluations. Durably retain each task specification, typed exact full-payload "
          "digest and only the four original rating-normalization vectors, not matrix/teacher grids. The "
          "ordinary sampler and student composite/point backward calls are unchanged. Independently regenerate "
          "every new task and teacher with its compact state and require exact digest equality before any "
          "qualification. Never substitute seed-only ratings, relax equality, reinterpret old caches or "
          "claim full optimizer-trajectory replay. Release the previous batch before capturing the next.")
FORMAT = "complete-task-digest-and-rating-vectors-v1"
ACTIVE = ContextVar("single_model_compact_training_cache", default=None)


@contextmanager
def runtime():
    if ACTIVE.get() is not None:
        raise ValueError("Never nest a compact training cache")
    current = {}
    token = ACTIVE.set(current)
    try:
        yield current
    finally:
        current.clear()
        ACTIVE.reset(token)


def capsule_for(payload):
    if set(payload) != {"episode", "teacher", "routing", "task"}:
        raise ValueError("Bind the entire original task payload, not selected grids")
    scaling = payload["episode"]["scaling"]
    if scaling is not None:
        compact.validate_scaling(scaling)
    return {"format": FORMAT, "task": deepcopy(payload["task"]), "content_sha256": content_digest(payload),
            "scaling": deepcopy(scaling)}


def validate_capsule(capsule):
    if (set(capsule) != {"format", "task", "content_sha256", "scaling"} or capsule["format"] != FORMAT
            or not isinstance(capsule["task"], dict) or not isinstance(capsule["content_sha256"], str)
            or len(capsule["content_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in capsule["content_sha256"])):
        raise ValueError("Require an exact typed task digest and compact-only replay state")
    if capsule["scaling"] is not None:
        compact.validate_scaling(capsule["scaling"])


def persist_task(payload, path):
    current = ACTIVE.get()
    path = Path(path)
    if current is None:
        raise ValueError("Capture complete task payloads only inside the bounded memory cache")
    if str(path) in current or path.exists() or path.is_symlink():
        raise FileExistsError("Never replace a compact task or capture one task twice")
    capsule = capsule_for(payload)
    digest = replicated.original.save_tensor(capsule, path)
    current[str(path)] = {"payload": payload, "content_sha256": capsule["content_sha256"], "sha256": digest}
    return digest


def load_current(record):
    current = ACTIVE.get()
    if current is None or record["artifact"] not in current:
        raise ValueError("Finite-point acceptance must use the current batch's original complete payload")
    item = current[record["artifact"]]
    if (record["sha256"] != item["sha256"] or sha256(record["artifact"]) != item["sha256"]
            or content_digest(item["payload"]) != item["content_sha256"]):
        raise ValueError("Current finite-point payload and durable capsule must remain unchanged")
    return item["payload"]


def regenerate(capsule, sampler, teachers, criterion, backend):
    validate_capsule(capsule)
    with compact.runtime(capsule["scaling"]):
        episode = sampler.sample(step=1, **capsule["task"]["specification"])
    teacher, routing = backend.teacher_target(teachers, criterion, episode)
    payload = {"episode": backend.cache_episode(episode), "teacher": teacher.detach().float().cpu(),
               "routing": routing, "task": deepcopy(capsule["task"])}
    if content_digest(payload) != capsule["content_sha256"]:
        raise ValueError("Independent complete task/teacher regeneration must match every typed byte")
    return payload


CAPTURE_SOURCE = replace_once(replicated.CAPTURE_SOURCE,
                             '        digest = save_tensor({"episode":',
                             '        digest = persist_task({"episode":')
CAPTURE_SOURCE_SHA256 = hashlib.sha256(CAPTURE_SOURCE.encode()).hexdigest()
_capture_namespace = {**replicated._capture.__globals__, "persist_task": persist_task}
exec(compile(CAPTURE_SOURCE, __file__, "exec"), _capture_namespace)
_capture = _capture_namespace["capture_batch"]

EVALUATE_SOURCE = replace_once(inspect.getsource(replicated.original.evaluate_training),
    'cached = torch.load(record["artifact"], map_location="cpu", weights_only=False)',
    'cached = load_current(record)')
EVALUATE_SOURCE_SHA256 = hashlib.sha256(EVALUATE_SOURCE.encode()).hexdigest()
_evaluate_namespace = {**replicated.original.evaluate_training.__globals__, "load_current": load_current}
exec(compile(EVALUATE_SOURCE, __file__, "exec"), _evaluate_namespace)
evaluate_training = _evaluate_namespace["evaluate_training"]


def capture_batch(output, directory, batch, model, optimizer, criterion, teachers, sampler, gradients):
    tasks, current = batch["tasks"], ACTIVE.get()
    if current is None:
        raise ValueError("Capture requires a scoped compact training cache")
    if (not tasks or [task["index"] for task in tasks] != list(range(len(tasks)))
            or len({task["name"] for task in tasks}) != len(tasks)
            or gradients.ndim != 2 or gradients.shape[0] < 3 * len(tasks)):
        raise ValueError("Reserve every original task and possible composite/point gradient")
    current.clear()
    return _capture(output, directory, batch, model, optimizer, criterion, teachers, sampler, gradients)
