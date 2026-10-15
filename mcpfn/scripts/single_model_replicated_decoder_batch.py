"""Replicate fresh training draws without changing the original domain masses or capture implementation."""

from copy import deepcopy
import hashlib
import inspect
import math

from single_model_registered_adaptation import replace_once
import train_single_matrixformer_finite_guarded as original


REPLICATES = 2
CAPTURE_SOURCE = replace_once(inspect.getsource(original.capture_batch),
                              'entries = [None] * 14', 'entries = [None] * len(batch["tasks"])')
CAPTURE_SOURCE_SHA256 = hashlib.sha256(CAPTURE_SOURCE.encode()).hexdigest()
_capture_namespace = dict(original.capture_batch.__globals__)
exec(compile(CAPTURE_SOURCE, __file__, "exec"), _capture_namespace)
_capture = _capture_namespace["capture_batch"]


def replicate_batch(batch, data_seed, gradient_seed):
    tasks = batch["tasks"]
    if (type(batch["step"]) is not int or batch["step"] < 1 or len(tasks) != 14
            or [task["index"] for task in tasks] != list(range(14))
            or len({task["name"] for task in tasks}) != 14
            or any(not math.isfinite(task["weight"]) or task["weight"] <= 0 for task in tasks)
            or not math.isclose(math.fsum(task["weight"] for task in tasks), 1., rel_tol=0, abs_tol=1e-15)):
        raise ValueError("Require the complete ordered original fourteen-cohort batch and positive unit mass")
    if any(type(seed) is not int or seed < 0 for seed in (data_seed, gradient_seed)):
        raise ValueError("Require explicit nonnegative integer seed streams")
    result = {"step": batch["step"], "tasks": []}
    for replicate in range(REPLICATES):
        for source in tasks:
            task = deepcopy(source)
            index = len(result["tasks"])
            offset = ((batch["step"] - 1) * REPLICATES * len(tasks) + index) * 100
            task.update(index=index, name=f'{source["name"]}_draw_{replicate}',
                        cohort=source["name"], replicate=replicate,
                        weight=source["weight"] / REPLICATES, gradient_seed=gradient_seed + offset)
            task["specification"].get("structured", task["specification"])["seed"] = data_seed + offset
            result["tasks"].append(task)
    seeds = original.alignment.fixed.seeds_in(result)
    if len(seeds) != 2 * REPLICATES * len(tasks):
        raise ValueError("Every replicated data/backward seed must be distinct")
    return result


def capture_batch(output, directory, batch, model, optimizer, criterion, teachers, sampler, gradients):
    tasks = batch["tasks"]
    if (not tasks or [task["index"] for task in tasks] != list(range(len(tasks)))
            or len({task["name"] for task in tasks}) != len(tasks)
            or gradients.ndim != 2 or gradients.shape[0] < 3 * len(tasks)):
        raise ValueError("Reserve distinct task slots and every possible composite/point gradient")
    return _capture(output, directory, batch, model, optimizer, criterion, teachers, sampler, gradients)
