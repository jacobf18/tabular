"""Add training-only native-anchor retention without replacing any teacher or point loss."""

from contextlib import contextmanager
from functools import partial
import hashlib
import inspect
from pathlib import Path

import torch

from dwip_teacher_mean_training import compute_context
from recsys_evaluation_common import sha256, write_json
import single_model_query_objective as query
from single_model_query_objective import column, query_budget_majorizer, TRAIN_OPTIONS, streamed_training_loss
from single_model_registered_adaptation import replace_once
from single_model_structured_teacher import verify_model
from streamed_large_matrix_training import STREAM_OPTIONS, atomic_torch_save
from tabimpute.streamed_inference import streamed_point_predictions


REFERENCE_SHA256 = "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
SPECIFICATION = {"weight": 10., "loss": "mean squared normalized prediction difference over every original query",
    "legacy_components": [6, 7], "structured_component": "anchor_latent_mcar",
    "reference_sha256": REFERENCE_SHA256, "new_normalization": False,
    "changes_original_teacher": False, "changes_point_metrics": False, "inference_retention_teacher": False}
POLICY = ("On every freshly generated native-anchor imputation task, add weight10 times the original-query "
    "mean squared difference between student and frozen V29 initializer normalized point predictions. "
    "Eligibility uses only the existing generator component, including both legacy anchor variants and "
    "every structured missingness family, never observed failure columns, query counts, errors or thresholds. "
    "Keep original general/specialist targets, Huber/NLL/CDF/query-majorizer terms, true-target point metrics, "
    "masks, column denominators and normalization unchanged. Spectral imputation and ratings retain their "
    "original backward callback exactly. The additional reference is frozen, detached and training-only; "
    "ordinary single-checkpoint inference is unchanged. Preserve complete reference grids and all task "
    "eligibility decisions for independent replay. No validation, selection, real data or gradient seed changes.")


def eligible(metadata):
    domain, structured = metadata.get("domain"), metadata.get("structured", False)
    if (metadata.get("gradient_data") != "synthetic" or domain not in ("imputation", "recsys")
            or type(structured) is not bool or type(metadata.get("seed")) is not int or metadata["seed"] < 0):
        raise ValueError("Require an explicitly seeded synthetic task and original domain/structured identity")
    if domain == "recsys":
        if structured:
            raise ValueError("Never relabel ratings as structured imputation")
        return False
    component = metadata.get("component")
    if structured:
        if component not in ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar"):
            raise ValueError("Require an original structured generator identity")
        return component == "anchor_latent_mcar"
    if type(component) is not int or component not in range(8):
        raise ValueError("Require an original legacy generator index")
    if component in (6, 7):
        expected = {6: "anchor_latent_mcar", 7: "anchor_latent_profile_mixed"}[component]
        if metadata.get("prior") != expected:
            raise ValueError("Native legacy index and prior name must agree")
        return True
    return False


def validate_target(episode, reference):
    if (not eligible(episode.metadata) or not isinstance(reference, torch.Tensor)
            or reference.shape != episode.target.shape or reference.device != episode.target.device
            or not reference.is_floating_point() or episode.mask.dtype != torch.bool
            or episode.mask.shape != episode.target.shape or not episode.mask.any()
            or not torch.isfinite(reference[episode.mask]).all()):
        raise ValueError("Require the full finite frozen reference on every original native-anchor query")


BACKWARD_SOURCE = inspect.getsource(query.backward_episode)
BACKWARD_SOURCE = replace_once(BACKWARD_SOURCE, "def backward_episode(", "def _backward_episode(")
BACKWARD_SOURCE = replace_once(BACKWARD_SOURCE, "*, query_seed_offset,", "*, query_seed_offset, retention_target,")
BACKWARD_SOURCE = replace_once(BACKWARD_SOURCE,
    '    if episode.metadata.get("gradient_data") != "synthetic":',
    '    validate_target(episode, retention_target)\n'
    '    retention_values = retention_target.detach().float().reshape(-1)\n'
    '    if episode.metadata.get("gradient_data") != "synthetic":')
BACKWARD_SOURCE = replace_once(BACKWARD_SOURCE,
    '        return original + column.COLUMN_OBJECTIVE["weight"] * cell_weights[lookup] * (point - truth.float()).square()',
    '        original = original + column.COLUMN_OBJECTIVE["weight"] * cell_weights[lookup] * (point - truth.float()).square()\n'
    '        return original + SPECIFICATION["weight"] * (point - retention_values[lookup]).square()')
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, "exec"), globals())


@contextmanager
def runtime(backend, directory, reference, identity, *, query_seed_offset):
    directory = Path(directory)
    original = backend.backward_episode
    if (not isinstance(original, partial) or original.func is not query.backward_episode
            or original.args or original.keywords != {"query_seed_offset": query_seed_offset}
            or type(query_seed_offset) is not int or query_seed_offset < 0
            or identity.get("sha256") != REFERENCE_SHA256):
        raise ValueError("Bind the exact original query objective and declared frozen V29 reference")
    verified = verify_model(reference, identity)
    directory.mkdir()
    records, seeds = [], set()
    complete = False

    def backward(student, criterion, episode, teacher, weight):
        selected = eligible(episode.metadata)
        seed = episode.metadata["seed"]
        if seed in seeds:
            raise ValueError("Never reuse a synthetic retention task or replace its archived reference")
        seeds.add(seed)
        record = {"seed": seed, "eligible": selected, "shape": list(episode.target.shape),
                  "queries": int(episode.mask.sum()), "metadata": dict(episode.metadata)}
        if not selected:
            value = original(student, criterion, episode, teacher, weight)
        else:
            if (student is reference or {value.data_ptr() for value in student.parameters()}.intersection(
                    value.data_ptr() for value in reference.parameters())):
                raise ValueError("The trainable student must not alias its frozen retention reference")
            with torch.no_grad(), compute_context(episode.source.device):
                prediction = streamed_point_predictions(reference, episode.source, criterion, **STREAM_OPTIONS).detach()
            validate_target(episode, prediction)
            path = directory / f"reference_{seed}.pt"
            atomic_torch_save({"prediction": prediction.float().cpu(), "source": episode.source.detach().cpu(),
                "mask": episode.mask.detach().cpu(), "identity": identity, "specification": SPECIFICATION,
                "metadata": dict(episode.metadata)}, path)
            record.update(artifact=str(path), sha256=sha256(path))
            records.append(record)
            value = _backward_episode(student, criterion, episode, teacher, weight,
                query_seed_offset=query_seed_offset, retention_target=prediction)
        if not selected:
            records.append(record)
        return value

    backend.backward_episode = backward
    try:
        yield
        if verify_model(reference, identity) != verified:
            raise ValueError("The additional reference must remain frozen and bitwise unchanged")
        complete = True
    finally:
        backend.backward_episode = original
        write_json(directory / "manifest.json", {"complete": complete, "policy": POLICY,
            "specification": SPECIFICATION, "reference": verified, "records": records,
            "backward_source_sha256": BACKWARD_SOURCE_SHA256, "inference_ensemble": False})
