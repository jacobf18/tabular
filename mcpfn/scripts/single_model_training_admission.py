"""Inspect structured training draws before any teacher, student or optimizer work."""

from copy import deepcopy
import math

import torch

from probe_single_model_regenerated_training import content_digest
from single_model_distillation import Episode


POLICY = ("Training-only batch admission, never a held-out sampler. Draw every scheduled structured episode "
    "once with its original seed, shape, mechanism and observed-only normalization. Preserve each complete "
    "raw-array digest, normalized episode digest, specification and actual support counts. If any structured "
    "episode has no queries or no observations, reject the entire batch before teacher/student inference or "
    "an optimizer proposal; log a data-admission no-op, not a finite-metric rejection or accepted update. "
    "Do not redraw, repair a mask, resize, replace a case, redistribute weights, fabricate a loss/metric or "
    "claim deferred legacy/rating tasks were generated. For admitted batches, reuse the same structured "
    "episodes exactly once and delegate legacy/ratings sampling unchanged. Unexpected exceptions still fail. "
    "A future trainer must explicitly account for rejected attempts and data-only evidence in its independent "
    "audit. This helper alone registers no experiment and performs no training, selection or qualification.")


def episode_digest(episode):
    if episode.ratings is not None:
        raise ValueError("Structured admission must not reinterpret ratings contexts")
    return content_digest({"source": episode.source, "target": episode.target,
                           "mask": episode.mask, "metadata": episode.metadata})


def validate_draw(episode, arrays, specification):
    source, target, query = episode.source, episode.target, episode.mask
    if (source.ndim != 3 or source.shape[0] != 1 or source.shape != target.shape or source.shape != query.shape
            or source.numel() == 0 or query.dtype != torch.bool or source.device.type != "cpu"
            or target.device.type != "cpu" or query.device.type != "cpu"
            or not torch.isfinite(target).all() or torch.isinf(source).any()
            or not torch.equal(torch.isnan(source), query)
            or not torch.equal(arrays["query"], query[0])):
        raise ValueError("Require complete original CPU structured draws, not repaired or malformed arrays")
    metadata = episode.metadata
    observations, queries = int((~query).sum()), int(query.sum())
    eligible = observations > 0 and queries > 0
    if (metadata.get("gradient_data") != "synthetic" or metadata.get("domain") != "imputation"
            or metadata.get("seed") != specification["seed"]
            or metadata.get("family") != specification["family"]
            or metadata.get("component") != specification["component"]
            or metadata.get("requested_shape") != specification["shape"]
            or metadata.get("regime") != specification["regime"]
            or metadata.get("requested_missingness") != specification["requested_missingness"]
            or metadata.get("usable_for_training") is not eligible
            or metadata.get("queries") != queries or metadata.get("observations") != observations
            or (metadata.get("rows"), metadata.get("columns")) != tuple(source.shape[1:])):
        raise ValueError("Bind actual support and full original task identity, including intrinsic latent shapes")
    return (["no_queries"] if queries == 0 else []) + (["no_observations"] if observations == 0 else [])


class TrainingBatchAdmission:
    def __init__(self, sampler, batch):
        tasks = batch["tasks"]
        if (type(batch["step"]) is not int or batch["step"] < 1 or not tasks
                or [task["index"] for task in tasks] != list(range(len(tasks)))
                or len({task["name"] for task in tasks}) != len(tasks)
                or any(not math.isfinite(task["weight"]) or task["weight"] <= 0 for task in tasks)
                or not math.isclose(math.fsum(task["weight"] for task in tasks), 1., rel_tol=0, abs_tol=1e-15)):
            raise ValueError("Require the complete ordered positively weighted registered batch")
        self._sampler = sampler
        self._episodes, self._specifications, self._digests = {}, {}, {}
        self._delivered = set()
        rows, rejected, deferred = [], [], []
        for task in deepcopy(tasks):
            specification = task["specification"]
            if specification.get("domain") not in {"imputation", "recsys"}:
                raise ValueError("Only declared synthetic imputation and ratings tasks are supported")
            key = content_digest(specification)
            if key in self._specifications:
                raise ValueError("Never duplicate a scheduled sampling specification")
            self._specifications[key] = task
            row = {"task": task, "sampling_key": key}
            if "structured" in specification:
                if set(specification) != {"domain", "structured"} or specification["domain"] != "imputation":
                    raise ValueError("Structured admission must not change ratings or ignore sampling arguments")
                episode, arrays = sampler.structured_prior.draw(**specification["structured"])
                reasons = validate_draw(episode, arrays, specification["structured"])
                episode.metadata = {**episode.metadata, "structured": True}
                digest = episode_digest(episode)
                row.update(state="structured_draw_inspected", metadata=deepcopy(episode.metadata),
                           raw_arrays_content_sha256=content_digest(arrays), episode_content_sha256=digest,
                           eligible=not reasons, rejection_reasons=reasons)
                if reasons:
                    rejected.append(task["index"])
                self._episodes[key], self._digests[key] = episode, digest
            else:
                row.update(state="deferred_until_batch_admitted", generated=False)
                deferred.append(task["index"])
            rows.append(row)
        if len(deferred) == len(tasks):
            raise ValueError("This admission rule requires at least one original structured training draw")
        self._eligible = not rejected
        if not self.eligible:
            self._episodes.clear()
        self._record = {"policy": POLICY, "step": batch["step"], "eligible": self.eligible,
                        "task_count": len(tasks), "structured_draws": len(tasks) - len(deferred),
                        "deferred_task_indices": deferred, "rejected_task_indices": rejected, "tasks": rows,
                        "teacher_inference_performed": False, "student_inference_performed": False,
                        "optimizer_proposal_performed": False, "masks_changed": False, "draws_replaced": False,
                        "held_out_evaluation_performed": False, "qualification_changed": False}

    @property
    def eligible(self):
        return self._eligible

    @property
    def record(self):
        return deepcopy(self._record)

    def sample(self, domain, step, *, structured=None, **arguments):
        if not self.eligible:
            raise ValueError("Rejected data-admission batches permit no teacher/student or optimizer work")
        if type(step) is not int or step != 1:
            raise ValueError("Only the original explicit-seed training capture invocation is admitted")
        specification = {"domain": domain, **arguments}
        if structured is not None:
            specification["structured"] = structured
        key = content_digest(specification)
        if key not in self._specifications or key in self._delivered:
            raise ValueError("Use each exact registered training specification once, never a held-out or replacement case")
        if structured is None:
            episode = self._sampler.sample(domain, step, **arguments)
        else:
            episode = self._episodes[key]
            if episode_digest(episode) != self._digests[key]:
                raise ValueError("The original admitted episode changed before capture")
            episode = Episode(episode.source.to(self._sampler.device), episode.target.to(self._sampler.device),
                              episode.mask.to(self._sampler.device), deepcopy(episode.metadata))
            del self._episodes[key]
        self._delivered.add(key)
        return episode

    def assert_consumed(self):
        expected = set(self._specifications) if self.eligible else set()
        if self._delivered != expected or self._episodes:
            raise ValueError("Account for every admitted task, or perform no sampling for a rejected batch")
