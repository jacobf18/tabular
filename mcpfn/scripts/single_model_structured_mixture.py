"""Complete structured-mask rehearsal with separate legacy and family-level validation."""

from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import math
import os
import random
import time

import numpy as np
import torch

from dwip_teacher_mean_training import compute_context
from recsys_evaluation_common import atomic_npz, sha256, write_json
from single_model_area_prior import AreaMixedSampler, regime_for_seed
from single_model_complete_missingness_prior import (
    COMPACT_SHAPES, COMPONENTS, NEURAL_FAMILIES, SPECTRAL_SHAPES, CompleteMissingnessSampler,
)
from single_model_distillation import teacher_target as reference_teacher_target
import single_model_column_objective as column
import single_model_query_objective as query
import single_model_truth_defined_columns as defined
from diagnose_single_model_missingness_coverage import FAMILIES
from streamed_large_matrix_training import STREAM_OPTIONS, atomic_torch_save
from tabimpute.streamed_inference import streamed_point_predictions
from train_single_matrixformer_parent_teacher import parent_teacher_target
from train_single_matrixformer import setup as original_setup, storage_guard
import train_single_matrixformer_continuation as backend


STRUCTURED_METRICS = ("missbench_nrmse", "uci_nrmse", "common_nrmse", "rmse")
SEEDS = {"seed": 70100601, "data_seed": 71100601, "validation_seed": 72100601,
         "confirmation_seed": 73100601, "preflight_seed": 74100601,
         "structured_validation_seed": 75100601, "structured_confirmation_seed": 76100601,
         "coverage_order_seed": 77100601, "regime_seed": 78100601, "query_seed_offset": 79100601}
POLICY = {
    "training": "Every even update uses one structured imputation episode; odd updates retain V8's legacy/area/support mixture; every update retains its original rating episode; both domain weights remain0.5",
    "coverage": "Cycles of all48 family/component pairs, independently shuffled; explicit family-appropriate shapes, amplitude regime and requested missingness sampled with private seeds",
    "selection": "Minimize the worst of four original primary ratios and twelve structured-family common-NRMSE ratios to original frozen references",
    "retention": "All six original legacy metrics, all four structured-overall metrics, and every structured-family common NRMSE must retain initialization, with no tolerance",
    "confirmation": "Fresh original92 plus structured192 cases; same initialization retention, all six original and four structured-overall metrics retain general, and strict extended minimax improvement",
    "coverage_semantics": "Canonical scores retain every case; canonical scored-column counts may depend on predictions; common guard uses identical truth-defined columns/cases, with nulls retained and all-undefined families rejected",
    "reference": "Original general and observed-error blend on identical imputation inputs; original specialist rating reference; no teacher or ensemble at student inference",
}


def structured_bank(seed, *, confirmation=False):
    result = []
    for family_index, family in enumerate(FAMILIES):
        shapes = COMPACT_SHAPES if family in NEURAL_FAMILIES else SPECTRAL_SHAPES
        for component_index, component in enumerate(COMPONENTS):
            for shape_index, shape in enumerate(shapes):
                regimes = ("legacy", "area_scaled") if confirmation else (
                    ("legacy", "area_scaled")[(family_index + component_index + shape_index) % 2],)
                for regime in regimes:
                    specification = {"family": family, "component": component, "shape": list(shape), "regime": regime,
                                     "requested_missingness": (.2, .6)[shape_index], "seed": seed + 100 * len(result)}
                    result.append({"domain": "imputation", "structured": specification})
    return result


def training_cases(steps=400):
    if type(steps) is not int or steps != 400:
        raise ValueError("Require the fixed400-update experiment")
    pairs = [(family, component) for family in FAMILIES for component in COMPONENTS]
    orders, result = {}, []
    for step in range(2, steps + 1, 2):
        index = step // 2 - 1
        cycle, within = divmod(index, len(pairs))
        if cycle not in orders:
            orders[cycle] = list(pairs)
            random.Random(SEEDS["coverage_order_seed"] + cycle).shuffle(orders[cycle])
        family, component = orders[cycle][within]
        seed = SEEDS["data_seed"] + 16 * step
        chooser = random.Random(seed)
        shapes = COMPACT_SHAPES if family in NEURAL_FAMILIES else SPECTRAL_SHAPES
        specification = {"family": family, "component": component, "seed": seed,
                         "shape": list(chooser.choice(shapes)), "regime": chooser.choice(("legacy", "area_scaled")),
                         "requested_missingness": chooser.choice((.2, .6))}
        result.append({"step": step, "case": specification})
    return result


class StructuredSampler(AreaMixedSampler):
    def __init__(self, config, *, device="cuda"):
        super().__init__(config, device=device)
        self.structured_prior = CompleteMissingnessSampler()

    def sample(self, domain, step, *, structured=None, **arguments):
        if structured is None:
            return super().sample(domain, step, **arguments)
        if domain != "imputation" or arguments:
            raise ValueError("Structured sampling must not change ratings or silently ignore arguments")
        episode = self.structured_prior.sample(device=self.device, **structured)
        episode.metadata["structured"] = True
        return episode


def aggregate_structured(rows, field):
    if not rows:
        raise ValueError("Never omit structured validation")
    result = {"canonical_metric_cases": len(rows)}
    for name in ("rmse", "mae", "missbench_nrmse", "uci_nrmse"):
        values = [row[field][name] for row in rows]
        if any(value is None or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Every canonical structured case must be retained")
        result[name] = float(np.mean(values))
    common = []
    for row in rows:
        value = row[field]["common_nrmse"]
        count = row[field]["column_coverage"]["common_nonconstant_columns"]
        if (value is None) != (count == 0):
            raise ValueError("Undefined common scores must be truth-defined")
        if value is not None:
            if not math.isfinite(value) or value < 0:
                raise ValueError("Nonfinite common score")
            common.append(value)
    if not common:
        raise ValueError("Cannot waive a structured-family guard with no defined cases")
    result.update(common_nrmse=float(np.mean(common)), common_nrmse_cases=len(common))
    return result


def aggregate(cases, field):
    legacy = [case for case in cases if not case.get("structured", False)]
    structured = [case for case in cases if case.get("structured", False)]
    if not legacy or {case.get("family") for case in structured} != set(FAMILIES):
        raise ValueError("Keep legacy validation and all twelve structured families separately")
    result = defined.aggregate(legacy, field)
    families = {family: aggregate_structured([case for case in structured if case["family"] == family], field)
                for family in FAMILIES}
    if len({values["canonical_metric_cases"] for values in families.values()}) != 1:
        raise ValueError("Structured families require balanced full coverage")
    result["structured"] = {"overall": aggregate_structured(structured, field), "families": families}
    return result


def metric_value(metrics, group, name):
    value = metrics["structured"][group][name] if group == "overall" else metrics["structured"]["families"][group][name]
    if not math.isfinite(value) or value < 0:
        raise ValueError("Require finite nonnegative structured metrics")
    return value


def selection_score(metrics, baseline, references):
    _, ratios, retained = column.selection_score(metrics, baseline, references)
    for values in (metrics, baseline, references):
        if set(values["structured"]["families"]) != set(FAMILIES):
            raise ValueError("All twelve family references are mandatory")
    for name in STRUCTURED_METRICS:
        retained = (metric_value(metrics, "overall", name) <= metric_value(baseline, "overall", name)) and retained
    for family in FAMILIES:
        value = metric_value(metrics, family, "common_nrmse")
        ratios[f"structured_{family}"] = value / max(metric_value(references, family, "common_nrmse"), 1e-12)
        retained = (value <= metric_value(baseline, family, "common_nrmse")) and retained
    return max(ratios.values()), ratios, retained


def confirmation_gate(student, initialization, general, references):
    score, ratios, retained = selection_score(student, initialization, references)
    initial_score, _, _ = selection_score(initialization, initialization, references)
    _, _, legacy_general_retained = column.selection_score(student, general, references)
    structured_general_retained = all(metric_value(student, "overall", name) <= metric_value(general, "overall", name)
                                      for name in STRUCTURED_METRICS)
    return {"passed": retained and legacy_general_retained and structured_general_retained and score < initial_score,
            "score": score, "initialization_score": initial_score, "ratios": ratios,
            "initialization_retained": retained, "legacy_general_retained": legacy_general_retained,
            "structured_general_retained": structured_general_retained, "real_benchmark_success": False,
            "policy": POLICY, "structured_family_ratios_to_initialization": {
                family: metric_value(student, family, "common_nrmse") / max(metric_value(initialization, family, "common_nrmse"), 1e-12)
                for family in FAMILIES}}


def require_truth_coverage(cases, fields):
    for case in cases:
        if case["domain"] != "imputation":
            continue
        signatures = {tuple(case[field]["column_coverage"][key] for key in
                            ("queried_columns", "common_nonconstant_columns")) for field in fields}
        if len(signatures) != 1:
            raise ValueError("Truth-only column coverage differs across models")


@contextmanager
def runtime(config, output):
    names = ("setup", "training_episode", "teacher_target", "backward_episode", "episode_metrics", "aggregate",
             "selection_score", "prepare_bank", "evaluate_bank")
    previous = {name: getattr(backend, name) for name in names}
    parent = []

    def setup(current):
        result = original_setup(current)
        parent[:] = [deepcopy(result[0]).eval().requires_grad_(False)]
        return (*result[:-1], StructuredSampler(current))

    def training_episode(sampler, current, domain, step):
        if domain == "recsys":
            shape = tuple(current["rating_training_shapes"][(step - 1) % len(current["rating_training_shapes"])])
            return sampler.sample(domain, step, shape=shape)
        if domain != "imputation":
            raise ValueError("Unknown training domain")
        if step % 2 == 0:
            record = current["training_structured_cases"][step // 2 - 1]
            if record["step"] != step:
                raise ValueError("Structured training schedule changed")
            return sampler.sample(domain, step, structured=record["case"])
        seed = current["data_seed"] + 16 * step
        return sampler.sample(domain, step, regime=regime_for_seed(seed, current["regime_seed"]))

    def training_target(teachers, criterion, episode, options=STREAM_OPTIONS):
        if not parent:
            raise RuntimeError("Initialize the frozen parent before training")
        return parent_teacher_target(parent[0], teachers, criterion, episode, options)

    @torch.no_grad()
    def prepare_bank(sampler, teachers, criterion, specifications, directory):
        bank, cases = [], []
        prediction_directory = directory / "reference_predictions"
        prediction_directory.mkdir()
        for index, specification in enumerate(specifications):
            episode = sampler.sample(step=1, **specification)
            teacher, routing = reference_teacher_target(teachers, criterion, episode)
            with compute_context(episode.source.device):
                general = streamed_point_predictions(teachers[0], episode.source, criterion, **STREAM_OPTIONS)
            destination = prediction_directory / f"case_{index:03d}.npz"
            atomic_npz(destination, general=general.float().cpu().numpy(), teacher=teacher.float().cpu().numpy())
            cases.append({"index": index, **episode.metadata, "general": defined.episode_metrics(general, episode),
                          "teacher": defined.episode_metrics(teacher, episode), "routing": routing,
                          "prediction_file": str(destination), "prediction_sha256": sha256(destination)})
            bank.append(backend.cache_episode(episode))
            write_json(directory / "status.json", {"state": "building_synthetic_bank", "completed": len(bank),
                "expected": len(specifications), "pid": os.getpid(), "updated_unix": time.time()})
            del episode, teacher, general
        require_truth_coverage(cases, ("general", "teacher"))
        general_metrics, teacher_metrics = aggregate(cases, "general"), aggregate(cases, "teacher")
        references = {"imputation": {name: min(general_metrics["imputation"][name], teacher_metrics["imputation"][name])
                                     for name in STRUCTURED_METRICS}, "recsys": teacher_metrics["recsys"],
                      "structured": {"overall": {name: min(metric_value(general_metrics, "overall", name), metric_value(teacher_metrics, "overall", name))
                                                   for name in STRUCTURED_METRICS},
                                     "families": {family: {"common_nrmse": min(metric_value(general_metrics, family, "common_nrmse"),
                                                                                metric_value(teacher_metrics, family, "common_nrmse"))}
                                                  for family in FAMILIES}}}
        storage_guard(directory)
        atomic_torch_save(bank, directory / "synthetic_bank.pt")
        reference = {"bank_sha256": sha256(directory / "synthetic_bank.pt"), "cases": cases, "general": general_metrics,
                     "teacher": teacher_metrics, "references": references}
        write_json(directory / "reference.json", reference)
        return bank, reference

    @torch.no_grad()
    def evaluate_bank(model, criterion, bank, *, device="cuda"):
        existing = list(output.glob("student_predictions_*"))
        pass_index = max((int(path.name.rsplit("_", 1)[1]) for path in existing), default=-1) + 1
        directory = output / f"student_predictions_{pass_index:03d}"
        directory.mkdir()
        was_training, cases = model.training, []
        model.eval()
        try:
            for index, cached in enumerate(bank):
                episode = backend.restore_episode(cached, device)
                with compute_context(episode.source.device):
                    prediction = streamed_point_predictions(model, episode.source, criterion, **STREAM_OPTIONS)
                destination = directory / f"case_{index:03d}.npz"
                atomic_npz(destination, student=prediction.float().cpu().numpy())
                cases.append({"index": index, **episode.metadata, "student": defined.episode_metrics(prediction, episode),
                              "prediction_file": str(destination), "prediction_sha256": sha256(destination)})
                del episode, prediction
        finally:
            model.train(was_training)
        return aggregate(cases, "student"), cases

    callbacks = {"setup": setup, "training_episode": training_episode, "teacher_target": training_target,
        "backward_episode": partial(query.backward_episode, query_seed_offset=config["query_seed_offset"]),
        "episode_metrics": defined.episode_metrics, "aggregate": aggregate, "selection_score": selection_score,
        "prepare_bank": prepare_bank, "evaluate_bank": evaluate_bank}
    for name, callback in callbacks.items():
        setattr(backend, name, callback)
    try:
        yield
    finally:
        for name, callback in previous.items():
            setattr(backend, name, callback)
