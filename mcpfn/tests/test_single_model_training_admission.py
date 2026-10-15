from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import single_model_training_admission as admission
from single_model_complete_missingness_prior import CompleteMissingnessSampler
from single_model_distillation import Episode


def structured_specification(seed=71):
    return {"domain": "imputation", "structured": {"family": "mar_block_neural", "component": "anchor_latent_mcar",
        "shape": [48, 24], "regime": "area_scaled", "requested_missingness": .6, "seed": seed}}


def batch():
    specifications = [{"domain": "imputation", "seed": 70, "component": 0, "shape": [256, 128]},
                      structured_specification(), structured_specification(72),
                      {"domain": "recsys", "seed": 73, "shape": [1000, 1000]}]
    return {"step": 1, "tasks": [{"index": index, "name": f"task_{index}", "weight": .25,
                                 "specification": specification, "gradient_seed": 100 + index}
                                for index, specification in enumerate(specifications)]}


class FakePrior:
    def __init__(self, modes=None):
        self.calls = []
        self.modes = modes or {}

    def draw(self, **specification):
        self.calls.append(deepcopy(specification))
        mode = self.modes.get(specification["seed"], "valid")
        if mode == "error":
            raise RuntimeError("unexpected generation error")
        query = torch.tensor([[False, True], [False, False]])
        if mode == "no_queries":
            query.zero_()
        elif mode == "no_observations":
            query.fill_(True)
        target = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        source = target.masked_fill(query, float("nan"))
        metadata = {"domain": "imputation", "gradient_data": "synthetic", "rows": 2, "columns": 2,
                    **specification, "requested_shape": specification["shape"],
                    "queries": int(query.sum()), "observations": int((~query).sum()),
                    "usable_for_training": bool(query.any() and (~query).any())}
        episode = Episode(source.unsqueeze(0), target.unsqueeze(0), query.unsqueeze(0), metadata)
        return episode, {"query": query, "model_source": source, "model_target": target, "raw_source": target.clone()}


def sampler(modes=None):
    calls = []
    original = SimpleNamespace(device="cpu", structured_prior=FakePrior(modes))

    def sample(domain, step, **arguments):
        calls.append((domain, step, deepcopy(arguments)))
        return {"delegated": domain, "arguments": arguments}

    original.sample = sample
    return original, calls


def test_valid_admission_reuses_each_structured_draw_and_delegates_legacy_ratings_unchanged():
    original, calls = sampler()
    scheduled = batch()
    saved = deepcopy(scheduled)
    checker = admission.TrainingBatchAdmission(original, scheduled)
    assert checker.eligible and len(original.structured_prior.calls) == 2 and calls == []
    assert scheduled == saved
    assert checker.record["deferred_task_indices"] == [0, 3]
    for task in scheduled["tasks"]:
        episode = checker.sample(step=1, **task["specification"])
        if "structured" in task["specification"]:
            assert admission.episode_digest(episode) == checker.record["tasks"][task["index"]]["episode_content_sha256"]
    checker.assert_consumed()
    assert len(original.structured_prior.calls) == 2
    assert len(calls) == 2
    assert calls[0][:2] == ("imputation", 1)
    assert calls[1][:2] == ("recsys", 1)
    assert calls[0][2] == {key: value for key, value in saved["tasks"][0]["specification"].items() if key != "domain"}
    assert calls[1][2] == {key: value for key, value in saved["tasks"][3]["specification"].items() if key != "domain"}


@pytest.mark.parametrize("mode", ["no_queries", "no_observations"])
def test_degenerate_draw_rejects_whole_batch_without_redraw_or_deferred_sampling(mode):
    original, calls = sampler({71: mode})
    checker = admission.TrainingBatchAdmission(original, batch())
    assert not checker.eligible and checker.record["rejected_task_indices"] == [1]
    assert len(original.structured_prior.calls) == 2 and calls == []
    assert checker.record["tasks"][1]["rejection_reasons"] == [mode]
    assert not checker.record["tasks"][1]["metadata"]["usable_for_training"]
    assert not checker.record["masks_changed"] and not checker.record["draws_replaced"]
    for task in batch()["tasks"]:
        with pytest.raises(ValueError, match="Rejected data-admission"):
            checker.sample(step=1, **task["specification"])
    checker.assert_consumed()
    assert calls == [] and not checker._episodes


def test_unexpected_generation_errors_are_not_converted_into_data_noops():
    original, calls = sampler({71: "error"})
    with pytest.raises(RuntimeError, match="unexpected generation"):
        admission.TrainingBatchAdmission(original, batch())
    assert calls == []


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "weight", "mass", "domain", "ratings_structured", "arguments"])
def test_no_scope_changes_or_invalid_batches_are_admitted(mutation):
    original, calls = sampler()
    scheduled = batch()
    if mutation == "omit":
        scheduled["tasks"].pop()
    elif mutation == "duplicate":
        scheduled["tasks"][2]["specification"] = deepcopy(scheduled["tasks"][1]["specification"])
    elif mutation == "weight":
        scheduled["tasks"][0]["weight"] = float("nan")
    elif mutation == "mass":
        scheduled["tasks"][0]["weight"] = .1
    elif mutation == "domain":
        scheduled["tasks"][0]["specification"]["domain"] = "real_benchmark"
    elif mutation == "ratings_structured":
        scheduled["tasks"][1]["specification"]["domain"] = "recsys"
    else:
        scheduled["tasks"][1]["specification"]["ignored"] = True
    with pytest.raises(ValueError):
        admission.TrainingBatchAdmission(original, scheduled)
    assert calls == []


def test_each_case_is_delivered_once_and_no_unregistered_or_held_out_seed_can_be_used():
    original, calls = sampler()
    checker = admission.TrainingBatchAdmission(original, batch())
    with pytest.raises(ValueError, match="Account for every"):
        checker.assert_consumed()
    checker.sample(step=1, **structured_specification())
    with pytest.raises(ValueError, match="exact registered"):
        checker.sample(step=1, **structured_specification())
    with pytest.raises(ValueError, match="exact registered"):
        checker.sample(step=1, **structured_specification(99))
    with pytest.raises(ValueError, match="original explicit-seed"):
        checker.sample(step=2, **structured_specification(72))


def test_modified_cached_episode_is_rejected_and_record_mutation_cannot_change_admission():
    original, calls = sampler()
    checker = admission.TrainingBatchAdmission(original, batch())
    report = checker.record
    report["eligible"] = False
    report["tasks"][1]["metadata"]["queries"] = 999
    assert checker.record["eligible"] and checker.record["tasks"][1]["metadata"]["queries"] == 1
    key = admission.content_digest(structured_specification())
    checker._episodes[key].target[0, 0, 0] += 1
    with pytest.raises(ValueError, match="original admitted episode changed"):
        checker.sample(step=1, **structured_specification())


def test_admission_decision_cannot_be_changed_by_assignment():
    original, calls = sampler({71: "no_queries"})
    checker = admission.TrainingBatchAdmission(original, batch())
    with pytest.raises(AttributeError):
        checker.eligible = True
    assert not checker.eligible and not checker.record["eligible"] and calls == []


def test_original_v28_empty_query_case_is_kept_unmodified_and_never_becomes_training_data():
    original = SimpleNamespace(device="cpu", structured_prior=CompleteMissingnessSampler())
    specification = structured_specification(631274001)
    scheduled = {"step": 62, "tasks": [{"index": 0, "name": "failed_draw", "weight": 1.,
                                      "specification": specification, "gradient_seed": 632274001}]}
    checker = admission.TrainingBatchAdmission(original, scheduled)
    record = checker.record["tasks"][0]
    assert not checker.eligible and record["rejection_reasons"] == ["no_queries"]
    assert (record["metadata"]["rows"], record["metadata"]["columns"]) == (11, 5)
    assert record["metadata"]["queries"] == 0 and record["metadata"]["observations"] == 55
    assert record["raw_arrays_content_sha256"] == "91875908f878e71091eea8c1a9fb97966e927185489929a7bc4c962cbb40f1ac"
    checker.assert_consumed()
