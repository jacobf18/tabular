from copy import deepcopy

import pytest
import torch

import single_model_four_draw_retention as objective
import single_model_standardized_teacher_huber as original
import test_single_model_anchor_retention as fixtures
import test_single_model_column_anchor_retention as columns
from test_single_model_four_draw_protection import tasks


@pytest.mark.parametrize("cohort", ["native", *objective.FAMILIES, "recsys"])
def test_all_task_and_effective_masses_halve_but_raw_gradient_multipliers_do_not(cohort):
    metadata = fixtures.metadata(domain="recsys") if cohort == "recsys" else fixtures.metadata() if cohort == "native" else (
        fixtures.metadata("low_rank_mcar", structured=True, family=cohort))
    old, new = original.retention_weighting(metadata), objective.retention_weighting(metadata)
    assert new["task_weight"] == .5 * old["task_weight"]
    assert new["effective_task_weight"] == .5 * old["effective_task_weight"]
    assert new["multiplier"] == old["multiplier"]
    assert objective.reference_role(metadata) == original.reference_role(metadata)


def test_complete_four_draw_mixture_and_specification_preserve_all_domain_masses():
    batch = tasks()
    for task in batch:
        task["cohort"] = task["name"].rsplit("_draw_", 1)[0]
    objective.validate_task_mixture(batch)
    assert sum(task["weight"] for task in batch if task["cohort"] == "legacy") == .5
    assert sum(task["weight"] for task in batch if task["cohort"] == "recsys") == .25
    assert sum(task["weight"] for task in batch if task["cohort"] in objective.FAMILIES) == pytest.approx(.25)
    assert objective.STRUCTURED_TASK_MASS * 48 == .25
    assert original.REPLICAS == 2 and objective.REPLICAS == 4
    assert objective.SPECIFICATION["structured_retention_task_mass"] == original.STRUCTURED_TASK_MASS / 2
    assert objective.REFERENCE_SHA256S == original.REFERENCE_SHA256S
    assert objective.MEAN_SPECIFICATION == original.MEAN_SPECIFICATION
    assert objective.BACKWARD_SOURCE_SHA256 == original.BACKWARD_SOURCE_SHA256
    batch[0]["weight"] *= 2
    with pytest.raises(ValueError, match="half"):
        objective.validate_task_mixture(batch)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("cohort", ["native", *objective.FAMILIES])
def test_each_actual_streamed_parameter_gradient_and_loss_are_preserved(monkeypatch, transposed, cohort):
    old_model, source, truth, mask = columns.example(shape=(1, 9, 7))
    new_model = deepcopy(old_model)
    if transposed:
        source, truth, mask = (value.transpose(1, 2).contiguous() for value in (source, truth, mask))
    criterion = columns.FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    metadata = fixtures.metadata() if cohort == "native" else fixtures.metadata("low_rank_mcar", structured=True, family=cohort)
    teacher, reference = torch.randn_like(truth, requires_grad=True), torch.randn_like(truth, requires_grad=True)
    episode = columns.Episode(source, truth, mask, metadata)
    options = {"attention_batch_size": 2, "cell_batch_size": 13,
        "encoder_row_batch_size": 3, "decoder_cell_batch_size": 5}
    for module in (original, objective):
        monkeypatch.setattr(module, "TRAIN_OPTIONS", dict(options))
    with objective.stronger.training_objective():
        old_loss = original._backward_episode(old_model, criterion, episode, teacher, 1.,
            query_seed_offset=5, retention_target=reference)
        new_loss = objective._backward_episode(new_model, criterion, episode, teacher, 1.,
            query_seed_offset=5, retention_target=reference)
    assert new_loss == old_loss
    for (name, old_parameter), (new_name, new_parameter) in zip(old_model.named_parameters(), new_model.named_parameters()):
        assert name == new_name and old_parameter.grad is not None and new_parameter.grad is not None
        torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=0, atol=0, msg=name)
    assert teacher.grad is reference.grad is None


def test_new_runtime_and_loss_use_local_weighting_without_changing_originals():
    assert objective._backward_episode.__globals__["retention_weighting"] is objective.retention_weighting
    assert objective.runtime.__wrapped__.__globals__["retention_weighting"] is objective.retention_weighting
    assert original._backward_episode.__globals__["retention_weighting"] is original.retention_weighting
    assert objective.RUNTIME_SOURCE == original.RUNTIME_SOURCE
