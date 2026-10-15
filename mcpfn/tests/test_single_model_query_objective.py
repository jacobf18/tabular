from copy import deepcopy

import pytest
import torch

from single_model_distillation import Episode
import single_model_query_objective as objective
from single_model_query_budget import budget_for_seed, query_budget_majorizer
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from test_streamed_training import assert_gradients_match, example


@pytest.mark.parametrize("budget", [2, 4, 8, 16])
@pytest.mark.parametrize("transposed", [False, True])
def test_streamed_query_objective_matches_dense_loss_and_gradients(monkeypatch, budget, transposed):
    reference, source, truth, mask = example(shape=(1, 9, 7))
    if transposed:
        source, truth, mask = (value.transpose(1, 2).contiguous() for value in (source, truth, mask))
    original_source, original_mask = source.clone(), mask.clone()
    model = deepcopy(reference)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teacher = torch.randn_like(truth, requires_grad=True)
    seed = next(seed for seed in range(100) if budget_for_seed(seed + 1000) == budget)
    weights, offset, _ = query_budget_majorizer(truth, mask, teacher, seed=seed + 1000)
    logits = reference(source, decode_mask=mask)
    expected = objective.column.balanced.balanced_loss(
        criterion, logits, truth[mask], teacher[mask], domain="imputation",
        probability_scales=objective.column.balanced.PROBABILITY_SCALES)
    expected = (expected + weights[mask] * (criterion.mean(logits) - truth[mask]).square()).mean() + offset
    monkeypatch.setattr(objective, "TRAIN_OPTIONS", {"attention_batch_size": 2, "cell_batch_size": 13,
                                                    "encoder_row_batch_size": 3, "decoder_cell_batch_size": 5})
    episode = Episode(source, truth, mask, {"domain": "imputation", "gradient_data": "synthetic", "seed": seed})
    actual = objective.backward_episode(model, criterion, episode, teacher, .5, query_seed_offset=1000)
    (.5 * expected).backward()
    assert actual == pytest.approx(float(expected.detach()), rel=1e-6, abs=1e-6)
    assert_gradients_match(reference, model)
    assert teacher.grad is None and episode.metadata["query_budget_objective"]["budget"] == budget
    torch.testing.assert_close(source, original_source, equal_nan=True)
    assert torch.equal(mask, original_mask)


def test_rating_loss_and_gradients_are_identical_to_the_original():
    reference, source, truth, mask = example(shape=(1, 9, 7))
    model = deepcopy(reference)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teacher = torch.randn_like(truth)
    episode = Episode(source, truth, mask, {"domain": "recsys", "gradient_data": "synthetic"})
    expected = objective.column.BASE_BACKWARD(reference, criterion, episode, teacher, .5,
                                              probability_scales=objective.column.balanced.PROBABILITY_SCALES)
    actual = objective.backward_episode(model, criterion, episode, teacher, .5, query_seed_offset=1000)
    assert actual == expected
    assert_gradients_match(reference, model)
    assert "query_budget_objective" not in episode.metadata and "column_objective" not in episode.metadata


@pytest.mark.parametrize("failure", [False, True])
def test_training_adapter_preserves_evaluation_and_restores_callbacks(failure):
    runner = objective.column.continuation
    callbacks = (runner.backward_episode, objective.column.balanced.backward_episode)
    evaluation = (runner.episode_metrics, runner.selection_score, runner.teacher_target, runner.setup)
    try:
        with objective.training_objective(1000):
            assert runner.backward_episode is objective.column.balanced.backward_episode
            assert runner.backward_episode.func is objective.backward_episode
            assert evaluation == (runner.episode_metrics, runner.selection_score, runner.teacher_target, runner.setup)
            if failure:
                raise RuntimeError("interrupted training")
    except RuntimeError:
        assert failure
    assert callbacks == (runner.backward_episode, objective.column.balanced.backward_episode)
    assert evaluation == (runner.episode_metrics, runner.selection_score, runner.teacher_target, runner.setup)


def test_non_synthetic_episodes_are_rejected_before_model_execution():
    episode = Episode(None, None, None, {"domain": "imputation", "gradient_data": "real"})
    with pytest.raises(ValueError, match="synthetic"):
        objective.backward_episode(None, None, episode, None, .5, query_seed_offset=1000)
