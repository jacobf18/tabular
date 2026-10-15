from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from single_model_distillation import (
    Episode, blend_predictions, distillation_loss, episode_metrics, selection_score,
    teacher_target, validation_specs,
)
from streamed_large_matrix_training import per_cell_objective
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.routing import fit_routing, observed_scales
from tabimpute.streamed_training import streamed_training_loss
from test_streamed_training import assert_gradients_match, example
from train_single_matrixformer import backward_episode
from confirm_single_matrixformer import aggregate, confirmation_specs


OPTIONS = dict(attention_batch_size=2, cell_batch_size=13, encoder_row_batch_size=3)


def test_synthetic_blend_matches_registered_observed_only_router():
    source = torch.tensor([[[1., float("nan")], [3., 5.], [5., 7.]]])
    general = torch.tensor([[[0., 100.], [2., 4.], [4., 6.]]])
    specialist = general + 3.
    prediction, decision = blend_predictions(source, general, specialist)
    matrix = source[0].numpy()
    expected = fit_routing(matrix, general[0].numpy(), specialist[0].numpy(), np.isfinite(matrix), observed_scales(matrix))
    assert decision == expected
    assert decision["specialist_weight"] == pytest.approx(1 / 3)
    torch.testing.assert_close(prediction, general + 1.)
    changed = specialist.clone()
    changed[torch.isnan(source)] = -999.
    assert blend_predictions(source, general, changed)[1] == decision
    assert blend_predictions(torch.full_like(source, torch.nan), general, specialist)[1]["specialist_weight"] == 0.


def test_distillation_matches_dense_loss_gradients_and_detaches_teacher():
    reference, source, target, mask = example()
    actual = deepcopy(reference)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    auxiliary = torch.randn_like(target, requires_grad=True)
    logits = reference(source, decode_mask=mask)
    expected = per_cell_objective(criterion, logits, target[mask], auxiliary[mask]).mean()
    observed = streamed_training_loss(
        actual, source, target, mask,
        lambda logits, truth, teacher: distillation_loss(criterion, logits, truth, teacher, nll_weight=1.),
        auxiliary_targets=auxiliary, decoder_cell_batch_size=7, **OPTIONS,
    )
    torch.testing.assert_close(observed, expected)
    expected.backward()
    observed.backward()
    assert_gradients_match(reference, actual)
    assert auxiliary.grad is None


def test_specialist_target_uses_only_frozen_specialist():
    model, source, target, mask = example(shape=(1, 9, 7))
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teachers = [deepcopy(model).eval().requires_grad_(False) for _ in range(2)]
    episode = Episode(source, target, mask, {"domain": "recsys", "gradient_data": "synthetic"})
    prediction, decision = teacher_target(teachers, criterion, episode, OPTIONS)
    with torch.no_grad():
        expected = criterion.mean(teachers[1](source))
    torch.testing.assert_close(prediction, expected, rtol=1e-5, atol=1e-6)
    assert not prediction.requires_grad
    assert decision["specialist_weight"] == 1.
    teachers[0].train()
    with pytest.raises(ValueError, match="frozen"):
        teacher_target(teachers, criterion, episode, OPTIONS)


def test_rating_metrics_are_user_macro_in_original_units():
    truth = torch.tensor([[[2., 4.], [3., 5.]]])
    scale = torch.tensor([[[2., 2.], [1., 1.]]])
    location = torch.tensor([[[3., 4.], [2., 3.]]])
    ratings = SimpleNamespace(means=torch.tensor([2., 3.]), row_offsets=torch.tensor([1., 0.]),
                              stds=torch.tensor([2., 2.]), row_scales=torch.tensor([1., .5]))
    mask = torch.tensor([[[True, True], [True, False]]])
    episode = Episode(torch.zeros_like(truth), (truth - location) / scale, mask, {"domain": "recsys"}, ratings)
    prediction = (torch.tensor([[[1., 2.], [5., 2.]]]) - location) / scale
    metrics = episode_metrics(prediction, episode)
    assert metrics["rmse"] == pytest.approx(((2.5 ** .5) + 2.) / 2.)
    assert metrics["mae"] == pytest.approx(1.75)


def test_selection_keeps_step_zero_and_rejects_domain_regression():
    baseline = {domain: {"rmse": 1., "mae": .8} for domain in ("imputation", "recsys")}
    assert selection_score(baseline, baseline, baseline) == (1., True)
    improved = deepcopy(baseline)
    improved["recsys"]["rmse"] = .9
    assert selection_score(improved, baseline, baseline) == (.95, True)
    improved["imputation"]["rmse"] = 1.031
    assert selection_score(improved, baseline, baseline)[1] is False
    improved["imputation"]["rmse"] = .9
    improved["recsys"]["mae"] = .801
    assert selection_score(improved, baseline, baseline)[1] is False
    improved["recsys"]["rmse"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        selection_score(improved, baseline, baseline)


def test_non_synthetic_gradient_episode_rejected_before_forward():
    episode = Episode(None, None, None, {"gradient_data": "real"})
    with pytest.raises(ValueError, match="synthetic"):
        backward_episode(None, None, episode, None, .5)


def test_validation_covers_both_priors_and_large_transposed_context():
    specifications = validation_specs()
    assert len(specifications) == 14
    assert len([case for case in specifications if case["domain"] == "recsys"]) == 4
    assert any(case.get("transpose") and case["shape"] == (2000, 1000) for case in specifications)


def test_confirmation_bank_and_equal_case_aggregation():
    specifications = confirmation_specs()
    assert len(specifications) == 46
    assert sum(case["domain"] == "recsys" for case in specifications) == 16
    cases = [{"domain": domain, **{role: {"rmse": value, "mae": value / 2}
                                  for role in ("student", "parent", "teacher")}}
             for domain, value in (("imputation", 1.), ("imputation", 3.), ("recsys", 7.))]
    metrics = aggregate(cases)
    assert metrics["student"]["imputation"]["rmse"] == 2.
    assert metrics["parent"]["recsys"]["mae"] == 3.5
