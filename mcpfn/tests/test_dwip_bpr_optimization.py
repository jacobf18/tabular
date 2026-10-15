import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dwip_bpr_evaluation import compare_goal, selection_key
from dwip_bpr_experiment import episode_arguments, ranking_episode
import dwip_bpr_experiment as bpr_experiment
from tabimpute.prior.recsys_rating_episode import context_crop_indices, matched_ranking_queries, sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import raw_query_predictions, user_pairwise_loss, synthetic_ranking_objective


def test_matched_negatives_cover_every_eligible_positive_user():
    ratings = torch.tensor([[5., 2., 1., 3.], [2., 4., 1., 5.], [3., 2., 4., 1.]])
    queries = torch.tensor([[True, False, False, False], [False, True, False, False], [True, False, False, False]])
    eligible = ~queries
    generator = torch.Generator().manual_seed(42)
    negative = matched_ranking_queries(queries, ratings, eligible, generator, 2)
    assert negative.sum(1).tolist() == [2, 2, 0]
    assert not (negative & queries).any()
    repeated = matched_ranking_queries(queries, ratings, eligible, torch.Generator().manual_seed(42), 2)
    assert torch.equal(negative, repeated)


def test_matched_negatives_handle_insufficient_candidates():
    queries = torch.tensor([[True, False], [True, False]])
    eligible = torch.tensor([[False, True], [False, False]])
    negative = matched_ranking_queries(queries, torch.full((2, 2), 5.), eligible, torch.Generator().manual_seed(1), 16)
    assert torch.equal(negative, eligible)


def test_catalog_negatives_can_focus_on_a_bounded_uniform_user_subset():
    queries = torch.zeros(20, 30, dtype=torch.bool)
    queries[:, 0] = True
    negative = matched_ranking_queries(queries, torch.full((20, 30), 5.), ~queries,
                                       torch.Generator().manual_seed(92), 100, max_users=4)
    assert negative.any(1).sum() == 4
    assert negative.sum() == 4 * 29
    assert not (negative & queries).any()
    with pytest.raises(ValueError):
        matched_ranking_queries(queries, torch.full((20, 30), 5.), ~queries, torch.Generator(), 16, max_users=-1)


def test_catalog_crop_preserves_user_window_and_has_unique_columns():
    observed = torch.rand(40, 120, generator=torch.Generator().manual_seed(93)) < .2
    queries = ~observed & (torch.rand(40, 120, generator=torch.Generator().manual_seed(94)) < .1)
    blocked_rows, blocked_cols = context_crop_indices(observed, 20, 30, torch.Generator().manual_seed(95), "first_observation", queries)
    catalog_rows, catalog_cols = context_crop_indices(observed, 20, 30, torch.Generator().manual_seed(95), "first_observation_catalog", queries)
    assert torch.equal(blocked_rows, catalog_rows)
    assert catalog_cols.unique().numel() == 30
    assert not torch.equal(blocked_cols, catalog_cols)
    assert queries[catalog_rows[:, None], catalog_cols].any()
    repeated_rows, repeated_cols = context_crop_indices(observed, 20, 30, torch.Generator().manual_seed(95), "first_observation_catalog", queries)
    assert torch.equal(catalog_rows, repeated_rows) and torch.equal(catalog_cols, repeated_cols)


def test_catalog_prior_remains_synthetic_known_only_and_has_finite_targets():
    episode = sample_rating_episode(96, rows=24, cols=32, parent_rows=64, parent_cols=128,
                                     context_ordering="first_observation_catalog", exclude_unknown_queries=True,
                                     ranking_query_mode="user_matched", negatives_per_user=100, ranking_max_users=4,
                                     normalization="two_way_user_scale", density_range=(.1, .1))
    assert episode.metadata["gradient_data"] == "synthetic"
    assert episode.metadata["context_ordering"] == "first_observation_catalog"
    assert episode.metadata["global_unknown_rating_queries"] == 0
    assert episode.rating_query.any() and torch.isfinite(episode.target).all()
    assert torch.isnan(episode.source[episode.query]).all()


@pytest.mark.parametrize("mode", ["global", "user_matched"])
def test_synthetic_episode_queries_are_hidden_and_provenance_explicit(mode):
    episode = sample_rating_episode(88, rows=32, cols=40, parent_rows=64, query_limit=100,
                                     density_range=(.3, .3), ranking_query_mode=mode, exclude_unknown_queries=True)
    assert episode.metadata["gradient_data"] == "synthetic"
    assert torch.isnan(episode.source[episode.query]).all()
    assert torch.all(episode.query | ~episode.rating_query)
    if mode == "user_matched":
        truth = episode.target[0] * episode.stds + episode.means
        positive_users = (episode.rating_query[0] & (truth.round() >= 4)).any(1)
        negative_users = (episode.query[0] & ~episode.rating_query[0]).any(1)
        assert not (negative_users & ~positive_users).any()


def test_matched_sampling_preserves_rating_prior_and_context_bitwise():
    arguments = {"seed": 89, "rows": 32, "cols": 40, "parent_rows": 64, "parent_cols": 80,
                 "query_limit": 100, "density_range": (.2, .2), "normalization": "two_way_user_scale",
                 "point_target": "expected_rating", "ordinal_targets": True}
    original = sample_rating_episode(**arguments, ranking_query_mode="global")
    matched = sample_rating_episode(**arguments, ranking_query_mode="user_matched")
    for name in ("source", "target", "rating_query", "means", "stds", "point_target", "row_offsets",
                 "row_scales", "ordinal_probabilities", "global_query_counts"):
        torch.testing.assert_close(getattr(original, name), getattr(matched, name), rtol=0, atol=0, equal_nan=True)


def test_complete_synthetic_ranking_objective_has_finite_gradients():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, targets):
            return (logits[:, 0] - targets).square()

    episode = sample_rating_episode(90, rows=24, cols=20, parent_rows=48, parent_cols=80,
                                     density_range=(.2, .2), ordinal_targets=True, point_target="expected_rating",
                                     normalization="two_way_user_scale", ranking_query_mode="user_matched")
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    loss, diagnostics = synthetic_ranking_objective(logits, episode, Criterion(), hard_negatives=4)
    assert diagnostics["ranking_positive_users"] == diagnostics["ranking_paired_users"] > 0
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_synthetic_teacher_anchor_is_user_macro_and_rejects_live_targets():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, targets):
            return (logits[:, 0] - targets).square()

    episode = sample_rating_episode(91, rows=24, cols=20, parent_rows=48, parent_cols=80,
                                     density_range=(.2, .2), ordinal_targets=True, point_target="expected_rating",
                                     normalization="two_way_user_scale", ranking_query_mode="user_matched")
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    prediction, _, users, _ = raw_query_predictions(logits, episode, Criterion())
    selected = episode.rating_query[episode.query]
    offsets = 1 + users[selected].float() % 3
    teacher = prediction[selected].detach() + offsets
    loss, diagnostics = synthetic_ranking_objective(logits, episode, Criterion(), ranking_weight=0, rating_weight=0,
                                                     teacher_prediction=teacher, teacher_anchor_weight=2)
    expected = torch.stack([offsets[users[selected] == user].square().mean() for user in users[selected].unique()]).mean()
    torch.testing.assert_close(loss, 2 * expected)
    assert diagnostics["synthetic_teacher_anchor"] == pytest.approx(expected.item())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="detached"):
        synthetic_ranking_objective(logits, episode, Criterion(), teacher_prediction=teacher.requires_grad_(), teacher_anchor_weight=1)


def test_pairwise_loss_matches_explicit_user_macro_and_gradients():
    prediction = torch.tensor([4., 2., 3., 5., 2., 1.], requires_grad=True)
    truth = torch.tensor([5., 1., 1., 4., 5., 1.])
    users = torch.tensor([0, 0, 0, 1, 1, 1])
    items = torch.tensor([0, 1, 2, 0, 1, 2])
    selected = torch.tensor([True, False, False, True, True, False])
    actual, diagnostics = user_pairwise_loss(prediction, truth, users, items, selected, (2, 3))
    expected = (torch.nn.functional.softplus(prediction[1:3] - prediction[0]).mean()
                + torch.nn.functional.softplus(prediction[5] - prediction[3:5]).mean()) / 2
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert diagnostics["ranking_paired_users"] == 2


def test_hard_negative_loss_selects_highest_scored_negative():
    prediction = torch.tensor([4., 2., 3.], requires_grad=True)
    loss, _ = user_pairwise_loss(prediction, torch.tensor([5., 1., 1.]), torch.zeros(3, dtype=torch.long),
                                torch.arange(3), torch.tensor([True, False, False]), (1, 3), hard_negatives=1)
    torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(-1.)))
    loss.backward()
    assert prediction.grad[1] == 0
    assert prediction.grad[0] < 0 and prediction.grad[2] > 0


def test_negative_only_ranking_preserves_loss_but_does_not_raise_positive_scores():
    prediction = torch.tensor([4., 2., 3.], requires_grad=True)
    arguments = (prediction, torch.tensor([5., 1., 1.]), torch.zeros(3, dtype=torch.long),
                 torch.arange(3), torch.tensor([True, False, False]), (1, 3))
    original, _ = user_pairwise_loss(*arguments, hard_negatives=1)
    negative_only, _ = user_pairwise_loss(*arguments, hard_negatives=1, negative_only=True)
    torch.testing.assert_close(original, negative_only)
    negative_only.backward()
    assert prediction.grad[0] == 0 and prediction.grad[1] == 0 and prediction.grad[2] > 0


def test_pairwise_loss_does_not_compare_different_users():
    prediction = torch.tensor([4., 5.], requires_grad=True)
    loss, diagnostics = user_pairwise_loss(prediction, torch.tensor([5., 1.]), torch.tensor([0, 1]),
                                          torch.tensor([0, 1]), torch.tensor([True, False]), (2, 2))
    assert loss.item() == 0 and diagnostics["ranking_paired_users"] == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros(2))


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_ranking_temperature_is_rejected(temperature):
    with pytest.raises(ValueError):
        user_pairwise_loss(None, None, None, None, None, (2, 2), temperature=temperature)


@pytest.mark.parametrize("provenance", ["real", None, "synthetic_and_real"])
def test_real_or_unknown_gradient_data_is_rejected_before_objective(provenance):
    episode = SimpleNamespace(metadata={"gradient_data": provenance})
    with pytest.raises(ValueError, match="synthetic"):
        synthetic_ranking_objective(None, episode, None)


def comparison_fixture():
    return {dataset: {"MAE": .8, "RMSE": .9, "NDCG@10": .1, "Recall@10": .1}
            for dataset in ("movielens_100k", "netflix_small")}


def test_selection_enforces_strict_rating_guardrail_before_ranking():
    baseline = comparison_fixture()
    feasible = copy.deepcopy(baseline)
    for values in feasible.values():
        values.update(MAE=.79, RMSE=.89)
    infeasible = copy.deepcopy(feasible)
    infeasible["netflix_small"].update(RMSE=.9)
    for values in infeasible.values():
        values.update({"NDCG@10": 1, "Recall@10": 1})
    assert selection_key(feasible, baseline, baseline) < selection_key(infeasible, baseline, baseline)
    assert not compare_goal(infeasible, baseline, baseline)["netflix_small"]["RMSE"]["better"]


def test_selection_improves_worst_ranking_ratio_and_requires_both_datasets():
    baseline = comparison_fixture()
    candidate = copy.deepcopy(baseline)
    for values in candidate.values():
        values.update({"MAE": .7, "RMSE": .8, "NDCG@10": .11, "Recall@10": .12})
    assert selection_key(candidate, baseline, baseline)[1] < 1
    assert all(value["better"] for values in compare_goal(candidate, baseline, baseline).values() for value in values.values())
    del candidate["netflix_small"]
    with pytest.raises(ValueError, match="both"):
        selection_key(candidate, baseline, baseline)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_selection_rejects_nonfinite_results(value):
    baseline = comparison_fixture()
    candidate = copy.deepcopy(baseline)
    candidate["netflix_small"]["NDCG@10"] = value
    with pytest.raises(ValueError):
        selection_key(candidate, baseline, baseline)


def test_training_defaults_preserve_registered_prior_geometry_and_scaling():
    args = episode_arguments(SimpleNamespace(seed=99, ranking_query_mode="user_matched"))
    assert args.normalization == "two_way_user_scale"
    assert (args.rows, args.cols, args.parent_rows, args.parent_cols) == (1000, 1000, 12288, 4096)
    assert args.point_objective == "expected_user_rmse"
    assert args.selection_objective == "rating_guarded_bpr"
    assert args.seed == 99


def test_catalog_mixture_only_changes_synthetic_crop_selection(monkeypatch):
    monkeypatch.setattr(bpr_experiment, "training_episode", lambda arguments, step: (arguments.context_ordering, step))
    args = SimpleNamespace(seed=61, catalog_fraction=0, context_ordering="first_observation")
    assert ranking_episode(args, 20) == ("first_observation", 20)
    args.catalog_fraction = 1
    assert ranking_episode(args, 20) == ("first_observation_catalog", 20)
    assert args.context_ordering == "first_observation"
    args.catalog_fraction = .5
    assert ranking_episode(args, 20) == ranking_episode(args, 20)
    args.catalog_fraction = -1
    with pytest.raises(ValueError):
        ranking_episode(args, 20)
