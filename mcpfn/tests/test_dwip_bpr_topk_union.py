from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import dwip_bpr_experiment as experiment
from tabimpute.prior.recsys_rating_episode import sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import (
    synthetic_ranking_objective,
    topk_expected_pairwise_loss,
    topk_union_candidate_weights,
)


def slate_loss(prediction, probabilities, **kwargs):
    count = prediction.numel()
    return topk_expected_pairwise_loss(
        prediction, probabilities, torch.zeros(count, dtype=torch.long), torch.arange(count),
        torch.zeros(count, dtype=torch.bool), (1, count), candidate_scope="topk_union", **kwargs)


@pytest.mark.parametrize("tail_count", [0, 20, 180, 1980])
@pytest.mark.parametrize("negative_only", [False, True])
def test_easy_tail_does_not_dilute_correction_gradients(tail_count, negative_only):
    hard_scores = torch.cat((torch.linspace(4.5, 4.41, 10), torch.linspace(4.4, 4.31, 10)))
    hard_relevance = torch.tensor([.02] * 10 + [.2] * 10)
    reference = hard_scores.clone().requires_grad_()
    expected, _ = slate_loss(reference, hard_relevance, temperature=.1, negative_only=negative_only)
    expected.backward()
    prediction = torch.cat((hard_scores, torch.full((tail_count,), 2.))).requires_grad_()
    relevance = torch.cat((hard_relevance, torch.full((tail_count,), .0005)))
    actual, diagnostics = slate_loss(prediction, relevance, temperature=.1, negative_only=negative_only)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(prediction.grad[:20], reference.grad)
    assert torch.count_nonzero(prediction.grad[20:]) == 0
    assert diagnostics["ranking_effective_candidates_mean"] == 20
    assert prediction.grad[:10].sum() > 0
    if negative_only:
        assert torch.count_nonzero(prediction.grad[10:]) == 0
    else:
        assert prediction.grad[10:20].sum() < 0
        torch.testing.assert_close(prediction.grad.sum(), torch.tensor(0.), atol=1e-6, rtol=0)


def test_union_preserves_full_catalog_rank_discounts_and_unique_pairs():
    prediction = torch.tensor([4., 3.8, 3.6, 3.4, 3.2, 3.], dtype=torch.float64, requires_grad=True)
    relevance = torch.tensor([.1, .2, .05, .8, .6, .01], dtype=torch.float64)
    loss, diagnostics = slate_loss(prediction, relevance, top_k=2, temperature=.3)
    numerator = prediction.sum() * 0
    denominator = 0
    for anchor in range(2):
        for candidate in (1, 3, 4):
            if candidate <= anchor:
                continue
            difference = relevance[anchor] - relevance[candidate]
            discount = 1 / torch.tensor(anchor + 2., dtype=torch.float64).log2()
            if candidate < 2:
                discount -= 1 / torch.tensor(candidate + 2., dtype=torch.float64).log2()
            weight = difference.abs() * discount
            numerator = numerator + weight * torch.nn.functional.softplus(
                -difference.sign() * (prediction[anchor] - prediction[candidate]) / .3)
            denominator += weight
    torch.testing.assert_close(loss, numerator / denominator)
    actual_gradient = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(numerator / denominator, prediction)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert diagnostics["ranking_effective_candidates_mean"] == 4


def test_teacher_cutoff_ties_share_remaining_slots_without_index_bias():
    relevance = torch.tensor([[.05, .04, .8, .6, .6, .6, .01]], dtype=torch.float64)
    weights = topk_union_candidate_weights(relevance, torch.ones_like(relevance, dtype=torch.bool), 2)
    torch.testing.assert_close(weights, torch.tensor([[1., 1., 1., 1 / 3, 1 / 3, 1 / 3, 0.]], dtype=torch.float64))
    assert weights.sum() == 4


def test_teacher_cutoff_tie_loss_is_invariant_to_tail_item_permutation():
    prediction = torch.tensor([4.5, 4.4, 3., 2.5, 2., 1.5, 1.], requires_grad=True)
    relevance = torch.tensor([.05, .04, .8, .6, .6, .6, .01])
    expected, _ = slate_loss(prediction, relevance, top_k=2)
    expected.backward()
    permutation = torch.tensor([0, 1, 5, 6, 3, 2, 4])
    shuffled = prediction.detach()[permutation].requires_grad_()
    actual, _ = slate_loss(shuffled, relevance[permutation], top_k=2)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(shuffled.grad, prediction.grad[permutation])


def test_zero_teacher_tail_is_not_reintroduced_by_cutoff_ties():
    relevance = torch.tensor([[0., .4, 0., 0., .8, 0., 0.]])
    weights = topk_union_candidate_weights(relevance, torch.ones_like(relevance, dtype=torch.bool), 3)
    torch.testing.assert_close(weights, torch.tensor([[1., 1., 1., 0., 1., 0., 0.]]))


def test_sparse_catalog_uses_only_valid_entries_and_does_not_double_count_overlap():
    relevance = torch.tensor([[.8, .4, 0., 0., 0.], [.2, .9, .8, .3, .7]])
    valid = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])
    weights = topk_union_candidate_weights(relevance, valid, 4)
    torch.testing.assert_close(weights, valid.float())


@pytest.mark.parametrize("top_k", [1, 2, 10])
@pytest.mark.parametrize("mode", ["empty", "rating_only", "single", "zero", "equal"])
def test_degenerate_slates_have_finite_zero_gradients(top_k, mode):
    count = 0 if mode == "empty" else 1 if mode == "single" else 4
    prediction = torch.linspace(2., 4., count).requires_grad_()
    relevance = torch.full((count,), 0. if mode == "zero" else .2)
    selected = torch.full((count,), mode == "rating_only", dtype=torch.bool)
    loss, diagnostics = topk_expected_pairwise_loss(
        prediction, relevance, torch.zeros(count, dtype=torch.long), torch.arange(count), selected,
        (1, count), top_k=top_k, candidate_scope="topk_union")
    assert loss.item() == 0 and diagnostics["ranking_paired_users"] == 0
    assert diagnostics["ranking_candidate_scope"] == "topk_union"
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_user_macro_weighting_is_not_affected_by_other_users_catalog_sizes():
    prediction = torch.tensor([4., 3., 2., 1., 4., 3., 9.], requires_grad=True)
    relevance = torch.tensor([.1, .8, .01, .001, .2, .9, .7])
    users = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    items = torch.tensor([0, 2, 4, 6, 1, 3, 0])
    selected = torch.tensor([True, False, False, False, True, False, True])
    actual, diagnostics = topk_expected_pairwise_loss(
        prediction, relevance, users, items, selected, (3, 8), top_k=1, candidate_scope="topk_union")
    expected = (slate_loss(prediction[:4], relevance[:4], top_k=1)[0]
                + slate_loss(prediction[4:6], relevance[4:6], top_k=1)[0]) / 2
    torch.testing.assert_close(actual, expected)
    assert diagnostics["ranking_paired_users"] == 2
    actual.backward()
    assert prediction.grad[-1] == 0


@pytest.mark.parametrize("top_k", [4, 10])
@pytest.mark.parametrize("negative_only", [False, True])
def test_full_slate_union_matches_unrestricted_loss_and_gradients(top_k, negative_only):
    prediction = torch.tensor([3., 4., 2., 1.], requires_grad=True)
    arguments = (prediction, torch.tensor([.1, .8, 0., .3]), torch.zeros(4, dtype=torch.long),
                 torch.arange(4), torch.tensor([True, False, False, False]), (1, 4))
    original, _ = topk_expected_pairwise_loss(*arguments, top_k=top_k, negative_only=negative_only)
    union, _ = topk_expected_pairwise_loss(*arguments, top_k=top_k, negative_only=negative_only, candidate_scope="topk_union")
    torch.testing.assert_close(union, original, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(union, prediction, retain_graph=True)[0],
                               torch.autograd.grad(original, prediction)[0], atol=0, rtol=0)


def test_union_gradient_matches_finite_differences_away_from_score_ties():
    prediction = torch.tensor([4., 3.7, 3.4, 3.1, 2.8], dtype=torch.float64, requires_grad=True)
    relevance = torch.tensor([.1, .05, .8, .6, 0.], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda scores: slate_loss(scores, relevance, top_k=2)[0], (prediction,))


@pytest.mark.parametrize("scope,center", [("unknown", False), ("unknown", True)])
def test_invalid_scope_is_rejected(scope, center):
    with pytest.raises(ValueError, match="scope|centered"):
        topk_expected_pairwise_loss(None, None, None, None, None, (1, 2),
                                   candidate_scope=scope, negative_only=True, center_negative_only=center)


class Criterion:
    def mean(self, logits):
        return logits[:, 0]

    def __call__(self, logits, target):
        return (logits[:, 0] - target).square()


@pytest.fixture
def synthetic_episode():
    return sample_rating_episode(
        219, rows=16, cols=24, parent_rows=64, parent_cols=96, density_range=(.2, .2),
        ranking_query_mode="user_catalog", relevance_targets=True, relevance_target="visible_item_frequency",
        ranking_max_users=4, negatives_per_user=100, query_limit=50, ordinal_targets=True,
        point_target="expected_rating")


@pytest.mark.parametrize("centered", [False, True])
def test_union_objective_preserves_rating_loss_teacher_anchor_and_synthetic_guards(synthetic_episode, centered):
    episode = synthetic_episode
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    teacher = torch.full((int(episode.rating_query.sum()),), 3.)
    options = {"ranking_objective": "topk_expected_pairwise", "ranking_top_k": 2,
               "teacher_prediction": teacher, "teacher_anchor_weight": 5, "return_components": True,
               "ranking_boundary_weight": 1, "negative_only_ranking": centered,
               "center_negative_only_ranking": centered}
    original = synthetic_ranking_objective(logits, episode, Criterion(), **options)
    loss, diagnostics, components = synthetic_ranking_objective(
        logits, episode, Criterion(), ranking_candidate_scope="topk_union", **options)
    torch.testing.assert_close(components["protected"], original[2]["protected"], atol=0, rtol=0)
    torch.testing.assert_close(loss, components["protected"] + components["ranking"])
    assert diagnostics["visible_mask_teacher_verified"] is True
    assert diagnostics["ranking_candidate_scope"] == "topk_union"
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    for metadata in ({**episode.metadata, "gradient_data": "real"},):
        with pytest.raises(ValueError, match="synthetic"):
            synthetic_ranking_objective(logits, replace(episode, metadata=metadata), Criterion(),
                                        ranking_candidate_scope="topk_union", **options)
    with pytest.raises(ValueError, match="mask"):
        synthetic_ranking_objective(logits, replace(episode, relevance_probabilities=episode.relevance_probabilities + .01),
                                    Criterion(), ranking_candidate_scope="topk_union", **options)


@pytest.mark.parametrize("objective", ["pairwise", "expected_listwise"])
def test_union_requires_topk_objective(synthetic_episode, objective):
    with pytest.raises(ValueError, match="top-k"):
        synthetic_ranking_objective(None, synthetic_episode, Criterion(), ranking_objective=objective,
                                    ranking_candidate_scope="topk_union")


@pytest.mark.parametrize("objective,center", [("pairwise", False), ("expected_listwise", False),
                                            ("topk_expected_pairwise", True)])
def test_cli_configuration_rejects_incompatible_flags_before_loading_a_model(tmp_path, monkeypatch, objective, center):
    (tmp_path / "config.json").write_text('{"arguments": {}}')
    monkeypatch.setattr(experiment, "INITIAL", tmp_path / "checkpoint.pth")
    with pytest.raises(ValueError, match="top-k|centered"):
        experiment.episode_arguments(SimpleNamespace(ranking_candidate_scope="topk_union", ranking_objective=objective,
                                                    center_negative_only_ranking=center))


@pytest.mark.parametrize("count", [20, 40, 200, 2000])
def test_centered_union_removes_common_offset_without_easy_tail_dilution(count):
    prediction = torch.cat((torch.linspace(4.5, 4.41, 10), torch.linspace(4.4, 4.31, 10),
                            torch.full((count - 20,), 2.))).double().requires_grad_()
    relevance = torch.tensor([.02] * 10 + [.2] * 10 + [.0005] * (count - 20), dtype=torch.float64)
    original, _ = slate_loss(prediction, relevance, temperature=.1, negative_only=True)
    centered, diagnostics = slate_loss(prediction, relevance, temperature=.1, negative_only=True, center_negative_only=True)
    torch.testing.assert_close(centered, original, atol=0, rtol=0)
    raw_gradient = torch.autograd.grad(original, prediction, retain_graph=True)[0]
    actual = torch.autograd.grad(centered, prediction)[0]
    torch.testing.assert_close(actual, raw_gradient - raw_gradient.mean(), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(actual.sum(), torch.tensor(0., dtype=torch.float64), atol=1e-13, rtol=0)
    assert actual[:20].norm() > .7 * raw_gradient[:20].norm()
    if count > 20:
        torch.testing.assert_close(actual[20:], -raw_gradient.mean().expand(count - 20))
        assert torch.all(actual[20:] < 0)
    assert diagnostics["ranking_gradient_center_scope"] == "all_valid_queries_per_user"
    assert diagnostics["ranking_effective_candidates_mean"] == 20


def test_centered_union_cannot_optimize_a_pure_common_score_offset():
    offset = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    prediction = torch.tensor([4., 3.5, 3., 2.], dtype=torch.float64) + offset
    relevance = torch.tensor([.1, .8, .3, .01], dtype=torch.float64)
    original, _ = slate_loss(prediction, relevance, top_k=1, negative_only=True)
    centered, _ = slate_loss(prediction, relevance, top_k=1, negative_only=True, center_negative_only=True)
    torch.testing.assert_close(centered, original, atol=0, rtol=0)
    assert torch.autograd.grad(original, offset, retain_graph=True)[0] > 0
    torch.testing.assert_close(torch.autograd.grad(centered, offset)[0], torch.tensor(0., dtype=torch.float64), atol=1e-14, rtol=0)


def test_union_gradient_centering_is_user_local_and_ignores_absent_candidates():
    prediction = torch.tensor([4., 3., 2., 1., 4., 3., 9.], dtype=torch.float64, requires_grad=True)
    relevance = torch.tensor([.1, .8, .01, .001, .2, .9, .7], dtype=torch.float64)
    users = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    items = torch.tensor([0, 2, 4, 6, 1, 3, 0])
    selected = torch.tensor([True, False, False, False, True, False, True])
    arguments = (prediction, relevance, users, items, selected, (3, 8))
    original, _ = topk_expected_pairwise_loss(*arguments, top_k=1, candidate_scope="topk_union", negative_only=True)
    centered, _ = topk_expected_pairwise_loss(*arguments, top_k=1, candidate_scope="topk_union", negative_only=True,
                                            center_negative_only=True)
    expected = torch.autograd.grad(original, prediction, retain_graph=True)[0]
    for user in (0, 1):
        expected[users == user] -= expected[users == user].mean()
    actual = torch.autograd.grad(centered, prediction)[0]
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    assert actual[-1] == 0


def test_cli_accepts_centered_union_with_explicit_one_sided_gradients(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"arguments": {}}')
    monkeypatch.setattr(experiment, "INITIAL", tmp_path / "checkpoint.pth")
    args = experiment.episode_arguments(SimpleNamespace(ranking_candidate_scope="topk_union",
        ranking_objective="topk_expected_pairwise", negative_only_ranking=True, center_negative_only_ranking=True))
    assert args.negative_only_ranking and args.center_negative_only_ranking
