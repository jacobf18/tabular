import pytest
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.prior.recsys_rating_episode import sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import raw_query_predictions, synthetic_ranking_objective


def synthetic_case(context_ordering, seed):
    episode = sample_rating_episode(
        seed, rows=24, cols=32, parent_rows=64, parent_cols=96, device='cpu',
        query_limit=96, density_range=(.15, .15), normalization='two_way_user_scale',
        point_target='expected_rating', ordinal_targets=True, context_ordering=context_ordering,
        ranking_query_mode='user_catalog', negatives_per_user=32, ranking_max_users=8,
        ranking_user_scope='parent_positive', relevance_targets=True,
        relevance_target='visible_item_frequency', exclude_unknown_queries=True,
    )
    criterion = FullSupportBarDistribution(torch.linspace(-3, 3, 33))
    logits = torch.randn((int(episode.query.sum()), 32), generator=torch.Generator().manual_seed(seed))
    selected = episode.rating_query[episode.query]
    prediction = raw_query_predictions(logits, episode, criterion)[0]
    return episode, criterion, logits, selected, prediction[selected].detach().clone()


def options():
    return dict(ranking_objective='topk_soft_choice', ranking_candidate_scope='topk_union',
                ranking_frequency_pseudocount=.5, ranking_boundary_weight=1, temperature=.1,
                teacher_anchor_weight=50, return_components=True)


@pytest.mark.parametrize('context_ordering', ['first_observation', 'first_observation_catalog'])
@pytest.mark.parametrize('seed', [71, 72])
def test_matching_detached_teacher_preserves_initial_loss_and_gradient(context_ordering, seed):
    episode, criterion, logits, _, teacher = synthetic_case(context_ordering, seed)
    logits.requires_grad_()
    baseline, _, baseline_components = synthetic_ranking_objective(
        logits, episode, criterion, **{**options(), 'teacher_anchor_weight': 0},
    )
    anchored, diagnostics, components = synthetic_ranking_objective(
        logits, episode, criterion, teacher_prediction=teacher, **options(),
    )
    assert diagnostics['synthetic_teacher_anchor'] == 0
    assert diagnostics['visible_mask_teacher_verified']
    torch.testing.assert_close(anchored, baseline, rtol=0, atol=0)
    torch.testing.assert_close(components['protected'], baseline_components['protected'], rtol=0, atol=0)
    actual = torch.autograd.grad(anchored, logits, retain_graph=True)[0]
    expected = torch.autograd.grad(baseline, logits)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('context_ordering', ['first_observation', 'first_observation_catalog'])
@pytest.mark.parametrize('seed', [71, 72])
def test_fixed_initial_teacher_restores_perturbed_rating_predictions(context_ordering, seed):
    episode, criterion, original, selected, teacher = synthetic_case(context_ordering, seed)
    logits = original.clone()
    logits[:, 0] += .5
    logits.requires_grad_()
    loss, diagnostics, _ = synthetic_ranking_objective(
        logits, episode, criterion, teacher_prediction=teacher,
        **{**options(), 'ranking_weight': 0, 'rating_weight': 0},
    )
    assert diagnostics['synthetic_teacher_anchor'] > 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[~selected]) == 0
    assert torch.count_nonzero(logits.grad[selected]) > 0
    direction = -logits.grad / logits.grad.norm()
    shifted = logits.detach() + .0001 * direction
    after, _, _ = synthetic_ranking_objective(
        shifted, episode, criterion, teacher_prediction=teacher,
        **{**options(), 'ranking_weight': 0, 'rating_weight': 0},
    )
    assert after < loss.detach()
    assert not teacher.requires_grad
