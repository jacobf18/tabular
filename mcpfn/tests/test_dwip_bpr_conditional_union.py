import pytest
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.prior.recsys_rating_episode import sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import raw_query_predictions, synthetic_ranking_objective


@pytest.mark.parametrize('context_ordering', ['first_observation', 'first_observation_catalog'])
@pytest.mark.parametrize('seed', [71, 72, 73])
def test_conditional_union_changes_only_synthetic_ranking_targets_and_gradients(context_ordering, seed):
    arguments = dict(seed=seed, rows=24, cols=32, parent_rows=64, parent_cols=96, device='cpu',
                     query_limit=96, density_range=(.15, .15), normalization='two_way_user_scale',
                     point_target='expected_rating', ordinal_targets=True, context_ordering=context_ordering,
                     ranking_query_mode='user_catalog', negatives_per_user=32, ranking_max_users=8,
                     ranking_user_scope='parent_positive', relevance_targets=True, exclude_unknown_queries=True)
    visible = sample_rating_episode(**arguments, relevance_target='visible_item_frequency')
    conditional = sample_rating_episode(**arguments, relevance_target='conditional')
    for name, value in vars(visible).items():
        if name == 'relevance_probabilities':
            continue
        counterpart = getattr(conditional, name)
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, counterpart, rtol=0, atol=0, equal_nan=True)
        elif name == 'metadata':
            assert {key: entry for key, entry in value.items() if key != 'relevance_target'} == {
                key: entry for key, entry in counterpart.items() if key != 'relevance_target'}
        else:
            assert value == counterpart
    assert conditional.metadata['gradient_data'] == 'synthetic'
    assert conditional.metadata['relevance_target'] == 'conditional'
    assert torch.isfinite(conditional.relevance_probabilities).all()
    assert torch.all((conditional.relevance_probabilities >= 0) & (conditional.relevance_probabilities <= 1))
    assert not conditional.relevance_probabilities.requires_grad
    assert not torch.equal(conditional.relevance_probabilities, visible.relevance_probabilities)
    assert not torch.equal(conditional.relevance_probabilities[:, 0], conditional.relevance_probabilities[:, 1])
    logits = torch.randn((int(visible.query.sum()), 32), generator=torch.Generator().manual_seed(seed)).requires_grad_()
    criterion = FullSupportBarDistribution(torch.linspace(-3, 3, 33))
    selected = visible.rating_query[visible.query]
    teacher = raw_query_predictions(logits, visible, criterion)[0][selected].detach() + .02
    options = dict(ranking_objective='topk_expected_pairwise', ranking_candidate_scope='topk_union',
                   ranking_boundary_weight=1, temperature=.1, teacher_prediction=teacher,
                   teacher_anchor_weight=50, return_components=True)
    visible_loss, _, visible_components = synthetic_ranking_objective(logits, visible, criterion, **options)
    conditional_loss, diagnostics, conditional_components = synthetic_ranking_objective(
        logits, conditional, criterion, **options)
    assert torch.isfinite(visible_loss) and torch.isfinite(conditional_loss)
    assert diagnostics['ranking_candidate_scope'] == 'topk_union'
    assert diagnostics['ranking_paired_users'] > 0
    torch.testing.assert_close(conditional_components['protected'], visible_components['protected'], rtol=0, atol=0)
    visible_protected = torch.autograd.grad(visible_components['protected'], logits, retain_graph=True)[0]
    conditional_protected = torch.autograd.grad(conditional_components['protected'], logits, retain_graph=True)[0]
    torch.testing.assert_close(conditional_protected, visible_protected, rtol=0, atol=0)
    visible_ranking = torch.autograd.grad(visible_components['ranking'], logits, retain_graph=True)[0]
    conditional_ranking = torch.autograd.grad(conditional_components['ranking'], logits)[0]
    assert torch.isfinite(visible_ranking).all() and torch.isfinite(conditional_ranking).all()
    assert not torch.equal(conditional_ranking, visible_ranking)
