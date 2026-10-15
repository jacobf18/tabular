import pytest
import torch
import torch.nn.functional as functional

from tabimpute.prior.recsys_soft_choice_objective import topk_soft_choice_loss


def slate_loss(scores, frequencies, top_k=10, smoothing=.00025, temperature=.1):
    users = torch.zeros(scores.numel(), dtype=torch.long)
    items = torch.arange(scores.numel())
    return topk_soft_choice_loss(
        scores, frequencies, users, items, torch.zeros_like(users, dtype=torch.bool),
        (1, scores.numel()), frequency_smoothing=smoothing, temperature=temperature, top_k=top_k,
    )


@pytest.mark.parametrize("frequencies", [[.05, .049], [.05, .005], [.05, 0.], [1., 0.]])
def test_smoothed_frequency_scores_have_finite_stationary_margin(frequencies):
    frequencies = torch.tensor(frequencies, dtype=torch.float64)
    scores = (3.5 + .1 * (frequencies + .00025).log()).requires_grad_()
    loss, diagnostics = slate_loss(scores, frequencies, top_k=2)
    loss.backward()
    assert torch.isfinite(loss)
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores), atol=1e-12, rtol=0)
    assert diagnostics["ranking_paired_users"] == 1


@pytest.mark.parametrize("margin,expected_sign", [(-.14, -1), (.8, 1)])
def test_wrong_order_and_overseparation_have_opposite_correction_directions(margin, expected_sign):
    margin = torch.tensor(margin, dtype=torch.float64, requires_grad=True)
    scores = torch.stack((3.5 + margin / 2, 3.5 - margin / 2))
    loss, _ = slate_loss(scores, torch.tensor([.05, .005], dtype=torch.float64), top_k=2)
    loss.backward()
    assert margin.grad.item() * expected_sign > 0


def test_full_union_at_finite_teacher_scores_is_stationary():
    frequencies = torch.linspace(.001, .1, 30, dtype=torch.float64)
    scores = (3.5 + .1 * (frequencies + .00025).log()).requires_grad_()
    loss, diagnostics = slate_loss(scores, frequencies)
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores), atol=1e-12, rtol=0)
    assert diagnostics["ranking_effective_candidates_mean"] == 10


@pytest.mark.parametrize("tail_count", [0, 20, 180, 1980])
def test_soft_choice_keeps_union_tail_exclusion_and_gradient_scale(tail_count):
    hard_scores = torch.cat((torch.full((10,), 4.14), torch.full((10,), 4.))).double()
    hard_rates = torch.cat((torch.full((10,), .005), torch.full((10,), .05))).double()
    reference = hard_scores.clone().requires_grad_()
    expected, _ = slate_loss(reference, hard_rates)
    expected.backward()
    scores = torch.cat((hard_scores, torch.full((tail_count,), 2., dtype=torch.float64))).requires_grad_()
    frequencies = torch.cat((hard_rates, torch.full((tail_count,), .0005, dtype=torch.float64)))
    actual, diagnostics = slate_loss(scores, frequencies)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(scores.grad[:20], reference.grad)
    assert torch.count_nonzero(scores.grad[20:]) == 0
    assert diagnostics["ranking_effective_candidates_mean"] == 20


def test_union_matches_explicit_soft_pair_enumeration():
    scores = torch.tensor([4., 3.8, 3.6, 3.4, 3.2, 3.], dtype=torch.float64, requires_grad=True)
    frequencies = torch.tensor([.1, .2, .05, .8, .6, .01], dtype=torch.float64)
    actual, _ = slate_loss(scores, frequencies, top_k=2, smoothing=.001, temperature=.3)
    numerator, denominator = scores.sum() * 0, 0
    for anchor in range(2):
        for candidate in (1, 3, 4):
            if candidate <= anchor:
                continue
            discount = 1 / torch.tensor(anchor + 2., dtype=torch.float64).log2()
            if candidate < 2:
                discount -= 1 / torch.tensor(candidate + 2., dtype=torch.float64).log2()
            weight = (frequencies[anchor] - frequencies[candidate]).abs() * discount
            choice = (frequencies[anchor] + .001) / (frequencies[anchor] + frequencies[candidate] + .002)
            gap = (scores[anchor] - scores[candidate]) / .3
            numerator = numerator + weight * (choice * functional.softplus(-gap) + (1 - choice) * functional.softplus(gap))
            denominator += weight
    expected = numerator / denominator
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, scores, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, scores)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    torch.testing.assert_close(actual_gradient.sum(), torch.tensor(0., dtype=torch.float64), atol=1e-12, rtol=0)


def test_cutoff_ties_preserve_item_permutation_equivariance():
    scores = torch.tensor([4.5, 4.4, 3., 2.5, 2., 1.5, 1.], dtype=torch.float64, requires_grad=True)
    frequencies = torch.tensor([.05, .04, .8, .6, .6, .6, .01], dtype=torch.float64)
    expected, _ = slate_loss(scores, frequencies, top_k=2)
    expected.backward()
    permutation = torch.tensor([0, 1, 5, 6, 3, 2, 4])
    shuffled = scores.detach()[permutation].requires_grad_()
    actual, _ = slate_loss(shuffled, frequencies[permutation], top_k=2)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(shuffled.grad, scores.grad[permutation])


def test_macro_users_sparse_queries_and_independent_offsets():
    scores = torch.tensor([4.3, 3.9, 3.2, 4.2, 4., 3.4, 3.], dtype=torch.float64, requires_grad=True)
    rates = torch.tensor([.01, .1, .03, .05, .01, .1, .03], dtype=torch.float64)
    users = torch.tensor([0, 0, 0, 2, 2, 2, 2])
    items = torch.tensor([0, 2, 4, 0, 1, 3, 4])
    selected = torch.zeros(7, dtype=torch.bool)
    actual, diagnostics = topk_soft_choice_loss(scores, rates, users, items, selected, (3, 5), frequency_smoothing=.00025, temperature=.1, top_k=2)
    expected = torch.stack([slate_loss(scores[users == user], rates[users == user], top_k=2)[0] for user in (0, 2)]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    for user in (0, 2):
        torch.testing.assert_close(scores.grad[users == user].sum(), torch.tensor(0., dtype=torch.float64), atol=1e-12, rtol=0)
    shifted, _ = topk_soft_choice_loss(scores.detach() + torch.tensor([.2, .2, .2, -.3, -.3, -.3, -.3]), rates, users, items, selected, (3, 5), frequency_smoothing=.00025, temperature=.1, top_k=2)
    torch.testing.assert_close(shifted, actual)
    assert diagnostics["ranking_paired_users"] == 2


@pytest.mark.parametrize("rates", [[0., 0., 0.], [.1, .1, .1]])
def test_zero_pair_mass_has_finite_differentiable_zero(rates):
    scores = torch.tensor([4., 3., 2.], requires_grad=True)
    loss, diagnostics = slate_loss(scores, torch.tensor(rates), top_k=2)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(scores.grad) == 0
    assert diagnostics["ranking_paired_users"] == 0


def test_no_additional_queries_gives_differentiable_zero():
    scores = torch.tensor([4., 3.], requires_grad=True)
    loss, _ = topk_soft_choice_loss(scores, torch.tensor([.1, .2]), torch.zeros(2, dtype=torch.long), torch.arange(2), torch.ones(2, dtype=torch.bool), (1, 2), frequency_smoothing=.00025)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(scores.grad) == 0


@pytest.mark.parametrize("smoothing", [0., -1., float('inf'), float('nan')])
def test_invalid_smoothing_is_rejected(smoothing):
    with pytest.raises(ValueError, match="smoothing"):
        slate_loss(torch.tensor([4., 3.]), torch.tensor([.1, .2]), smoothing=smoothing)


@pytest.mark.parametrize("temperature", [0., -1., float('inf'), float('nan')])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        slate_loss(torch.tensor([4., 3.]), torch.tensor([.1, .2]), temperature=temperature)


@pytest.mark.parametrize("frequencies", [torch.tensor([.1, .2], requires_grad=True), torch.tensor([-.1, .2]), torch.tensor([.1, 1.1]), torch.tensor([float('nan'), .2])])
def test_invalid_or_live_teacher_is_rejected(frequencies):
    with pytest.raises(ValueError, match="detached"):
        slate_loss(torch.tensor([4., 3.]), frequencies)


def test_nonfinite_scores_and_misaligned_queries_are_rejected():
    with pytest.raises(ValueError, match="finite query"):
        slate_loss(torch.tensor([float('inf'), 3.]), torch.tensor([.1, .2]))
    with pytest.raises(ValueError, match="aligned"):
        slate_loss(torch.tensor([4., 3.]), torch.tensor([.1]))
