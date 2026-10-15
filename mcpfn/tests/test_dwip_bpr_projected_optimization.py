import pytest
import torch

from dwip_bpr_optimization import protected_optimizer_step
from tabimpute.prior.recsys_rating_episode import sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import synthetic_ranking_objective


def test_projection_protects_actual_adam_update_and_keeps_free_direction():
    parameter = torch.nn.Parameter(torch.tensor([1., 1.]))
    optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=0)
    parameter.grad = torch.tensor([-1., -1.])
    before = parameter.detach().clone()
    report = protected_optimizer_step(optimizer, (parameter,), (torch.tensor([1., 0.]),))
    assert report["protected_step_projected"] and report["protected_first_order_passed"]
    assert parameter[0] <= before[0] and parameter[1] > before[1]
    assert report["protected_dot_after"] <= 1e-7


def test_nonconflicting_optimizer_update_is_unchanged():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = torch.optim.AdamW([parameter], lr=.01, weight_decay=.02)
    ordinary = torch.optim.AdamW([reference], lr=.01, weight_decay=.02)
    parameter.grad = torch.tensor([1., 2.])
    reference.grad = parameter.grad.clone()
    ordinary.step()
    report = protected_optimizer_step(optimizer, (parameter,), (torch.tensor([1., 1.]),))
    assert not report["protected_step_projected"]
    torch.testing.assert_close(parameter, reference, atol=0, rtol=0)


def test_protection_accounts_for_adam_momentum_not_just_current_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=.1, weight_decay=0)
    for _ in range(5):
        parameter.grad = torch.tensor([-1.])
        optimizer.step()
    before = parameter.detach().clone()
    parameter.grad = torch.tensor([.1])
    report = protected_optimizer_step(optimizer, (parameter,), (torch.ones(1),))
    assert report["protected_step_projected"]
    assert parameter.item() <= before.item()


def test_projection_rejects_nonfinite_protection_before_changing_weights():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=.1)
    parameter.grad = torch.ones(1)
    with pytest.raises(ValueError, match="Nonfinite"):
        protected_optimizer_step(optimizer, (parameter,), (torch.tensor([float("nan")]),))
    assert parameter.item() == 1


def test_quantized_projection_stall_restores_protected_weights_without_relaxing_tolerance():
    parameter = torch.nn.Parameter(torch.tensor([1., 1e8]))
    independent = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.SGD([parameter, independent], lr=.1, momentum=.9)
    parameter.grad = torch.tensor([-1., 0.])
    independent.grad = torch.tensor([-1.])
    before = parameter.detach().clone()
    report = protected_optimizer_step(optimizer, (parameter, independent), (torch.tensor([1., 1000.]), None))
    assert report["protected_step_projected"] and report["protected_step_rolled_back"]
    assert report["protected_dot_after_projection"] > 1e-7
    assert report["protected_dot_after"] == 0 and report["protected_first_order_passed"]
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert independent.item() > 1
    torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], parameter.grad)


def test_synthetic_objective_components_support_two_backward_passes():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(205, rows=16, cols=20, parent_rows=32, parent_cols=64,
                                     density_range=(.2, .2), ordinal_targets=True, point_target="expected_rating",
                                     ranking_query_mode="user_catalog", relevance_targets=True, ranking_max_users=4,
                                     negatives_per_user=100, normalization="two_way_user_scale")
    logits = torch.nn.Parameter(torch.zeros(int(episode.query.sum()), 1))
    loss, _, components = synthetic_ranking_objective(logits, episode, Criterion(), ranking_objective="expected_listwise", return_components=True)
    protected = torch.autograd.grad(components["protected"], (logits,), retain_graph=True)
    loss.backward()
    optimizer = torch.optim.AdamW([logits], lr=.01)
    report = protected_optimizer_step(optimizer, (logits,), protected)
    assert report["protected_first_order_passed"]
    assert torch.isfinite(logits).all()
