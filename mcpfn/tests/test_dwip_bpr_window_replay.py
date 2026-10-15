import copy
from types import SimpleNamespace

import pytest
import torch

from dwip_bpr_experiment import episode_arguments, student_training_logits
from dwip_bpr_window_training import recomputed_window_logits, windowed_training_logits
from tabimpute.prior.recsys_ranking_objective import topk_expected_pairwise_loss
from tabimpute.prior.recsys_soft_choice_objective import topk_soft_choice_loss


class FeatureEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 3)
        self.last_num_cls_rows = 0
        self.last_num_cls_cols = 0

    def forward(self, source):
        features = self.linear(source[..., None])
        return features + .1 * features.mean(2, keepdim=True)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(.25)

    def forward(self, features, **kwargs):
        return features + self.dropout(self.linear(features))


class WindowModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_encoder = FeatureEncoder()
        self.transformer_blocks = torch.nn.ModuleList([Block(), Block()])
        self.decoder = torch.nn.Linear(3, 2)

    def forward(self, source, decode_mask):
        features = self.feature_encoder(source)
        for block in self.transformer_blocks:
            features = block(features)
        return self.decoder(features[decode_mask])


def coupled_loss(logits, objective="dense"):
    if objective == "soft_choice":
        count = logits.shape[0]
        ranking = topk_soft_choice_loss(
            logits[:, 0], torch.linspace(.01, .99, count), torch.zeros(count, dtype=torch.long),
            torch.arange(count), torch.zeros(count, dtype=torch.bool), (1, count), top_k=2,
            frequency_smoothing=.00025)[0]
        return ranking + .1 * logits.square().mean()
    if objective in ("topk_union", "centered_union"):
        count = logits.shape[0]
        ranking = topk_expected_pairwise_loss(
            logits[:, 0], torch.linspace(.01, .99, count), torch.zeros(count, dtype=torch.long),
            torch.arange(count), torch.zeros(count, dtype=torch.bool), (1, count), top_k=2,
            candidate_scope="topk_union", negative_only=objective == "centered_union",
            center_negative_only=objective == "centered_union")[0]
        return ranking + .1 * logits.square().mean() if objective == "centered_union" else ranking
    return logits.square().mean() + (logits[::2].mean() - logits[1::2].mean()).square() + (logits * torch.rand_like(logits)).mean()


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("width", [0, 2, 3])
@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("objective", ["dense", "topk_union", "centered_union", "soft_choice"])
def test_replay_preserves_joint_loss_gradients_optimizer_update_and_rng(checkpointing, width, offload, objective):
    torch.manual_seed(271)
    original = WindowModel()
    replayed = copy.deepcopy(original)
    source = torch.randn(1, 4, 5)
    query = torch.rand(source.shape) > .3
    torch.manual_seed(272)
    expected = windowed_training_logits(original, source, query, checkpointing, width)
    expected_loss = coupled_loss(expected, objective)
    expected_loss.backward()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(272)
    actual, replay = recomputed_window_logits(replayed, source, query, checkpointing, width, offload)
    assert actual.is_leaf
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_loss = coupled_loss(actual, objective)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
    actual_loss.backward()
    assert all(parameter.grad is None for parameter in replayed.parameters())
    replay.backward(actual.grad)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for before, after in zip(original.parameters(), replayed.parameters()):
        torch.testing.assert_close(after.grad, before.grad, atol=1e-6, rtol=1e-5)
    torch.optim.AdamW(original.parameters(), lr=.001).step()
    torch.optim.AdamW(replayed.parameters(), lr=.001).step()
    for before, after in zip(original.parameters(), replayed.parameters()):
        torch.testing.assert_close(after, before, atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError, match="already"):
        replay.backward(actual.grad)


def test_replay_skips_empty_windows_and_keeps_partial_tail_queries():
    model = WindowModel()
    source = torch.randn(1, 3, 5)
    query = torch.zeros_like(source, dtype=torch.bool)
    query[:, :, :2] = True
    query[:, :, 4] = True
    logits, replay = recomputed_window_logits(model, source, query, window_cols=2)
    assert [(entry[0], entry[1]) for entry in replay.entries] == [(0, 2), (4, 5)]
    logits.square().sum().backward()
    assert replay.backward(logits.grad) == 2


def test_autocast_replay_keeps_logits_exact_with_only_accumulation_roundoff():
    torch.manual_seed(273)
    original = WindowModel()
    replayed = copy.deepcopy(original)
    source = torch.randn(1, 4, 6)
    query = torch.ones_like(source, dtype=torch.bool)

    def context():
        return torch.autocast("cpu", dtype=torch.bfloat16)

    torch.manual_seed(274)
    with context():
        expected = windowed_training_logits(original, source, query, window_cols=3)
    expected.float().square().mean().backward()
    torch.manual_seed(274)
    with context():
        actual, replay = recomputed_window_logits(replayed, source, query, window_cols=3)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.float().square().mean().backward()
    replay.backward(actual.grad, context)
    expected_gradient = torch.cat([parameter.grad.flatten() for parameter in original.parameters()])
    actual_gradient = torch.cat([parameter.grad.flatten() for parameter in replayed.parameters()])
    assert (actual_gradient - expected_gradient).norm() / expected_gradient.norm() < .02
    assert torch.nn.functional.cosine_similarity(actual_gradient, expected_gradient, dim=0) > .999


def test_replay_rejects_training_mode_changes():
    model = WindowModel()
    source = torch.randn(1, 3, 4)
    logits, replay = recomputed_window_logits(model, source, torch.ones_like(source, dtype=torch.bool), window_cols=2)
    logits.square().sum().backward()
    model.eval()
    with pytest.raises(ValueError, match="training mode changed"):
        replay.backward(logits.grad)


def test_replay_refuses_changed_logits_before_optimizer_update():
    model = WindowModel()
    source = torch.randn(1, 3, 4)
    query = torch.ones_like(source, dtype=torch.bool)
    logits, replay = recomputed_window_logits(model, source, query, window_cols=2)
    logits.square().mean().backward()
    with torch.no_grad():
        model.decoder.bias.add_(1)
    with pytest.raises(ValueError, match="Recomputed window logits differ"):
        replay.backward(logits.grad)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_offloaded_reference_logits_remain_detached_and_corruption_is_rejected(checkpointing):
    model = WindowModel()
    source = torch.randn(1, 4, 8)
    query = torch.ones_like(source, dtype=torch.bool)
    logits, replay = recomputed_window_logits(model, source, query, checkpointing, 2, True)
    assert len(replay.entries) == 4 and logits.is_leaf and logits.requires_grad
    assert all(entry[3].device.type == "cpu" and not entry[3].requires_grad for entry in replay.entries)
    logits.square().mean().backward()
    replay.entries[2][3].add_(1)
    with pytest.raises(ValueError, match="Recomputed window logits differ"):
        replay.backward(logits.grad)


@pytest.mark.parametrize("gradient", [None, torch.ones(1), torch.full((12, 2), torch.nan),
    torch.ones(12, 2, dtype=torch.float64), torch.ones(12, 2, device="meta")])
def test_replay_rejects_missing_misaligned_and_nonfinite_gradients(gradient):
    model = WindowModel()
    source = torch.randn(1, 3, 4)
    logits, replay = recomputed_window_logits(model, source, torch.ones_like(source, dtype=torch.bool), window_cols=2)
    with pytest.raises(ValueError, match="finite gradients"):
        replay.backward(gradient)


def test_replay_rejects_mutating_running_statistics_and_disabled_grad_context():
    source = torch.randn(1, 3, 4)
    query = torch.ones_like(source, dtype=torch.bool)
    with pytest.raises(ValueError, match="running-stat"):
        recomputed_window_logits(torch.nn.BatchNorm1d(3), source, query, window_cols=2)
    with torch.no_grad(), pytest.raises(ValueError, match="enabled training"):
        recomputed_window_logits(WindowModel(), source, query, window_cols=2)


def test_training_row_override_and_projection_guard():
    args = episode_arguments(SimpleNamespace(training_rows=2000))
    assert args.rows == 2000
    with pytest.raises(ValueError, match="two synthetic training rows"):
        episode_arguments(SimpleNamespace(training_rows=1))
    with pytest.raises(ValueError, match="cannot be combined"):
        episode_arguments(SimpleNamespace(recompute_window_gradients=True, project_rating_gradient=True))
    with pytest.raises(ValueError, match="requires window-gradient replay"):
        episode_arguments(SimpleNamespace(offload_window_activations=True, recompute_window_gradients=False))


def test_student_wrapper_defaults_to_original_graph_and_enables_replay_explicitly():
    model = WindowModel()
    source = torch.randn(1, 3, 4)
    episode = SimpleNamespace(source=source, query=torch.ones_like(source, dtype=torch.bool))
    args = SimpleNamespace(activation_checkpointing=False)
    ordinary, plan = student_training_logits(model, episode, args, 2)
    assert plan is None and not ordinary.is_leaf
    args.recompute_window_gradients = True
    replayed, plan = student_training_logits(model, episode, args, 2)
    assert replayed.is_leaf and plan is not None
