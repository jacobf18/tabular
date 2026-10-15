import json
import sys
from types import SimpleNamespace

import pytest
import torch

import dwip_recsys_experiment as experiment
import dwip_recsys_provenance as provenance
import evaluate_dwip_recsys_candidate as evaluation
import preflight_dwip_recsys_training as preflight
from tabimpute.prior.recsys_rating_episode import RatingEpisode, rating_objective, sample_rating_episode


@pytest.fixture
def clean_root(tmp_path, monkeypatch):
    checkpoint = tmp_path / "root.pth"
    checkpoint.write_bytes(b"audited synthetic root fixture")
    monkeypatch.setattr(provenance, "AUDITED_ROOTS", {provenance.file_sha256(checkpoint): "fixture"})
    return checkpoint


def child_checkpoint(parent, folder):
    folder.mkdir()
    checkpoint = folder / "best_validation_checkpoint.pth"
    checkpoint.write_bytes(b"synthetic fixture " + str(folder).encode())
    config = {"training_policy": provenance.POLICY, "training_data": provenance.TRAINING_DATA,
              "initial_checkpoint": str(parent.resolve()), "initial_sha256": provenance.file_sha256(parent)}
    (folder / "config.json").write_text(json.dumps(config))
    provenance.write_synthetic_receipt(checkpoint)
    return checkpoint


def test_synthetic_lineage_accepts_audited_roots_and_verified_descendants(clean_root, tmp_path):
    child = child_checkpoint(clean_root, tmp_path / "child")
    grandchild = child_checkpoint(child, tmp_path / "grandchild")
    audit = provenance.require_synthetic_checkpoint(grandchild)
    assert audit["eligible"] is True
    assert audit["audited_root"] == provenance.file_sha256(clean_root)
    assert audit["checkpoint_sha256"] == provenance.file_sha256(grandchild)


def test_unknown_checkpoint_rejected_even_with_synthetic_config_claim(tmp_path):
    checkpoint = tmp_path / "unknown.pth"
    checkpoint.write_bytes(b"unknown weights")
    (tmp_path / "config.json").write_text(json.dumps({"training_data": provenance.TRAINING_DATA}))
    with pytest.raises(ValueError, match="Unknown checkpoint provenance"):
        provenance.require_synthetic_checkpoint(checkpoint)


def test_real_trained_checkpoint_and_synthetic_descendants_rejected(tmp_path, monkeypatch):
    checkpoint = tmp_path / "renamed-real.pth"
    checkpoint.write_bytes(b"real gradient updates")
    monkeypatch.setattr(provenance, "INELIGIBLE_HASHES", {provenance.file_sha256(checkpoint)})
    child = child_checkpoint(checkpoint, tmp_path / "synthetic-child")
    grandchild = child_checkpoint(child, tmp_path / "synthetic-grandchild")
    for candidate in (checkpoint, child, grandchild):
        with pytest.raises(ValueError, match="Ineligible real-trained"):
            provenance.require_synthetic_checkpoint(candidate)


@pytest.mark.parametrize("target", ["checkpoint", "config", "parent"])
def test_lineage_rejects_changed_content(clean_root, tmp_path, target):
    child = child_checkpoint(clean_root, tmp_path / "child")
    path = {"checkpoint": child, "config": child.parent / "config.json", "parent": clean_root}[target]
    with path.open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch|Unknown checkpoint provenance"):
        provenance.require_synthetic_checkpoint(child)


@pytest.mark.parametrize("change", [
    {"training_data": "Masked inner-training ratings plus synthetic regularization"},
    {"arguments": {"real_adaptation": True}},
    {"adaptation_provenance": {"real": "data"}},
    {"adaptation_schedule": "mixed"},
    {"training_policy": "other"},
])
def test_non_synthetic_manifest_rejected_even_with_current_receipt(clean_root, tmp_path, change):
    child = child_checkpoint(clean_root, tmp_path / "child")
    config_path = child.parent / "config.json"
    config = json.loads(config_path.read_text())
    config.update(change)
    config_path.write_text(json.dumps(config))
    provenance.write_synthetic_receipt(child)
    with pytest.raises(ValueError, match="non-synthetic"):
        provenance.require_synthetic_checkpoint(child)


def test_lineage_cycle_rejected(clean_root, tmp_path):
    child = child_checkpoint(clean_root, tmp_path / "child")
    config_path = child.parent / "config.json"
    config = json.loads(config_path.read_text())
    config["initial_checkpoint"] = str(child.resolve())
    config_path.write_text(json.dumps(config))
    provenance.write_synthetic_receipt(child)
    with pytest.raises(ValueError, match="Cyclic"):
        provenance.require_synthetic_checkpoint(child)


def test_training_checkpoint_writer_produces_verifiable_receipt(clean_root, tmp_path, monkeypatch):
    child = child_checkpoint(clean_root, tmp_path / "child")
    monkeypatch.setattr(experiment.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3))
    experiment.save_training_checkpoint(child, {"model": torch.ones(1), "step": 1})
    assert provenance.require_synthetic_checkpoint(child)["eligible"] is True


def test_every_training_step_uses_only_synthetic_sampler(monkeypatch):
    args = SimpleNamespace(seed=56, rows=24, cols=20, normalization="two_way", mae_weight=.1,
                           density_low=.002, density_high=.08, parent_rows=80, parent_cols=64,
                           point_target="expected_rating", item_popularity_low=1.4, item_popularity_high=2.8,
                           context_ordering="first_observation", query_sampling="user_balanced")
    seeds = []

    def sampler(seed, **kwargs):
        seeds.append(seed)
        return SimpleNamespace(metadata={"gradient_data": "synthetic"})

    def forbidden(*args, **kwargs):
        pytest.fail("Training accessed real evaluation data")

    monkeypatch.setattr(experiment, "sample_rating_episode", sampler)
    monkeypatch.setattr(experiment, "load_split", forbidden)
    for step in range(1, 11):
        provenance.require_synthetic_episode(experiment.training_episode(args, step))
    assert seeds == list(range(56000001, 56000011))
    with pytest.raises(TypeError):
        experiment.training_episode(args, 1, {"real": torch.ones(2, 2)})
    assert not hasattr(experiment, "load_adaptation_sources")


def test_synthetic_generator_explicitly_labels_gradient_data():
    episode = sample_rating_episode(52, 24, 20, device="cpu", parent_rows=48)
    provenance.require_synthetic_episode(episode)


@pytest.mark.parametrize("metadata", [{}, {"gradient_data": "inner_training_ratings"}, {"gradient_data": "unknown"}])
def test_non_synthetic_episode_rejected(metadata):
    with pytest.raises(ValueError, match="Only explicitly synthetic"):
        provenance.require_synthetic_episode(SimpleNamespace(metadata=metadata))


def test_training_rejects_prohibited_mode_before_cuda_or_data_access():
    with pytest.raises(ValueError, match="prohibited"):
        experiment.train(SimpleNamespace(real_adaptation=True))


def test_training_rejects_unknown_initialization_before_cuda_or_data_access(tmp_path):
    checkpoint = tmp_path / "unknown.pth"
    checkpoint.write_bytes(b"unknown")
    with pytest.raises(ValueError, match="Unknown checkpoint provenance"):
        experiment.train(SimpleNamespace(checkpoint=checkpoint))


@pytest.mark.parametrize("module,arguments", [
    (experiment, ["train"]), (preflight, ["--checkpoint", "unused", "--output", "unused"]),
])
def test_real_adaptation_cli_flag_is_rejected(module, arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__file__, *arguments, "--real-adaptation"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


@pytest.mark.parametrize("module", [preflight, evaluation])
def test_entrypoints_reject_unknown_weights_before_creating_outputs(module, tmp_path, monkeypatch):
    checkpoint = tmp_path / "unknown.pth"
    checkpoint.write_bytes(b"unknown")
    output = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", [module.__file__, "--checkpoint", str(checkpoint), "--output", str(output)])
    with pytest.raises(ValueError, match="Unknown checkpoint provenance"):
        module.main()
    assert not output.exists()


def test_configurable_storage_guard_keeps_default_and_honors_explicit_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment.shutil, "disk_usage", lambda path: SimpleNamespace(free=18 * 1024**3))
    path = tmp_path / "checkpoint.pth"
    with pytest.raises(RuntimeError, match="18 GiB"):
        experiment.save_checkpoint(path, {"model": torch.ones(1)})
    experiment.save_checkpoint(path, {"model": torch.ones(1)}, reserve_gib=17)
    assert path.exists()
    with pytest.raises(ValueError, match="finite positive"):
        experiment.save_checkpoint(tmp_path / "invalid.pth", {}, reserve_gib=float("nan"))


def test_user_rmse_objective_matches_macro_over_users_not_root_of_macro_mse():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    query = torch.tensor([[[True, False], [True, True]]])
    episode = RatingEpisode(source=torch.full((1, 2, 2), torch.nan), target=torch.ones((1, 2, 2)),
                            query=query, rating_query=query, means=torch.zeros(2), stds=torch.ones(2), metadata={})
    logits = torch.tensor([[2.], [4.], [5.]], requires_grad=True)
    loss, diagnostics = rating_objective(logits, episode, Criterion(), nll_weight=0, point_objective="user_rmse")
    expected = ((torch.tensor(1.) + 1e-6).sqrt() + (torch.tensor(12.5) + 1e-6).sqrt()) / 2
    torch.testing.assert_close(loss, expected)
    assert diagnostics["mse"] == pytest.approx(6.75)
    assert loss.item() != pytest.approx(6.75**.5)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    perfect = torch.ones_like(logits, requires_grad=True)
    exact, _ = rating_objective(perfect, episode, Criterion(), nll_weight=0, point_objective="user_rmse")
    exact.backward()
    assert torch.isfinite(perfect.grad).all()


def test_synthetic_known_rating_ranking_uses_low_ratings_and_averages_users():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    target = torch.tensor([[[5., 2., 4.], [1., 5., 3.]]])
    query = torch.ones_like(target, dtype=torch.bool)
    episode = RatingEpisode(source=torch.full_like(target, torch.nan), target=target, query=query,
                            rating_query=query, means=torch.zeros(3), stds=torch.ones(3), metadata={})
    logits = torch.tensor([[4.8], [2.9], [3.8], [1.5], [4.5], [2.7]], requires_grad=True)
    loss, diagnostics = rating_objective(logits, episode, Criterion(), nll_weight=0, ranking_weight=.05,
                                         ranking_negatives="observed_low")
    expected = (torch.nn.functional.softplus(torch.tensor([2.9 - 4.8, 2.9 - 3.8])).mean()
                + torch.nn.functional.softplus(torch.tensor([1.5 - 4.5, 2.7 - 4.5])).mean()) / 2
    assert diagnostics["ranking"] == pytest.approx(expected.item())
    _, control = rating_objective(logits, episode, Criterion(), nll_weight=0, ranking_weight=.05)
    assert control["ranking"] == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
