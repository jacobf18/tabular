import copy

import pytest
import torch

from dwip_bpr_optimization import EXPOSURE_PARAMETER_NAMES, configure_trainable_parameters, verify_frozen_parameter_state
from dwip_bpr_window_training import recomputed_window_logits, windowed_training_logits
from dwip_recsys_experiment import candidate_parameters, update_ema
from tabimpute.model.model_new_stable import TabImputeModelStable


def small_dwip():
    return TabImputeModelStable(embedding_size=16, num_attention_heads=2, mlp_hidden_size=32,
                               num_layers=2, num_outputs=5, num_cls=2, attention_dropout=0,
                               ffn_dropout=0, drop_path_rate=0)


def synthetic_queries():
    source = torch.randn(1, 4, 6)
    query = torch.zeros_like(source, dtype=torch.bool)
    query[:, 1, ::2] = True
    query[:, 2, 1::2] = True
    return source.masked_fill(query, torch.nan), query


def test_exposure_scope_is_exact_and_does_not_change_state():
    model = small_dwip()
    original = copy.deepcopy(model.state_dict())
    manifest = configure_trainable_parameters(model, "exposure_embeddings")
    assert set(manifest["parameter_names"]) == EXPOSURE_PARAMETER_NAMES
    assert manifest["parameter_tensors"] == 4
    assert manifest["parameter_elements"] == 7 * 16
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == EXPOSURE_PARAMETER_NAMES
    assert original.keys() == model.state_dict().keys()
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_synthetic_exposure_updates_and_ema_preserve_every_other_state_tensor(replay, activation_checkpointing):
    torch.manual_seed(310)
    model = small_dwip().train()
    initial = copy.deepcopy(model.state_dict())
    configure_trainable_parameters(model, "exposure_embeddings")
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=.01, weight_decay=.02)
    averaged = copy.deepcopy(initial)
    source, query = synthetic_queries()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        if replay:
            logits, plan = recomputed_window_logits(model, source, query, activation_checkpointing, window_cols=3)
        else:
            logits = windowed_training_logits(model, source, query, activation_checkpointing, window_cols=3)
        loss = logits.square().mean() + (logits[:3].mean() - logits[3:].mean()).square()
        loss.backward()
        if replay:
            assert plan.backward(logits.grad) == 2
        for name, parameter in model.named_parameters():
            if name in EXPOSURE_PARAMETER_NAMES:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            else:
                assert parameter.grad is None
        optimizer.step()
        update_ema(averaged, model.state_dict(), .95)
    expected_frozen = len(initial) - len(EXPOSURE_PARAMETER_NAMES)
    assert verify_frozen_parameter_state(model.state_dict(), initial, "exposure_embeddings") == expected_frozen
    assert verify_frozen_parameter_state(averaged, initial, "exposure_embeddings") == expected_frozen
    assert any(not torch.equal(model.state_dict()[name], initial[name]) for name in EXPOSURE_PARAMETER_NAMES)
    with candidate_parameters(model, averaged):
        assert verify_frozen_parameter_state(model.state_dict(), initial, "exposure_embeddings") == expected_frozen
    assert verify_frozen_parameter_state(model.state_dict(), initial, "exposure_embeddings") == expected_frozen


def test_full_scope_restores_original_trainability_and_update():
    torch.manual_seed(311)
    model = small_dwip().train()
    reference = copy.deepcopy(model)
    configure_trainable_parameters(model, "exposure_embeddings")
    manifest = configure_trainable_parameters(model)
    assert manifest["frozen_parameter_tensors"] == 0
    assert all(parameter.requires_grad for parameter in model.parameters())
    source, query = synthetic_queries()
    for selected in (model, reference):
        optimizer = torch.optim.AdamW(selected.parameters(), lr=.001)
        selected(source, decode_mask=query).square().mean().backward()
        optimizer.step()
    assert all(torch.equal(value, reference.state_dict()[name]) for name, value in model.state_dict().items())


def test_scope_rejects_unknown_or_missing_names_without_mutation():
    model = torch.nn.Linear(2, 3)
    for scope in ("unknown", "exposure_embeddings"):
        with pytest.raises(ValueError):
            configure_trainable_parameters(model, scope)
        assert all(parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(ValueError):
        configure_trainable_parameters(torch.nn.Identity())


def test_frozen_state_audit_rejects_changed_frozen_weights_and_buffers():
    model = small_dwip()
    initial = copy.deepcopy(model.state_dict())
    current = copy.deepcopy(initial)
    name = next(name for name in current if name not in EXPOSURE_PARAMETER_NAMES)
    current[name].add_(1)
    with pytest.raises(ValueError, match="Frozen model state changed"):
        verify_frozen_parameter_state(current, initial, "exposure_embeddings")
    assert verify_frozen_parameter_state(current, initial, "full") == 0
    with pytest.raises(ValueError, match="Unknown"):
        verify_frozen_parameter_state(current, initial, "unknown")
    with pytest.raises(ValueError, match="state keys"):
        verify_frozen_parameter_state({}, initial, "full")
    with pytest.raises(ValueError, match="parameter names"):
        verify_frozen_parameter_state({}, {}, "exposure_embeddings")
