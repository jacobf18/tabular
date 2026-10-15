from copy import deepcopy
import pytest
import torch

import train_single_matrixformer_top_block as training
import audit_single_model_top_block_training as audit


def test_complete_new_schedule_preserves_mixture_shapes_and_reserves_independent_seeds():
    batches = training.batches()
    previous = training.baseline.batches()
    assert len(batches) == len(previous) == 64
    assert training.BOUNDARIES == (0, 10, 25, 50, 64)
    for batch, reference in zip(batches, previous):
        assert batch['step'] == reference['step']
        assert len(batch['tasks']) == 28
        for task, original in zip(batch['tasks'], reference['tasks']):
            normalized = deepcopy(task)
            normalized['gradient_seed'] = original['gradient_seed']
            normalized['specification'].get('structured', normalized['specification'])['seed'] = (
                original['specification'].get('structured', original['specification'])['seed'])
            assert normalized == original
    seeds = training.original.alignment.fixed.seeds_in(batches)
    confirmation = training.original.alignment.fixed.seeds_in(training.confirmation_cases())
    assert len(seeds) == 3584 and len(confirmation) == 284
    assert not seeds.intersection(confirmation)
    assert not seeds.intersection(training.original.alignment.fixed.seeds_in(previous))
    assert not confirmation.intersection(training.original.alignment.fixed.seeds_in(training.baseline.confirmation_cases()))
    assert training.PARENT == training.original.alignment.fixed.PARENT


def test_optimizer_scope_storage_and_checkpoint_provenance_are_honest():
    source = training.ADAPTED_SOURCE
    assert 'view = top_block.TopBlockParameterView(model)' in source
    assert 'with top_block.capture_coordinates(model, view):' in source
    assert 'preflight.frozen_inventory(' in source
    assert '"decoder_only_training": False, "top_block_training": True' in source
    assert 'compact_cache.runtime()' in source
    assert 'compact_cache.capture_batch(' in source and 'compact_cache.evaluate_training(' in source
    assert 'paired.protected_update(' in source
    assert 'record["raw_endpoint_sha256"] = save_endpoint(' in source
    assert 'teacher_policy.runtime(' in source
    assert 'top_block_training' not in training.baseline.ADAPTED_SOURCE
    assert training.RATE == 1e-5
    assert all(training.constant_rate(step) == 1e-5 for step in range(1, 65))


@pytest.mark.parametrize('step', [0, 65, 1.5, True])
def test_rate_cannot_drift_outside_registered_attempts(step):
    with pytest.raises(ValueError):
        training.constant_rate(step)


def test_independent_audit_uses_exact_new_inventory_and_all_compact_teacher_replays():
    source = audit.ADAPTED_SOURCE
    assert 'len(selected) != 34' in source and '19496843' in source
    assert 'frozen["frozen_tensors"] != 220' in source
    assert '"changed_decoder_tensors"' not in source
    assert '"changed_trainable_tensors"' in source
    assert 'load_regenerated_task(path)' in source
    assert 'training.audit_endpoint(payload, raw_path)' in source
    assert 'training.constant_rate(step)' in source
    assert 'len(decision["comparisons"]) != 22' in source
    assert '152 * (3 + len(boundaries))' in source
    assert 'training.preflight.check_routing(payload, task, config)' in source
    assert audit.run_audit.__globals__ is audit.__dict__
    assert audit.load_regenerated_task.__globals__ is audit.__dict__
    assert audit.regeneration_evidence.__globals__ is audit.__dict__
    assert audit.training is training
    assert '284' in training.QUALIFICATION and '1136' in training.QUALIFICATION


def test_scope_inventory_rejects_decoder_only_or_added_buffers():
    parameters = [[f'transformer_blocks.7.weight_{index}', [1], 1] for index in range(28)]
    parameters.extend([[f'decoder.weight_{index}', [1], 1] for index in range(6)])
    parameters[-1][2] += 19496843 - 34
    parameters.append(['feature_encoder.weight', [1], 1])
    selected = parameters[:-1]
    audit.check_inventories(parameters, selected, parameters)
    with pytest.raises(ValueError):
        audit.check_inventories(parameters, selected[28:], parameters)
    with pytest.raises(ValueError):
        audit.check_inventories(parameters, selected + [['transformer_blocks.7.rope.inv_freq', [12], 12]], parameters)


def test_endpoint_keeps_all_protected_rows_and_exact_composite_prefix(tmp_path, monkeypatch):
    output, scratch = tmp_path / 'run', tmp_path / 'scratch'
    output.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(training, 'OUTPUT', output)
    monkeypatch.setattr(training, 'SCRATCH', scratch)
    monkeypatch.setattr(training.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': 100 * 1024**3})())
    gradients = torch.arange(84 * 8, dtype=torch.float32).reshape(84, 8)
    gradients[0, 0] = -0.
    payload = {'gradients': gradients, 'snapshot': {'task_gradients': gradients[:28].clone()},
               'record': {'composite_tasks': 28, 'protected_tasks': 84}, 'batch': {'tasks': [None] * 28}}
    path = output / 'raw_final_update.pt'
    digest = training.save_endpoint(payload, path)
    restored = torch.load(path, weights_only=True, mmap=True)
    assert training.sha256(path) == digest
    assert training.composite.byte_equal(restored['gradients'], gradients)
    training.verify_hashes(training.audit_endpoint(restored, path))
    with pytest.raises(FileExistsError):
        training.save_endpoint(payload, path)


@pytest.mark.parametrize('filesystem,reserve,budget', [('checkpoint', 18, 1.5), ('root', 6, 0), ('scratch', 50, 12)])
def test_training_admission_preserves_original_reserves(monkeypatch, filesystem, reserve, budget):
    paths = {'checkpoint': training.ROOT, 'root': training.Path('/'), 'scratch': training.SCRATCH.parent}
    free = {path: 100 * 1024**3 for path in paths.values()}
    free[paths[filesystem]] = int((reserve + budget) * 1024**3) - 1
    monkeypatch.setattr(training.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': free[path]})())
    with pytest.raises(OSError):
        training.require_capacity()
    free[paths[filesystem]] += 1
    training.require_capacity()
