from copy import deepcopy
import pytest
import torch

import train_single_matrixformer_top_two_blocks as training
import audit_single_model_top_two_blocks as audit


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
    assert 'view = top_two.TopTwoBlockParameterView(model)' in source
    assert 'with top_two.capture_coordinates(model, view):' in source
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
    assert 'len(selected) != 62' in source and '27766670' in source
    assert 'frozen["frozen_tensors"] != 192' in source
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
    parameters.extend([[f'transformer_blocks.6.weight_{index}', [1], 1] for index in range(28)])
    parameters.extend([[f'decoder.weight_{index}', [1], 1] for index in range(6)])
    parameters[-1][2] += 27766670 - 62
    parameters.append(['feature_encoder.weight', [1], 1])
    selected = parameters[:-1]
    audit.check_inventories(parameters, selected, parameters)
    with pytest.raises(ValueError):
        audit.check_inventories(parameters, selected[56:], parameters)
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


@pytest.mark.parametrize('filesystem,reserve,budget', [('checkpoint', 18, 1.5), ('root', 6, 16 / 1024), ('scratch', 50, 16)])
def test_training_admission_preserves_original_reserves(monkeypatch, filesystem, reserve, budget):
    monkeypatch.setattr(training.diagnostics.storage, 'require_mount', lambda: None)
    paths = {'checkpoint': training.ROOT, 'root': training.Path('/'), 'scratch': training.SCRATCH.parent}
    free = {path: 100 * 1024**3 for path in paths.values()}
    free[paths[filesystem]] = int((reserve + budget) * 1024**3) - 1
    monkeypatch.setattr(training.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': free[path]})())
    with pytest.raises(OSError):
        training.require_capacity()
    free[paths[filesystem]] += 1
    training.require_capacity()


def valid_preflight():
    frozen = {"frozen_tensors": 192, "changed_trainable_tensors": [str(index) for index in range(62)],
              "unselected_parameters_and_all_buffers_bitwise_unchanged": True}
    result = {"preflight_only": True, "record": {"accepted": True, "step": 1, "rate": 1e-5,
              "finite": {"all_retained": True, "accepted": True}},
              "trainable_parameters": 27766670, "backbone_parameters": 50231826, "frozen": frozen}
    audit_report = {"complete": True, "training_tasks": 28, "prediction_grids": 56,
                    "teacher_replay": {"complete": True, "full_grids": 28, "bitwise_equal": True},
                    "raw_endpoint": {"gradient_vectors": 84, "accepted_updates": 1, "accepted": True},
                    "autograd_replayed": False, "cuda_initialized": False, "frozen": deepcopy(frozen)}
    gate = {"passed": True, "preflight_only": True, "audit_sha256": "digest"}
    return [result, audit_report, gate, {"state": "complete", "accepted": True}]


@pytest.mark.parametrize("entry,path,value", [
    (0, ("trainable_parameters",), 19496843),
    (0, ("record", "rate"), 2e-5),
    (0, ("record", "accepted"), False),
    (0, ("record", "finite", "all_retained"), False),
    (0, ("frozen", "frozen_tensors"), 220),
    (0, ("frozen", "unselected_parameters_and_all_buffers_bitwise_unchanged"), False),
    (1, ("training_tasks",), 27),
    (1, ("prediction_grids",), 55),
    (1, ("teacher_replay", "full_grids"), 27),
    (1, ("teacher_replay", "complete"), False),
    (1, ("teacher_replay", "bitwise_equal"), False),
    (1, ("raw_endpoint", "gradient_vectors"), 28),
    (1, ("raw_endpoint", "accepted_updates"), 0),
    (1, ("raw_endpoint", "accepted"), False),
    (1, ("autograd_replayed",), True),
    (1, ("cuda_initialized",), True),
    (1, ("frozen", "changed_trainable_tensors"), []),
    (2, ("passed",), False),
    (2, ("audit_sha256",), "substituted"),
    (3, ("state",), "running"),
])
def test_preflight_evidence_cannot_be_incomplete_substituted_or_rejected(entry, path, value):
    records = valid_preflight()
    training.validate_preflight(*records, "digest")
    current = records[entry]
    for field in path[:-1]:
        current = current[field]
    current[path[-1]] = value
    with pytest.raises(ValueError, match="independently audited"):
        training.validate_preflight(*records, "digest")


def test_fresh_streams_do_not_reuse_preflight_or_consumed_v25_confirmation():
    seeds = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    consumed = training.original.alignment.fixed.seeds_in([
        training.preflight.batch(), training.preflight.preceding.training.confirmation_cases()])
    assert len(seeds) == 3868 and not seeds.intersection(consumed)
    assert training.DATA_SEED == 561100901 and training.GRADIENT_SEED == 562100901
    assert training.CONFIRMATION_SEEDS == (571100901, 572100901)


def test_scratch_diagnostics_are_registered_before_any_boundary_and_never_reused():
    source = training.ADAPTED_SOURCE
    assert 'diagnostics.require_empty_validation(config["diagnostic_storage"], directory)' in source
    assert 'diagnostics.verify(config["diagnostic_storage"], "training")' in source
    assert 'config["parent_selection_qualified"] is not True' in source
    assert '"top_two_block_training": True' in source
    assert 'diagnostics.prepare(config["diagnostic_storage"], "training")' in training.REGISTER_SOURCE
    assert training.PARENT == training.preflight.PARENT == training.original.alignment.fixed.PARENT
