from copy import deepcopy

import pytest
import torch

import preflight_single_model_top_block as preflight


def test_full_batch_retains_completed_v23_shapes_masses_and_teachers_with_fresh_seeds():
    actual = preflight.batch()
    previous = preflight.preceding.training.batches()[0]
    assert len(actual["tasks"]) == len(previous["tasks"]) == 28
    for task, original in zip(actual["tasks"], previous["tasks"]):
        normalized = deepcopy(task)
        normalized["gradient_seed"] = original["gradient_seed"]
        normalized["specification"].get("structured", normalized["specification"])["seed"] = (
            original["specification"].get("structured", original["specification"])["seed"])
        assert normalized == original
    seeds = preflight.training.alignment.fixed.seeds_in(actual)
    assert len(seeds) == 56
    assert not seeds.intersection(preflight.training.alignment.fixed.seeds_in(previous))
    assert not seeds.intersection(preflight.training.alignment.fixed.seeds_in(preflight.preceding.training.confirmation_cases()))
    roles = [preflight.expected_role(task) for task in actual["tasks"]]
    assert roles.count("general") == 24
    assert roles.count("initialization") == roles.count("specialist") == 2
    assert preflight.PARENT == preflight.training.alignment.fixed.PARENT
    assert preflight.RATE == 2e-5


def test_execution_and_audit_bind_existing_top_block_without_mutating_registered_sources():
    source = preflight.ADAPTED_SOURCE
    assert 'view = top_block.TopBlockParameterView(model)' in source
    assert 'with top_block.capture_coordinates(model, view):' in source
    assert source.count('row[0].startswith(PREFIXES)') == 2
    assert source.count('frozen = frozen_inventory(') == 2
    assert 'paired.protected_update(' in source and 'paired.check_endpoint(' in source
    assert 'hashes.update(audit_raw(payload, raw_path))' in source
    assert 'teacher_evidence = audit_teacher_replay(output, config)' in source
    assert '"selection_qualified": False' in source
    assert preflight.PREFIXES == ('transformer_blocks.7.', 'decoder.')
    assert 'TopBlockParameterView' not in preflight.template.ADAPTED_SOURCE
    assert preflight.inspect.unwrap(preflight.replay_targets).__globals__ is preflight.__dict__
    assert preflight.audit_teacher_replay.__globals__ is preflight.__dict__


def inventory():
    current = {f'transformer_blocks.7.weight_{index:02d}': torch.ones(2) for index in range(33)}
    current.update({'decoder.weight': torch.ones(3), 'transformer_blocks.6.weight': torch.zeros(2),
                    'other_buffer': torch.zeros(2),
                    'transformer_blocks.7.self_attention_between_features.rope.inv_freq': torch.zeros(2)})
    mutable = {name for name in current if name.startswith(preflight.PREFIXES) and not name.endswith('inv_freq')}
    return current, mutable


@pytest.mark.parametrize('buffer_name', ['other_buffer', 'transformer_blocks.7.self_attention_between_features.rope.inv_freq'])
def test_frozen_inventory_tracks_both_mutable_modules_and_preserves_signed_zero(monkeypatch, buffer_name):
    initial, mutable = inventory()
    monkeypatch.setattr(preflight, 'read', lambda path: [[name] for name in mutable])
    current = deepcopy(initial)
    current['transformer_blocks.7.weight_00'].add_(1)
    current['decoder.weight'].add_(1)
    result = preflight.frozen_inventory(initial, current, mutable)
    assert result['frozen_tensors'] == 3
    assert result['changed_trainable_tensors'] == ['transformer_blocks.7.weight_00', 'decoder.weight']
    current[buffer_name][0] = -0.
    with pytest.raises(ValueError, match='Frozen parameter or buffer'):
        preflight.frozen_inventory(initial, current, mutable)


@pytest.mark.parametrize('mutation', ['omitted', 'wrong_scope', 'shape', 'nonfinite'])
def test_invalid_state_inventory_cannot_pass(monkeypatch, mutation):
    initial, mutable = inventory()
    registered = [[name] for name in mutable]
    monkeypatch.setattr(preflight, 'read', lambda path: registered)
    current = deepcopy(initial)
    if mutation == 'omitted':
        current.pop('decoder.weight')
    elif mutation == 'wrong_scope':
        mutable.remove('decoder.weight')
        mutable.add('other_buffer')
    elif mutation == 'shape':
        current['decoder.weight'] = torch.ones(4)
    else:
        current['decoder.weight'][0] = float('nan')
    with pytest.raises(ValueError):
        preflight.frozen_inventory(initial, current, mutable)


def test_raw_scratch_evidence_preserves_every_gradient_and_exact_proposal_alias(tmp_path, monkeypatch):
    output, scratch = tmp_path / 'canonical', tmp_path / 'scratch'
    output.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(preflight, 'OUTPUT', output)
    monkeypatch.setattr(preflight, 'SCRATCH', scratch)
    gradients = torch.arange(84 * 8, dtype=torch.float32).reshape(84, 8)
    gradients[0, 0] = -0.
    payload = {'gradients': gradients, 'snapshot': {'task_gradients': gradients[:28].clone()},
               'record': {'composite_tasks': 28, 'protected_tasks': 84}, 'batch': {'tasks': [None] * 28}}
    path = output / 'raw_update.pt'
    digest = preflight.save_raw(payload, path)
    restored = torch.load(path, weights_only=True, mmap=True)
    assert digest == preflight.sha256(path)
    assert preflight.composite.byte_equal(restored['gradients'], gradients)
    assert path.is_symlink() and path.resolve() == scratch / path.name
    evidence = preflight.audit_raw(restored, path)
    preflight.verify_hashes(evidence)
    with pytest.raises(FileExistsError):
        preflight.save_raw(payload, path)
    restored['snapshot']['task_gradients'] = restored['snapshot']['task_gradients'].clone()
    with pytest.raises(ValueError):
        preflight.audit_raw(restored, path)


@pytest.mark.parametrize('filesystem,reserve,budget', [('checkpoint', 18, 1), ('root', 6, 0), ('scratch', 50, 12)])
def test_capacity_checks_all_reserves_before_any_work(monkeypatch, filesystem, reserve, budget):
    paths = {'checkpoint': preflight.ROOT, 'root': preflight.Path('/'), 'scratch': preflight.SCRATCH.parent}
    free = {path: 100 * 1024**3 for path in paths.values()}
    free[paths[filesystem]] = (reserve + budget) * 1024**3 - 1
    monkeypatch.setattr(preflight.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': free[path]})())
    with pytest.raises(OSError, match=f'{reserve} GiB reserve'):
        preflight.require_space()
    free[paths[filesystem]] += 1
    preflight.require_space()


@pytest.mark.parametrize('mutation', ['incomplete', 'qualified', 'unconfirmed', 'live', 'campaign', 'review'])
def test_no_new_gpu_attempt_before_terminal_unsuccessful_v23(monkeypatch, mutation):
    result = {'complete': True, 'gate': {'passed': False, 'confirmation_evaluated': True},
              'campaign': None, 'review': None}
    current = {'state': 'complete', 'pid': 999999999}
    if mutation == 'incomplete':
        result['complete'] = False
    elif mutation == 'qualified':
        result['gate']['passed'] = True
    elif mutation == 'unconfirmed':
        result['gate']['confirmation_evaluated'] = False
    elif mutation == 'live':
        current['pid'] = preflight.os.getpid()
    else:
        result[mutation] = 'existing'
    monkeypatch.setattr(preflight, 'read', lambda path: result if path.name == 'result.json' else current)
    monkeypatch.setattr(preflight.preceding, 'prerequisite', lambda path: pytest.fail('Never bypass terminal prerequisite'))
    with pytest.raises(ValueError, match='terminal V23'):
        preflight.require_predecessor()
