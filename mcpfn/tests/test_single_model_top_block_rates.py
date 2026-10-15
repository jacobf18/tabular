from copy import deepcopy

import pytest
import torch

import probe_single_model_top_block_rates as probe


def paired_metrics(value):
    return {f"legacy_draw_{draw}/{metric}": value for draw in range(2) for metric in ("rmse", "common_nrmse")}


def test_fixed_smaller_rates_preserve_all_tasks_coordinates_and_paired_rules():
    assert probe.RATES == (2e-5, 1e-5, 5e-6, 2e-6)
    assert probe.source.RATE == probe.RATES[0]
    assert probe.source.PARENT == probe.source.training.alignment.fixed.PARENT
    assert probe.WAIT_FOR == probe.source.preceding.OUTPUT
    assert probe.REPEATS == (0, 1)
    source = probe.ADAPTED_SOURCE
    assert 'view = top_block.TopBlockParameterView(model)' in source
    assert 'paired.protected_update(' in source and 'paired.check_endpoint(' in source
    assert 'paired.check_record(' in source
    assert 'paired.metric_decision(' in source
    assert 'source.frozen_inventory(' in source
    assert source.count('check_reference(original, record, snapshot, directory / "candidate_parameters.pt")') == 2
    assert 'restore_snapshot(payload["snapshot"], original["gradients"], 28)' in source
    assert '"training_tasks": 28' in source and 'grids += 56' in source
    assert 'TopBlockParameterView' not in probe.inspect.getsource(probe.template)


def test_paired_repeat_feasibility_retains_every_repeat_and_reports_individual_regressions():
    before = {str(repeat): paired_metrics(1.) for repeat in probe.REPEATS}
    after = deepcopy(before)
    for current in after.values():
        current['legacy_draw_0/rmse'] = 1.05
        current['legacy_draw_1/rmse'] = .9
    decision = probe.repeat_decision(before, after)
    assert decision['finite_feasible']
    assert not decision['repetitions']['0']['individual_draw_diagnostics']['all_retained']
    after['1']['legacy_draw_1/rmse'] = .96
    assert not probe.repeat_decision(before, after)['finite_feasible']
    with pytest.raises(ValueError, match='Both'):
        probe.repeat_decision(before, {'0': after['0']})


def test_compact_gradient_reference_requires_exact_bytes_including_signed_zero():
    gradients = torch.zeros(6, 2)
    snapshot = {'task_gradients': gradients[:2].clone()}
    assert 'task_gradients' not in probe.compact_snapshot(snapshot, gradients, 2)
    snapshot['task_gradients'][0, 0] = -0.
    with pytest.raises(ValueError, match='exactly duplicated'):
        probe.compact_snapshot(snapshot, gradients, 2)


@pytest.mark.parametrize('rate', probe.RATES)
@pytest.mark.parametrize('accepted', [True, False])
def test_actual_paired_adam_roundtrip_and_exact_rejection_reset(rate, accepted):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    parent = deepcopy(model.state_dict())
    gradients = torch.ones(6, 2)
    optimizer = probe.fresh_optimizer(model, .02)
    initial_optimizer = deepcopy(optimizer.state_dict())
    record, snapshot = probe.paired.protected_update(model, optimizer, gradients, [.5, .5], rate,
        paired_metrics(1.), lambda current: paired_metrics(float(current.weight[0, 0].detach()) if accepted else 1.1))
    record.update(step=1, accepted_updates=int(record['accepted']))
    assert record['accepted'] is accepted
    compact = probe.compact_snapshot(snapshot, gradients, 2)
    restored = probe.restore_snapshot(compact, gradients, 2)
    probe.paired.check_record(record, [.5, .5], paired_metrics(1.), 6)
    assert snapshot['optimizer_before']['param_groups'][0]['lr'] == rate
    expected_optimizer = snapshot['optimizer_after'] if accepted else initial_optimizer
    probe.source.checks.optimizer_equal(optimizer.state_dict(), expected_optimizer)
    for name in ('before', 'tentative', 'after', 'task_gradients', 'combined_gradient', 'applied_gradient'):
        assert probe.composite.byte_equal(restored[name], snapshot[name])
    if not accepted:
        assert torch.equal(model.weight, parent['weight'])
        assert not optimizer.state
        assert optimizer.param_groups[0]['lr'] == probe.source.RATE
    assert not probe.fresh_optimizer(model, .02).state


def test_reference_replay_checks_rejected_candidate_not_just_rolled_back_endpoint(tmp_path, monkeypatch):
    source, current = tmp_path / 'source', tmp_path / 'candidate'
    source.mkdir()
    current.mkdir()
    monkeypatch.setattr(probe.source, 'OUTPUT', source)
    candidate = torch.tensor([0., 1.])
    torch.save(candidate, source / 'candidate_parameters.pt')
    path = current / 'candidate_parameters.pt'
    torch.save(candidate.clone(), path)
    optimizer = probe.fresh_optimizer(torch.nn.Linear(2, 1, bias=False), .02)
    snapshot = {name: torch.ones(2) for name in ('before', 'tentative', 'after', 'combined_gradient',
                                               'applied_gradient', 'task_gradients')}
    snapshot.update(optimizer_before=optimizer.state_dict(), optimizer_after=optimizer.state_dict())
    record = {'accepted': False, 'reason': 'finite_metrics_rejected'}
    original = {'record': record, 'snapshot': deepcopy(snapshot)}
    probe.check_reference(original, record, snapshot, path)
    candidate[0] = -0.
    torch.save(candidate, path)
    with pytest.raises(ValueError, match='rejected candidate parameters'):
        probe.check_reference(original, record, snapshot, path)


def test_scratch_endpoint_preserves_bytes_and_cannot_overwrite_or_change_placement(tmp_path, monkeypatch):
    output, scratch = tmp_path / 'canonical', tmp_path / 'scratch'
    (output / 'rate_0').mkdir(parents=True)
    scratch.mkdir()
    monkeypatch.setattr(probe, 'OUTPUT', output)
    monkeypatch.setattr(probe, 'SCRATCH', scratch)
    monkeypatch.setattr(probe.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': 100 * 1024**3})())
    payload = {'vector': torch.tensor([-0., 1.]), 'declared': 'compact endpoint'}
    path = output / 'rate_0/endpoint.pt'
    digest = probe.save_endpoint(payload, path)
    restored = torch.load(path, weights_only=True, mmap=True)
    assert probe.composite.byte_equal(payload['vector'], restored['vector'])
    assert digest == probe.sha256(path)
    probe.verify_hashes(probe.audit_endpoint_storage(path))
    with pytest.raises(FileExistsError):
        probe.save_endpoint(payload, path)
    receipt = path.with_suffix('.scratch-write.json')
    report = probe.read(receipt)
    report['scratch_is_provider_persistent'] = True
    probe.write_json(receipt, report)
    with pytest.raises(ValueError, match='declared scratch'):
        probe.audit_endpoint_storage(path)


@pytest.mark.parametrize('filesystem,reserve,budget', [('checkpoint', 18, 1), ('root', 6, 0), ('scratch', 50, 4)])
def test_capacity_admission_preserves_all_storage_reserves(monkeypatch, filesystem, reserve, budget):
    paths = {'checkpoint': probe.source.ROOT, 'root': probe.Path('/'), 'scratch': probe.SCRATCH.parent}
    free = {path: 100 * 1024**3 for path in paths.values()}
    free[paths[filesystem]] = (reserve + budget) * 1024**3 - 1
    monkeypatch.setattr(probe.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': free[path]})())
    with pytest.raises(OSError, match=f'{reserve} GiB reserve'):
        probe.require_capacity()
    free[paths[filesystem]] += 1
    probe.require_capacity()
