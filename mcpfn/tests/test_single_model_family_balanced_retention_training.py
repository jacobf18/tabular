import inspect

import test_single_model_cohort_anchor_retention_training as family_reference_test_blueprint


SOURCE = family_reference_test_blueprint.SOURCE.replace('cohort_anchor_retention', 'family_balanced_retention')
exec(compile(SOURCE, __file__, 'exec'), globals())
for name in ('test_full_trial_records_the_exact_cohort_assignments',
        'test_full_replay_never_swaps_or_mutates_references',
        'test_qualification_requires_cohort_bindings_as_well_as_full_grid_hashes'):
    adapted_test = inspect.getsource(getattr(family_reference_test_blueprint, name)).replace(
        'integration.retention.BACKWARD_SOURCE_SHA256 == integration.previous.retention.BACKWARD_SOURCE_SHA256',
        'integration.retention.BACKWARD_SOURCE_SHA256 != integration.previous.retention.BACKWARD_SOURCE_SHA256')
    exec(compile(adapted_test, __file__, 'exec'), globals())


@pytest.mark.parametrize('mutation', ['decision_multiplier', 'gradient_scale', 'task_weight', 'archive_weight'])
def test_full_replay_rejects_changed_retention_weights(tmp_path, monkeypatch, mutation):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    path = Path(next(iter(state.tasks)))
    payload = torch.load(path, weights_only=True)
    task, decision, _ = state.tasks[str(path)]
    if mutation == 'decision_multiplier':
        decision['weighting']['multiplier'] *= 2
    elif mutation == 'gradient_scale':
        decision['gradient_scale'] = .25
    elif mutation == 'task_weight':
        task['weight'] *= 2
        payload['task']['weight'] = task['weight']
    else:
        target_path = Path(decision['artifact'])
        target = torch.load(target_path, weights_only=True)
        target['weighting']['effective_task_weight'] *= 2
        torch.save(target, target_path)
        decision['sha256'] = probe.sha256(target_path)
    with pytest.raises(ValueError):
        state.check(path, payload)
    assert not state.summary_path.exists()


@pytest.mark.parametrize('mutation', ['multiplier', 'task_weight', 'gradient_scale', 'registered_weight'])
def test_qualification_rejects_rehashed_weight_changes(tmp_path, monkeypatch, mutation):
    training, config, counts, references, backend, calls = fixture(tmp_path, monkeypatch)
    state = integration.RetentionReplay(training, config, counts, references, None, backend)
    replay_all(state)
    hashes = state.finish()
    rows = [json.loads(line) for line in state.rows_path.read_text().splitlines()]
    if mutation == 'multiplier':
        rows[0]['weighting']['multiplier'] *= 2
    elif mutation == 'task_weight':
        rows[0]['task_weight'] *= 2
    elif mutation == 'gradient_scale':
        rows[0]['gradient_scale'] = .25
    else:
        config['batches'][0]['tasks'][0]['weight'] *= 2
        probe.write_json(training.OUTPUT / 'config.json', config)
    state.rows_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    hashes[str(state.rows_path)] = probe.sha256(state.rows_path)
    report = probe.read(state.summary_path)
    report['artifact_sha256'][str(state.rows_path)] = probe.sha256(state.rows_path)
    report['training_config_sha256'] = probe.sha256(training.OUTPUT / 'config.json')
    report['artifact_sha256'][str(training.OUTPUT / 'config.json')] = report['training_config_sha256']
    probe.write_json(state.summary_path, report)
    hashes[str(state.summary_path)] = probe.sha256(state.summary_path)
    with pytest.raises(ValueError):
        integration.require_report(training, {'artifact_sha256': hashes, 'admission_attempts': counts})
