from copy import deepcopy
import inspect

import pytest
import torch

import compare_single_model_consumed_banks as comparison


def fixtures():
    bank, specifications, records = [], [], []
    for index in range(284):
        structured = index >= 92
        domain = 'recsys' if 60 <= index < 92 else 'imputation'
        metadata = {'seed': 1000 + index, 'domain': domain, 'gradient_data': 'synthetic', 'rows': 3, 'columns': 2}
        specification = {'domain': domain}
        if structured:
            metadata.update(structured=True, family=comparison.mixture.FAMILIES[(index - 92) // 16])
            specification['structured'] = {'seed': 1000 + index}
        else:
            specification['seed'] = 1000 + index
        bank.append({'metadata': metadata})
        specifications.append(specification)
        metrics = {'rmse': 1., 'mae': .8}
        if domain == 'imputation':
            metrics.update(missbench_nrmse=.9, uci_nrmse=.9, common_nrmse=.9,
                           column_coverage={'queried_columns': 2, 'common_nonconstant_columns': 2})
        records.append({'index': index, **metadata, 'student': metrics,
                        'prediction_file': '', 'prediction_sha256': 'unused'})
    report = {'cases': records, 'student': comparison.mixture.aggregate(records, 'student')}
    return bank, specifications, report


def history_fixture(label):
    trial = comparison.TRIALS[label]
    result = {'complete': True, 'gate': {'passed': False, 'confirmation_evaluated': True}, 'campaign': None, 'review': None}
    audit = {'complete': True, 'complete_training_audit': True, 'through_step': 64, 'attempted_batches': 64,
             'accepted_updates': trial['updates'], 'best': {'step': trial['step']}, 'prediction_grids': trial['grids'],
             'independently_regenerated_training_tasks': 1792, 'independently_regenerated_rating_contexts': 128,
             'top_two_block_training': True}
    replay = {'complete': True, 'endpoint_grids': 56, 'selected_grids': 152}
    confirmation = {'complete': True, 'cases': 284, 'prediction_grids': 1136}
    manifest = {'selected_step': trial['step'], 'checkpoint_sha256': trial['checkpoint_sha256']}
    return trial, result, audit, replay, confirmation, manifest


def test_exact_four_cells_reuse_diagonal_and_infer_only_two_missing_cells():
    cells = comparison.cells()
    assert len(cells) == 4 and len({(cell['bank'], cell['model']) for cell in cells}) == 4
    assert {cell['name'] for cell in cells if cell['reuse']} == {'v26_on_v26', 'v27_on_v27'}
    assert {cell['name'] for cell in cells if not cell['reuse']} == {'v27_on_v26', 'v26_on_v27'}
    assert 'not the teacher-policy change alone' in comparison.POLICY
    assert 'never future training or fresh confirmation' in comparison.POLICY
    assert 'no inference ensemble' in comparison.POLICY


@pytest.mark.parametrize('label', ['v26', 'v27'])
@pytest.mark.parametrize('record,key,value', [(1, 'campaign', 'published'), (1, 'complete', False),
    (2, 'through_step', 63), (2, 'independently_regenerated_training_tasks', 1791),
    (2, 'top_two_block_training', False), (3, 'selected_grids', 151), (4, 'prediction_grids', 1135),
    (5, 'checkpoint_sha256', 'substituted')])
def test_history_requires_full_stopped_unqualified_original_evidence(label, record, key, value):
    arguments = list(history_fixture(label))
    comparison.validate_history(*arguments)
    arguments[record][key] = value
    with pytest.raises(ValueError, match='complete audited but unqualified'):
        comparison.validate_history(*arguments)


@pytest.mark.parametrize('mutation', ['short', 'order', 'seed', 'domain', 'structured'])
def test_cached_bank_must_match_every_original_case(mutation):
    bank, specifications, report = fixtures()
    comparison.validate_bank(bank, specifications)
    if mutation == 'short':
        bank.pop()
    elif mutation == 'order':
        bank[0], bank[1] = bank[1], bank[0]
    elif mutation == 'seed':
        bank[0]['metadata']['seed'] += 1
    elif mutation == 'domain':
        bank[0]['metadata']['domain'] = 'recsys'
    else:
        bank[92]['metadata']['structured'] = False
    with pytest.raises(ValueError):
        comparison.validate_bank(bank, specifications)


@pytest.mark.parametrize('mutation', [None, 'short', 'metadata', 'path', 'pass_index', 'aggregate'])
def test_prediction_report_binds_full_metadata_paths_and_macro_metrics(tmp_path, mutation):
    bank, specifications, report = fixtures()
    for index, record in enumerate(report['cases']):
        record['prediction_file'] = str(tmp_path / 'student_predictions_000' / f'case_{index:03d}.npz')
    if mutation == 'short':
        report['cases'].pop()
    elif mutation == 'metadata':
        report['cases'][0]['rows'] += 1
    elif mutation == 'path':
        report['cases'][0]['prediction_file'] = str(tmp_path.parent / 'another-run.npz')
    elif mutation == 'pass_index':
        report['cases'][0]['prediction_file'] = str(tmp_path / 'student_predictions_001/case_000.npz')
    elif mutation == 'aggregate':
        report['student']['imputation']['rmse'] += .01
    if mutation:
        with pytest.raises(ValueError):
            comparison.validate_report(report, bank, tmp_path, True)
    else:
        comparison.validate_report(report, bank, tmp_path, True)


def paired_fixture():
    bank, specifications, first = fixtures()
    second = deepcopy(first)
    for record in second['cases']:
        record['student']['rmse'] += .1
        record['student']['mae'] -= .1
    second['student'] = comparison.mixture.aggregate(second['cases'], 'student')
    reference = {'general': deepcopy(first['student']), 'teacher': deepcopy(first['student']), 'cases': []}
    for record in first['cases']:
        reference['cases'].append({**record, 'general': deepcopy(record['student']), 'teacher': deepcopy(record['student'])})
    return {'v26': first, 'v27': second}, {'reference': reference, 'initialization': deepcopy(first)}


def test_complete_paired_comparison_preserves_improvements_and_regressions():
    reports, original = paired_fixture()
    summary, paired = comparison.comparisons('v26', reports, original)
    assert len(summary) == 72 and len(paired) == 284
    assert {row['group'] for row in summary} == {'legacy', 'recsys', 'structured_overall', *comparison.mixture.FAMILIES}
    assert [row['index'] for row in paired] == list(range(284))
    for row in summary:
        if row['metric'] == 'rmse':
            assert row['v27_minus_v26'] == pytest.approx(.1)
        elif row['metric'] == 'mae':
            assert row['v27_minus_v26'] == pytest.approx(-.1)
        else:
            assert row['v27_minus_v26'] == 0
    assert all(row['v27_minus_v26']['rmse'] > 0 for row in paired)
    assert all(row['v27_minus_v26']['mae'] < 0 for row in paired)


@pytest.mark.parametrize('mutation', ['metadata', 'coverage'])
def test_paired_case_identity_and_truth_coverage_cannot_drift(mutation):
    reports, original = paired_fixture()
    if mutation == 'metadata':
        reports['v27']['cases'][0]['seed'] += 1
    else:
        reports['v27']['cases'][0]['student']['column_coverage']['common_nonconstant_columns'] = 1
    with pytest.raises(ValueError):
        comparison.comparisons('v27', reports, original)


def test_evaluation_is_one_checkpoint_per_cell_no_grad_and_no_bank_generation():
    source = inspect.getsource(comparison.evaluate)
    assert '@torch.no_grad()' in source
    assert 'model.eval().requires_grad_(False)' in source
    assert 'model.load_state_dict(checkpoint["model"], strict=True)' in source
    assert 'mixture.backend.evaluate_bank(model, criterion, bank)' in source
    assert 'prepare_bank' not in source and 'backward' not in source and 'optimizer.step' not in source
    assert 'torch.save' not in source and 'best_validation_checkpoint' not in source
    assert 'if cell["reuse"]:' in source
    assert '1792' in inspect.getsource(comparison.validate_history)


@pytest.mark.parametrize('initialized,visible', [(True, ''), (False, '0'), (False, None)])
def test_cpu_audit_rejects_cuda_or_ambiguous_device_access(monkeypatch, initialized, visible):
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: initialized)
    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(ValueError, match='CPU-only'):
        comparison.audit()


def test_existing_output_cannot_be_registered_again(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, 'OUTPUT', tmp_path)
    with pytest.raises(FileExistsError):
        comparison.register()


@pytest.mark.parametrize('volume,reserve,budget', [('checkpoint', 18, 64), ('root', 6, 16), ('scratch', 50, 1536)])
def test_capacity_checks_precede_directory_creation_and_preserve_every_reserve(tmp_path, monkeypatch, volume, reserve, budget):
    monkeypatch.setattr(comparison, 'OUTPUT', tmp_path / 'absent')
    monkeypatch.setattr(comparison, 'SCRATCH', tmp_path / 'also_absent')
    monkeypatch.setattr(comparison.storage, 'require_mount', lambda: None)
    paths = {'checkpoint': comparison.ROOT, 'root': comparison.Path('/'), 'scratch': comparison.storage.SCRATCH}

    def disk_usage(path):
        free = 100 * 1024**3
        if path == paths[volume]:
            free = reserve * 1024**3 + budget * 1024**2 - 1
        return type('Usage', (), {'free': free})()

    monkeypatch.setattr(comparison.shutil, 'disk_usage', disk_usage)
    with pytest.raises(OSError, match='reserves'):
        comparison.require_capacity()
    monkeypatch.setattr(comparison.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': 100 * 1024**3})())
    comparison.require_capacity()
    assert not comparison.OUTPUT.exists() and not comparison.SCRATCH.exists()
