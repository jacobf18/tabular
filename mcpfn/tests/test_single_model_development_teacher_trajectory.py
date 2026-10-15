from copy import deepcopy

import numpy as np
import pytest
import torch

import analyze_single_model_development_teacher_trajectory as analysis


@pytest.mark.parametrize('step,expected', [(10, (0, 10)), (25, (0, 10, 25)),
    (50, (0, 10, 25, 50)), (64, (0, 10, 25, 50, 64))])
def test_include_every_preceding_registered_boundary(step, expected):
    assert analysis.requested_boundaries(step) == expected


@pytest.mark.parametrize('step', [0, 1, 11, 63, 65, True, 10.])
def test_no_arbitrary_boundary_or_untrained_only_analysis(step):
    with pytest.raises(ValueError):
        analysis.requested_boundaries(step)


@pytest.mark.parametrize('steps', [[0], [0, 25], [0, 10, 10], [10, 0], [0, 10, 50]])
def test_reject_missing_unpublished_reordered_or_duplicate_boundaries(steps):
    with pytest.raises(ValueError):
        analysis.published_prefix([{'step': step} for step in steps], (0, 10))


def test_later_appended_boundaries_do_not_change_an_earlier_snapshot():
    records = [{'step': step} for step in analysis.BOUNDARIES]
    assert analysis.published_prefix(records, (0, 10)) == [{'step': 0}, {'step': 10}]


@pytest.mark.parametrize('fault', [None, 'step', 'updates', 'metrics', 'too_many_updates', 'fractional_updates'])
def test_report_matches_its_original_publication(fault):
    report = {'step': 10, 'accepted_updates': 8, 'student': {'rmse': 1.}}
    record = deepcopy(report)
    if fault == 'step':
        report['step'] = 25
    elif fault == 'updates':
        record['accepted_updates'] = 7
    elif fault == 'metrics':
        report['student']['rmse'] = 0.
    elif fault == 'too_many_updates':
        report['accepted_updates'] = record['accepted_updates'] = 11
    elif fault == 'fractional_updates':
        report['accepted_updates'] = record['accepted_updates'] = 8.5
    if fault:
        with pytest.raises(ValueError):
            analysis.validate_published(report, record, 10)
    else:
        analysis.validate_published(report, record, 10)


def fixture_case(tmp_path):
    teacher = tmp_path / 'teacher.npz'
    student = tmp_path / 'student.npz'
    truth = np.zeros((1, 2, 2), dtype=np.float32)
    np.savez(teacher, general=truth, teacher=truth)
    np.savez(student, student=truth + 2.)
    record = {'seed': 123, 'domain': 'imputation', 'rows': 2, 'columns': 2,
        'prediction_file': str(teacher), 'general': {'rmse': 0.}, 'teacher': {'rmse': 0.}}
    reports = {'teacher': {'cases': [record]}, **{role: {'cases': [
        {'prediction_file': str(student), 'student': {'rmse': 2.}}]} for role in ('student', 'actual_v29', 'v8')}}
    bank = [{'metadata': {'seed': 123, 'gradient_data': 'synthetic'}, 'target': torch.from_numpy(truth),
        'source': torch.tensor([[[3., float('nan')], [float('nan'), 5.]]]),
        'mask': torch.tensor([[[True, True], [False, False]]])}]
    return bank, reports, [record]


def test_score_every_role_and_split_with_original_overlap_and_rmse(tmp_path):
    bank, reports, reference = fixture_case(tmp_path)
    rows, checks = analysis.score_bank(bank, reports, reference, 'run', 10, 8)
    assert len(rows) == 10 and checks == 5
    assert all(row['observed_query_overlap'] == 1 and row['accepted_updates'] == 8 for row in rows)
    student = {row['split']: row for row in rows if row['role'] == 'student'}
    assert student['query']['target_mse'] == 4. and student['observed']['target_mse'] == 5.
    assert student['query']['teacher_huber'] == 1.5
    assert student['query']['teacher_mse'] == 4. and student['query']['v8_mse'] == 0.


@pytest.mark.parametrize('fault', ['missing_case', 'seed', 'real', 'geometry', 'score'])
def test_no_truth_or_score_mismatches_are_silently_ignored(tmp_path, fault):
    bank, reports, reference = fixture_case(tmp_path)
    if fault == 'missing_case':
        bank.clear()
    elif fault == 'seed':
        bank[0]['metadata']['seed'] = 124
    elif fault == 'real':
        bank[0]['metadata']['gradient_data'] = 'real'
    elif fault == 'geometry':
        bank[0]['target'] = torch.zeros(1, 3, 2)
    else:
        reports['student']['cases'][0]['student']['rmse'] = 1.
    with pytest.raises(ValueError):
        analysis.score_bank(bank, reports, reference, 'run', 10, 8)


def complete_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'original'
    root.mkdir()
    bank, reports, reference = fixture_case(root)
    common = root / 'domain_clipping_v10'
    common.mkdir()
    bank_path = common / 'synthetic_bank.pt'
    torch.save(bank, bank_path)
    record = reference[0]
    record.update(index=0, gradient_data='synthetic', routing={'target': 'same-input observed blend'},
        prediction_sha256=analysis.metrics.digest(record['prediction_file']))
    teacher = {'bank_sha256': analysis.metrics.digest(bank_path), 'cases': [record]}
    student_case = {**record, **reports['student']['cases'][0]}
    student_case['prediction_sha256'] = analysis.metrics.digest(student_case['prediction_file'])
    analysis.metrics.write(common / 'reference.json', teacher)
    analysis.metrics.write(common / 'validation_0.json', {'cases': [student_case]})
    configurations = {}
    for label in analysis.RUNS:
        run = root / label
        run.mkdir()
        config = {'boundaries': list(analysis.BOUNDARIES),
            'parent_checkpoint': str(root / 'legacy_strata_v29/selected_checkpoint.pth'),
            'input_sha256': {str(bank_path): analysis.metrics.digest(bank_path)}}
        analysis.metrics.write(run / 'config.json', config)
        digest = analysis.metrics.digest(run / 'config.json')
        configurations[label] = digest
        analysis.metrics.write(run / 'registration.json', {'config_sha256': digest})
        trajectory = []
        for step in (0, 10):
            report = {'step': step, 'accepted_updates': 0 if step == 0 else 8,
                'student': {'rmse': 2.}, 'cases': [student_case]}
            analysis.metrics.write(run / f'validation_{step}.json', report)
            trajectory.append({key: report[key] for key in ('step', 'accepted_updates', 'student')})
        (run / 'validation.jsonl').write_text(''.join(analysis.metrics.json.dumps(row) + '\n' for row in trajectory))
    audit = root / 'all_imputation_blend_v45_audit'
    audit.mkdir()
    analysis.metrics.write(audit / 'audit.json', {'complete': True,
        'artifact_sha256': {record['prediction_file']: record['prediction_sha256'],
                           student_case['prediction_file']: student_case['prediction_sha256']}})
    monkeypatch.setattr(analysis, 'HISTORICAL_AUDIT_SHA256', analysis.metrics.digest(audit / 'audit.json'))
    monkeypatch.setattr(analysis, 'RUNS', configurations)
    monkeypatch.setattr(analysis, 'COUNTS', {'legacy': 1})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    return root


def test_complete_orchestration_preserves_original_evidence_and_audit_limitations(tmp_path, monkeypatch):
    root = complete_fixture(tmp_path, monkeypatch)
    before = {str(path): analysis.metrics.digest(path) for path in root.rglob('*') if path.is_file()}
    output = tmp_path / 'diagnostic'
    analysis.analyze(root, output, 10)
    result = analysis.metrics.read(output / 'result.json')
    assert result['complete'] is True and result['case_role_split_rows'] == 40
    assert result['raw_imputation_rmse_checks'] == 20 and result['boundaries'] == [0, 10]
    for key in ('v46_full_independent_audit_claimed', 'cuda_initialized', 'model_inference_performed',
            'real_data_used', 'training_performed', 'new_seeds_consumed', 'selection_performed', 'gate_changed'):
        assert result[key] is False
    after = {str(path): analysis.metrics.digest(path) for path in root.rglob('*') if path.is_file()}
    assert before == after
    with pytest.raises(ValueError, match='new external'):
        analysis.analyze(root, output, 10)


@pytest.mark.parametrize('fault', ['unpublished', 'grid_changed', 'wrong_score', 'unsafe_output', 'visible_cuda'])
def test_actual_diagnostic_rejects_incomplete_or_mutated_inputs(tmp_path, monkeypatch, fault):
    root = complete_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'diagnostic'
    current = root / 'stronger_mean_distillation_v46'
    if fault == 'unpublished':
        path = current / 'validation.jsonl'
        path.write_text(path.read_text().splitlines()[0] + '\n')
    elif fault == 'grid_changed':
        (root / 'student.npz').write_bytes(b'changed')
    elif fault == 'wrong_score':
        path = current / 'validation_10.json'
        report = analysis.metrics.read(path)
        report['cases'][0]['student']['rmse'] = 3.
        analysis.metrics.write(path, report)
    elif fault == 'unsafe_output':
        output = root / 'diagnostic'
    else:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(ValueError):
        analysis.analyze(root, output, 10)
    if output.exists():
        assert analysis.metrics.read(output / 'status.json')['state'] == 'failed'
    else:
        assert not output.exists()
