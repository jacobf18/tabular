from copy import deepcopy

import pytest
import torch

import relocate_single_model_balanced_native_capacity as capacity
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize('candidate', capacity.GRID)
@pytest.mark.parametrize('mutation', [None, 'step', 'epoch', 'config', 'candidate', 'updates',
    'derived', 'optimizer', 'model', 'elements', 'nonfinite', 'unknown'])
def test_only_complete_original_nonselected_grid_states(candidate, mutation):
    payload = {'model': {'weight': torch.ones(3)}, 'step': 400, 'epoch': 400,
        'config_sha256': capacity.GRID_CONFIG_SHA256, 'derived_checkpoint': True,
        'gradient_updates_performed_here': 0, 'candidate': deepcopy(candidate)}
    kind = candidate['id']
    if mutation in {'step', 'epoch'}:
        payload[mutation] += 1
    elif mutation == 'config':
        payload['config_sha256'] = 'wrong'
    elif mutation == 'candidate':
        payload['candidate']['alpha'] += .01
    elif mutation == 'updates':
        payload['gradient_updates_performed_here'] = 1
    elif mutation == 'derived':
        payload['derived_checkpoint'] = False
    elif mutation == 'optimizer':
        payload['optimizer'] = {}
    elif mutation == 'model':
        payload['model'] = {}
    elif mutation == 'elements':
        payload['model']['weight'] = torch.ones(2)
    elif mutation == 'nonfinite':
        payload['model']['weight'][0] = float('nan')
    elif mutation == 'unknown':
        kind = 'selected'
    if mutation:
        with pytest.raises(ValueError, match='derived-grid states'):
            capacity.validate_payload(payload, kind, model_tensors=1, model_elements=3)
    else:
        capacity.validate_payload(payload, kind, model_tensors=1, model_elements=3)


def closed_grid():
    config = {'grid': deepcopy(capacity.GRID), 'gradient_updates_performed_here': 0}
    arithmetic = {candidate['id']: {'tensors': 254, 'elements': 77998688, 'changed_tensors': 238,
        'arithmetic_independently_reconstructed': True, 'allowed_output_ulps': 1} for candidate in capacity.GRID}
    build = {'complete': True, 'single_parameter_set_per_candidate': True, 'cuda_initialized': False,
        'candidates': {identity: {**row, 'checkpoint_sha256': 'a' * 64} for identity, row in arithmetic.items()}}
    decision = {'selected': None, 'eligible_candidates': [], 'candidates_evaluated': 12, 'passed': False}
    audit = {'complete': True, 'phase': 'selection', 'cases': 152, 'prediction_grids_rescored': 2280,
        'candidates_audited': 12, 'arithmetic': arithmetic, 'research_selection_tolerance': 0,
        'cuda_initialized': False, 'goal_success_not_assessed': True,
        'decision_sha256': capacity.GRID_HASHES[str(capacity.ROOT / 'displacement_grid_v1_qualification/selection/decision.json')],
        'artifact_sha256': {str(capacity.ROOT / name): 'a' * 64 for name in capacity.TARGETS}}
    gate = {'passed': False, 'selected_candidate': None, 'checkpoint_sha256': None, 'selected_step': 0,
        'confirmation_evaluated': False, 'real_benchmark_success': False,
        'gradient_updates_performed_here': 0, 'derived_checkpoint': True}
    return config, build, audit, decision, gate


@pytest.mark.parametrize('mutation', [None, (0, 'gradient_updates_performed_here', 1), (0, 'grid', []),
    (1, 'complete', False), (1, 'candidates', {}), (1, 'single_parameter_set_per_candidate', False),
    (1, 'cuda_initialized', True), (2, 'complete', False), (2, 'phase', 'confirmation'),
    (2, 'cases', 151), (2, 'prediction_grids_rescored', 2279), (2, 'candidates_audited', 11),
    (2, 'arithmetic', {}), (2, 'research_selection_tolerance', .01), (2, 'cuda_initialized', True),
    (2, 'goal_success_not_assessed', False), (2, 'decision_sha256', 'wrong'),
    (3, 'selected', 'candidate_00'), (3, 'eligible_candidates', ['candidate_01']),
    (3, 'candidates_evaluated', 11), (3, 'passed', True), (4, 'passed', True),
    (4, 'selected_candidate', 'candidate_01'), (4, 'checkpoint_sha256', 'a' * 64),
    (4, 'selected_step', 400), (4, 'confirmation_evaluated', True),
    (4, 'real_benchmark_success', True), (4, 'gradient_updates_performed_here', 1),
    (4, 'derived_checkpoint', False)])
def test_all_twelve_audited_rejections_remain_unchanged(mutation):
    reports = closed_grid()
    if mutation is None:
        capacity.validate_grid(*reports)
    else:
        index, key, value = mutation
        reports[index][key] = value
        with pytest.raises(ValueError):
            capacity.validate_grid(*reports)


@pytest.mark.parametrize('mutation', ['hash', 'reconstruction', 'tensors', 'elements', 'changed', 'tolerance'])
def test_every_original_candidate_byte_and_arithmetic_identity_is_bound(mutation):
    config, build, audit, decision, gate = closed_grid()
    if mutation == 'hash':
        audit['artifact_sha256'][str(capacity.ROOT / next(iter(capacity.TARGETS)))] = 'b' * 64
    else:
        key = {'reconstruction': 'arithmetic_independently_reconstructed', 'tensors': 'tensors',
            'elements': 'elements', 'changed': 'changed_tensors', 'tolerance': 'allowed_output_ulps'}[mutation]
        audit['arithmetic']['candidate_00'][key] = False if mutation == 'reconstruction' else -1
    with pytest.raises(ValueError, match='every candidate'):
        capacity.validate_grid(config, build, audit, decision, gate)


def test_exact_copy_refuses_overwrite(tmp_path, monkeypatch):
    source, destination = tmp_path / 'original.pt', tmp_path / 'copy.pt'
    source.write_bytes(b'exact original serialized bytes')
    row = {'source': str(source), 'destination': str(destination), 'bytes': source.stat().st_size,
        'sha256': capacity.sha256(source), 'kind': 'candidate_00'}
    checked = []
    monkeypatch.setattr(capacity, 'validate_file', lambda path, kind: checked.append((path, kind)))
    capacity.copy_and_verify(row)
    assert destination.read_bytes() == source.read_bytes() and checked == [(destination, 'candidate_00')]
    with pytest.raises(FileExistsError):
        capacity.copy_and_verify(row)


def test_cloned_helpers_and_scope_are_correct():
    assert len(capacity.TARGETS) == 12 and len(set(capacity.TARGETS.values())) == 12
    assert capacity.STATES['displacement_grid_v1'] == 'rejected'
    assert capacity.SELECTED == capacity.probe.PARENT
    assert capacity.TRAINING_ADMISSION_BYTES == 18 * 1024**3 + 2432 * 1024**2
    assert "'canonical_artifacts_checked': 12" in capacity.RECHECK_SOURCE
    for function in vars(capacity).values():
        if callable(function) and getattr(function, '__globals__', None) is vars(capacity):
            assert unresolved_globals(function) == [], function.__name__


def complete_probe():
    gate = {'passed': True, 'preflight_only': True, 'real_benchmark_success': False}
    audit = {'complete': True, 'training_tasks': 28, 'prediction_grids': 56,
        'teacher_replay': {'component_grids': 52}, 'cuda_initialized': False}
    retention = {'complete': True, 'reference_grids': 26, 'component_grids': 50,
        'component_role_counts': {'v8': 24, 'v29': 26}}
    mean = {'complete': True, 'tasks': 28, 'imputation_reweighted_tasks': 26,
        'unchanged_rating_tasks': 2, 'cuda_initialized': False}
    review = {'complete': True, 'accepted': True, 'config_sha256': capacity.PROBE_CONFIG_SHA256,
        'mean_tasks': 28, 'goal_achieved': False, 'full_trial_authorized': False}
    return gate, audit, retention, mean, review


@pytest.mark.parametrize('mutation', [None, (0, 'passed', False), (0, 'preflight_only', False),
    (0, 'real_benchmark_success', True), (1, 'complete', False), (1, 'training_tasks', 27),
    (1, 'prediction_grids', 55), (1, 'teacher_replay', {'component_grids': 51}),
    (1, 'cuda_initialized', True), (2, 'complete', False), (2, 'reference_grids', 25),
    (2, 'component_grids', 49), (2, 'component_role_counts', {'v8': 26, 'v29': 24}),
    (3, 'complete', False), (3, 'tasks', 27), (3, 'imputation_reweighted_tasks', 25),
    (3, 'unchanged_rating_tasks', 1), (3, 'cuda_initialized', True), (4, 'complete', False),
    (4, 'accepted', False), (4, 'config_sha256', 'wrong'), (4, 'mean_tasks', 27),
    (4, 'goal_achieved', True), (4, 'full_trial_authorized', True)])
def test_capacity_does_not_authorize_itself_from_pending_or_rejected_feasibility(mutation):
    reports = complete_probe()
    if mutation is None:
        capacity.validate_probe(*reports)
    else:
        index, key, value = mutation
        reports[index][key] = value
        with pytest.raises(ValueError, match='accepted full-context probe'):
            capacity.validate_probe(*reports)
