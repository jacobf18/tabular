from copy import deepcopy

import pytest

import relocate_single_model_idle_root_tails as relocation


def test_only_two_old_nonselected_root_tails_are_in_scope():
    assert relocation.OWNERS == {'mixed_distill_v1': (200, 150), 'mixed_distill_v2': (400, 350)}
    assert relocation.SOURCE == relocation.REPO / 'runs/single_model_goal_20261005'
    assert not relocation.DESTINATION.is_relative_to(relocation.SOURCE)
    assert 'No published or selected model moves' in relocation.POLICY
    assert 'may not survive VM replacement' in relocation.POLICY


@pytest.mark.parametrize('mutation', [None, 'live_state', 'wrong_step', 'selected_tail', 'wrong_config', 'no_optimizer', 'wrong_score'])
def test_owner_identity_excludes_selected_incomplete_or_unbound_checkpoints(mutation):
    current = {'state': 'training_complete', 'step': 200, 'best_step': 150, 'best_score': 1.1}
    payload = {'step': 200, 'best_step': 150, 'best_score': 1.1, 'config_sha256': 'original',
               'model': {'weights': 'unchanged'}, 'optimizer': {'state': 'unchanged'}}
    if mutation == 'live_state':
        current['state'] = 'training'
    elif mutation == 'wrong_step':
        payload['step'] = 199
    elif mutation == 'selected_tail':
        payload['best_step'] = 200
    elif mutation == 'wrong_config':
        payload['config_sha256'] = 'different'
    elif mutation == 'no_optimizer':
        payload['optimizer'] = {}
    elif mutation == 'wrong_score':
        payload['best_score'] = 1.2
    original = deepcopy(payload)
    if mutation is None:
        relocation.checkpoint_identity(payload, current, (200, 150), 'original')
    else:
        with pytest.raises(ValueError, match='exact terminal nonselected'):
            relocation.checkpoint_identity(payload, current, (200, 150), 'original')
    assert payload == original
