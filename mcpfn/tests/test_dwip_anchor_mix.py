from copy import deepcopy
import json

import pytest
import torch

from run_dwip_anchor_mix import build_commands, rehearsal_prior
from run_dwip_spectral_horizon import build_commands as control_commands
from run_spectral_query_experiments import checkpoint_schedule
from train_direct_pure_rabel_curriculum import CurriculumPhase
from train_dwip_anchor_mix import ShapeAwareMixedMatrixPrior


def test_anchor_mix_preserves_commands_except_declared_prior_and_adapter(tmp_path):
    trials, controls, prior, anchor = control_commands(tmp_path)
    new_trials, commands, new_prior, new_anchor = build_commands(tmp_path)
    assert (trials, anchor) == (new_trials, new_anchor)
    command = commands['dwip1e5']
    assert command[1] == 'scripts/train_dwip_anchor_mix.py'
    restored = list(command)
    restored[1] = controls['dwip1e5'][1]
    position = command.index('--prior-overrides-json') + 1
    assert json.loads(command[position]) == new_prior
    restored[position] = controls['dwip1e5'][position]
    assert restored == controls['dwip1e5']
    assert checkpoint_schedule(command) == [4000, 12000, 24000, 36000, 48000]
    assert new_prior['validation_spectral_specs'] == prior['validation_spectral_specs']


def test_anchor_mix_has_fixed_mass_preserves_spectral_components_and_mask_balance(tmp_path):
    _, _, prior, _ = control_commands(tmp_path)
    original = deepcopy(prior)
    changed = rehearsal_prior(prior)
    assert prior == original
    assert changed['mixed_priors'][:6] == prior['mixed_priors']
    components = changed['mixed_priors']
    total = sum(component['weight'] for component in components)
    assert sum(component['weight'] for component in components[6:]) / total == .25
    assert sum(component['weight'] for component in components if component['missingness_type'] == 'mcar') / total == .5


def test_anchor_children_use_small_shapes_without_changing_spectral_shapes(tmp_path):
    _, _, prior, _ = build_commands(tmp_path)
    phase = CurriculumPhase(name='test', end_step=1, rows_low=80, rows_high=90, cols_low=35, cols_high=40, batch_size=2)
    generator = ShapeAwareMixedMatrixPrior(phase, 'cpu', 'profile_mixed', prior)
    for child in generator.children[:6]:
        assert (child.config['num_rows_low'], child.config['num_rows_high']) == (80, 90)
        assert (child.config['num_cols_low'], child.config['num_cols_high']) == (35, 40)
    for child in generator.children[6:]:
        assert (child.config['num_rows_low'], child.config['num_rows_high']) == (10, 50)
        assert (child.config['num_cols_low'], child.config['num_cols_high']) == (5, 50)
        (source, target, _, _, _), _ = child.get_batch()
        assert source.shape == target.shape
        assert 10 <= source.shape[1] <= 50 and 5 <= source.shape[2] <= 50
        assert torch.isfinite(target).all()
        assert torch.isfinite(source).any()


def test_shape_override_rejects_invalid_dimensions(tmp_path):
    _, _, prior, _ = build_commands(tmp_path)
    prior['mixed_priors'][-1]['shape_override']['num_rows_low'] = 100
    phase = CurriculumPhase(name='test', end_step=1, rows_low=80, rows_high=90, cols_low=35, cols_high=40, batch_size=2)
    with pytest.raises(ValueError, match='ordered'):
        ShapeAwareMixedMatrixPrior(phase, 'cpu', 'profile_mixed', prior)
