import ast
from copy import deepcopy

import pytest

import train_single_matrixformer_general_imputation_teacher as training
import audit_single_model_general_imputation_teacher as audit
from test_single_model_top_two_blocks_trial import valid_preflight


def test_schedule_preserves_every_shape_regime_mass_and_fresh_seed():
    batches = training.batches()
    previous = training.template.batches()
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
        roles = [training.preflight.expected_role(task) for task in batch['tasks']]
        assert roles.count('general') == 26 and roles.count('specialist') == 2
    seeds = training.original.alignment.fixed.seeds_in(batches)
    confirmation = training.original.alignment.fixed.seeds_in(training.confirmation_cases())
    consumed = training.original.alignment.fixed.seeds_in([
        previous, training.template.confirmation_cases(), training.preflight.batch()])
    assert len(seeds) == 3584 and len(confirmation) == 284
    assert not seeds.intersection(confirmation) and not (seeds | confirmation).intersection(consumed)
    assert (training.DATA_SEED, training.GRADIENT_SEED) == (591100901, 592100901)
    assert training.CONFIRMATION_SEEDS == (601100901, 602100901)


def test_initializer_is_unqualified_v26_but_original_teachers_and_baselines_are_retained():
    assert training.PARENT == training.template.OUTPUT / 'selected_checkpoint.pth'
    assert training.PARENT == training.preflight.PARENT
    assert training.PARENT != training.original.alignment.fixed.PARENT
    assert training.PARENT_SHA256 == training.preflight.PARENT_SHA256
    assert training.PARENT_UPDATES == 64
    assert training.teacher_policy.POLICY == {'legacy': 'general', 'structured': 'general', 'recsys': 'specialist'}
    assert training.template.teacher_policy.POLICY['legacy'] == 'initialization'
    source = training.ADAPTED_SOURCE
    assert 'teacher_policy.identities(config["base"], original.alignment.fixed.PARENT)' in source
    assert 'config["parent_selection_qualified"] is not False' in source
    assert 'config["input_sha256"][str(PARENT)] != PARENT_SHA256' in source
    assert 'read(INITIAL_RUN / "validation_64.json")["student"]' in source
    assert 'teacher_policy.identities(base, original.alignment.fixed.PARENT)' in training.REGISTER_SOURCE
    assert '"retention_baseline": str(original.alignment.fixed.PARENT)' in training.REGISTER_SOURCE
    assert 'sha256(PARENT) != PARENT_SHA256' in training.REGISTER_SOURCE


def test_optimizer_training_math_and_original_selection_guards_are_unchanged():
    actual_tree, previous_tree = [ast.parse(module.ADAPTED_SOURCE) for module in (training, training.template)]
    for name in ('setup', 'train', 'selected_boundary', 'confirmation_cases', 'batches'):
        actual = next(node for node in actual_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        previous = next(node for node in previous_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(actual) == ast.dump(previous)
    assert training.PREFIXES == ('transformer_blocks.6.', 'transformer_blocks.7.', 'decoder.')
    assert all(training.constant_rate(step) == 1e-5 for step in range(1, 65))
    for name in ('replicated_batches', 'require_evidence', 'validate_preflight', 'save_endpoint', 'audit_endpoint'):
        assert getattr(training, name).__globals__ is training.__dict__


@pytest.mark.parametrize('step', [0, 65, 1.5, True])
def test_no_learning_rate_or_attempt_budget_drift(step):
    with pytest.raises(ValueError):
        training.constant_rate(step)


@pytest.mark.parametrize('entry,field,value', [
    (0, 'trainable_parameters', 19496843), (1, 'prediction_grids', 55),
    (1, 'cuda_initialized', True), (1, 'autograd_replayed', True),
    (2, 'passed', False), (2, 'audit_sha256', 'changed'), (3, 'state', 'running')])
def test_require_complete_accepted_independent_preflight(entry, field, value):
    records = valid_preflight()
    training.validate_preflight(*records, 'digest')
    records[entry][field] = value
    with pytest.raises(ValueError, match='independently audited'):
        training.validate_preflight(*records, 'digest')


def test_audit_covers_v26_parent_grids_and_all_original_reference_grids():
    source = audit.ADAPTED_SOURCE
    assert audit.training is training
    assert 'saved["parent_training_updates"] != training.PARENT_UPDATES' in source
    assert 'training.INITIAL_RUN / "validation_64.json"' in source
    assert '(parent_report, ("student",))' in source
    assert '152 * (4 + len(boundaries))' in source
    assert 'len(selected) != 62' in source and '27766670' in source
    assert 'frozen["frozen_tensors"] != 192' in source
    assert 'len(decision["comparisons"]) != 22' in source
    assert 'training.preflight.check_routing(payload, task, config)' in source
    assert 'load_regenerated_task(path)' in source
    for name in ('load_regenerated_task', 'regeneration_evidence', 'run_audit'):
        assert getattr(audit, name).__globals__ is audit.__dict__
    assert '1368' in training.QUALIFICATION and '1136' in training.QUALIFICATION
    assert '1792' in training.QUALIFICATION and '284' in training.QUALIFICATION


def test_independent_task_replay_uses_the_new_teacher_and_original_v8_identity():
    assert 'teacher_policy' in audit.run_audit.__code__.co_names
    assert audit.run_audit.__globals__['training'] is training
    assert training.preflight.expected_role({'specification': {'domain': 'imputation'}}) == 'general'
    assert training.preflight.expected_role({'specification': {'domain': 'recsys'}}) == 'specialist'
    assert 'diagnostics.prepare(config["diagnostic_storage"], "training")' in training.REGISTER_SOURCE
