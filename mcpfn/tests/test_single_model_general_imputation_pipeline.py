import ast
import os

import pytest

import run_single_model_general_imputation_teacher_pipeline as pipeline
import train_single_matrixformer_general_imputation_teacher as training


def identity():
    return {'pid': os.getpid(), 'start_ticks': 42,
            'command': 'bash -c set -e; python -u scripts/train_single_matrixformer_general_imputation_teacher.py '
                       '--stage train; ' + pipeline.AUDITOR_COMMAND}


def test_registered_chain_and_exact_final_auditor_exec_are_both_valid(monkeypatch):
    expected = identity()
    monkeypatch.setattr(pipeline, 'process_identity', lambda pid: expected)
    assert pipeline.chain_is_live(expected)
    observed = {**expected, 'command': pipeline.AUDITOR_COMMAND}
    monkeypatch.setattr(pipeline, 'process_identity', lambda pid: observed)
    monkeypatch.chdir(pipeline.REPO)
    assert pipeline.chain_is_live(expected)


@pytest.mark.parametrize('mutation', ['missing', 'pid', 'start', 'auditor', 'cwd', 'suffix', 'trainer'])
def test_dead_reused_or_unexpected_exec_cannot_pass(monkeypatch, tmp_path, mutation):
    expected = identity()
    observed = {**expected, 'command': pipeline.AUDITOR_COMMAND}
    monkeypatch.chdir(pipeline.REPO)
    if mutation == 'missing':
        observed = None
    elif mutation == 'pid':
        observed['pid'] += 1
    elif mutation == 'start':
        observed['start_ticks'] += 1
    elif mutation == 'auditor':
        observed['command'] += ' --changed'
    elif mutation == 'cwd':
        monkeypatch.chdir(tmp_path)
    elif mutation == 'suffix':
        expected['command'] += '; another-command'
    else:
        expected['command'] = expected['command'].replace('general_imputation_teacher.py --stage train', 'other.py --stage train')
    monkeypatch.setattr(pipeline, 'process_identity', lambda pid: observed)
    assert not pipeline.chain_is_live(expected)


def test_all_original_scientific_gates_cases_and_full_campaign_are_preserved():
    assert pipeline.training is training
    assert pipeline.training_audit.training is training
    assert pipeline.REPLAY.name == 'general_imputation_teacher_v27_endpoint_replay'
    assert pipeline.CONFIRMATION.name == 'general_imputation_teacher_v27_confirmation'
    source = pipeline.ADAPTED_SOURCE
    assert 'audit["through_step"] != 64' in source
    assert 'audit["training_episodes"] != 1792' in source
    assert 'audit["prediction_grids"] != 1368' in source
    assert 'result["frozen"]["frozen_tensors"] != 192' in source
    assert pipeline.STAGE_WRITE_BUDGETS == pipeline.template.STAGE_WRITE_BUDGETS
    actual_tree, previous_tree = [ast.parse(module.ADAPTED_SOURCE) for module in (pipeline, pipeline.template)]
    for name in ('replay', 'audit_replay', 'confirm', 'audit_confirmation', 'run'):
        actual = next(node for node in actual_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        previous = next(node for node in previous_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(actual) == ast.dump(previous)
    for name in ('register_chain_identity', 'wait_for_training_audit', 'require_compact_audit', 'materialize_endpoint_tasks'):
        assert getattr(pipeline, name).__globals__ is pipeline.__dict__
    assert 'chain_is_live' in pipeline.wait_for_training_audit.__code__.co_names


def test_endpoint_teacher_materialization_binds_new_policy_and_unmodified_full_inventory():
    assert pipeline.materialize_endpoint_tasks.__globals__['training'] is training
    assert 'teacher_policy' in pipeline.materialize_endpoint_tasks.__code__.co_names
    assert training.teacher_policy.POLICY == {'legacy': 'general', 'structured': 'general', 'recsys': 'specialist'}
    assert 'teacher_policy' in pipeline.require_compact_audit.__code__.co_names
    assert '284' in pipeline.POLICY and '1136' in pipeline.POLICY
