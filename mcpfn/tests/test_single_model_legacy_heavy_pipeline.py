"""Test full V30 scope, actual materialization and cross-process dependency identities."""

import builtins
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

import run_single_model_legacy_heavy_pipeline as pipeline
import test_single_model_legacy_strata_pipeline as inherited
import test_single_model_legacy_strata_replay_recovery as materialization_tests
from recover_single_model_legacy_strata_replay import code_globals


def test_complete_original_qualification_and_all_runtime_globals():
    assert pipeline.training.OUTPUT.name == "legacy_heavy_v30"
    assert pipeline.OUTPUT.name == "legacy_heavy_v30_pipeline"
    assert pipeline.REPLAY.name == "legacy_heavy_v30_endpoint_replay"
    assert pipeline.CONFIRMATION.name == "legacy_heavy_v30_confirmation"
    assert len(pipeline.training.confirmation_cases()) == 284
    assert pipeline.PARENT_CACHE == pipeline.template.PARENT_CACHE
    assert pipeline.STAGE_WRITE_BUDGETS == pipeline.template.STAGE_WRITE_BUDGETS
    assert pipeline.MINIMUM_FREE == 18 * 1024**3
    assert pipeline.deepcopy is deepcopy and "deepcopy" not in vars(pipeline.template)
    for module in (pipeline, pipeline.training, pipeline.training_audit):
        for name, function in vars(module).items():
            if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(module):
                assert not code_globals(inspect.unwrap(function).__code__) - vars(module).keys() - vars(builtins).keys(), name
    for text in ('"prediction_grids": count, "cases": 284', 'run_single_matrixformer_cached_benchmarks.py',
                 'review_single_matrixformer_campaign.py', 'len(report["cases"]) != 152', '"research_selection_tolerance": 0'):
        assert text in pipeline.ADAPTED_SOURCE
    assert pipeline.no_confirmation_reason.__code__.co_code == pipeline.template.no_confirmation_reason.__code__.co_code
    assert 'require_compact_audit(audit)' in pipeline.ADAPTED_SOURCE


def test_generated_sources_and_confirmation_identities_stable_across_fresh_interpreter():
    payload = {"pipeline": pipeline.ADAPTED_SOURCE_SHA256, "training": pipeline.training.ADAPTED_SOURCE_SHA256,
               "audit": pipeline.training_audit.ADAPTED_SOURCE_SHA256, "cases": pipeline.training.confirmation_cases()}
    command = ("import json, runpy; module=runpy.run_path('scripts/run_single_model_legacy_heavy_pipeline.py', run_name='identity_test'); "
               "print(json.dumps({'pipeline':module['ADAPTED_SOURCE_SHA256'], 'training':module['training'].ADAPTED_SOURCE_SHA256, "
               "'audit':module['training_audit'].ADAPTED_SOURCE_SHA256, 'cases':module['training'].confirmation_cases()}))")
    result = subprocess.check_output([sys.executable, "-c", command], cwd=pipeline.REPO,
        env={**os.environ, "PYTHONHASHSEED": "124", "CUDA_VISIBLE_DEVICES": ""}, text=True)
    assert json.loads(result) == payload


def test_actual_v30_task_materialization_executes_all_28_tasks(tmp_path, monkeypatch):
    def namespace():
        result = dict(vars(pipeline))
        function = pipeline.materialize_endpoint_tasks
        result["materialize_endpoint_tasks"] = types.FunctionType(function.__code__, result, function.__name__)
        return result

    monkeypatch.setattr(materialization_tests.recovery, "execution_namespace", namespace)
    fixture = materialization_tests.materialization.__wrapped__(tmp_path)
    materialization_tests.test_execute_all_28_materialized_tasks_without_mutating_inputs(fixture)


def test_v30_actual_endpoint_can_precede_terminal_ordinal(monkeypatch):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_last_real_endpoint_can_precede_terminal_ordinal_without_fabricating_a_batch(monkeypatch)


def test_v30_empty_endpoint_proof_does_not_infer_or_select(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_empty_endpoint_is_audited_without_fake_neural_replay_or_selection(tmp_path, monkeypatch)


@pytest.mark.parametrize("mutation", [None, "fixed_task_count", "omitted_admission", "rating_count"])
def test_v30_requires_exact_independent_admission_and_task_counts(monkeypatch, mutation):
    monkeypatch.setattr(inherited, "pipeline", pipeline)
    inherited.test_pipeline_requires_exact_admitted_counts_and_all_scheduled_structured_draws(monkeypatch, mutation)


def test_only_exact_registered_shell_exec_handoff_is_accepted(monkeypatch, tmp_path):
    expected = {"pid": os.getpid(), "start_ticks": 42, "command":
                "bash -c python scripts/train_single_matrixformer_legacy_heavy.py --stage train; " + pipeline.AUDITOR_COMMAND}
    observed = {**expected, "command": pipeline.AUDITOR_COMMAND}
    monkeypatch.setattr(pipeline, "process_identity", lambda pid: observed)
    monkeypatch.setattr(pipeline, "REPO", Path.cwd())
    assert pipeline.chain_is_live(expected)
    observed["start_ticks"] += 1
    assert not pipeline.chain_is_live(expected)
    observed["start_ticks"] -= 1
    monkeypatch.setattr(pipeline, "REPO", tmp_path)
    assert not pipeline.chain_is_live(expected)
