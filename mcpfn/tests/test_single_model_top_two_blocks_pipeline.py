import ast
from copy import deepcopy

import pytest

import run_single_model_top_two_blocks_pipeline as pipeline


def test_full_trial_scope_and_original_real_campaign_are_preserved():
    source = pipeline.ADAPTED_SOURCE
    for expression in ('audit["through_step"] != 64', 'audit["training_episodes"] != 1792',
                       'audit["prediction_grids"] != 1216', 'require_compact_audit(audit)',
                       'view = top_two.TopTwoBlockParameterView(model)',
                       'result["frozen"]["frozen_tensors"] != 192',
                       'len(decision["comparisons"]) != 22', 'len(bank) != 284', 'count != 1136',
                       'wait_for_training_audit(output)', 'cached = check_materialized_task(row)',
                       'scripts/run_single_matrixformer_cached_benchmarks.py',
                       'scripts/review_single_matrixformer_campaign.py'):
        assert expression in source
    assert 'run_child(output, "training_audit"' not in source
    assert pipeline.training is pipeline.training_audit.training
    assert pipeline.PARENT_CACHE == pipeline.template.PARENT_CACHE
    for name in ("OUTPUT", "AUDIT", "REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        assert getattr(pipeline, name).name.startswith("top_two_blocks_v26")


def test_only_optimizer_scope_changes_in_replay_without_relaxing_confirmation_or_promotion():
    original_source = pipeline.template.ADAPTED_SOURCE.replace(
        "view = top_block.TopBlockParameterView(model)", "view = top_two.TopTwoBlockParameterView(model)")
    original_source = original_source.replace(
        'result["frozen"]["frozen_tensors"] != 220', 'result["frozen"]["frozen_tensors"] != 192')
    original_source = original_source.replace(
        'checkpoint.get("top_block_training") is not True',
        'checkpoint.get("top_block_training") is not True or checkpoint.get("top_two_block_training") is not True')
    actual_tree, original_tree = ast.parse(pipeline.ADAPTED_SOURCE), ast.parse(original_source)
    for name in ("confirm", "audit_confirmation", "confirmation_setup", "no_confirmation_reason",
                 "replay", "audit_replay", "endpoint_setup"):
        actual = next(node for node in actual_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        original = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(actual) == ast.dump(original)
    for name in ("materialize_endpoint_tasks", "check_materialized_task", "wait_for_training_audit", "process_identity"):
        assert getattr(pipeline, name).__code__.co_code == getattr(pipeline.template, name).__code__.co_code
        assert getattr(pipeline, name).__globals__ is pipeline.__dict__
    assert pipeline.no_confirmation_reason({"step": 0}, {}) is not None
    assert pipeline.no_confirmation_reason({"step": 10}, {"selected_validation_passed": False}) is not None


def test_only_the_live_new_train_audit_chain_can_be_bound():
    constants = pipeline.register_chain_identity.__code__.co_consts
    assert 'train_single_matrixformer_top_two_blocks.py --stage train' in constants
    assert 'python scripts/audit_single_model_top_two_blocks.py' in constants
    assert pipeline.register_chain_identity.__globals__ is pipeline.__dict__


def test_registered_qualification_layout_matches_training_and_budgets():
    plan = pipeline.diagnostics.layout(pipeline.training.OUTPUT, pipeline.training.SCRATCH, pipeline.training.BOUNDARIES)
    assert {row["canonical"] for row in plan["directories"]["qualification"]} == {str(pipeline.REPLAY), str(pipeline.CONFIRMATION)}
    source = pipeline.ADAPTED_SOURCE
    assert 'diagnostics.prepare(read(training.OUTPUT / "config.json")["diagnostic_storage"], "qualification")' in source
    assert 'for directory in (REPLAY, CONFIRMATION):\n        directory.mkdir()' not in source
    assert 'diagnostics.verify(read(training.OUTPUT / "config.json")["diagnostic_storage"], "qualification")' in source
    assert pipeline.STAGE_WRITE_BUDGETS["confirm"]["scratch"] == 2560 * 1024**2
    assert pipeline.STAGE_WRITE_BUDGETS["audit-confirmation"]["checkpoint"] >= 312096899


@pytest.mark.parametrize("stage", list(pipeline.STAGE_WRITE_BUDGETS))
def test_every_stage_checks_actual_volume_budgets_and_declared_links(monkeypatch, stage):
    calls = []
    monkeypatch.setattr(pipeline.diagnostics, "require_capacity", lambda label: calls.append(("capacity", label)))
    monkeypatch.setattr(pipeline.diagnostics, "verify", lambda plan, phase: calls.append(("verify", plan, phase)))
    monkeypatch.setattr(pipeline, "read", lambda path: {"diagnostic_storage": "registered-layout"})
    pipeline.require_stage_capacity(stage)
    assert calls == [("capacity", stage), ("verify", "registered-layout", "qualification")]
    with pytest.raises(ValueError):
        pipeline.require_stage_capacity("unknown")


def test_complete_compact_audit_and_both_top_block_flags_are_mandatory(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "AUDIT", tmp_path / "audit")
    pipeline.write_json(pipeline.AUDIT / "audit.json", {})
    directory = tmp_path / "audit_task_replay"
    pipeline.write_json(directory / "status.json", {"state": "complete"})
    pipeline.write_json(directory / "result.json", {"complete": True, "tasks": 1792,
                        "audit_sha256": pipeline.sha256(pipeline.AUDIT / "audit.json")})
    pipeline.write_json(directory / "config.json", {"source_sha256": {}})
    report = {"teacher_identity_checked_tasks": 1792, "training_teacher_policy": pipeline.training.teacher_policy.POLICY,
              "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
              "exact_complete_task_and_teacher_digest_checks": True, "decoder_only_training": False,
              "top_block_training": True, "top_two_block_training": True,
              "autograd_replayed": False, "full_optimizer_trajectory_replayed": False}
    pipeline.require_compact_audit(report)
    for key, wrong in {"teacher_identity_checked_tasks": 2688, "independently_regenerated_training_tasks": 2688,
                       "independently_regenerated_rating_contexts": 192, "decoder_only_training": True,
                       "top_block_training": False, "top_two_block_training": False,
                       "autograd_replayed": True, "full_optimizer_trajectory_replayed": True}.items():
        with pytest.raises(ValueError):
            pipeline.require_compact_audit({**report, key: wrong})


def test_new_reservation_cannot_be_replaced_with_consumed_v25_confirmation():
    reservation = {"pipeline_config_sha256": "pipeline", "run_config_sha256": "run", "policy": pipeline.POLICY,
                   "confirmation_cases": pipeline.training.confirmation_cases()}
    pipeline.verify_reservation(reservation, "pipeline", "run", True)
    altered = deepcopy(reservation)
    altered["confirmation_cases"] = pipeline.template.training.confirmation_cases()
    with pytest.raises(ValueError, match="reservation changed"):
        pipeline.verify_reservation(altered, "pipeline", "run", True)
