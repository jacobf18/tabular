import ast
from copy import deepcopy

import pytest

import train_single_matrixformer_top_block_continuation as training
import audit_single_model_top_block_continuation as audit


def test_complete_mixture_schedule_has_fresh_data_and_reserved_confirmation():
    batches = training.batches()
    schedule = training.schedule.batches()[:96]
    previous = training.template.batches()
    assert training.STEPS == len(batches) == 96
    assert training.BOUNDARIES == (0, 16, 32, 64, 96)
    for batch, original in zip(batches, schedule):
        expected = training.replicated.replicate_batch(original, training.DATA_SEED, training.GRADIENT_SEED)
        assert batch == expected and len(batch["tasks"]) == 28
    current = training.original.alignment.fixed.seeds_in(batches)
    confirmation = training.original.alignment.fixed.seeds_in(training.confirmation_cases())
    assert len(current) == 5376 and len(confirmation) == 284 and not current.intersection(confirmation)
    consumed = training.original.alignment.fixed.seeds_in([previous, training.template.confirmation_cases()])
    assert not (current | confirmation).intersection(consumed)
    assert training.PARENT == training.template.OUTPUT / "selected_checkpoint.pth"
    assert training.PARENT != training.original.alignment.fixed.PARENT


def test_optimizer_and_train_loop_are_unchanged_with_fresh_moments():
    actual = ast.parse(training.ADAPTED_SOURCE)
    previous = ast.parse(training.template.ADAPTED_SOURCE)
    for name in ("setup", "train", "selected_boundary"):
        current = next(node for node in actual.body if isinstance(node, ast.FunctionDef) and node.name == name)
        original = next(node for node in previous.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(current) == ast.dump(original)
    assert all(training.constant_rate(step) == 1e-5 for step in range(1, 97))
    assert training.PREFIXES == training.template.PREFIXES
    for name in ("save_endpoint", "audit_endpoint", "constant_rate", "register_trial"):
        assert getattr(training, name).__globals__ is training.__dict__
    for name in ("load_regenerated_task", "regeneration_evidence", "run_audit"):
        assert getattr(audit, name).__globals__ is audit.__dict__
    assert audit.training is training and audit.sys is not None and audit.time is not None


@pytest.mark.parametrize("step", [0, 97, True, 1.5])
def test_unchanged_rate_cannot_escape_registered_attempts(step):
    with pytest.raises(ValueError):
        training.constant_rate(step)


def test_parent_is_honestly_unqualified_and_never_becomes_a_teacher_or_baseline():
    source, registration = training.ADAPTED_SOURCE, training.REGISTER_SOURCE
    for expression in ('teacher_policy.identities(config["base"], original.alignment.fixed.PARENT)',
                       'config["parent_selection_qualified"] is not False',
                       'config["parent_training_updates"] != PARENT_UPDATES',
                       '"parent_training_updates": PARENT_UPDATES',
                       'read(INITIAL_RUN / "validation_64.json")["student"]',
                       'diagnostics.require_empty_validation(config["diagnostic_storage"], directory)',
                       'diagnostics.verify(config["diagnostic_storage"], "training")',
                       '"decoder_only_training": False, "top_block_training": True'):
        assert expression in source
    for expression in ('"parent_selection_qualified": False',
                       '"retention_baseline": str(original.alignment.fixed.PARENT)',
                       'teacher_policy.identities(base, original.alignment.fixed.PARENT)',
                       'len(fresh) != 5660', 'fresh.intersection(used)',
                       'diagnostics.prepare(config["diagnostic_storage"], "training")'):
        assert expression in registration
    assert 'teacher_policy.identities(base, PARENT)' not in registration
    assert training.PARENT_UPDATES == 61


def test_audit_requires_full_continuation_and_original_v8_guards():
    source = audit.ADAPTED_SOURCE
    for expression in ('saved["parent_training_updates"] != training.PARENT_UPDATES',
                       'parent_report_path = training.INITIAL_RUN / "validation_64.json"',
                       '(parent_report, ("student",))', '152 * (4 + len(boundaries))',
                       'len(selected) != 34', '19496843', 'frozen["frozen_tensors"] != 220',
                       'len(decision["comparisons"]) != 22', 'load_regenerated_task(path)',
                       'training.audit_endpoint(payload, raw_path)',
                       '"decoder_only_training": False, "top_block_training": True'):
        assert expression in source
    assert '2688' in training.QUALIFICATION and '1136' in training.QUALIFICATION
    assert 'full historical autograd' in training.QUALIFICATION


@pytest.fixture
def history():
    result = {"complete": True, "gate": {"passed": False, "confirmation_evaluated": True}, "campaign": None, "review": None}
    report = {"complete": True, "complete_training_audit": True, "through_step": 64, "accepted_updates": 61,
              "best": {"step": 64}, "training_episodes": 1792, "prediction_grids": 1216,
              "decoder_only_training": False, "top_block_training": True,
              "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
              "artifact_sha256": {str(training.PARENT): training.PARENT_SHA256}}
    replay = {"complete": True, "endpoint_grids": 56, "selected_grids": 152}
    confirmation = {"complete": True, "cases": 284, "prediction_grids": 1136}
    return result, report, replay, confirmation, training.PARENT_SHA256


def test_complete_failed_v24_is_training_input_not_qualification(history):
    training.validate_history(*history)
    for index, key, wrong in ((0, "campaign", "published"), (1, "accepted_updates", 60),
                              (1, "prediction_grids", 1215), (1, "top_block_training", False),
                              (1, "independently_regenerated_training_tasks", 1791),
                              (2, "selected_grids", 151), (3, "prediction_grids", 1135)):
        altered = deepcopy(history)
        altered[index][key] = wrong
        with pytest.raises(ValueError, match="unqualified V24 lifecycle"):
            training.validate_history(*altered)
    with pytest.raises(ValueError):
        training.validate_history(*history[:-1], "other-checkpoint")


def test_scratch_qualification_is_declared_but_not_created_by_training_registration():
    plan = training.diagnostics.layout(training.OUTPUT, training.SCRATCH, training.BOUNDARIES)
    assert len(plan["directories"]["training"]) == 5
    assert len(plan["directories"]["qualification"]) == 2
    assert all("top_block_continuation_v25" in record["canonical"] for phase in plan["directories"].values() for record in phase)
    assert plan["scratch_is_provider_persistent"] is False
    assert 'diagnostics.prepare(config["diagnostic_storage"], "qualification")' not in training.REGISTER_SOURCE
