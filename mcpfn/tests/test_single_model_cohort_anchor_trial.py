from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import single_model_cohort_anchor_trial as trial
import train_single_matrixformer_legacy_blend_teacher as previous_training
from single_model_registered_adaptation import replace_once


def test_uniform_arithmetic_binding_preserves_original_teacher_component_verification():
    original = previous_training.ADAPTED_SOURCE
    clause = '\n            or config["legacy_blend_replay"] != blend_replay.specification(SCRATCH, config["batches"])'
    baseline = replace_once(original, clause, "")
    adapted = trial.adapt_training(baseline)
    assert adapted == replace_once(original, 'or config["teacher_policy"] != teacher_policy.POLICY',
        'or config["teacher_policy"] != teacher_policy.POLICY\n'
        '            or config["arithmetic"] != all_trial.ARITHMETIC')
    compile(adapted, "prospective_training", "exec")
    assert trial.reference_bytes is trial.previous.reference_bytes
    assert trial.ARITHMETIC == {"blas_threads": 24, "torch_threads": 4}


@pytest.mark.parametrize("failure", [False, True])
def test_cli_arithmetic_context_wraps_original_behavior_and_preserves_failures(failure):
    calls = []

    @contextmanager
    def arithmetic():
        calls.append("enter")
        try:
            yield
        finally:
            calls.append("exit")

    source = '\nif __name__ == "__main__":\n    calls.append("original_cli")\n    if failure:\n        raise RuntimeError("preserved")\n'
    adapted = trial.wrap_cli(source)
    namespace = {"__name__": "__main__", "calls": calls, "failure": failure,
        "arithmetic_recovery": SimpleNamespace(previous=SimpleNamespace(original_arithmetic=arithmetic))}
    if failure:
        with pytest.raises(RuntimeError, match="preserved"):
            exec(compile(adapted, "cli", "exec"), namespace)
    else:
        exec(compile(adapted, "cli", "exec"), namespace)
    assert calls == ["enter", "original_cli", "exit"]
    namespace.update(__name__="import_only", calls=[])
    exec(compile(adapted, "cli", "exec"), namespace)
    assert namespace["calls"] == []


@pytest.mark.parametrize("source", ["pass\n", '\nif __name__ == "__main__":\n    pass\n' * 2])
def test_cli_wrapper_rejects_missing_or_ambiguous_entrypoints(source):
    with pytest.raises(ValueError):
        trial.wrap_cli(source)


def test_recipe_policy_preserves_full_qualification_and_discloses_adverse_probe_result():
    for text in ("64 fresh", "0/10/25/50/64", "all22", "284 fresh", "32 original", "every frozen real",
            "every soft retention penalty increased", "No probe", "scientific retry", "no additional inference model"):
        assert text in trial.POLICY


def test_recipe_integrates_into_the_complete_inherited_training_definition_without_running_it():
    namespace = {"__name__": "unregistered_cohort_recipe_test", "__file__": trial.__file__,
        "all_trial": trial, "fixed_trial": previous_training.fixed_trial, "previous_training": previous_training}
    exec(compile(previous_training.ADAPTATION_SOURCE, trial.__file__, "exec"), namespace)
    assert namespace["all_trial"] is trial and namespace["POLICY"] == trial.POLICY
    assert 'with arithmetic_recovery.previous.original_arithmetic():' in namespace["SOURCE"]
    assert 'config["arithmetic"] != all_trial.ARITHMETIC' in namespace["ADAPTED_SOURCE"]
    assert '"arithmetic": all_trial.ARITHMETIC' in namespace["REGISTER_SOURCE"]
    assert 'test_single_model_cohort_anchor_trial.py' in namespace["REGISTER_SOURCE"]
    assert namespace["STEPS"] == 64 and namespace["BOUNDARIES"] == previous_training.BOUNDARIES
    assert namespace["PARENT"] == previous_training.PARENT
