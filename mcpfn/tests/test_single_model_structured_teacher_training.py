from copy import deepcopy

import audit_single_model_structured_teacher as audit
import train_single_matrixformer_structured_teacher as training


def remove_seeds(value):
    if isinstance(value, dict):
        return {key: remove_seeds(item) for key, item in value.items() if "seed" not in key}
    if isinstance(value, list):
        return [remove_seeds(item) for item in value]
    return value


def test_full_prior_and_optimizer_schedule_unchanged_except_teachers_and_seeds():
    batches = training.batches()
    original = training.baseline.batches()
    assert len(batches) == 64 and all(len(batch["tasks"]) == 28 for batch in batches)
    assert remove_seeds(batches) == remove_seeds(original)
    assert remove_seeds(training.confirmation_cases()) == remove_seeds(training.baseline.confirmation_cases())
    assert len(training.confirmation_cases()) == 284
    seeds = training.original.alignment.fixed.seeds_in([batches, training.confirmation_cases()])
    assert len(seeds) == 3868
    earlier = training.original.alignment.fixed.seeds_in([original, training.baseline.confirmation_cases(), training.preflight.batch()])
    assert not seeds.intersection(earlier)
    assert training.PARENT == training.original.alignment.fixed.PARENT
    assert training.INITIAL_RUN == training.original.alignment.fixed.PRODUCER
    assert training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert all(training.constant_rate(step) == 2e-5 for step in range(1, 65))


def test_only_training_runtime_changes_and_all_evidence_is_required():
    source = training.ADAPTED_SOURCE
    assert 'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]):' in source
    assert 'with original.alignment.mixture.runtime(config["base"], directory):' in source
    assert 'paired.protected_update(' in source
    assert 'original.alignment.fixed.selection_decision(' in source
    assert 'shutil.disk_usage(ROOT).free < 31 * 1024**3' in source
    assert 'preflight.require_predecessor()' in source
    assert 'audit["teacher_replay"]["full_grids"] != 28' in source
    assert training.baseline.QUALIFICATION in training.QUALIFICATION
    assert 'training.preflight.check_routing(payload, task, config)' in audit.ADAPTED_SOURCE
    assert audit.training is training
    assert 'teacher_policy.runtime(' not in training.baseline.ADAPTED_SOURCE


def test_new_teacher_assignment_never_overwrites_baseline_guard_decisions():
    baseline = {"step": 25, "score": 1.1}
    decision = {"passed": False, "score": 1.0}
    original = deepcopy(baseline)
    assert training.selected_boundary(baseline, 50, 47, decision) == original
