import inspect

import test_single_model_anchor_retention_trial as original_template
import train_single_matrixformer_all_imputation_retention as previous_training


def fixed_preflight():
    records = original_tests.valid_preflight()
    records[0]["record"].update(proposal_policy=training.paired.POLICY, displacement_fraction=.5)
    records[1]["raw_endpoint"]["fixed_half_step"] = {"fraction": .5, "evaluated": True,
        "parameters": 77998496, "actual_saved_displacements_checked": True}
    return records


SOURCE = inspect.getsource(original_template)
for old, new in {
    "train_single_matrixformer_anchor_retention": "train_single_matrixformer_legacy_blend_teacher",
    "881100901": "1031100901", "882100901": "1032100901",
    "891100901": "1041100901", "892100901": "1042100901",
    "training.paired is training.predecessor.paired": "training.paired is training.preflight.paired",
    "training.teacher_policy is training.predecessor.teacher_policy": "training.teacher_policy is training.preflight.teacher_policy",
    "records = original_tests.valid_preflight()": "records = fixed_preflight()",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


def test_new_teacher_only_recipe_keeps_exact_original_losses_priors_and_qualification():
    assert training.OUTPUT.name == "legacy_blend_teacher_v41"
    assert training.PARENT == previous_training.PARENT and training.PARENT_SHA256 == previous_training.PARENT_SHA256
    assert training.teacher_policy.POLICY == {"legacy": "observed_blend", "structured": "general", "recsys": "specialist"}
    assert previous_training.teacher_policy.POLICY["legacy"] == "general"
    assert training.retention is previous_training.retention and training.retention_training is previous_training.retention_training
    assert training.paired is previous_training.paired and training.RATE == previous_training.RATE
    assert training.TRIAL_QUALIFICATION == previous_training.TRIAL_QUALIFICATION
    assert training.diagnostics is previous_training.diagnostics
    assert training.capacity.probe is training.preflight
    assert training.STORAGE_ACCOUNTING["scratch"] == {**previous_training.STORAGE_ACCOUNTING["scratch"],
        "legacy_blend_component_allowance": 2121943040}
    for volume in ("checkpoint", "root"):
        assert training.STORAGE_ACCOUNTING[volume] == previous_training.STORAGE_ACCOUNTING[volume]
    assert sum(training.STORAGE_ACCOUNTING["scratch"].values()) == 38340399232
    current = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    previous = training.original.alignment.fixed.seeds_in([previous_training.batches(), previous_training.confirmation_cases(), training.preflight.batch()])
    assert len(current) == 3868 and not current.intersection(previous)
    for batch, original in zip(training.batches(), previous_training.batches()):
        for task, prior in zip(batch["tasks"], original["tasks"]):
            copied = deepcopy(task)
            copied["gradient_seed"] = prior["gradient_seed"]
            copied["specification"].get("structured", copied["specification"])["seed"] = prior["specification"].get("structured", prior["specification"])["seed"]
            assert copied == prior


def test_registration_freezes_new_component_budget_sources_and_training_only_policy():
    assert '"legacy_blend_replay": blend_replay.specification(SCRATCH, batches())' in training.REGISTER_SOURCE
    assert 'config["legacy_blend_replay"] != blend_replay.specification(SCRATCH, config["batches"])' in training.ADAPTED_SOURCE
    for name in ("legacy_blend_full_replay", "legacy_blend_teacher_trial", "legacy_blend_teacher_audit", "legacy_blend_teacher_pipeline"):
        assert f"tests/test_single_model_{name}.py" in training.REGISTER_SOURCE
    for name in ("train", "verify_registration", "register_trial", "setup"):
        assert getattr(training, name).__globals__["teacher_policy"] is training.preflight.teacher_policy
    assert '1e4d520861c2fc0490e68e14396e62ffd25ccfdaeaf623550bfb0bcc1f0ed891' in training.ADAPTATION_SOURCE
    assert '"displacement_fraction": paired.FRACTION' in training.REGISTER_SOURCE
    assert 'config["displacement_fraction"] != paired.FRACTION' in training.ADAPTED_SOURCE
