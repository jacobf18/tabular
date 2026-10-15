import inspect

import test_single_model_anchor_retention_trial as anchor_template
import train_single_matrixformer_legacy_blend_teacher as previous_training


def fixed_preflight():
    records = original_tests.valid_preflight()
    records[0]["record"].update(proposal_policy=training.paired.POLICY, displacement_fraction=.5)
    records[1]["raw_endpoint"]["fixed_half_step"] = {"fraction": .5, "evaluated": True,
        "parameters": 77998496, "actual_saved_displacements_checked": True}
    return records


ADAPTED_TEST_SOURCE = inspect.getsource(anchor_template)
for old, new in {
    "train_single_matrixformer_anchor_retention": "train_single_matrixformer_cohort_anchor_retention",
    "881100901": "1061100901", "882100901": "1062100901",
    "891100901": "1071100901", "892100901": "1072100901",
    "training.paired is training.predecessor.paired": "training.paired is training.preflight.paired",
    "training.teacher_policy is training.predecessor.teacher_policy": "training.teacher_policy is training.preflight.teacher_policy",
    "records = original_tests.valid_preflight()": "records = fixed_preflight()",
    "training.retention.REFERENCE_SHA256": 'training.retention.REFERENCE_SHA256S["v29"]',
    'result = deepcopy(model).eval().requires_grad_(False)':
        'result = {role: deepcopy(model).eval().requires_grad_(False) for role in ("v8", "v29")}',
    'current.decoder.weight.item() == 0.': 'all(model.decoder.weight.item() == 0. for model in current.values())',
}.items():
    assert old in ADAPTED_TEST_SOURCE
    ADAPTED_TEST_SOURCE = ADAPTED_TEST_SOURCE.replace(old, new)
exec(compile(ADAPTED_TEST_SOURCE, __file__, "exec"), globals())


def test_reference_assignment_only_trial_preserves_every_other_scientific_setting():
    assert training.OUTPUT.name == "cohort_anchor_retention_v42"
    assert training.PARENT == previous_training.PARENT and training.PARENT_SHA256 == previous_training.PARENT_SHA256
    assert training.retention is training.preflight.retention and training.retention is not previous_training.retention
    assert training.retention_training is not previous_training.retention_training
    assert training.retention.BACKWARD_SOURCE_SHA256 == previous_training.retention.BACKWARD_SOURCE_SHA256
    assert training.teacher_policy is previous_training.teacher_policy and training.paired is previous_training.paired
    assert training.RATE == previous_training.RATE and training.TRIAL_QUALIFICATION == previous_training.TRIAL_QUALIFICATION
    assert training.STORAGE_ACCOUNTING == previous_training.STORAGE_ACCOUNTING
    assert training.diagnostics is previous_training.diagnostics and training.capacity.probe is training.preflight
    current = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    previous = training.original.alignment.fixed.seeds_in([previous_training.batches(), previous_training.confirmation_cases(), training.preflight.batch()])
    assert len(current) == 3868 and not current.intersection(previous)
    for batch, original in zip(training.batches(), previous_training.batches()):
        for task, prior in zip(batch["tasks"], original["tasks"]):
            copied = deepcopy(task)
            copied["gradient_seed"] = prior["gradient_seed"]
            copied["specification"].get("structured", copied["specification"])["seed"] = prior["specification"].get("structured", prior["specification"])["seed"]
            assert copied == prior


def test_registration_binds_both_references_exact_arithmetic_and_all_new_sources():
    assert '"retention_references": retention_training.reference_identities(PARENT, base)' in training.REGISTER_SOURCE
    assert '"arithmetic": all_trial.ARITHMETIC' in training.REGISTER_SOURCE
    assert 'config["arithmetic"] != all_trial.ARITHMETIC' in training.ADAPTED_SOURCE
    assert 'with arithmetic_recovery.previous.original_arithmetic():' in training.SOURCE
    assert 'CAPACITY_RECEIPT_SHA256' in training.ADAPTATION_SOURCE
    assert 'receipt["canonical_artifacts_checked"] != 7' in training.ADAPTATION_SOURCE
    for name in ("trial", "audit", "pipeline", "training"):
        assert f"tests/test_single_model_cohort_anchor_retention_{name}.py" in training.REGISTER_SOURCE
    for name in ("train", "verify_registration", "register_trial", "setup"):
        assert getattr(training, name).__globals__["retention_training"] is training.retention_training


def test_full_trial_cannot_register_without_the_completed_capacity_receipt(monkeypatch):
    monkeypatch.setattr(training, "CAPACITY_RECEIPT_SHA256", None)
    with pytest.raises(ValueError, match="storage-preservation receipt"):
        training.retention_evidence()
