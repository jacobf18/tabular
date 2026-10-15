from pathlib import Path

from test_single_model_anchor_retention import metadata


TEST_SOURCE = Path(__file__).with_name("test_single_model_parent_structured_mean_probe_audit.py").read_text()
for old, new in {
    "audit_single_model_parent_structured_mean_probe": "audit_single_model_native_parent_mean_probe",
    "single_model_parent_structured_retention": "single_model_native_parent_retention",
    "parent_structured_retention_preflight_v1_mean_audit": "native_parent_retention_preflight_v1_mean_audit",
    "test_single_model_parent_structured_mean_probe_audit.py": "test_single_model_native_parent_mean_probe_audit.py",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit native-parent mean-audit test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_exact_native_parent_probe_and_both_reference_components_are_bound():
    assert audit.CONFIG_SHA256 == "fc04a4288d5dbbbf925ccb3dc6c1d81bdc7b23c6d528a949fb9370040d681cb3"
    assert '"component_role_counts"] != {"v8": 24, "v29": 26}' in audit.AUDIT_SOURCE
    assert '"component_grids"] != 50' in audit.AUDIT_SOURCE
    assert '"reference_grids"] != 26' in audit.AUDIT_SOURCE
    assert 'Path(original_math.__file__).resolve()' in audit.AUDIT_SOURCE
    assert audit.probe.retention.reference_weights("v29") == {"v29": 1.}
    assert audit.probe.retention.reference_weights("structured_mean") == {"v8": .5, "v29": .5}
    assert 'native = native_task(registered, metadata)' in audit.AUDIT_SOURCE
    assert 'reference_role == "general"' not in audit.AUDIT_SOURCE


@pytest.mark.parametrize("component", range(8))
def test_every_native_component_is_classified_by_cohort_not_reference_label(component):
    fields = metadata(component, prior="anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar")
    assert audit.probe.retention.reference_role(fields) == "v29"
    assert audit.native_task({"cohort": "legacy"}, fields) is True
    with pytest.raises(ValueError):
        audit.native_task({"cohort": "mar"}, fields)


@pytest.mark.parametrize("family", audit.probe.retention.FAMILIES)
def test_structured_and_rating_tasks_are_never_native(family):
    fields = metadata("low_rank_mcar", structured=True, family=family)
    assert audit.native_task({"cohort": family}, fields) is False
    assert audit.native_task({"cohort": "recsys"}, metadata(domain="recsys")) is False
    with pytest.raises(ValueError):
        audit.native_task({"cohort": "legacy"}, fields)


def test_registered28_task_mixture_contains_exactly_two_native_tasks():
    classifications = []
    for task in audit.probe.batch()["tasks"]:
        cohort = task["cohort"]
        fields = metadata() if cohort == "legacy" else metadata(domain="recsys") if cohort == "recsys" else metadata(
            "low_rank_mcar", structured=True, family=cohort)
        classifications.append(audit.native_task(task, fields))
    assert len(classifications) == 28 and sum(classifications) == 2
