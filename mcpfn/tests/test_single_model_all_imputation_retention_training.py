import test_single_model_fixed_half_shrunk_retention_training as previous_tests


SOURCE = previous_tests.SOURCE
for old, new in {
    "single_model_fixed_half_shrunk_retention_training": "single_model_all_imputation_retention_training",
    "preflight_single_model_fixed_half_shrunk_retention": "preflight_single_model_all_imputation_retention",
    "test_single_model_anchor_retention_preflight as base_fixtures": "test_single_model_all_imputation_retention_preflight as base_fixtures",
    'len(calls) == 2 and state.reference_grids == 2': 'len(calls) == 26 and state.reference_grids == 26',
    'report["reference_grids"] == 2': 'report["reference_grids"] == 26',
    '[4., 5.]]])': '[4., 3.5]]])',
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


def test_every_imputation_component_is_eligible_and_ratings_still_excluded():
    assert integration.retention is probe.retention
    assert integration.retention.SPECIFICATION["legacy_components"] == list(range(8))
    assert len(integration.retention.SPECIFICATION["structured_components"]) == 4
    assert integration.retention.SPECIFICATION["prior_count"] == 1.
    assert integration.retention.REFERENCE_SHA256 == probe.PARENT_SHA256
