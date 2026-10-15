import test_single_model_column_anchor_retention_training as previous_tests


SOURCE = previous_tests.SOURCE
for old, new in {
    "single_model_column_anchor_retention_training": "single_model_fixed_half_shrunk_retention_training",
    "preflight_single_model_column_anchor_retention": "preflight_single_model_fixed_half_shrunk_retention",
}.items():
    assert old in SOURCE
    SOURCE = SOURCE.replace(old, new)
exec(compile(SOURCE, __file__, "exec"), globals())


def test_all_reference_replays_use_unchanged_count_shrunk_loss_and_frozen_parent():
    assert integration.retention is probe.retention
    assert integration.retention.SPECIFICATION["prior_count"] == 1.
    assert integration.retention.SPECIFICATION["prior_variance"] == 1.
    assert integration.retention.REFERENCE_SHA256 == probe.PARENT_SHA256
