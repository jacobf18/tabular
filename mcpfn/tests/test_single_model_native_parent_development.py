from pathlib import Path

import test_single_model_development_teacher_trajectory as previous_tests


TEST_SOURCE = Path(previous_tests.__file__).read_text()
for old, new in {
    "import analyze_single_model_development_teacher_trajectory as analysis":
        "import analyze_single_model_native_parent_development as analysis",
    "all_imputation_blend_v45": "dual_parent_v50",
    "stronger_mean_distillation_v46": "native_parent_v51",
    "v46_full_independent_audit_claimed": "v51_full_independent_audit_claimed",
}.items():
    if old not in TEST_SOURCE:
        raise ValueError(f"Missing explicit native-parent diagnostic test adaptation: {old}")
    TEST_SOURCE = TEST_SOURCE.replace(old, new)
exec(compile(TEST_SOURCE, __file__, "exec"), globals())


def test_native_parent_comparison_binds_frozen_configs_and_historical_audit_only():
    assert analysis.RUNS == {
        "dual_parent_v50": "0d07291d0000fbf0fc6ff1b9b7cfd77ec298139dc29ec10900b91421f7561223",
        "native_parent_v51": "f8fcbee841f39f6c7bb33f02c5ce595b3734ded91e51a3ddb07c4c54fa6f5bd0",
    }
    assert analysis.HISTORICAL_AUDIT_SHA256 == "1935ed5ac8b6dfb3c6bfb176366589456f361ec7ff69b1ec2bb16ca1dbce4ec7"
    assert "actual_v29_shrunk_mse" in analysis.metrics.METRICS
    assert "structured_mean_shrunk_mse" in analysis.metrics.METRICS


def test_complete_result_includes_all_added_distances_without_altering_training(tmp_path, monkeypatch):
    root = complete_fixture(tmp_path, monkeypatch)
    output = tmp_path / "diagnostic"
    analysis.analyze(root, output, 10)
    configuration = analysis.metrics.read(output / "config.json")
    assert configuration["v51_full_independent_audit_claimed"] is False
    assert str(Path(analysis.blueprint.__file__).resolve()) in configuration["source_sha256"]
    assert str(Path(analysis.metrics.previous.__file__).resolve()) in configuration["source_sha256"]
    for filename in ("cases.csv", "summary.csv"):
        columns = (output / filename).read_text().splitlines()[0].split(",")
        assert all(name in columns for name in analysis.metrics.METRICS)
