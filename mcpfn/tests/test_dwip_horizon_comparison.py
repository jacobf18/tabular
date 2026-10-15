import json
from pathlib import Path

import pytest

from analyze_dwip_horizon_comparison import MATCHED_SOURCES, verify_matched_commands, verify_source_hashes
from run_dwip_spectral_geometry import build_commands as control_commands
from run_dwip_spectral_horizon import build_commands as horizon_commands


def commands():
    return (control_commands(Path("/tmp/control"))[1]["dwip1e5"],
            horizon_commands(Path("/tmp/horizon"))[1]["dwip1e5"])


def test_horizon_comparison_allows_only_expected_schedule_and_location_changes():
    control, candidate = commands()
    assert verify_matched_commands(control, candidate) == {"control": 24000, "candidate": 48000}


@pytest.mark.parametrize("flag,value", [("--data-seed", "999"), ("--point-loss-weight", "3"), ("--warmup-steps", "800")])
def test_horizon_comparison_rejects_confounding_changes(flag, value):
    control, candidate = commands()
    candidate[candidate.index(flag) + 1] = value
    with pytest.raises(ValueError, match="beyond the declared horizon"):
        verify_matched_commands(control, candidate)


def test_horizon_comparison_rejects_different_training_shapes():
    control, candidate = commands()
    position = candidate.index("--phases-json") + 1
    phases = json.loads(candidate[position])
    phases[-1]["rows_high"] = 400
    candidate[position] = json.dumps(phases)
    with pytest.raises(ValueError, match="beyond the declared horizon"):
        verify_matched_commands(control, candidate)


def test_source_check_allows_only_irrelevant_orchestration_changes():
    control = {path: "recorded" for path in MATCHED_SOURCES}
    candidate = {**control, "scripts/run_spectral_query_experiments.py": "new_horizon_support"}
    assert verify_source_hashes(control, candidate) == control


@pytest.mark.parametrize("value", [None, "changed"])
def test_source_check_rejects_missing_or_changed_training_source(value):
    control = {path: "recorded" for path in MATCHED_SOURCES}
    candidate = {**control, MATCHED_SOURCES[0]: value}
    with pytest.raises(ValueError, match="source hashes differ"):
        verify_source_hashes(control, candidate)
