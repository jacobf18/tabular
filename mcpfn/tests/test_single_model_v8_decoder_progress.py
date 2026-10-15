from copy import deepcopy

import pytest

import report_single_model_v8_decoder_progress as progress
from test_single_model_guard_contributions import fixture


def test_v8_report_binds_identical_training_and_retention_baseline():
    assert progress.training.PARENT == progress.training.original.alignment.fixed.PARENT
    assert progress.training.OUTPUT == progress.training.ROOT / "v8_decoder_v17"
    assert progress.training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert 'parent["decision"]' not in progress.ADAPTED_SOURCE
    assert 'initial_path = training.original.alignment.fixed.PRODUCER / "validation_0.json"' in progress.ADAPTED_SOURCE


def test_v8_report_decomposes_without_a_parent_decision():
    initial, current, guards = fixture()
    before = deepcopy((initial, current, guards))
    result = progress.decompose(initial, current, guards)
    assert (initial, current, guards) == before
    assert len(result["guards"]) == len(guards)
    assert len(result["cases"]) == len(initial)
    assert sum(row["guard_delta_contribution"] for row in result["components"]) == pytest.approx(-1 / 3)


def test_v8_report_keeps_all_cases_and_guards_and_performs_no_inference():
    assert 'len(current["cases"]) != 152' in progress.ADAPTED_SOURCE
    assert 'len(current["decision"]["comparisons"]) != 22' in progress.ADAPTED_SOURCE
    assert '"prediction_performed": False, "selection_performed": False' in progress.ADAPTED_SOURCE
    assert 'Path(template.__file__).resolve()' in progress.ADAPTED_SOURCE
    assert 'single_model_registered_adaptation.py' in progress.ADAPTED_SOURCE


@pytest.mark.parametrize("mutation", ["incomplete", "not_decoder", "tolerance", "parent", "baseline", "step"])
def test_v8_report_rejects_incompatible_audits(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(progress.torch.cuda, "is_initialized", lambda: False)
    audit = {"complete": True, "decoder_only_training": True, "research_selection_tolerance": 0,
             "training_initialization": str(progress.training.PARENT),
             "retention_baseline": str(progress.training.PARENT), "through_step": 10}
    if mutation == "incomplete":
        audit["complete"] = False
    elif mutation == "not_decoder":
        audit["decoder_only_training"] = False
    elif mutation == "tolerance":
        audit["research_selection_tolerance"] = .001
    elif mutation == "parent":
        audit["training_initialization"] = "another-checkpoint"
    elif mutation == "baseline":
        audit["retention_baseline"] = "another-checkpoint"
    else:
        audit["through_step"] = 12
    monkeypatch.setattr(progress.training, "read", lambda path: audit)
    with pytest.raises(ValueError, match="unchanged completed audit"):
        progress.report(tmp_path / "audit.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()
