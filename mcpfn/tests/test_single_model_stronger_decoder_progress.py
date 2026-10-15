import report_single_model_decoder_progress as previous
import report_single_model_stronger_decoder_progress as stronger
import report_single_model_targeted_decoder_progress as targeted


def test_stronger_report_preserves_distinct_v8_baseline_and_v14_initialization():
    assert stronger.training.PARENT == targeted.training.PARENT
    assert stronger.training.INITIAL_RUN == targeted.training.INITIAL_RUN
    assert stronger.training.OUTPUT == stronger.training.ROOT / "stronger_decoder_v16"
    assert stronger.training.original.alignment.fixed.PARENT == targeted.training.original.alignment.fixed.PARENT
    assert stronger.training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert targeted.training.BOUNDARIES == (0, 10, 25, 50, 100)
    assert stronger.POLICY == targeted.POLICY


def test_stronger_report_reuses_every_guard_component_and_case_without_predictions():
    assert stronger.paired_progress is previous.paired_progress
    assert 'len(current["cases"]) != 152' in stronger.ADAPTED_SOURCE
    assert 'len(current["decision"]["comparisons"]) != 22' in stronger.ADAPTED_SOURCE
    assert '"prediction_performed": False, "selection_performed": False' in stronger.ADAPTED_SOURCE
    assert 'scripts/single_model_registered_adaptation.py' in stronger.ADAPTED_SOURCE
