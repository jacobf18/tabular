import report_single_model_decoder_progress as previous
import report_single_model_targeted_decoder_progress as targeted


def test_targeted_report_keeps_v8_baseline_but_uses_v14_training_input():
    assert targeted.training.PARENT == targeted.training.ROOT / "decoder_repair_v14/latest_checkpoint.pth"
    assert targeted.training.INITIAL_RUN == targeted.training.ROOT / "decoder_repair_v14"
    assert targeted.training.OUTPUT == targeted.training.ROOT / "targeted_decoder_v15"
    assert targeted.training.original.alignment.fixed.PARENT == previous.training.original.alignment.fixed.PARENT
    assert targeted.training.BOUNDARIES == (0, 10, 25, 50, 100)
    assert "V8 as the sole retention baseline" in targeted.POLICY
    assert "V14-inherited" in targeted.POLICY


def test_targeted_report_reuses_the_tested_complete_paired_decomposition():
    assert targeted.paired_progress is previous.paired_progress
