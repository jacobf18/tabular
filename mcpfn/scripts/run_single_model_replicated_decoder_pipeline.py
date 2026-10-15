"""Qualify the doubled-draw V18 trial before any unchanged frozen real benchmark campaign."""

import run_single_model_targeted_decoder_pipeline as template
import run_single_model_v8_decoder_pipeline as previous
from single_model_registered_adaptation import install, replace_once


POLICY_TEXT = replace_once(previous.POLICY, "V17 attempts", "V18 attempts")
POLICY_TEXT = replace_once(POLICY_TEXT, "896 cached training tasks", "1792 cached training tasks")
POLICY_TEXT = replace_once(POLICY_TEXT, "on fourteen tasks", "on twenty-eight tasks")


install(template, globals(), {
    'import audit_single_model_targeted_decoder as training_audit': 'import audit_single_model_replicated_decoder as training_audit',
    'import train_single_matrixformer_targeted_decoder as training': 'import train_single_matrixformer_replicated_decoder as training',
    'OUTPUT = ROOT / "targeted_decoder_v15_pipeline"': 'OUTPUT = ROOT / "replicated_decoder_v18_pipeline"',
    'AUDIT = ROOT / "targeted_decoder_v15_audit"': 'AUDIT = ROOT / "replicated_decoder_v18_audit"',
    'REPLAY = ROOT / "targeted_decoder_v15_endpoint_replay"': 'REPLAY = ROOT / "replicated_decoder_v18_endpoint_replay"',
    'CONFIRMATION = ROOT / "targeted_decoder_v15_confirmation"': 'CONFIRMATION = ROOT / "replicated_decoder_v18_confirmation"',
    'CAMPAIGN = ROOT / "targeted_decoder_v15_benchmarks"': 'CAMPAIGN = ROOT / "replicated_decoder_v18_benchmarks"',
    'REVIEW = ROOT / "targeted_decoder_v15_review"': 'REVIEW = ROOT / "replicated_decoder_v18_review"',
    'REPO / "tests/test_single_model_targeted_decoder_audit.py"': 'REPO / "tests/test_single_model_replicated_decoder_training.py"',
    'REPO / "tests/test_single_model_targeted_decoder_pipeline.py"': 'REPO / "tests/test_single_model_replicated_decoder_pipeline.py"',
    'audit["through_step"] != 100': 'audit["through_step"] != 64',
    'audit["attempted_batches"] != 100 or audit["training_episodes"] != 1400': 'audit["attempted_batches"] != 64 or audit["training_episodes"] != 1792',
    'audit["prediction_grids"] != 1368': 'audit["prediction_grids"] != 1216',
    'expected_count = 14 if payload["record"]["point_after"] is None else 28':
        'expected_count = 28 if payload["record"]["point_after"] is None else 56',
    'Independent confirmation must initialize from V8, never the V14 training input':
        'Independent confirmation must initialize from the original V8 baseline, never the trained candidate',
}, {"POLICY": POLICY_TEXT})


if __name__ == "__main__":
    main()
