"""Replay paired training, then apply unchanged synthetic and frozen-real qualification."""

import run_single_model_targeted_decoder_pipeline as template
import run_single_model_replicated_decoder_pipeline as previous
from single_model_registered_adaptation import install, replace_once


POLICY_TEXT = (replace_once(previous.POLICY, "V18 attempts", "V19 attempts")
               + " Endpoint training checks use the prospectively registered paired-mean rule and retain every individual draw; "
                 "all development, confirmation, promotion and real-benchmark requirements remain unchanged.")


install(template, globals(), {
    'import audit_single_model_targeted_decoder as training_audit': 'import audit_single_model_paired_decoder as training_audit',
    'import train_single_matrixformer_targeted_decoder as training': 'import train_single_matrixformer_paired_decoder as training',
    'OUTPUT = ROOT / "targeted_decoder_v15_pipeline"': 'OUTPUT = ROOT / "paired_decoder_v19_pipeline"',
    'AUDIT = ROOT / "targeted_decoder_v15_audit"': 'AUDIT = ROOT / "paired_decoder_v19_audit"',
    'REPLAY = ROOT / "targeted_decoder_v15_endpoint_replay"': 'REPLAY = ROOT / "paired_decoder_v19_endpoint_replay"',
    'CONFIRMATION = ROOT / "targeted_decoder_v15_confirmation"': 'CONFIRMATION = ROOT / "paired_decoder_v19_confirmation"',
    'CAMPAIGN = ROOT / "targeted_decoder_v15_benchmarks"': 'CAMPAIGN = ROOT / "paired_decoder_v19_benchmarks"',
    'REVIEW = ROOT / "targeted_decoder_v15_review"': 'REVIEW = ROOT / "paired_decoder_v19_review"',
    'REPO / "tests/test_single_model_targeted_decoder_audit.py"': 'REPO / "tests/test_single_model_paired_decoder_training.py"',
    'REPO / "tests/test_single_model_targeted_decoder_pipeline.py"': 'REPO / "tests/test_single_model_paired_decoder_pipeline.py"',
    'audit["through_step"] != 100': 'audit["through_step"] != 64',
    'audit["attempted_batches"] != 100 or audit["training_episodes"] != 1400': 'audit["attempted_batches"] != 64 or audit["training_episodes"] != 1792',
    'audit["prediction_grids"] != 1368': 'audit["prediction_grids"] != 1216',
    'expected_count = 14 if payload["record"]["point_after"] is None else 28':
        'expected_count = 28 if payload["record"]["point_after"] is None else 56',
    'Independent confirmation must initialize from V8, never the V14 training input':
        'Independent confirmation must initialize from original V8, never the trained candidate',
    'displacement_decision(products, norm, record["adam_norm"])':
        'training.paired.paired_displacement(products, norm, record["adam_norm"], record["point_before"], record["composite_tasks"])',
    'finite = training.original.guarded.metric_decision(measured["before"], measured["candidate"])':
        'finite = training.paired.metric_decision(measured["before"], measured["candidate"])',
    'compare_nested(training.original.guarded.metric_decision(measured["before"], measured["candidate"]),':
        'compare_nested(training.paired.metric_decision(measured["before"], measured["candidate"]),',
}, {"POLICY": POLICY_TEXT})


if __name__ == "__main__":
    main()
