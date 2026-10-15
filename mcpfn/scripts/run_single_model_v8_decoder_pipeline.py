"""Qualify V17 using the complete original V8 guards, independent confirmation and frozen real campaigns."""

import run_single_model_targeted_decoder_pipeline as template
from single_model_registered_adaptation import install


POLICY_TEXT = ("Wait for all64 registered V17 attempts; independently audit all896 cached training tasks, all22 "
               "original V8 retention guards and every scheduled development prediction. The training input is "
               "V8 itself, so count its original reference once:1216 case/role grids, not a duplicated parent "
               "reference. Verify the final decoder-only optimizer/projection endpoint and frozen V8 backbone. "
               "Independently reconstruct/replay all before/candidate scores on fourteen tasks and CPU-audit "
               "those grids. Replay any sole selected checkpoint on152 development cases without reselection. "
               "Confirm once on284 reserved fresh cases against V8 with the unchanged gate. Independently audit "
               "all1136 confirmation grids before alias publication, every frozen real suite and the complete "
               "comparison review. No fallback, guard relaxation, training restart or inference ensemble.")


install(template, globals(), {
    'import audit_single_model_targeted_decoder as training_audit': 'import audit_single_model_v8_decoder as training_audit',
    'import train_single_matrixformer_targeted_decoder as training': 'import train_single_matrixformer_v8_decoder as training',
    'OUTPUT = ROOT / "targeted_decoder_v15_pipeline"': 'OUTPUT = ROOT / "v8_decoder_v17_pipeline"',
    'AUDIT = ROOT / "targeted_decoder_v15_audit"': 'AUDIT = ROOT / "v8_decoder_v17_audit"',
    'REPLAY = ROOT / "targeted_decoder_v15_endpoint_replay"': 'REPLAY = ROOT / "v8_decoder_v17_endpoint_replay"',
    'CONFIRMATION = ROOT / "targeted_decoder_v15_confirmation"': 'CONFIRMATION = ROOT / "v8_decoder_v17_confirmation"',
    'CAMPAIGN = ROOT / "targeted_decoder_v15_benchmarks"': 'CAMPAIGN = ROOT / "v8_decoder_v17_benchmarks"',
    'REVIEW = ROOT / "targeted_decoder_v15_review"': 'REVIEW = ROOT / "v8_decoder_v17_review"',
    'REPO / "tests/test_single_model_targeted_decoder_audit.py"': 'REPO / "tests/test_single_model_v8_decoder_training.py"',
    'REPO / "tests/test_single_model_targeted_decoder_pipeline.py"': 'REPO / "tests/test_single_model_v8_decoder_pipeline.py"',
    'audit["through_step"] != 100': 'audit["through_step"] != 64',
    'audit["attempted_batches"] != 100 or audit["training_episodes"] != 1400': 'audit["attempted_batches"] != 64 or audit["training_episodes"] != 896',
    'audit["prediction_grids"] != 1368': 'audit["prediction_grids"] != 1216',
    'Independent confirmation must initialize from V8, never the V14 training input': 'Independent confirmation must initialize from the original V8 baseline, never from the trained candidate',
}, {"POLICY": POLICY_TEXT})


if __name__ == "__main__":
    main()
