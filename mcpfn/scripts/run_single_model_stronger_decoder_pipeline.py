"""Qualify the stronger-rate trial with all original V8 gates and frozen real campaigns."""

import run_single_model_targeted_decoder_pipeline as template
from single_model_registered_adaptation import install


install(template, globals(), {
    'import audit_single_model_targeted_decoder as training_audit': 'import audit_single_model_stronger_decoder as training_audit',
    'import train_single_matrixformer_targeted_decoder as training': 'import train_single_matrixformer_stronger_decoder as training',
    'OUTPUT = ROOT / "targeted_decoder_v15_pipeline"': 'OUTPUT = ROOT / "stronger_decoder_v16_pipeline"',
    'AUDIT = ROOT / "targeted_decoder_v15_audit"': 'AUDIT = ROOT / "stronger_decoder_v16_audit"',
    'REPLAY = ROOT / "targeted_decoder_v15_endpoint_replay"': 'REPLAY = ROOT / "stronger_decoder_v16_endpoint_replay"',
    'CONFIRMATION = ROOT / "targeted_decoder_v15_confirmation"': 'CONFIRMATION = ROOT / "stronger_decoder_v16_confirmation"',
    'CAMPAIGN = ROOT / "targeted_decoder_v15_benchmarks"': 'CAMPAIGN = ROOT / "stronger_decoder_v16_benchmarks"',
    'REVIEW = ROOT / "targeted_decoder_v15_review"': 'REVIEW = ROOT / "stronger_decoder_v16_review"',
    'REPO / "tests/test_single_model_targeted_decoder_audit.py"': 'REPO / "tests/test_single_model_stronger_decoder.py"',
    'REPO / "tests/test_single_model_targeted_decoder_pipeline.py"': 'REPO / "tests/test_single_model_stronger_decoder_pipeline.py"',
    'audit["through_step"] != 100': 'audit["through_step"] != 64',
    'audit["attempted_batches"] != 100 or audit["training_episodes"] != 1400': 'audit["attempted_batches"] != 64 or audit["training_episodes"] != 896',
}, {"POLICY": template.POLICY.replace("one hundred", "sixty-four")})


if __name__ == "__main__":
    main()
