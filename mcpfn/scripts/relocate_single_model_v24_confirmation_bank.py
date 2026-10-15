"""Preserve the completed, consumed V24 confirmation bank without moving model checkpoints."""

import relocate_single_model_v23_confirmation_bank as template
from single_model_registered_adaptation import install


POLICY_TEXT = template.POLICY.replace("failed V23", "failed V24").replace(
    "stopped V23 owners", "stopped V24 owners").replace(
    "while unrelated V24 training continues untouched", "without starting or interrupting any scientific run")

install(template, globals(), {
    'OWNER = ROOT / "compact_continuation_v23_confirmation"': 'OWNER = ROOT / "top_block_v24_confirmation"',
    'OUTPUT = ROOT / "v23_failed_confirmation_bank_relocation_v1"':
        'OUTPUT = ROOT / "v24_failed_confirmation_bank_relocation_v1"',
    'single_model_goal_20261005_owned/v23_failed_confirmation_bank_v1/synthetic_bank.pt':
        'single_model_goal_20261005_owned/v24_failed_confirmation_bank_v1/synthetic_bank.pt',
    'ROOT / "compact_continuation_v23_pipeline"': 'ROOT / "top_block_v24_pipeline"',
    'Require both completed, absent original V23 owners': 'Require both completed, absent original V24 owners',
    'Require the unchanged independently audited failed V23 confirmation':
        'Require the unchanged independently audited failed V24 confirmation',
    'sources = [Path(__file__).resolve(), Path(bounded.__file__).resolve(), Path(scratch.__file__).resolve(),':
        'sources = [Path(__file__).resolve(), Path(template.__file__).resolve(),\n'
        '               REPO / "scripts/single_model_registered_adaptation.py",\n'
        '               Path(bounded.__file__).resolve(), Path(scratch.__file__).resolve(),',
    'tests/test_single_model_v23_bank_relocation.py': 'tests/test_single_model_v24_bank_relocation.py',
}, {"POLICY": POLICY_TEXT})


if __name__ == "__main__":
    main()
