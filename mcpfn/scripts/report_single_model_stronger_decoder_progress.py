"""Decompose audited V16 development scores without new prediction or selection."""

import report_single_model_targeted_decoder_progress as template
from single_model_registered_adaptation import install


install(template, globals(), {
    'import train_single_matrixformer_targeted_decoder as training': 'import train_single_matrixformer_stronger_decoder as training',
    'Require a completed V15 audit': 'Require a completed V16 audit',
    'sources = [Path(__file__).resolve(),':
        'sources = [Path(__file__).resolve(), REPO / "scripts/report_single_model_targeted_decoder_progress.py", REPO / "scripts/single_model_registered_adaptation.py",',
    'REPO / "tests/test_single_model_targeted_decoder_progress.py"': 'REPO / "tests/test_single_model_stronger_decoder_progress.py"',
}, {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.audit.resolve(), arguments.output.resolve())
