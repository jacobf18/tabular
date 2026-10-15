"""Decompose audited V17 scores against its identical V8 training and retention baseline."""

import report_single_model_guard_contributions as template
from single_model_registered_adaptation import install


install(template, globals(), {
    'import train_single_matrixformer_finite_guarded as training':
        'import train_single_matrixformer_v8_decoder as training',
    'if not audit["complete"] or audit["research_selection_tolerance"] != 0:':
        'if (not audit["complete"] or not audit["decoder_only_training"] or audit["research_selection_tolerance"] != 0\n'
        '            or audit["training_initialization"] != str(training.PARENT)\n'
        '            or audit["retention_baseline"] != str(training.PARENT)\n'
        '            or audit["through_step"] not in training.BOUNDARIES[1:]):',
    'initial_path = training.alignment.fixed.PRODUCER / "validation_0.json"':
        'initial_path = training.original.alignment.fixed.PRODUCER / "validation_0.json"',
    'source_paths = [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "tests/test_single_model_guard_contributions.py"]':
        'source_paths = [Path(__file__).resolve(), Path(template.__file__).resolve(),\n'
        '                    Path(__file__).resolve().with_name("single_model_registered_adaptation.py"),\n'
        '                    Path(__file__).resolve().parents[1] / "tests/test_single_model_guard_contributions.py",\n'
        '                    Path(__file__).resolve().parents[1] / "tests/test_single_model_v8_decoder_progress.py"]',
    '"config_sha256": sha256(output / "config.json")})':
        '"prediction_performed": False, "selection_performed": False,\n'
        '                                        "config_sha256": sha256(output / "config.json")})',
    '# Audited synthetic retention contributions': '# Audited V8-start decoder retention contributions',
}, {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report(arguments.audit.resolve(), arguments.output.resolve())
