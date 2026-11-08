"""Read-only terminal review of the registered full-context native-coverage probe."""

import argparse
from pathlib import Path
import types

import recheck_single_model_standardized_teacher_huber_probe as blueprint
from single_model_registered_adaptation import replace_assignment, replace_once


SOURCE = Path(blueprint.__file__).read_text()
for old, new in {
    'standardized_teacher_huber_preflight_v1': 'balanced_native_coverage_preflight_v1',
    'preflight_single_model_standardized_teacher_huber': 'preflight_single_model_balanced_native_coverage',
    'audit_single_model_standardized_teacher_huber_probe.py': 'audit_single_model_balanced_native_coverage_probe.py',
    'test_single_model_standardized_teacher_huber_probe_recheck.py': 'test_single_model_balanced_native_coverage_probe_recheck.py',
    'Standardized teacher-Huber probe: terminal review': 'Balanced native-coverage probe: terminal review',
    'balanced mixed priors; standardized': 'balanced mixed priors, full-size native contexts; standardized',
}.items():
    if old not in SOURCE:
        raise ValueError(f'Missing explicit terminal-review adaptation: {old}')
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_assignment(SOURCE, 'CONFIG_SHA256', None)
SOURCE = replace_once(SOURCE,
    '    require(config[\'batch\'] == probe.batch()',
    '    require(config["native_coverage"] == probe.native.SPECIFICATION\n'
    '        and config["native_probe_shape"] == probe.NATIVE_SHAPE, "Keep the registered native probe")\n'
    '    require(config[\'batch\'] == probe.batch()')
SOURCE = replace_once(SOURCE, "if __name__ == '__main__':\n    main()", '')
exec(compile(SOURCE, __file__, 'exec'), globals())
_review = main


def review(config_sha256):
    if (not isinstance(config_sha256, str) or len(config_sha256) != 64
            or any(character not in '0123456789abcdef' for character in config_sha256)):
        raise ValueError('Require the exact original configuration SHA256')
    implementation = types.FunctionType(_review.__code__, {**globals(), 'CONFIG_SHA256': config_sha256},
        'review', _review.__defaults__)
    return implementation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-sha256', required=True)
    review(parser.parse_args().config_sha256)


if __name__ == '__main__':
    main()
