"""Attribute V55's closed failed confirmation using its unchanged V29 initializer."""

import diagnose_single_model_standardized_huber_initializer as diagnostic_blueprint
from single_model_registered_adaptation import replace_once


DIAGNOSTIC_SOURCE = diagnostic_blueprint.DIAGNOSTIC_SOURCE
for old, new in {
    "run_single_model_standardized_huber_pipeline": "run_single_model_balanced_native_quarter_pipeline",
    "standardized_teacher_huber_v54": "balanced_native_quarter_v55",
    "test_single_model_standardized_huber_initializer_diagnosis.py": "test_single_model_balanced_native_quarter_initializer_diagnosis.py",
    "V54": "V55",
    "step50": "step64",
    'replay_gate["selected_step"] != 50': 'replay_gate["selected_step"] != 64',
    'gate["selected_step"] != 50': 'gate["selected_step"] != 64',
    'training_audit["accepted_updates"] != 57': 'training_audit["accepted_updates"] != 40',
    "seven failed": "five failed",
    "3693644e3cba0407cc69cf52581f671f5b9534a6a85583bedf5fd9789868fcd1":
        "7282100044e413e12240c58ae48298625e3988f810d47c806154a1f17d954e51",
    "d327dee4de4a35b7d6f609509754724bf685b99ff9693a4e33f51c6e5cbcfd9a":
        "ee6632ca8febd150295f73c529f121bb36a64cd86eb5c454437489ea2c3256c8",
    "afd123f3ecff78906b74f11a16ad008f04f472895142634ecf2ae6eaec6cddc5":
        "acd1c19314ca6124d6e657561a5ac6cfedd6782ff24efd7d77c33c552e23e8bd",
    "77e3e3015fd8c0465818ec30a755f72e2c9741bf891d13f568201cd5f645acfd":
        "7ae73bf88f8f7ebe18278297b6e3d7f4f5c620b9ba40cc726758e8e856c3a941",
    "2df02bd91b9a34e8e998fe668245aaa141bb4220e762b0caab1003f1ad0915ac":
        "7a43ce1f8c3c09ba90786b541f1f0c54fca1468a5716f81dd122f0d550e9c3b6",
    "ec03449991a3f02f96d6c158f74570ffbf355714b7b6c174e688009490a0a67d":
        "b84427602511bfc0dc24d462af92e7783e33bae6ff2109015fe9d8c8c1b9e1d1",
    "c55dc61520f69c8c1bbbfe9390f31305fab11c379b66382438aee5d980e022c5":
        "50c87410944013c63e6bda5558388ef45047a7400edfa1b3c89167652b69e102",
}.items():
    if old not in DIAGNOSTIC_SOURCE:
        raise ValueError(f"Missing explicit V55 initializer diagnosis adaptation: {old}")
    DIAGNOSTIC_SOURCE = DIAGNOSTIC_SOURCE.replace(old, new)
DIAGNOSTIC_SOURCE = replace_once(DIAGNOSTIC_SOURCE,
    '        REPO / "tests/test_single_model_native_column_initializer_diagnosis.py",',
    '        REPO / "scripts/diagnose_single_model_native_column_initializer.py",\n'
    '        REPO / "tests/test_single_model_native_column_initializer_diagnosis.py",')
exec(compile(DIAGNOSTIC_SOURCE, __file__, "exec"), globals())
