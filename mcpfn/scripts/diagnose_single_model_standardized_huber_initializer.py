"""Attribute V54's closed failed confirmation using its unchanged V29 initializer."""

from pathlib import Path

import diagnose_single_model_native_column_initializer as diagnostic_blueprint
from single_model_registered_adaptation import replace_once


DIAGNOSTIC_SOURCE = Path(diagnostic_blueprint.__file__).read_text()
for old, new in {
    "run_single_model_native_column_mean_pipeline": "run_single_model_standardized_huber_pipeline",
    "native_column_mean_v48": "standardized_teacher_huber_v54",
    "test_single_model_native_column_initializer_diagnosis.py": "test_single_model_standardized_huber_initializer_diagnosis.py",
    "V48": "V54",
    "step10": "step50",
    'replay_gate["selected_step"] != 10': 'replay_gate["selected_step"] != 50',
    'gate["selected_step"] != 10': 'gate["selected_step"] != 50',
    'training_audit["accepted_updates"] != 56': 'training_audit["accepted_updates"] != 57',
    "four failed": "seven failed",
    "7e8b6d6c6b29bb339744ecd62f9d080a2c70de7e0c82c0ca7fa181bcc8f88b54":
        "3693644e3cba0407cc69cf52581f671f5b9534a6a85583bedf5fd9789868fcd1",
    "8c7742bfadb54a27d5fe1681a125a1faeb2dec482a2f49e9e60714d0b2a47b7c":
        "d327dee4de4a35b7d6f609509754724bf685b99ff9693a4e33f51c6e5cbcfd9a",
    "9b1c91649d171d1a79c38c49d7b4e416db8a43d5a88fb9128e0938a578fafef5":
        "afd123f3ecff78906b74f11a16ad008f04f472895142634ecf2ae6eaec6cddc5",
    "a0054e6290eb152933a90b05f918df0ba1ee6b64338b1fd9cc461413ca7933e8":
        "77e3e3015fd8c0465818ec30a755f72e2c9741bf891d13f568201cd5f645acfd",
    "820dc1b436bba15ad7755e42c2747e4d984faf2716094db4948d29a6e08e1be2":
        "2df02bd91b9a34e8e998fe668245aaa141bb4220e762b0caab1003f1ad0915ac",
    "d62175eaacb476ef3f9f31a9e04741c68743b5f33c4c3f862ebd71408adf9dc1":
        "ec03449991a3f02f96d6c158f74570ffbf355714b7b6c174e688009490a0a67d",
    "4aa6289010d3dd0e8c367554b1292dfe709bf6c054ac909e14b8ca6a1fd9d699":
        "c55dc61520f69c8c1bbbfe9390f31305fab11c379b66382438aee5d980e022c5",
}.items():
    if old not in DIAGNOSTIC_SOURCE:
        raise ValueError(f"Missing explicit V54 initializer diagnosis adaptation: {old}")
    DIAGNOSTIC_SOURCE = DIAGNOSTIC_SOURCE.replace(old, new)
DIAGNOSTIC_SOURCE = replace_once(DIAGNOSTIC_SOURCE,
    '    sources = [Path(__file__).resolve(), Path(blueprint.__file__).resolve(), Path(predecessor.__file__).resolve(),',
    '    sources = [Path(__file__).resolve(), Path(diagnostic_blueprint.__file__).resolve(),\n'
    '        REPO / "tests/test_single_model_native_column_initializer_diagnosis.py",\n'
    '        Path(blueprint.__file__).resolve(), Path(predecessor.__file__).resolve(),')
exec(compile(DIAGNOSTIC_SOURCE, __file__, "exec"), globals())
