"""Attribute V49's failed confirmation using the unchanged actual V29 initializer."""

from pathlib import Path

import diagnose_single_model_native_column_initializer as diagnostic_blueprint
from single_model_registered_adaptation import replace_once


DIAGNOSTIC_SOURCE = Path(diagnostic_blueprint.__file__).read_text()
for old, new in {
    "run_single_model_native_column_mean_pipeline": "run_single_model_parent_structured_pipeline",
    "native_column_mean_v48": "parent_structured_v49",
    "test_single_model_native_column_initializer_diagnosis.py": "test_single_model_parent_structured_initializer_diagnosis.py",
    "V48": "V49",
    "step10": "step25",
    'replay_gate["selected_step"] != 10': 'replay_gate["selected_step"] != 25',
    'gate["selected_step"] != 10': 'gate["selected_step"] != 25',
    'training_audit["accepted_updates"] != 56': 'training_audit["accepted_updates"] != 62',
    "four failed": "two failed",
    "7e8b6d6c6b29bb339744ecd62f9d080a2c70de7e0c82c0ca7fa181bcc8f88b54":
        "6b1942ab8e84a47bea23c52fac515c4b827e5678ebe1a11566e980b7af9dc990",
    "8c7742bfadb54a27d5fe1681a125a1faeb2dec482a2f49e9e60714d0b2a47b7c":
        "6d3ab8f6b9562d8cdf7e93d41c71d804f277ffa9e4c7adc1ce1eeb601ae595e6",
    "9b1c91649d171d1a79c38c49d7b4e416db8a43d5a88fb9128e0938a578fafef5":
        "fb5ee8dbe214285ef4bdd09ddd4ca7a8491896bad89bf19376e4eb47e87cff44",
    "a0054e6290eb152933a90b05f918df0ba1ee6b64338b1fd9cc461413ca7933e8":
        "d00bf590be2701eb6d40da3ed9ca7397120b0188e2dec9012b5fcdd9b3ba2ea3",
    "820dc1b436bba15ad7755e42c2747e4d984faf2716094db4948d29a6e08e1be2":
        "ea9c12c070c2f6572b79995a8dceebee2d4969403658a5d41b6e10ce5b2d69a1",
    "d62175eaacb476ef3f9f31a9e04741c68743b5f33c4c3f862ebd71408adf9dc1":
        "54a7d9733415f14cbbd99d2e934f9789208de76686b2ec604c8962f98518938a",
    "4aa6289010d3dd0e8c367554b1292dfe709bf6c054ac909e14b8ca6a1fd9d699":
        "a9aeb90bdba783f843d9a3703b73ec933800ba9644c252b338c23eeb550d4c4b",
}.items():
    if old not in DIAGNOSTIC_SOURCE:
        raise ValueError(f"Missing explicit V49 initializer diagnosis adaptation: {old}")
    DIAGNOSTIC_SOURCE = DIAGNOSTIC_SOURCE.replace(old, new)
DIAGNOSTIC_SOURCE = replace_once(DIAGNOSTIC_SOURCE,
    '    sources = [Path(__file__).resolve(), Path(blueprint.__file__).resolve(), Path(predecessor.__file__).resolve(),',
    '    sources = [Path(__file__).resolve(), Path(diagnostic_blueprint.__file__).resolve(),\n'
    '        REPO / "tests/test_single_model_native_column_initializer_diagnosis.py",\n'
    '        Path(blueprint.__file__).resolve(), Path(predecessor.__file__).resolve(),')
exec(compile(DIAGNOSTIC_SOURCE, __file__, "exec"), globals())
