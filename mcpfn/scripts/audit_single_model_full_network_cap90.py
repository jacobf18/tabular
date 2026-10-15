"""Audit V35's complete fresh synthetic trajectory with the original full-network checks."""

import inspect

import audit_single_model_full_network as predecessor


SOURCE = inspect.getsource(predecessor)
for old, new in {
    "import train_single_matrixformer_full_network as training": "import train_single_matrixformer_full_network_cap90 as training",
    "full_network_v34_audit": "full_network_cap90_v35_audit",
    "single_model_full_network_audit_replay": "single_model_full_network_cap90_audit_replay",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V35 audit adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = SOURCE.replace("V34", "V35")
exec(compile(SOURCE, __file__, "exec"), globals())
