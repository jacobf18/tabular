"""Independently audit the complete V36 full-network synthetic trajectory."""

import audit_single_model_full_network_cap90 as predecessor


SOURCE = predecessor.SOURCE
for old, new in {
    "import train_single_matrixformer_full_network_cap90 as training": "import train_single_matrixformer_full_network_cap90_v2 as training",
    "full_network_cap90_v35_audit": "full_network_cap90_v36_audit",
    "single_model_full_network_cap90_audit_replay": "single_model_full_network_cap90_v2_audit_replay",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V36 audit adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = SOURCE.replace("V35", "V36")
exec(compile(SOURCE, __file__, "exec"), globals())
