"""Losslessly reclaim the unused supplemental inventory for the next training continuation."""

import hashlib

import archive_single_model_supplemental_snapshots as baseline
from single_model_registered_adaptation import replace_once, replace_assignment


def require_unused_inventory():
    baseline.gradients.relocation.terminal(baseline.OUTPUT / "status.json")
    report = baseline.read(baseline.OUTPUT / "independent_recheck.json")
    if (not report["complete"] or report["archived_checkpoints"] != 0
            or not report["all_original_bytes_and_paths_preserved"]
            or baseline.sha256(baseline.OUTPUT / "config.json") != report["config_sha256"]):
        raise ValueError("Require the completed unused supplemental inventory; never rearchive its files")
    return {str(baseline.OUTPUT / name): baseline.sha256(baseline.OUTPUT / name)
            for name in ("config.json", "status.json", "independent_recheck.json")}


require_predecessors = baseline.require_predecessors
gradients = baseline.gradients
POLICY_TEXT = (baseline.POLICY + " This fresh additive run uses the same proven-idle candidate queues after the "
               "previous supplemental run archived zero files. It preserves all previous registrations and "
               "every original serialized checkpoint, while reclaiming root capacity for another continuation.")
ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, 'OUTPUT = ROOT / "supplemental_snapshot_capacity_v1"',
                             'OUTPUT = ROOT / "continuation_snapshot_capacity_v1"')
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, 'inputs.update(predecessors)',
                             'inputs.update(predecessors)\n        inputs.update(require_unused_inventory())')
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, 'REPO / "tests/test_single_model_root_capacity_supplement.py"',
                             'REPO / "tests/test_single_model_continuation_capacity.py"')
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, 'Path(baseline.__file__).resolve(), Path(gradients.__file__).resolve(),',
                             'Path(baseline.__file__).resolve(), Path(baseline.baseline.__file__).resolve(),\n'
                             '                        Path(gradients.__file__).resolve(),')
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            status("failed", error=repr(error))
        raise
