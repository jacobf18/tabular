"""Archive two additional completed legacy queues without touching selected checkpoints."""

import archive_single_model_legacy_snapshots as baseline
import archive_single_model_root_gradients as gradients
from single_model_registered_adaptation import install


EXTRA_NAMES = ("spectral_combination_20260913", "cell_query_20260909_checked")
EXTRA_POLICY = (baseline.POLICY.replace("Stop only at24 GiB", "Stop only at20 GiB")
                + " This additive capacity supplement owns only the completed spectral-combination and checked "
                  "cell-query queues after the original legacy and root-gradient archives finish. The cancelled "
                  "spectral-optimizer queue is excluded. No old source or archive registration is changed.")


def require_predecessors():
    hashes = {}
    for output in (baseline.OUTPUT, gradients.OUTPUT):
        gradients.relocation.terminal(output / "status.json")
        report = gradients.read(output / "independent_recheck.json")
        if not report["complete"] or not report["all_original_bytes_and_paths_preserved"]:
            raise ValueError("Require both completed lossless predecessor archives")
        if gradients.sha256(output / "config.json") != report["config_sha256"]:
            raise ValueError("Predecessor capacity registration changed")
        hashes.update({str(output / name): gradients.sha256(output / name)
                       for name in ("status.json", "independent_recheck.json", "config.json")})
    return hashes


install(baseline, globals(), {
    'OUTPUT = ROOT / "legacy_snapshot_capacity_v1"': 'OUTPUT = ROOT / "supplemental_snapshot_capacity_v1"',
    '        records, inputs, untouched = inventory()':
        '        predecessors = require_predecessors()\n'
        '        records, inputs, untouched = inventory()\n'
        '        inputs.update(predecessors)',
    'REPO / "tests/test_single_model_legacy_snapshot_archive.py"': 'REPO / "tests/test_single_model_root_capacity_supplement.py"',
    '        OUTPUT.mkdir()':
        '        sources.extend([Path(baseline.__file__).resolve(), Path(gradients.__file__).resolve(),\n'
        '                        Path(gradients.relocation.__file__).resolve()])\n'
        '        OUTPUT.mkdir()',
}, {"NAMES": EXTRA_NAMES, "TARGET_ROOT_FREE": 20 * 1024**3, "POLICY": EXTRA_POLICY,
    "WRITERS": baseline.WRITERS | {"run_spectral_combination_experiments.py", "run_cell_query_experiments.py"}})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            status("failed", error=repr(error))
        raise
