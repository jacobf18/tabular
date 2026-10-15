"""Relocate two terminal images after supplemental recovery without reducing any write floor."""

import archive_single_model_supplemental_snapshots as supplement
import relocate_single_model_mounted_archives as baseline
from single_model_registered_adaptation import install


PAIR_NAMES = baseline.NAMES[:2]
PAIR_POLICY = ("Storage only: use the original fully verified copy/fsync/hash/remount procedure for the finite-endpoint "
               "and decoder-checkpoint images, totalling13.5816 GiB, after the original capacity inventory and two "
               "additive lossless supplements finish. Leave the paired-endpoint image in place. Require measured "
               "capacity for both copies plus6 GiB root reserve and at least31 GiB checkpoint free after relocation. "
               "The training registration's31 GiB minimum and all18 GiB scientific write floors remain unchanged. "
               "No source bytes, mount units, original metadata, checkpoints or scientific protocols change.")


def supplemental_capacity_ready():
    hashes = supplement.require_predecessors()
    output = supplement.OUTPUT
    baseline.terminal(output / "status.json")
    report = baseline.read(output / "independent_recheck.json")
    if (not report["complete"] or not report["all_original_bytes_and_paths_preserved"]
            or (not report["capacity_target_reached"] and report["archived_checkpoints"] != report["candidate_checkpoints"])):
        raise ValueError("Supplement must reach capacity or finish its entire fixed inventory")
    if baseline.sha256(output / "config.json") != report["config_sha256"]:
        raise ValueError("Supplement registration changed")
    hashes.update({str(output / name): baseline.sha256(output / name)
                   for name in ("status.json", "independent_recheck.json", "config.json")})
    return hashes


install(baseline, globals(), {
    'OUTPUT = REPO / "runs/single_model_goal_20261005/mounted_archive_relocation_v1"':
        'OUTPUT = REPO / "runs/single_model_goal_20261005/mounted_archive_relocation_v2"',
    'inputs = capacity_ready()': 'inputs = supplemental_capacity_ready()',
    '    with ExitStack() as locks:':
        '    if shutil.disk_usage(ROOT).free + sum(sizes) < 31 * 1024**3:\n'
        '        raise OSError("Never lower the full training registration capacity requirement")\n'
        '    with ExitStack() as locks:',
    'REPO / "tests/test_single_model_mounted_archive_relocation.py"': 'REPO / "tests/test_single_model_root_capacity_supplement.py"',
    '        OUTPUT.mkdir()':
        '        sources.extend([Path(baseline.__file__).resolve(), Path(supplement.__file__).resolve(),\n'
        '                        Path(supplement.gradients.__file__).resolve(), Path(supplement.baseline.__file__).resolve(),\n'
        '                        REPO / "scripts/single_model_registered_adaptation.py"])\n'
        '        OUTPUT.mkdir()',
    '        if shutil.disk_usage(OUTPUT).free < RESERVE:':
        '        if shutil.disk_usage(ROOT).free < 31 * 1024**3:\n'
        '            raise OSError("Measured checkpoint free space must meet the unchanged full-trial requirement")\n'
        '        if shutil.disk_usage(OUTPUT).free < RESERVE:',
}, {"NAMES": PAIR_NAMES, "POLICY": PAIR_POLICY})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            status("failed", error=repr(error))
        raise
