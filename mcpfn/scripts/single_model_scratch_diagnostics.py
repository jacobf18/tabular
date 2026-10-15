"""Prospective scratch placement for new synthetic diagnostics, never model checkpoints."""

from pathlib import Path
import shutil

import relocate_single_model_idle_diagnostics_to_scratch as storage
from relocate_single_model_gradient_archive import sync_directory


RESERVES = {"checkpoint": 18 * 1024**3, "root": 6 * 1024**3, "scratch": 50 * 1024**3}
WRITE_BUDGETS = {stage: {volume: amount * 1024**2 for volume, amount in amounts.items()}
                for stage, amounts in {
    "training": {"checkpoint": 1536, "root": 16, "scratch": 16384},
    "replay": {"checkpoint": 16, "root": 16, "scratch": 768},
    "audit-replay": {"checkpoint": 16, "root": 16, "scratch": 32},
    "confirm": {"checkpoint": 16, "root": 16, "scratch": 2560},
    "audit-confirmation": {"checkpoint": 400, "root": 16, "scratch": 32},
    "benchmarks": {"checkpoint": 512, "root": 16, "scratch": 0},
    "review": {"checkpoint": 16, "root": 16, "scratch": 0},
}.items()}
POLICY = ("Prospectively place only a new trial's synthetic development, endpoint-replay and confirmation "
    "directories on the existing UUID-bound machine-local scratch filesystem. Establish empty target directories "
    "and canonical symlinks before the relevant stage starts; never relocate or overwrite any existing file, "
    "directory, active output or earlier evidence. Keep the ordinary selected/latest checkpoints and any published "
    "best alias, training configuration/status/audits and complete frozen-real campaign on checkpoint storage. "
    "The qualification gate and every scientific metric, case, teacher, selection and audit remain unchanged. "
    "Account for each stage's actual writing filesystem, including durable checkpoint publication during the "
    "confirmation audit, and preserve18 GiB checkpoint,6 GiB root and50 GiB scratch beyond declared write budgets. "
    "Scratch is machine-local, not provider-persistent storage or an external backup. This helper is preparation "
    "only: it performs no model training, evaluation, nomination, scientific registration or seed consumption.")


def layout(run, scratch, boundaries):
    run, scratch = Path(run), Path(scratch)
    boundaries = list(boundaries)
    if (not run.is_absolute() or not scratch.is_absolute() or run == scratch
            or run.is_relative_to(scratch) or scratch.is_relative_to(run)
            or ".." in run.parts or ".." in scratch.parts
            or not boundaries or boundaries[0] != 0
            or any(type(step) is not int or step < 0 for step in boundaries)
            or sorted(set(boundaries)) != boundaries):
        raise ValueError("Require distinct absolute storage roots and ordered unique development boundaries")
    directories = {"training": [{"canonical": str(run / f"validation_{step:03d}"),
                                 "target": str(scratch / "validation" / f"validation_{step:03d}")}
                                for step in boundaries],
                   "qualification": [{"canonical": str(run.with_name(run.name + suffix)),
                                      "target": str(scratch / "qualification" / name)}
                                     for suffix, name in (("_endpoint_replay", "endpoint_replay"),
                                                          ("_confirmation", "confirmation"))]}
    return {"policy": POLICY, "run": str(run), "scratch": str(scratch), "boundaries": boundaries,
            "directories": directories, "scratch_is_provider_persistent": False,
            "minimum_free_bytes": dict(RESERVES),
            "stage_write_budget_bytes": {stage: dict(amounts) for stage, amounts in WRITE_BUDGETS.items()}}


def require_plan(plan, phase):
    if phase not in {"training", "qualification"}:
        raise ValueError("Only the two explicitly declared diagnostic phases may create directories")
    if plan != layout(plan["run"], plan["scratch"], plan["boundaries"]):
        raise ValueError("Preserve the exact prospective layout, scope, reserves and write budgets")


def require_filesystems(plan):
    storage.require_mount()
    run, scratch = Path(plan["run"]), Path(plan["scratch"])
    if (run.parent != storage.ROOT or run.is_symlink() or not run.is_dir()
            or run.stat().st_dev != storage.ROOT.stat().st_dev
            or not scratch.is_relative_to(storage.SCRATCH) or scratch.is_symlink() or not scratch.is_dir()
            or scratch.resolve() != scratch or scratch.stat().st_dev != storage.SCRATCH.stat().st_dev
            or scratch.stat().st_dev == run.stat().st_dev):
        raise ValueError("Keep ordinary checkpoints on their original volume and diagnostics on UUID-bound scratch")


def prepare(plan, phase):
    require_plan(plan, phase)
    require_filesystems(plan)
    for record in plan["directories"][phase]:
        for path in (Path(record["canonical"]), Path(record["target"])):
            if path.exists() or path.is_symlink():
                raise FileExistsError("Only new empty diagnostic directories may be prepared; never retry or relocate")
    for record in plan["directories"][phase]:
        canonical, target = Path(record["canonical"]), Path(record["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve().is_relative_to(Path(plan["scratch"])):
            raise ValueError("Never escape the declared scratch namespace through a parent symlink")
        target.mkdir()
        sync_directory(target.parent)
        canonical.symlink_to(target, target_is_directory=True)
        sync_directory(canonical.parent)
    verify(plan, phase)


def verify(plan, phase):
    require_plan(plan, phase)
    require_filesystems(plan)
    for record in plan["directories"][phase]:
        canonical, target = Path(record["canonical"]), Path(record["target"])
        if (not canonical.is_symlink() or canonical.readlink() != target or canonical.resolve() != target
                or not target.is_dir() or target.is_symlink() or target.resolve() != target):
            raise ValueError("Preserve each exact canonical diagnostic link and its declared physical target")


def require_empty_validation(plan, directory):
    verify(plan, "training")
    directory = Path(directory)
    if str(directory) not in {record["canonical"] for record in plan["directories"]["training"]}:
        raise ValueError("Only a declared development boundary may use a prepared diagnostic directory")
    if any(directory.iterdir()):
        raise FileExistsError("Never reevaluate or append predictions to a started development boundary")


def require_capacity(stage, volumes=None):
    if stage not in WRITE_BUDGETS:
        raise ValueError("Every stage needs a prospectively declared per-volume budget")
    if volumes is None:
        volumes = {"checkpoint": storage.ROOT, "root": Path("/"), "scratch": storage.SCRATCH}
    if set(volumes) != set(RESERVES):
        raise ValueError("Never omit any storage volume from admission checks")
    for volume, path in volumes.items():
        free, budget = shutil.disk_usage(path).free, WRITE_BUDGETS[stage][volume]
        if free - budget < RESERVES[volume]:
            raise OSError(f"Preserve {RESERVES[volume]} bytes on {volume} before {stage}: free={free}, budget={budget}")
