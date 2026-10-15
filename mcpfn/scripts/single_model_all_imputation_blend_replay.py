"""Independently reconstruct observed-only blend targets for all imputation tasks."""

import inspect
import math

import single_model_legacy_blend_replay as replay_blueprint
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = (
    "Replay both frozen original teacher components for every synthetic imputation task, including "
    "all legacy and structured families. Refit observed-only convex weights and exactly reconstruct "
    "every cached FP32 target cell without hidden truth. Preserve the original rating-specialist "
    "callback, identity checks, single-model inference and all qualification guards. This standalone "
    "helper registers no scientific experiment, seeds, resource allowance or checkpoint selection.")

SOURCE = inspect.getsource(replay_blueprint)
SOURCE = replace_once(SOURCE,
    '"""Independently verify legacy blend components, observed weights and full targets."""',
    '"""Independently verify all-imputation blend components, observed weights and full targets."""')
SOURCE = replace_once(SOURCE,
    "import single_model_legacy_blend_teacher as teacher_policy",
    "import single_model_all_imputation_blend_teacher as teacher_policy")
SOURCE = replace_once(SOURCE,
    'return "general" if "structured" in specification else "observed_blend"',
    'return "observed_blend"')
SOURCE = replace_assignment(SOURCE, "POLICY", POLICY_TEXT)
SOURCE = replace_once(SOURCE, 'if len(tasks) != 2:', 'if len(tasks) != 26 or len(batch["tasks"]) != 28:')
SOURCE = replace_once(SOURCE, '"Keep both original legacy draws in the28-task probe"',
    '"Keep all26 imputation tasks in the original28-task probe"')
SOURCE = replace_once(SOURCE, 'shape = task["specification"]["shape"]',
    'specification = task["specification"]\n'
    '        shape = specification.get("structured", specification)["shape"]')
SOURCE = replace_once(SOURCE,
    '    if arrays + 1024**2 > ALLOWANCE:\n'
    '        raise ValueError("The complete raw components and serialization must fit the registered extra allowance")\n'
    '    return {"legacy_tasks": 2, "component_grids": 4, "array_bytes": arrays,\n'
    '        "serialization_allowance_bytes": 1024**2, "reserved_checkpoint_bytes": ALLOWANCE}',
    '    return {"blend_tasks": 26, "component_grids": 52, "array_bytes": arrays,\n'
    '        "serialization_allowance_bytes": 1024**2, "reserved_checkpoint_bytes": arrays + 1024**2}')
SOURCE = replace_once(SOURCE, 'ALLOWANCE = 4 * 1024**2\n', '')
SOURCE = SOURCE.replace("independently replayed legacy component grid", "independently replayed imputation component grid")
exec(compile(SOURCE, __file__, "exec"), globals())
