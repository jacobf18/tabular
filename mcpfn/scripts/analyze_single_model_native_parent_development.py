"""Compare every published V50/V51 development case without changing selection."""

from pathlib import Path

import analyze_single_model_development_teacher_trajectory as blueprint
from single_model_registered_adaptation import replace_assignment, replace_once


SOURCE = Path(blueprint.__file__).read_text()
for old, new in {
    "import analyze_single_model_teacher_alignment as metrics": "import single_model_native_parent_alignment as metrics",
    "all_imputation_blend_v45": "dual_parent_v50",
    "stronger_mean_distillation_v46": "native_parent_v51",
    "v46_full_independent_audit_claimed": "v51_full_independent_audit_claimed",
    "V45": "V50", "V46": "V51",
    "test_single_model_development_teacher_trajectory.py": "test_single_model_native_parent_development.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit native-parent diagnostic adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_assignment(SOURCE, "RUNS", {
    "dual_parent_v50": "0d07291d0000fbf0fc6ff1b9b7cfd77ec298139dc29ec10900b91421f7561223",
    "native_parent_v51": "f8fcbee841f39f6c7bb33f02c5ce595b3734ded91e51a3ddb07c4c54fa6f5bd0",
})
SOURCE = replace_assignment(SOURCE, "HISTORICAL_AUDIT_SHA256",
    "1935ed5ac8b6dfb3c6bfb176366589456f361ec7ff69b1ec2bb16ca1dbce4ec7")
SOURCE = replace_assignment(SOURCE, "POLICY", (
    "CPU-only description of every published original V50/V51 development boundary through "
    "the requested registered step, including all earlier boundaries and all152 cases. Keep "
    "the original synthetic truth, observed-source/query masks and their overlaps, observed-blend "
    "imputation/specialist rating teachers, general parent, V8 and actual V29 initializer distinct. "
    "Add output distances to actual V29 and the exact FP32 half-V8/half-V29 grid. These are "
    "normalized-network-coordinate equal-case descriptive distances, not original-unit/user-macro "
    "rating errors, canonical NRMSE, full training losses or parameter-gradient dominance. Rating "
    "distances do not imply a retention penalty; ratings still have none. Keep every adverse and "
    "improved case, observed and query split, original update count and selection decision. "
    "Trials have different training seeds and accepted trajectories; differences are not a "
    "seed-controlled causal effect of the native-reference change. Preserve V50's completed "
    "independent audit while making no full-audit claim for live V51. Reproduce all normalized "
    "imputation RMSEs from saved grids and verify every original input hash before and after "
    "scoring. No model loading/inference, CUDA, new seeds, training, confirmation, real data, "
    "selection, changed qualification, scientific retry or inference ensemble. This descriptive "
    "report cannot replace the mandatory original full-task/component/endpoint audits."
))
SOURCE = replace_once(SOURCE, "(__file__, metrics.__file__, Path(__file__)",
    "(__file__, metrics.__file__, blueprint.__file__, metrics.previous.__file__, "
    "Path(__file__).resolve().parents[1] / 'tests/test_single_model_native_parent_alignment.py', "
    "Path(__file__)")
exec(compile(SOURCE, __file__, "exec"), globals())
