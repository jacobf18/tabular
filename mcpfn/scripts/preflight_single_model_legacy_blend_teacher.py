"""Test a legacy observed-blend teacher once after the complete negative V40 trial."""

from pathlib import Path

import preflight_single_model_all_imputation_retention as previous_probe
import run_single_model_all_imputation_retention_pipeline as completed
import single_model_legacy_blend_teacher as new_teacher
import single_model_legacy_blend_replay as blend_replay
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = ("One fresh28-task probe after V40's complete64-attempt negative result,63 optimizer attempts, "
    "58 accepted updates, original data-only rejection40, full fixed-half audit,1764 task replays,1638 "
    "retention references and56 endpoint grids. Initialize only from V29 step25/98 inherited updates, never "
    "V40 or a probe endpoint. Change only the legacy training target from the single original general "
    "teacher to the original observed-error blend on the same normalized synthetic input. Keep general "
    "targets on all structured tasks and specialist targets on ratings. " + new_teacher.POLICY_TEXT
    + " This separate registration authorizes one feasibility update and its independent replays only, "
    "not full training. Preserve all-imputation weight10 count-shrunk V29 retention, one unit-variance "
    "pseudocount,1e-6 floor, all queries, original priors and.5/.25/.25 masses, Huber/NLL/CDF/query/point "
    "loss coefficients,238 tensors/77,998,496 trainable parameters,16 frozen buffers,84 gradient rows, "
    "30 finite protections, fresh5e-6 AdamW,.1 projection margin, fixed-half rounded displacement,2x norm "
    "cap and exact rollback. Preserve.9 allocator cap and18/6/50 GiB reserves with512 MiB/16 MiB/32 GiB "
    "write budgets. Add4 MiB explicit component-replay allowance within the existing checkpoint budget. "
    + blend_replay.POLICY + " Retain all22 development guards,32 confirmation comparisons and frozen real "
    "benchmarks for any separately authorized full trial. No hidden truth routing, changed inference, "
    "consumed-seed reuse, scientific retry, prior repair, source mutation or benchmark-success claim.")


def require_exact_predecessor():
    paths = {
        completed.training.OUTPUT / "config.json": "ace695e5abe41e57f785d5157acc9bb91b16b2226a49401df41497eac8f62b3c",
        completed.OUTPUT / "result.json": "13d9ef09ba2cf33d524493200b77dba6333d447a64de9587e8eab3d32292b7ed",
        completed.AUDIT / "audit.json": "54388632ce023a47ce805b3a93c38c4fac714d6735cb01b7a42a8111f8b3efaa",
        completed.REPLAY / "audit.json": "6b5dddc5a6298323b0f4580e33c4688007e44114b5f2f802cedbe9253e7ab608",
        completed.CONFIRMATION / "gate.json": "3016c6c32e38311fecae4a0c250e409167d7eb92827c715d1efecbc80ca71a94",
        Path("/mnt/matrixformer_scratch/all_imputation_retention_v40/anchor_retention_replay/summary.json"):
            "a6422909b086d961077e8a4636c780425d9e93c55b7ac5172f5ea03a1a25fe43",
    }
    expected = {str(path): digest for path, digest in paths.items()}
    previous_probe.verify_hashes(expected)
    return expected


SOURCE = previous_probe.SOURCE
for old, new in {
    "import run_single_model_fixed_half_shrunk_retention_pipeline as latest":
        "import run_single_model_all_imputation_retention_pipeline as latest",
    "import single_model_all_imputation_retention as retention":
        "import single_model_all_imputation_retention as retention\nimport single_model_legacy_blend_teacher as teacher_policy",
    "all_imputation_retention_preflight_v1": "legacy_blend_teacher_preflight_v1",
    "991100901, 992100901": "1021100901, 1022100901",
    "require_v39_predecessor": "require_v40_predecessor",
    '"accepted_updates": 59': '"accepted_updates": 58',
    '"independently_regenerated_training_tasks": 1792': '"independently_regenerated_training_tasks": 1764',
    '"independently_regenerated_rating_contexts": 128': '"independently_regenerated_rating_contexts": 126',
    'training_audit["admission_attempts"]["data_rejected_steps"] != []':
        'training_audit["admission_attempts"]["data_rejected_steps"] != [40]',
    'training_audit["admission_attempts"]["optimizer_attempts"] != 64':
        'training_audit["admission_attempts"]["optimizer_attempts"] != 63',
    "tests/test_single_model_all_imputation_retention_preflight.py":
        "tests/test_single_model_legacy_blend_teacher_preflight.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit legacy-blend feasibility adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, 'STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] += 128 * 1024**2',
    'STORAGE_ACCOUNTING["checkpoint_artifact_allowance_bytes"] += 128 * 1024**2 + blend_replay.ALLOWANCE')
SOURCE = replace_once(SOURCE, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    'REGISTER_SOURCE = replace_once(REGISTER_SOURCE, \'    identities = teacher_policy.identities(base, training.alignment.fixed.PARENT)\',\n'
    '    \'    replay_storage = blend_replay.storage_accounting(batch())\\n    identities = teacher_policy.identities(base, training.alignment.fixed.PARENT)\')\n'
    'REGISTER_SOURCE = replace_once(REGISTER_SOURCE, \'config = {"policy": POLICY,\',\n'
    '    \'config = {"policy": POLICY, "blend_replay_storage": replay_storage, "blend_replay_policy": blend_replay.POLICY, "teacher_replay_source_sha256": TEACHER_REPLAY_SOURCE_SHA256, "teacher_replay_audit_source_sha256": TEACHER_REPLAY_AUDIT_SOURCE_SHA256,\')\n'
    'REGISTER_SOURCE = replace_once(REGISTER_SOURCE, \'    sources.update(Path(module.__file__).resolve()\',\n'
    '    \'    sources.update({REPO / "tests/test_single_model_legacy_blend_teacher.py", REPO / "tests/test_single_model_legacy_blend_replay.py"})\\n    sources.update(Path(module.__file__).resolve()\')\n'
    'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
SOURCE = replace_once(SOURCE, 'ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)',
    'ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, \'or config["teacher_policy"] != teacher_policy.POLICY\',\n'
    '    \'or config["teacher_policy"] != teacher_policy.POLICY\\n\'\n'
    '    \'            or config["blend_replay_storage"] != blend_replay.storage_accounting(config["batch"])\\n\'\n'
    '    \'            or config["blend_replay_policy"] != blend_replay.POLICY\\n\'\n'
    '    \'            or config["teacher_replay_source_sha256"] != TEACHER_REPLAY_SOURCE_SHA256\\n\'\n'
    '    \'            or config["teacher_replay_audit_source_sha256"] != TEACHER_REPLAY_AUDIT_SOURCE_SHA256\')\n'
    'ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)')
SOURCE = replace_once(SOURCE, 'if __name__ == "__main__":',
    'expected_role, check_routing = blend_replay.expected_role, blend_replay.check_routing\n'
    'TEACHER_REPLAY_SOURCE = blend_replay.GPU_SOURCE\n'
    'TEACHER_REPLAY_AUDIT_SOURCE = blend_replay.AUDIT_SOURCE\n'
    'TEACHER_REPLAY_SOURCE_SHA256 = hashlib.sha256(TEACHER_REPLAY_SOURCE.encode()).hexdigest()\n'
    'TEACHER_REPLAY_AUDIT_SOURCE_SHA256 = hashlib.sha256(TEACHER_REPLAY_AUDIT_SOURCE.encode()).hexdigest()\n'
    'exec(compile(TEACHER_REPLAY_SOURCE, __file__, "exec"), globals())\n'
    'exec(compile(TEACHER_REPLAY_AUDIT_SOURCE, __file__, "exec"), globals())\n\n'
    'if __name__ == "__main__":')
SOURCE = SOURCE.replace("V39", "V40")
SOURCE = replace_assignment(SOURCE, "POLICY_TEXT", POLICY_TEXT)
exec(compile(SOURCE, __file__, "exec"), globals())
