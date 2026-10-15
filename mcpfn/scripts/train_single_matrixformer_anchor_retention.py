"""Train V37 with an additional frozen native-anchor reference and unchanged evaluation gates."""

import train_single_matrixformer_full_network_cap90_v2 as predecessor
from train_single_matrixformer_full_network_cap90_v2 import memory_evidence, cap90_recipe, ENTRY_POLICY, failed_entry_evidence
import single_model_anchor_retention as retention
import single_model_anchor_retention_training as retention_training
import relocate_single_model_anchor_retention_capacity as capacity
from single_model_registered_adaptation import replace_once


ANCHOR_POLICY = ("Initialize the single ordinary V37 student only from V29 step25 with98 inherited updates. "
    "Retain that same checkpoint separately as a frozen training-only native-anchor reference. "
    + retention.POLICY + " " + retention_training.POLICY
    + " Keep the original encoder, all eight blocks and decoder:238 trainable tensors/77,998,496 parameters, "
    "zero frozen parameters and16 bitwise-frozen buffers. Keep original general-imputation and specialist-rating "
    "teachers, observed-only normalization, native-anchor rehearsal, Huber/NLL/CDF/query/point losses, "
    ".5/.25/.25 prior masses, half-composite/half-point proposal,84 raw rows,30 point protections, fresh5e-6 "
    "AdamW,.1 inward margin,2x norm bound and exact rollback. Attempt64 fresh synthetic batches; validate "
    "at0/10/25/50/64 with all22 original comparisons, strict minimax selection, earlier ties and ineligible "
    "step0. Confirm only a sole eligible selection against all32 original fresh comparisons; preserve complete "
    "frozen real protocols and every individual regression. Keep the measured.9 allocator cap. Preserve18/6/50 "
    "GiB reserves and2.25 GiB/16 MiB/32 GiB training budgets, prospectively accounting128 MiB checkpoint "
    "metadata/audit allowance and2 GiB synthetic prediction/task allowance within those same budgets. Added "
    "reference grids use declared machine-local scratch, not provider-persistent storage or backup. No probe "
    "warm start, new inference module, ensemble, real-data training/selection, gate/reference relaxation, "
    "scientific retry or registered-source mutation. The completed retention probe's terminal status records "
    "retention_audited rather than accepted; require that actual status field while preserving every original "
    "acceptance check on its optimizer result, finite metrics, independent raw audit and qualification gate. "
    "Synthetic qualification is not benchmark success.")


def retention_evidence():
    evidence = capacity.require_probe()
    path = capacity.OUTPUT / "independent_recheck.json"
    receipt = capacity.previous.copying.read(path)
    configuration = capacity.OUTPUT / "config.json"
    if (receipt["complete"] is not True or receipt["canonical_artifacts_checked"] != 3
            or receipt["config_sha256"] != capacity.sha256(configuration)
            or receipt["all_original_serialized_bytes_and_paths_preserved"] is not True
            or receipt["selected_checkpoints_and_registered_sources_unchanged"] is not True
            or receipt["full_training_capacity_ready"] is not True
            or receipt["scientific_training_or_evaluation_performed"] is not False):
        raise ValueError("Bind the complete independent capacity restoration before training")
    return {**evidence, str(path): capacity.sha256(path), str(configuration): capacity.sha256(configuration)}


SOURCE = predecessor.SOURCE
for old, new in {
    "full_network_cap90_v36": "anchor_retention_v37",
    "851100901, 852100901": "881100901, 882100901",
    "861100901, 862100901": "891100901, 892100901",
    "import preflight_single_model_full_network as preflight": "import preflight_single_model_anchor_retention as preflight",
    "audit_single_model_full_network_cap90_v2.py": "audit_single_model_anchor_retention.py",
    "run_single_model_full_network_cap90_v2_pipeline.py": "run_single_model_anchor_retention_full_pipeline.py",
    "test_single_model_full_network_cap90_v2_trial.py": "test_single_model_anchor_retention_trial.py",
    "test_single_model_full_network_cap90_v2_audit.py": "test_single_model_anchor_retention_audit.py",
    "test_single_model_full_network_cap90_v2_pipeline.py": "test_single_model_anchor_retention_full_pipeline.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V37 training adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE,
    'TRIAL_POLICY += " " + memory_evidence.POLICY.replace("V" + "35", "V36") + " " + ENTRY_POLICY',
    'TRIAL_POLICY = ANCHOR_POLICY')
SOURCE = replace_once(SOURCE, '"metadata_and_audit_allowance": 64 * 1024**2',
                      '"metadata_and_audit_allowance": 128 * 1024**2')
SOURCE = replace_once(SOURCE, '"prediction_task_allowance": 1024**3', '"prediction_task_allowance": 2 * 1024**3')
SOURCE = replace_once(SOURCE, 'exec(compile(BASE_PREFLIGHT_SOURCE, __file__, "exec"), globals())',
    'BASE_PREFLIGHT_SOURCE = replace_once(BASE_PREFLIGHT_SOURCE, '
    '\'current.get("accepted") is not True\', \'current.get("retention_audited") is not True\')\n'
    'exec(compile(BASE_PREFLIGHT_SOURCE, __file__, "exec"), globals())')
SOURCE = replace_once(SOURCE, 'ADAPTED_SOURCE = cap90_recipe.adapt_verifier(ADAPTED_SOURCE)',
    'ADAPTED_SOURCE = cap90_recipe.adapt_verifier(ADAPTED_SOURCE)\n'
    'ADAPTED_SOURCE = retention_training.adapt_training(ADAPTED_SOURCE)')
SOURCE = replace_once(SOURCE, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
    'REGISTER_SOURCE = retention_training.adapt_registration(REGISTER_SOURCE)\n'
    'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
SOURCE = replace_once(SOURCE, '**memory_evidence.require_evidence(), **failed_entry_evidence()}',
                      '**memory_evidence.require_evidence(), **failed_entry_evidence(), **retention_evidence()}')
SOURCE = replace_once(SOURCE, 'REPO / "tests/test_single_model_full_network_cap90_recipe.py",',
    'REPO / "tests/test_single_model_full_network_cap90_recipe.py", '
    'REPO / "tests/test_single_model_anchor_retention.py", REPO / "tests/test_single_model_anchor_retention_training.py",')
SOURCE = SOURCE.replace("V36", "V37")
exec(compile(SOURCE, __file__, "exec"), globals())
