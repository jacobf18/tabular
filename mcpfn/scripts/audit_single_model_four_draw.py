"""Audit all V57 admissions, tasks, components, transfers and the actual endpoint."""

from contextvars import ContextVar
import hashlib
import inspect

import audit_single_model_balanced_native_long as blueprint
import train_single_matrixformer_four_draw as training
import single_model_four_draw_protection as checks
import single_model_four_draw_trial_retention as retention_training
import single_model_four_draw_trial_blend_replay as blend_replay
import single_model_four_draw_fixed_trial as fixed_trial
import single_model_four_draw_transfer_audit as transfer_audit
from single_model_four_draw_full_integration import clone_owned
from single_model_registered_adaptation import replace_once


clone_owned(blueprint, globals())
REPLAY = ContextVar("single_model_four_draw_full_audit_replay", default=None)
POLICY = training.POLICY + (
    " Independently replay every admission, admitted task, teacher and retention component; "
    "check every recorded defined-prefix gradient transfer, full optimizer coordinates, "
    "actual final endpoint and original validation grid. Preserve data-only rejections, "
    "undefined individual metrics and adverse contributions. This is not historical neural "
    "backward or full optimizer-trajectory replay and does not establish benchmark parity."
)
ADAPTED_SOURCE = blueprint.ADAPTED_SOURCE
for before, after in (
        ('import train_single_matrixformer_balanced_native_long as training', 'import train_single_matrixformer_four_draw as training'),
        ('import single_model_fixed_quarter_protection as checks', 'import single_model_four_draw_protection as checks'),
        ('full256-attempt qualification audit', 'full128-attempt qualification audit'),
        ('len(manifest["tasks"]) != 28', 'len(manifest["tasks"]) != 56'),
        ('all{28 * through_step} cached tasks', "all{counts['training_tasks']} admitted cached tasks"),
        ('    accepted, previous_rate = 0, config["base"]["peak_lr"]',
         '    transfers = transfer_audit.audit(training, config, rows)\n'
         '    hashes.update(transfers["artifact_sha256"])\n'
         '    accepted, previous_rate = 0, config["base"]["peak_lr"]'),
        ('"raw_endpoint": endpoint_result, "artifact_sha256": hashes',
         '"raw_endpoint": endpoint_result, "gradient_transfers": transfers, "artifact_sha256": hashes'),
):
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, before, after)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())
REGENERATION_SOURCE = inspect.getsource(blueprint.admission_source.regeneration_evidence)
for before, after in (("def regeneration_evidence(", "def original_regeneration_evidence("),
        ("range(28)", "range(56)"), ("24 * through_step", "48 * through_step")):
    REGENERATION_SOURCE = replace_once(REGENERATION_SOURCE, before, after)
exec(compile(REGENERATION_SOURCE, __file__, "exec"), globals())
RUN_AUDIT_SOURCE = blueprint.RUN_AUDIT_SOURCE
exec(compile(RUN_AUDIT_SOURCE, __file__, "exec"), globals())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=training.ROOT / "four_draw_v57_audit")
    arguments = parser.parse_args()
    with training.arithmetic_recovery.previous.original_arithmetic():
        run_audit(arguments.output.resolve())


if __name__ == "__main__":
    main()
