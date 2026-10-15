"""Bind every fixed-mean component to full training and independent grid replay."""

from collections import Counter
from pathlib import Path
import types

import single_model_parent_structured_training as predecessor
import single_model_dual_parent_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once
from train_single_matrixformer import GENERAL


INITIALIZER = predecessor.INITIALIZER
INITIALIZER_SHA256 = predecessor.INITIALIZER_SHA256
EVALUATION_BASELINE = predecessor.EVALUATION_BASELINE
EVALUATION_BASELINE_SHA256 = predecessor.EVALUATION_BASELINE_SHA256
POLICY_TEXT = ("Full-trial integration of fixed dual-parent structured retention. " + cohort_retention.POLICY
    + " Preserve every original task, teacher, supervised/retention weight and scale1 gradient. "
    "For each admitted28-task batch independently replay50 retention components (2 general, "
    "24 V8,24 V29) and26 final target grids. Preserve both components separately so offsetting "
    "errors cannot hide in their average. Data-only rejections and ratings create no retention "
    "inference. Keep original V8/general qualification thresholds and all teacher-component "
    "checks. No historical full-autograd or optimizer-trajectory replay claim. Actual execution "
    "requires separate full-trial registration with fresh synthetic seeds.")


def reference_identities(parent, base):
    identities = predecessor.reference_identities(parent, base)
    identities["v8"] = {"checkpoint": str(EVALUATION_BASELINE), "sha256": EVALUATION_BASELINE_SHA256}
    cohort_retention.validate_identities(identities)
    return identities


def verify_references(references, identities):
    cohort_retention.validate_identities(identities)
    if set(references) != set(identities) or len({id(reference) for reference in references.values()}) != len(references):
        raise ValueError("Independently replay all three distinct frozen training references")
    return {role: cohort_retention.verify_model(reference, identities[role]) for role, reference in references.items()}


for name in ("original_identity", "original_cohort_identity", "native_cohort_identity", "cohort_identity"):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError("Dual-parent full integration must not retain predecessor closures")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


def expected_component_counts(admitted_batches):
    if type(admitted_batches) is not int or admitted_batches < 0:
        raise ValueError("Require a nonnegative exact admitted-batch count")
    return {role: count * admitted_batches for role, count in (("general", 2), ("v8", 24), ("v29", 24))
            if admitted_batches}


def validate_saved_components(episode, cached):
    role = retention.reference_role(episode.metadata)
    if cached["component_weights"] != retention.reference_weights(role):
        raise ValueError("Keep the original fixed retention coefficients in every saved task")
    prediction = retention.combine_predictions(role, cached["components"])
    for component in cached["components"].values():
        retention.validate_target(episode, component)
    if not byte_equal(prediction, cached["prediction"]):
        raise ValueError("Each saved target must be the exact declared FP32 combination")


def require_report_components(report, rows, counts):
    expected = expected_component_counts(len(counts["admitted_steps"]))
    if report["component_role_counts"] != expected or report["component_grids"] != sum(expected.values()):
        raise ValueError("Full qualification must bind every component of every admitted batch")
    actual = Counter()
    for row in rows:
        if not row["eligible"]:
            if any(name in row for name in ("component_roles", "components_bitwise_equal")):
                raise ValueError("Ratings must not acquire retention-component inference")
            continue
        roles = list(retention.reference_weights(row["reference_role"]))
        if row["component_roles"] != roles or row["components_bitwise_equal"] is not True:
            raise ValueError("Qualification must retain both independently reproduced components")
        actual.update(roles)
    if dict(actual) != expected:
        raise ValueError("Qualification component counts must match every actual task row")


RETENTION_TRAINING_SOURCE = replace_once(predecessor.RETENTION_TRAINING_SOURCE,
    "single_model_parent_structured_retention as retention", "single_model_dual_parent_retention as retention")
RETENTION_TRAINING_SOURCE = replace_assignment(RETENTION_TRAINING_SOURCE, "POLICY", POLICY_TEXT)
for old, new in (
    ('        self.reference_grids = 0', '        self.reference_grids = 0\n        self.component_role_counts = Counter()'),
    ('cached["identity"] != self.expected_identity[reference_role]',
     'cached["identity"] != retention.target_identity(reference_role, self.expected_identity)'),
    ('            retention.validate_target(episode, cached["prediction"])',
     '            retention.validate_target(episode, cached["prediction"])\n            validate_saved_components(episode, cached)'),
    ('            device = next(self.reference[reference_role].parameters()).device\n'
     '            with torch.no_grad(), compute_context(device):\n'
     '                prediction = retention.streamed_point_predictions(self.reference[reference_role], episode.source.to(device),\n'
     '                    self.criterion, **retention.STREAM_OPTIONS).float().cpu()',
     '            components = {}\n'
     '            for component_role in retention.reference_weights(reference_role):\n'
     '                device = next(self.reference[component_role].parameters()).device\n'
     '                with torch.no_grad(), compute_context(device):\n'
     '                    prediction = retention.streamed_point_predictions(self.reference[component_role],\n'
     '                        episode.source.to(device), self.criterion, **retention.STREAM_OPTIONS).float().cpu()\n'
     '                if not byte_equal(prediction, cached["components"][component_role]):\n'
     '                    raise ValueError("Independent inference must reproduce each complete retention component")\n'
     '                components[component_role] = prediction\n'
     '                self.component_role_counts[component_role] += 1\n'
     '            prediction = retention.combine_predictions(reference_role, components)'),
    ('            atomic_npz(output, prediction=prediction.numpy())',
     '            atomic_npz(output, prediction=prediction.numpy(),\n'
     '                **{f"component_{name}": value.numpy() for name, value in components.items()})'),
    ('            self.reference_grids += 1',
     '            record.update(component_roles=list(components), components_bitwise_equal=True)\n'
     '            self.reference_grids += 1'),
    ('        self.hashes[str(self.rows_path)] = sha256(self.rows_path)',
     '        if dict(self.component_role_counts) != expected_component_counts(len(self.counts["admitted_steps"])):\n'
     '            raise ValueError("Replay every retention component from every admitted batch")\n'
     '        self.hashes[str(self.rows_path)] = sha256(self.rows_path)'),
    ('"reference_grids": self.reference_grids, "reference": self.verified,',
     '"reference_grids": self.reference_grids, "reference": self.verified,\n'
     '            "component_role_counts": dict(self.component_role_counts),\n'
     '            "component_grids": sum(self.component_role_counts.values()),'),
    ('    verify_hashes(report["artifact_sha256"])\n    return report',
     '    verify_hashes(report["artifact_sha256"])\n    require_report_components(report, rows, counts)\n    return report'),
):
    RETENTION_TRAINING_SOURCE = replace_once(RETENTION_TRAINING_SOURCE, old, new)
exec(compile(RETENTION_TRAINING_SOURCE, __file__, "exec"), globals())
