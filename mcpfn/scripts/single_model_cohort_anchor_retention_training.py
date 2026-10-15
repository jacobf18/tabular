"""Bind both cohort references to every full-trial task and independent replay."""

import ast
from collections import Counter

import single_model_all_imputation_retention_training as previous
import single_model_cohort_anchor_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = ("Prospective full-trial integration of the cohort-reference policy. " + cohort_retention.POLICY
    + " Bind both exact frozen identities in registration, every admitted batch and independent replay. "
    "Regenerate every original training task, validate every role decision, replay each selected reference "
    "on its complete original source/mask and preserve every original teacher grid. Check both frozen "
    "networks before and after replay. Ratings and data-only rejections add no reference inference. "
    "Qualification must bind all task rows, role counts, checkpoint identities and full-grid hashes; no "
    "inference ensemble, changed loss, dropped cohort or historical full-autograd replay claim. This "
    "implementation alone authorizes no training, seeds, selection or confirmation.")


def reference_identities(parent, base):
    identities = {"v8": {"checkpoint": base["parent_checkpoint"],
        "sha256": base["input_sha256"][base["parent_checkpoint"]]},
        "v29": {"checkpoint": str(parent), "sha256": cohort_retention.REFERENCE_SHA256S["v29"]}}
    cohort_retention.validate_identities(identities)
    return identities


def cohort_identity(config):
    identities = reference_identities(config["parent_checkpoint"], config["base"])
    if (config["retention_references"] != identities
            or any(config["input_sha256"].get(identity["checkpoint"]) != identity["sha256"] for identity in identities.values())
            or config["anchor_retention"] != retention.SPECIFICATION
            or config["retention_backward_sha256"] != retention.BACKWARD_SOURCE_SHA256
            or config["retention_storage"] != layout(config["scratch"])):
        raise ValueError("Bind both original checkpoint identities, unchanged penalty and declared replay paths")
    return identities


def verify_references(references, identities):
    retention.validate_identities(identities)
    if set(references) != set(identities) or references["v8"] is references["v29"]:
        raise ValueError("Independently replay both distinct frozen references")
    return {role: retention.verify_model(reference, identities[role]) for role, reference in references.items()}


SOURCE = previous.SOURCE
identity_node = next(node for node in ast.parse(SOURCE).body if isinstance(node, ast.FunctionDef) and node.name == "identity")
source_lines = SOURCE.splitlines(keepends=True)
SOURCE = "".join(source_lines[:identity_node.lineno - 1]) + "def identity(config):\n    return cohort_identity(config)\n" + "".join(source_lines[identity_node.end_lineno:])
for old, new in [
    ("import single_model_all_imputation_retention as retention", "import single_model_cohort_anchor_retention as retention"),
    ('    reference = deepcopy(model).eval().requires_grad_(False)\n    retention.verify_model(reference, expected)\n    return reference',
        '    return retention.make_references(model, expected)'),
    ('config = {"policy": POLICY, "anchor_retention":',
        'config = {"policy": POLICY, "retention_references": retention_training.reference_identities(PARENT, base), "anchor_retention":'),
    ('self.verified = retention.verify_model(reference, self.expected_identity)',
        'self.verified = verify_references(reference, self.expected_identity)'),
    ('        selected = retention.eligible(episode.metadata)',
        '        reference_role = retention.reference_role(episode.metadata)\n        selected = reference_role is not None'),
    ('or decision["metadata"] != episode.metadata or decision["eligible"] is not selected',
        'or decision["metadata"] != episode.metadata or decision["eligible"] is not selected\n'
        '                or decision["reference_role"] != reference_role'),
    ('"queries": int(episode.mask.sum()), "full_episode_and_identity_checked": True}',
        '"queries": int(episode.mask.sum()), "full_episode_and_identity_checked": True,\n'
        '                  "reference_role": reference_role, "metadata": dict(episode.metadata)}'),
    ('cached["identity"] != self.expected_identity or cached["specification"]',
        'cached["identity"] != self.expected_identity[reference_role] or cached["reference_role"] != reference_role\n'
        '                    or cached["specification"]'),
    ('device = next(self.reference.parameters()).device', 'device = next(self.reference[reference_role].parameters()).device'),
    ('retention.streamed_point_predictions(self.reference, episode.source.to(device),',
        'retention.streamed_point_predictions(self.reference[reference_role], episode.source.to(device),'),
    ('if retention.verify_model(self.reference, self.expected_identity) != self.verified:',
        'if verify_references(self.reference, self.expected_identity) != self.verified:'),
    ('"reference_grids": self.reference_grids, "reference": self.verified,',
        '"reference_grids": self.reference_grids, "reference": self.verified,\n'
        '            "reference_role_counts": dict(Counter(json.loads(line)["reference_role"] for line in self.rows_path.read_text().splitlines() if json.loads(line)["eligible"])),'),
    ('report["reference"] != {**reference, "state_tensors": 254, "bitwise_equal": True}',
        'report["reference"] != {role: {**identity, "state_tensors": 254, "bitwise_equal": True} for role, identity in reference.items()}'),
    ('or report["reference_grids"] != sum(row["eligible"] for row in rows)',
        'or report["reference_grids"] != sum(row["eligible"] for row in rows)\n'
        '            or any(row["reference_role"] != retention.reference_role(row["metadata"])\n'
        '                or row["eligible"] is not (row["reference_role"] is not None) for row in rows)\n'
        '            or report["reference_role_counts"] != dict(Counter(row["reference_role"] for row in rows if row["eligible"]))'),
]:
    if SOURCE.count(old) != 1:
        raise ValueError(f"Require one full-trial cohort adaptation, found {SOURCE.count(old)}: {old}")
    SOURCE = replace_once(SOURCE, old, new)
SOURCE = replace_assignment(SOURCE, "POLICY", POLICY_TEXT)
exec(compile(SOURCE, __file__, "exec"), globals())
