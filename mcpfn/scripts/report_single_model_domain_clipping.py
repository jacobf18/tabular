"""Report all V10 boundaries using the original strict scorer and completed audits."""

import argparse
from contextlib import contextmanager
from pathlib import Path

from audit_single_matrixformer_domain_clipping import check_record
from recsys_evaluation_common import sha256, write_json
import report_single_model_structured_run as original
from train_single_matrixformer_domain_clipping import validate_recipe


def validate_clipping_audit(audit, prediction_sha256):
    if (not audit["complete"] or audit["through_step"] != 400 or audit["logged_domain_gradients_checked"] != 800
            or audit["raw_prediction_audit_sha256"] != prediction_sha256
            or audit["research_selection_tolerance"] != 0 or audit["optimizer_trajectory_replayed"]):
        raise ValueError("Require the complete original800-domain-record audit and its matching prediction audit")
    if [row["step"] for row in audit["checked_rows"]] != list(range(1, 401)):
        raise ValueError("Keep every registered optimizer update")
    for row in audit["checked_rows"]:
        check_record(row)


@contextmanager
def report_label():
    previous = original.markdown

    def labeled(summary, rows):
        text = previous(summary, rows)
        title = "# V9 structured-mixture training outcome"
        if not text.startswith(title):
            raise ValueError("Unexpected original report format; do not mislabel an experiment")
        return text.replace(title, "# V10 per-domain-clipping training outcome", 1) + (
            "\nThe V10 optimizer clips isolated weighted domain gradients before summing them. The full prediction audit and all800 logged domain-clipping records pass.\n"
            "The paired V9 control uses the same registered seeds and protocols, but independently regenerated rating tensors have small floating-point differences; no bitwise-input or real-benchmark parity is claimed.\n")

    original.markdown = labeled
    try:
        yield
    finally:
        original.markdown = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, audit, output = args.run.resolve(), args.audit.resolve(), args.output.resolve()
    validate_recipe(original.read(run / "config.json"))
    audit_path = audit / "domain_clipping_audit.json"
    validate_clipping_audit(original.read(audit_path), sha256(audit / "audit.json"))
    fingerprint = sha256(audit_path)
    with report_label():
        original.report(run, audit, output)
    if sha256(audit_path) != fingerprint:
        raise ValueError("The clipping audit changed during reporting")
    write_json(output / "domain_clipping_report.json", {"complete": True, "through_step": 400,
        "clipping_audit": str(audit_path), "clipping_audit_sha256": fingerprint,
        "prediction_audit_sha256": sha256(audit / "audit.json"), "report_source_sha256": sha256(Path(__file__)),
        "training_performed": False, "checkpoint_selection_performed": False, "goal_success_not_assessed": True})


if __name__ == "__main__":
    main()
