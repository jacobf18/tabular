"""Report all V11 boundaries with unchanged strict scoring and complete original audits."""

import argparse
from contextlib import contextmanager
from pathlib import Path

from recsys_evaluation_common import sha256, write_json
from report_single_model_domain_clipping import validate_clipping_audit
import report_single_model_structured_run as original
from train_single_matrixformer_family_rehearsal import validate_recipe


@contextmanager
def report_label():
    previous = original.markdown

    def labeled(summary, rows):
        text = previous(summary, rows)
        title = "# V9 structured-mixture training outcome"
        if not text.startswith(title):
            raise ValueError("Unexpected original report format; do not mislabel an experiment")
        return text.replace(title, "# V11 family-rehearsal training outcome", 1) + (
            "\nV11 preserves the original legacy-imputation and rating episode specifications, objectives, teacher identities, domain clipping and single-network architecture. Of200 structured episodes,100 cycle all48 family/generator pairs and100 rehearse both development-failing families across all four generators.\n"
            "This is an exploratory prior-sampling intervention on a reused synthetic development bank, not a blinded comparison. Independently regenerated CUDA rating inputs are not claimed bitwise identical.\n"
            "The full prediction audit and all800 logged domain-clipping records pass; all22 original retention guards and all nine validation boundaries remain visible. No real-benchmark parity is established by this report.\n")

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
    fingerprint = sha256(audit_path)
    validate_clipping_audit(original.read(audit_path), sha256(audit / "audit.json"))
    with report_label():
        original.report(run, audit, output)
    if sha256(audit_path) != fingerprint:
        raise ValueError("The clipping audit changed during reporting")
    write_json(output / "family_rehearsal_report.json", {"complete": True, "through_step": 400,
        "clipping_audit": str(audit_path), "clipping_audit_sha256": fingerprint,
        "prediction_audit_sha256": sha256(audit / "audit.json"), "report_source_sha256": sha256(Path(__file__)),
        "training_performed": False, "checkpoint_selection_performed": False, "goal_success_not_assessed": True})


if __name__ == "__main__":
    main()
