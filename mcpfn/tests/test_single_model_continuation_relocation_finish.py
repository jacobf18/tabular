import pytest

import finish_single_model_continuation_relocation as finish


def fixture():
    records = [{"name": name, "image": f"/{name}", "destination": f"/root/{name}",
                "image_sha256": name, "image_bytes": 123} for name in finish.original.NAMES]
    copies = [{"source": record["image"], "destination": record["destination"], "sha256": record["image_sha256"],
               "bytes_relocated": record["image_bytes"], "canonical_path_preserved": True, "durable_copy_fsynced": True}
              for record in records]
    return [{"records": records}, {"state": "failed", "error": finish.ERROR},
            {"results": [{"all_original_artifact_hashes_unchanged": True, "destination": records[0]["destination"]}]}, copies]


def test_only_completed_copies_and_exact_interruption_are_accepted():
    finish.validate_copies(*fixture())


@pytest.mark.parametrize("field,value", [("source", "other"), ("destination", "other"), ("sha256", "changed"),
    ("bytes_relocated", 122), ("canonical_path_preserved", False), ("durable_copy_fsynced", False)])
def test_incomplete_or_changed_copy_cannot_be_finalized(field, value):
    records = fixture()
    records[-1][1][field] = value
    with pytest.raises(ValueError, match="durable copy"):
        finish.validate_copies(*records)


@pytest.mark.parametrize("field,value", [("state", "running"), ("error", "different failure")])
def test_unrelated_failure_or_active_run_is_rejected(field, value):
    records = fixture()
    records[1][field] = value
    with pytest.raises(ValueError, match="documented"):
        finish.validate_copies(*records)
