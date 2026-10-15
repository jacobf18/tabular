"""Replay all structured draws of the failed V28 batch without training or repairing data."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256
from single_model_complete_missingness_prior import CompleteMissingnessSampler
import single_model_training_admission as admission


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
OUTPUT = ROOT / "anchor_rehearsal_v28_batch_admission_diagnosis_v1.json"
DIAGNOSIS = ROOT / "anchor_rehearsal_v28_abort_diagnosis_v1.json"
DIAGNOSIS_SHA256 = "5a42ec52c5bae25bec7fbf94b343db5bc26435a25e9018abff3281a7af58ba65"
FAILED_DRAW_SHA256 = "91875908f878e71091eea8c1a9fb97966e927185489929a7bc4c962cbb40f1ac"


def check_record(record, batch):
    if (record["eligible"] is not False or record["step"] != 62
            or record["task_count"] != 28 or record["structured_draws"] != 24
            or record["deferred_task_indices"] != [0, 13, 14, 27]
            or record["rejected_task_indices"] != [23]
            or [row["task"] for row in record["tasks"]] != batch["tasks"]):
        raise ValueError("Preserve the complete original failed batch, including all four deferred tasks")
    failed = record["tasks"][23]
    if (failed["raw_arrays_content_sha256"] != FAILED_DRAW_SHA256
            or failed["rejection_reasons"] != ["no_queries"]
            or failed["metadata"]["queries"] != 0 or failed["metadata"]["observations"] != 55
            or (failed["metadata"]["rows"], failed["metadata"]["columns"]) != (11, 5)):
        raise ValueError("Keep the original failed native mask and all raw arrays unmodified")
    for flag in ("teacher_inference_performed", "student_inference_performed", "optimizer_proposal_performed",
                 "masks_changed", "draws_replaced", "held_out_evaluation_performed", "qualification_changed"):
        if record[flag] is not False:
            raise ValueError("This is a data-only diagnosis of used training seeds, never a new experiment")


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Preserve every diagnosis; do not overwrite or silently retry")
    if sha256(DIAGNOSIS) != DIAGNOSIS_SHA256:
        raise ValueError("Require the unchanged original terminal-failure diagnosis")
    diagnosis = json.loads(DIAGNOSIS.read_text())
    config_path = ROOT / "anchor_rehearsal_v28/config.json"
    config = json.loads(config_path.read_text())
    verify_hashes(diagnosis["input_sha256"])
    verify_hashes(config["source_sha256"])
    batch = config["batches"][61]
    records = []
    for repetition in range(2):
        sampler = SimpleNamespace(device="cpu", structured_prior=CompleteMissingnessSampler())
        checker = admission.TrainingBatchAdmission(sampler, batch)
        checker.assert_consumed()
        check_record(checker.record, batch)
        records.append(checker.record)
    if records[0] != records[1]:
        raise ValueError("Both complete CPU regenerations must agree, not only the failed draw")
    verify_hashes(diagnosis["input_sha256"])
    result = {"diagnosis_complete": True, "created_unix": time.time(), "matching_cpu_regenerations": 2,
              "training_performed": False, "original_v28_restarted": False,
              "registered_experiment_created": False, "record": records[0],
              "input_sha256": {str(DIAGNOSIS): DIAGNOSIS_SHA256, str(config_path): sha256(config_path)},
              "source_sha256": {str(Path(path).resolve()): sha256(path)
                                for path in (__file__, admission.__file__)}}
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "structured_draws": 24,
                      "deferred_tasks": 4, "rejected_task_indices": records[0]["rejected_task_indices"]}))


if __name__ == "__main__":
    main()
