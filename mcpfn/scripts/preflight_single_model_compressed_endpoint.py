"""Exercise actual RAM serialization, durable compression, persistent mount and mmap."""

import os
from pathlib import Path
import time

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import single_model_compressed_endpoint as endpoint
import single_model_composite_gradient_storage as composite


OUTPUT = Path("/mnt/checkpoints_200/single_model_goal_20261005/compressed_endpoint_preflight_v1")


def main():
    OUTPUT.mkdir()
    sources = [Path(__file__).resolve(), Path(endpoint.__file__).resolve(), Path(composite.__file__).resolve(),
               Path(endpoint.accounting.__file__).resolve(), REPO / "tests/test_single_model_compressed_endpoint.py"]
    config = {"policy": endpoint.POLICY, "source_sha256": {str(path): sha256(path) for path in sources},
              "purpose": "actual storage-only preflight with84 protected rows and28 proposal rows",
              "training_performed": False, "real_benchmarks_evaluated": False}
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "status.json", {"state": "running", "pid": os.getpid(), "updated_unix": time.time()})
    try:
        gradients = torch.arange(84 * 4096, dtype=torch.float32).reshape(84, 4096) / 1000
        gradients[0, 0] = -0.0
        complete, snapshot = composite.share(gradients, {"task_gradients": gradients[:28].clone()}, 28)
        payload = {"gradients": complete, "snapshot": snapshot,
                   "record": {"composite_tasks": 28, "protected_tasks": 84},
                   "extra_buffer": torch.tensor([float("nan"), -0.0, 0.0])}
        path = OUTPUT / "endpoint.pt"
        digest = endpoint.save(payload, path)
        restored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        hashes = endpoint.audit(restored, path)
        if (not composite.byte_equal(restored["gradients"], payload["gradients"])
                or not composite.byte_equal(restored["extra_buffer"], payload["extra_buffer"])):
            raise ValueError("Actual mounted mmap bytes must retain all gradients, NaN and signed zero")
        verify_hashes(config["source_sha256"])
        write_json(OUTPUT / "audit.json", {"complete": True, "config_sha256": sha256(OUTPUT / "config.json"),
                   "artifact_sha256": {str(path): digest, **hashes}, "protected_rows": 84, "proposal_rows": 28,
                   "actual_filesystem_roundtrip": True, "training_performed": False,
                   "qualification_changed": False, "goal_achieved": False})
        write_json(OUTPUT / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
    except BaseException as error:
        write_json(OUTPUT / "status.json", {"state": "failed", "pid": os.getpid(), "updated_unix": time.time(),
                   "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
