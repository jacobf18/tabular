"""One preregistered point2 cross-seed average, reusing the frozen synthetic gate."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import psutil

import evaluate_dwip_cross_seed_average as core
from analyze_rowcol_controlled_evidence import read_jsonl
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json

QUEUES = {"seed1": "dwip_spectral_point2_horizon48k_20260916",
          "seed2": "dwip_spectral_point2_seed2_48k_20260916"}
WEIGHTS = {"seed1": .5, "seed2": .5}


def verify_registration(registered, sources, manifests):
    if registered["queues"] != QUEUES or registered["weights"] != WEIGHTS:
        raise ValueError("Registered replicas or weights changed")
    if registered["source_sha256"] != sources or registered["queue_manifest_sha256"] != manifests:
        raise ValueError("Registered source or training manifest changed")


def require_live(identity):
    try:
        process = psutil.Process(identity["pid"])
        if process.create_time() != identity["created"] or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("Dependency handle is no longer live; no restart attempted")
    except psutil.NoSuchProcess as error:
        raise RuntimeError("Dependency handle is absent; no restart attempted") from error


def wait_for_idle_gpu(root):
    while active := gpu_processes():
        write_json(root / "status.json", {"state": "waiting_for_idle_gpu", "observed_gpu_pids": active})
        time.sleep(120)


def register(root, roots, sources, manifests):
    path = root / "prespecification.json"
    if path.exists():
        registered = json.loads(path.read_text())
        verify_registration(registered, sources, manifests)
        return registered
    second = roots["seed2"]
    state = json.loads((second / "status.json").read_text())
    if state["state"] != "training" or any((second / name).exists() for name in (
            "dwip1e5_mteb.json", "dwip1e5_mteb.log")):
        raise RuntimeError("Preregister while seed2 is still training, before its benchmark evaluation")
    identities = {}
    for name in ("launcher.pid", "transfer_launcher.pid"):
        process = psutil.Process(int((second / name).read_text()))
        identities[name] = {"pid": process.pid, "created": process.create_time()}
        require_live(identities[name])
    trainer = [process for process in psutil.Process(identities["launcher.pid"]["pid"]).children(recursive=True)
               if any(argument.endswith("train_dwip_spectral_geometry.py") for argument in process.cmdline())]
    if len(trainer) != 1:
        raise RuntimeError("Expected one live seed2 trainer")
    metrics = read_jsonl(WORK / f"tabimpute-{QUEUES['seed2']}-dwip1e5/metrics_stream.jsonl")
    registered = {"registered_unix": time.time(), "queues": QUEUES, "weights": WEIGHTS,
        "source_sha256": sources, "queue_manifest_sha256": manifests, "dependency_handles": identities,
        "seed2_status_at_registration": state, "seed2_sampled_step": metrics[-1]["step"] if metrics else None,
        "live_trainer": {"pid": trainer[0].pid, "created": trainer[0].create_time()},
        "hypothesis": "One equal-weight average of the two independently synthetic-selected point2/48k models may reduce optimization variance. No early snapshots or extra seeds are used.",
        "context": "The analogous original-loss average failed its strict gate; that result is preserved, not reinterpreted. Point2 has different gradient scale and broad-error behavior; improvement is not assumed.",
        "architecture": "One unchanged DWIP model: FP64 parameter accumulation into FP32 state, not prediction ensembling",
        "acceptance": "Freshly reproduce both raw fixed synthetic scores; average must have strictly lower macro RMSE than BOTH. Otherwise no benchmark evaluation of the average.",
        "benchmark_policy": "This registration precedes seed2 benchmark evaluation. No alternate weights, seeds, checkpoints or benchmark-driven choices.",
        "dependencies": "Prepare CPU weights only after training ends; GPU evaluation only after both transfer panels finish and the GPU is idle."}
    write_json(path, registered)
    return registered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="dwip_point2_cross_seed_average_20260916_v2")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.experiment
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    roots = {label: REPO / "runs/objective_queues" / queue for label, queue in QUEUES.items()}
    paths = [Path(__file__), Path(core.__file__), REPO / "scripts/evaluate_dwip_checkpoint_average.py",
             REPO / "scripts/rowcol_fixed_validation.py", REPO / "src/tabimpute/model/model_new_stable.py",
             REPO / "src/tabimpute/model/bar_distribution.py"]
    sources = {str(path): sha256(path) for path in paths}
    manifests = {label: sha256(path / "manifest.json") for label, path in roots.items()}
    registered = register(root, roots, sources, manifests)
    if args.register_only:
        if not (root / "status.json").exists():
            write_json(root / "status.json", {"state": "registered", "gpu_evaluation_started": False})
        return
    try:
        while True:
            state = json.loads((roots["seed2"] / "status.json").read_text())["state"]
            if state in {"evaluating", "evaluating_mteb", "complete"}:
                break
            if state != "training":
                raise RuntimeError(f"Unexpected dependency state: {state}")
            require_live(registered["dependency_handles"]["launcher.pid"])
            write_json(root / "status.json", {"state": "waiting_for_training", "updated_unix": time.time()})
            time.sleep(120)
        verify_registration(registered, {str(path): sha256(path) for path in paths},
                            {label: sha256(path / "manifest.json") for label, path in roots.items()})
        core.QUEUES = dict(QUEUES)
        _, work, checkpoints, config, borders, caches = core.prepare(root)
        while True:
            states = {label: json.loads((path / "transfer_evaluation/status.json").read_text())["state"] for label, path in roots.items()}
            if all(state == "complete" for state in states.values()):
                break
            if any(state == "error" for state in states.values()):
                raise RuntimeError(f"Transfer dependency failed: {states}")
            require_live(registered["dependency_handles"]["transfer_launcher.pid"])
            write_json(root / "status.json", {"state": "waiting_for_transfer", "updated_unix": time.time()})
            time.sleep(120)
        wait_for_idle_gpu(root)
        verify_registration(registered, {str(path): sha256(path) for path in paths},
                            {label: sha256(path / "manifest.json") for label, path in roots.items()})
        _, work, checkpoints, config, borders, caches = core.prepare(root)
        core.evaluate(root, work, checkpoints, config, borders, caches)
    except Exception as error:
        write_json(root / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
