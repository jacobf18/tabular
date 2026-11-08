"""Evaluate a user-returned checkpoint without claiming synthetic qualification."""

import argparse
import fcntl
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

from recsys_evaluation_common import REPO, sha256, write_csv, write_json


STAGES = ("tabular", "recsys", "llm")


def read(path):
    return json.loads(Path(path).read_text())


def verify(hashes):
    for filename, expected in hashes.items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Frozen artifact changed: {filename}")


def canonical_protocol(root, stage):
    return (root / f"{stage}_protocol_v1").resolve()


def report_only_manifest(protocol_path, checkpoint, authorization, role_key="student_has_one_checkpoint"):
    path = authorization / "evaluation_authorization.json"
    record = read(path)
    if (record.get("evaluation_only") is not True or record.get("training_authorized") is not False
            or record.get("independent_confirmation_passed") is not False
            or record["selected_step"] <= 0 or record["checkpoint"] != str(checkpoint)
            or record["checkpoint_sha256"] != sha256(checkpoint)):
        raise ValueError("Require explicit report-only authorization for the unchanged returned checkpoint")
    verify(record["input_sha256"])
    verify(record["source_sha256"])
    if record["protocol_sha256"].get(str(protocol_path / "protocol.json")) != sha256(protocol_path / "protocol.json"):
        raise ValueError("Unregistered evaluation protocol")
    return {"candidate_sha256": {str(checkpoint): sha256(checkpoint),
                                  str(checkpoint.parent / "config.json"): sha256(checkpoint.parent / "config.json")},
            "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path / "protocol.json"),
            "evaluation_authorization": str(path), "evaluation_authorization_sha256": sha256(path),
            "independent_confirmation_passed": False, "evaluation_only": True,
            "selected_step": record["selected_step"], role_key: True}


def adapted_source(source, role_key):
    prefix = "def evaluate(output, protocol_path, checkpoint, confirmation):"
    start = '    gate = read_json(confirmation / "gate.json")\n'
    end = '    manifest_path = output / "manifest.json"\n'
    if not source.startswith(prefix) or source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("Frozen evaluator authorization boundary changed")
    before, remainder = source.split(start)
    removed, unchanged = remainder.split(end)
    if 'gate["passed"]' not in removed or f'"{role_key}": True' not in removed:
        raise ValueError("Unexpected original authorization/manifest block")
    before = before.replace(prefix, prefix.replace("confirmation", "authorization"), 1)
    result = before + f'    manifest = report_only_manifest(protocol_path, checkpoint, authorization, {role_key!r})\n' + end + unchanged
    if result.split(end, 1)[1] != source.split(end, 1)[1] or "gate.json" in result:
        raise ValueError("Inference, parity, caching, scoring and audits must remain unchanged")
    return result


def evaluator(stage):
    module = importlib.import_module(f"evaluate_single_matrixformer_{stage}")
    role_key = "student_has_one_checkpoint" if stage == "recsys" else "single_checkpoint"
    source = adapted_source(inspect.getsource(module.evaluate), role_key)
    namespace = {**vars(module), "report_only_manifest": report_only_manifest}
    exec(compile(source, f"<report_only_{stage}>", "exec"), namespace)
    return module, namespace["evaluate"], source


def register(arguments):
    checkpoint, output, protocols = arguments.checkpoint, arguments.output, arguments.protocol_root
    stop_path = checkpoint.parent / "stop_summary.json"
    stopped = read(stop_path)
    selected_path = checkpoint.parent / f'validation_{stopped["checkpoint"]["selected_attempt"]}.json'
    selected = read(selected_path)
    if (sha256(checkpoint) != stopped["checkpoint"]["sha256"]
            or sha256(checkpoint) != selected["selected_checkpoint_sha256"]
            or selected["step"] != selected["best_step"] or selected["step"] <= 0):
        raise ValueError("Checkpoint does not match the previously returned synthetic selection")
    verify(read(checkpoint.parent / "config.json")["source_sha256"])
    source_paths = [Path(__file__).resolve(), REPO / "tests/test_stopped_single_matrixformer.py",
                    REPO / "scripts/reuse_single_matrixformer_parent_controls.py",
                    REPO / "scripts/review_single_matrixformer_campaign.py",
                    REPO / "scripts/run_single_matrixformer_benchmarks.py"]
    source_hashes = {str(path): sha256(path) for path in source_paths}
    protocol_hashes, adapted = {}, {}
    for stage in STAGES:
        module, _, source = evaluator(stage)
        path = canonical_protocol(protocols, stage)
        protocol = module.load_protocol(path)
        source_hashes.update(protocol["source_sha256"])
        source_hashes[str(Path(module.__file__).resolve())] = sha256(Path(module.__file__))
        protocol_hashes[str(path / "protocol.json")] = sha256(path / "protocol.json")
        adapted[stage] = source
    import reuse_single_matrixformer_parent_controls as cache

    _, controls, provenance = cache.validated_parent_controls(arguments.parent_cache, canonical_protocol(protocols, "recsys"))
    if len(controls) != 4:
        raise ValueError("Require all four audited unchanged-parent controls")
    record = {"evaluation_only": True, "training_authorized": False, "independent_confirmation_passed": False,
              "request": "User explicitly requested real benchmark evaluation of the returned checkpoint after the stop at100.",
              "policy": "All original benchmark cases and audits; no training, tuning, checkpoint reselection or synthetic gate fabrication.",
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "selected_step": selected["step"],
              "selected_before_real_evaluation": True, "protocol_root": str(protocols), "stages": list(STAGES),
              "protocol_sha256": protocol_hashes, "source_sha256": source_hashes,
              "input_sha256": {str(path): sha256(path) for path in [checkpoint, checkpoint.parent / "config.json", stop_path, selected_path]},
              "parent_cache": str(arguments.parent_cache), "parent_cache_provenance": provenance,
              "registered_unix": time.time()}
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources").mkdir()
    for path in source_paths:
        shutil.copy2(path, output / "sources" / path.name)
    for stage, source in adapted.items():
        (output / "sources" / f"report_only_{stage}.py").write_text(source)
    write_json(output / "evaluation_authorization.json", record)
    write_json(output / "status.json", {"state": "registered", "updated_unix": time.time(), "training_authorized": False})


def worker(output, stage):
    record = read(output / "evaluation_authorization.json")
    checkpoint = Path(record["checkpoint"])
    protocol = canonical_protocol(Path(record["protocol_root"]), stage)
    destination = output / stage
    module, evaluate, source = evaluator(stage)
    if source != (output / "sources" / f"report_only_{stage}.py").read_text():
        raise ValueError("Report-only evaluator changed since registration")
    report_only_manifest(protocol, checkpoint, output)
    if stage == "recsys" and not destination.exists():
        import reuse_single_matrixformer_parent_controls as cache

        namespace = {**vars(cache), "candidate_manifest": report_only_manifest}
        prepare = types.FunctionType(cache.prepare.__code__, namespace, cache.prepare.__name__)
        report = prepare(Path(record["parent_cache"]), protocol, checkpoint, output, destination)
        if len(report["controls"]) != 4 or report["student_predictions_copied"] or report["specialist_parity_copied"]:
            raise ValueError("Only audited unchanged-parent controls may be reused")
    destination.mkdir(exist_ok=True)
    with (destination / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate(destination, protocol, checkpoint, output)


def run(output):
    from run_single_matrixformer_benchmarks import require_full_audit
    import review_single_matrixformer_campaign as review

    record = read(output / "evaluation_authorization.json")
    environment = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}", "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TMPDIR": "/run"}
    audits = {}
    for stage in STAGES:
        verify(record["input_sha256"])
        verify(record["source_sha256"])
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", "worker", "--suite", stage, "--output", str(output)]
        with (output / f"{stage}.log").open("a") as stream:
            process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            write_json(output / "status.json", {"state": "evaluating", "stage": stage, "pid": os.getpid(),
                                                "child_pid": process.pid, "updated_unix": time.time()})
            try:
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if returncode:
            raise RuntimeError(f"{stage} exited with code{returncode}; see {output / (stage + '.log')}")
        path = output / stage / "audit.json"
        require_full_audit(stage, read(path))
        audits[stage] = {"path": str(path), "sha256": sha256(path)}
    comparisons = review.comparisons(output)
    verdict = review.decision(comparisons)
    write_csv(output / "comparisons.csv", comparisons)
    warning = "Report-only evaluation requested after the user stop. Independent synthetic confirmation and the full training audit remain incomplete. No training or checkpoint reselection was performed.\n\n"
    (output / "RESULTS.md").write_text(warning + review.markdown(comparisons, verdict))
    verify(record["input_sha256"])
    verify(record["source_sha256"])
    write_json(output / "result.json", {**verdict, "checkpoint_sha256": record["checkpoint_sha256"],
                                       "independent_confirmation_passed": False, "evaluation_only": True})
    write_json(output / "campaign_audit.json", {"all_evaluations_complete": True, "audits": audits,
                                               "evaluation_authorization_sha256": sha256(output / "evaluation_authorization.json"),
                                               "independent_confirmation_passed": False, "goal_success_not_assessed": True})
    write_json(output / "status.json", {"state": "complete", "stages": list(STAGES), "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "worker"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--protocol-root", type=Path)
    parser.add_argument("--parent-cache", type=Path)
    parser.add_argument("--suite", choices=STAGES)
    arguments = parser.parse_args()
    for name in ("output", "checkpoint", "protocol_root", "parent_cache"):
        value = getattr(arguments, name)
        if value is not None:
            setattr(arguments, name, value.resolve())
    if arguments.stage == "register":
        if any(getattr(arguments, name) is None for name in ("checkpoint", "protocol_root", "parent_cache")):
            parser.error("Registration requires --checkpoint, --protocol-root and --parent-cache")
        register(arguments)
        return
    if arguments.stage == "worker":
        if arguments.suite is None:
            parser.error("Worker requires --suite")
        worker(arguments.output, arguments.suite)
        return
    with (arguments.output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(arguments.output)
        except BaseException as error:
            write_json(arguments.output / "status.json", {"state": "failed", "error": repr(error),
                                                         "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
