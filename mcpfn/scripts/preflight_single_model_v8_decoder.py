"""Conditionally test a fresh decoder update directly from the frozen V8 baseline."""

import inspect

import preflight_single_model_targeted_decoder as template
import run_single_model_stronger_decoder_pipeline as preceding
from single_model_registered_adaptation import install


WAIT_FOR = preceding.OUTPUT
POLICY_TEXT = ("Only after the unchanged V16 qualification pipeline completes without qualification, test one "
               "fresh fourteen-task decoder-only fused-CUDA AdamW update directly from V8, not the unqualified "
               "V13/V14/V15/V16 backbone or a preflight endpoint. The V14/V15 saved-score evidence motivates "
               "testing whether inheriting their backbone is avoidable, not a claim that it caused the errors. "
               "Preserve the targeted first-batch prior weights/specifications except fresh data/backward seeds, "
               ".25 legacy/.25 structured/.5 rating mass, all twelve structured families, V8 imputation and "
               "original rating teachers, normalization and Huber/NLL/CDF/query objectives. Use fresh decoder "
               "Adam at2e-5 with unchanged0.1 joint protection,2x displacement cap, strict finite-point rollback "
               "and bitwise-frozen V8 nondecoder state. Save and independently CPU-audit every gradient, optimizer "
               "state and before/candidate prediction grid. This is optimizer feasibility, not validation, "
               "model selection, confirmation, promotion or benchmark performance. If V16 qualifies, skip this "
               "preflight rather than competing with its campaign. No inference ensemble or fallback model.")


def pipeline_handle(pid, expected_start=None):
    os.kill(pid, 0)
    process = Path("/proc") / str(pid)
    command = (process / "cmdline").read_bytes().decode().strip("\0").split("\0")
    script = str(Path(preceding.__file__).resolve())
    if script not in command or command[command.index(script) + 1:] != ["--stage", "run"]:
        raise ValueError("Bind the exact existing V16 qualification process, not an arbitrary PID")
    start = (process / "stat").read_text().rsplit(") ", 1)[1].split()[19]
    if expected_start is not None and start != expected_start:
        raise ValueError("Never mistake a reused PID for the registered waiter")
    return {"pid": pid, "command": command, "start_ticks": start}


def require_space():
    if shutil.disk_usage(ROOT).free < 22 * 1024**3:
        raise OSError("Reserve the complete raw preflight evidence plus18 GiB; no concurrent training writes")


def register_trial(output):
    if output != OUTPUT or output.exists():
        raise FileExistsError("Require the new declared V8-start preflight")
    require_space()
    preceding.verify(WAIT_FOR)
    current = read(WAIT_FOR / "status.json")
    if current["state"] in {"registered", "complete", "failed"}:
        raise ValueError("Register prospectively while the actual V16 pipeline is live")
    handle = pipeline_handle(current["pid"])
    base = read(training.alignment.fixed.PRODUCER / "config.json")
    if PARENT != training.alignment.fixed.PARENT or sha256(PARENT) != base["input_sha256"][str(PARENT)]:
        raise ValueError("Use the exact original frozen V8 baseline as the training input")
    previous = sorted(set(ROOT.glob("*/config.json")) | set((REPO / "runs/single_model_goal_20261005").glob("*/config.json")))
    used = set().union(*(training.alignment.fixed.seeds_in(read(path)) for path in previous))
    fresh = training.alignment.fixed.seeds_in(batch())
    if len(fresh) != 28 or fresh.intersection(used):
        raise ValueError("Reserve fourteen new data/backward seed pairs, not development or confirmation data")
    evidence = read(EVIDENCE / "result.json")
    if not evidence["complete"] or evidence["prediction_performed"] or evidence["selection_performed"]:
        raise ValueError("Use only the completed saved-score inheritance diagnosis")
    for key in ("input_sha256", "source_sha256"):
        verify_hashes(read(EVIDENCE / "config.json")[key])
    paths = [PARENT, training.alignment.fixed.PRODUCER / "config.json", EVIDENCE / "config.json", EVIDENCE / "result.json",
             ROOT / "domain_gradient_probe_v1/parameter_inventory.json", WAIT_FOR / "config.json", WAIT_FOR / "registration.json"]
    sources = {Path(__file__).resolve(), REPO / "tests/test_single_model_v8_decoder_preflight.py"}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
                   and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())
    config = {"policy": POLICY, "base": base, "parent": str(PARENT), "parent_selection_qualified": True,
              "retention_baseline_for_future_training": str(PARENT), "mutable_prefix": "decoder.",
              "batch": batch(), "rate": RATE, "wait_for": str(WAIT_FOR), "wait_handle": handle,
              "adapted_source_sha256": ADAPTED_SOURCE_SHA256,
              "input_sha256": {**base["input_sha256"], **{str(path): sha256(path) for path in paths}},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}},
              "seed_inventory": {str(path): sha256(path) for path in previous}}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time(),
               "optimizer_update_performed": False, "confirmation_or_real_benchmarks_evaluated": False})
    status(output, "registered")


def after_v16(result):
    if not result["complete"] or type(result["gate"]["passed"]) is not bool:
        raise ValueError("Require the completed unchanged V16 qualification outcome")
    return "skip_qualified_v16" if result["gate"]["passed"] else "run_preflight"


def wait_and_run(output):
    config = verify(output)
    if read(output / "status.json")["state"] != "registered":
        raise ValueError("Never restart a started prospective preflight waiter")
    handle = config["wait_handle"]
    while True:
        current = read(WAIT_FOR / "status.json")
        if current["state"] == "complete":
            break
        if current["state"] in {"failed", "interrupted"}:
            raise RuntimeError("Preserve the unresolved original pipeline failure; do not bypass its audit")
        pipeline_handle(handle["pid"], handle["start_ticks"])
        if current["pid"] != handle["pid"]:
            raise ValueError("The registered qualification process changed")
        status(output, "waiting_for_v16", pipeline_pid=handle["pid"], pipeline_state=current["state"])
        time.sleep(30)
    preceding.verify(WAIT_FOR)
    terminal = read(WAIT_FOR / "result.json")
    decision = after_v16(terminal)
    write_json(output / "wait_result.json", {"decision": decision, "pipeline_result_sha256": sha256(WAIT_FOR / "result.json"),
               "pipeline_status_sha256": sha256(WAIT_FOR / "status.json"), "neural_evaluation_performed": False})
    if decision == "skip_qualified_v16":
        status(output, "skipped_qualified_v16", optimizer_update_performed=False)
        return
    preceding.prerequisite(WAIT_FOR)
    replay = read(preceding.REPLAY / "audit.json")
    if not replay["complete"]:
        raise ValueError("Require the original complete endpoint-replay audit")
    verify_hashes(replay["artifact_sha256"])
    require_space()
    status(output, "ready_for_preflight")
    run(output)


install(template, globals(), {
    'OUTPUT = ROOT / "targeted_decoder_preflight_v1"': 'OUTPUT = ROOT / "v8_decoder_preflight_v1"',
    'PARENT = ROOT / "decoder_repair_v14/latest_checkpoint.pth"': 'PARENT = training.alignment.fixed.PARENT',
    'EVIDENCE = ROOT / "decoder_repair_v14_progress_025"': 'EVIDENCE = ROOT / "targeted_decoder_v15_progress_100"',
    'RATE = 2e-6': 'RATE = 2e-5',
    'DATA_SEED, GRADIENT_SEED = 231100901, 232100901': 'DATA_SEED, GRADIENT_SEED = 281100901, 282100901',
    'or config["parent"] != str(PARENT) or config["mutable_prefix"] != "decoder."':
        'or config["parent"] != str(PARENT) or config["mutable_prefix"] != "decoder."\n            or config["wait_for"] != str(WAIT_FOR) or config["adapted_source_sha256"] != ADAPTED_SOURCE_SHA256',
    inspect.getsource(template.register): 'def register(output):\n    return register_trial(output)\n',
    'if read(output / "status.json")["state"] != "registered":': 'if read(output / "status.json")["state"] != "ready_for_preflight":',
}, {"POLICY": POLICY_TEXT})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if arguments.stage == "run":
                    wait_and_run(OUTPUT)
                elif read(OUTPUT / "status.json")["state"] != "skipped_qualified_v16":
                    audit(OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
