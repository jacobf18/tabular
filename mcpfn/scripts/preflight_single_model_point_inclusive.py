"""Test a fresh point-inclusive Adam proposal without changing held-out qualification."""

import hashlib
import inspect

import preflight_single_model_structured_teacher as baseline
import run_single_model_structured_teacher_pipeline as preceding
import single_model_paired_protection as paired
import single_model_point_inclusive_proposal as inclusive
import single_model_replicated_decoder_batch as replicated
import single_model_shared_tensor_storage as shared_storage
import single_model_structured_teacher as teacher_policy
from single_model_registered_adaptation import replace_once, replace_assignment
from tabimpute.streamed_inference import streamed_point_predictions as teacher_predictions


POLICY_TEXT = ("One fresh28-task decoder-only feasibility test directly from original V8 after V20's completed "
               "unsuccessful independent qualification. Preserve V20's first-batch prior specifications, "
               ".25 legacy/.25 structured/.5 ratings mass, V8/general/specialist training teachers, normalization, "
               "Huber/NLL/CDF/query and exact point-metric definitions,2e-5 rate, paired protection,0.1 margin, "
               "2x norm cap and strict finite paired rollback. Change only the Adam proposal: half of each "
               "task's mass for its original composite, half shared by its defined ground-truth point gradients. "
               "Keep all248 nondecoder state tensors fixed. Preserve all raw gradients once through shared "
               "tensor storage, full actual optimizer evidence, all before/candidate grids and independently "
               "replayed teacher targets. No development, confirmation, selection, promotion, real benchmark "
               "or inference ensemble. The endpoint cannot initialize future training. Fresh data/backward "
               "seeds, not a matched-seed causal comparison with an earlier preflight.")
RAW_GRADIENT_STORAGE = "One lossless CPU tensor shared by gradients and snapshot.task_gradients; all raw rows retained"


def require_space():
    if shutil.disk_usage(ROOT).free < 23 * 1024**3:
        raise OSError("Reserve5 GiB for shared-gradient preflight evidence above the unchanged18 GiB write floor")


def shared_gradients(gradients, snapshot):
    saved = snapshot["task_gradients"]
    if (saved.device.type != "cpu" or saved.dtype != torch.float32 or gradients.shape != saved.shape
            or not torch.equal(gradients.detach().cpu(), saved)):
        raise ValueError("Share only the exact complete original proposal-gradient bytes")
    return saved


ADAPTED_SOURCE = baseline.ADAPTED_SOURCE
for old, new in {
    'OUTPUT = ROOT / "structured_teacher_preflight_v1"': 'OUTPUT = ROOT / "point_inclusive_preflight_v1"',
    'DATA_SEED, GRADIENT_SEED = 371100901, 372100901': 'DATA_SEED, GRADIENT_SEED = 401100901, 402100901',
    'or config["protection_source_sha256"] != paired.PROTECTED_SOURCE_SHA256':
        'or config["protection_source_sha256"] != inclusive.PROTECTED_SOURCE_SHA256',
    'or config["audit_source_sha256"] != paired.AUDIT_SOURCE_SHA256':
        'or config["audit_source_sha256"] != inclusive.AUDIT_SOURCE_SHA256\n'
        '            or config["proposal_policy"] != inclusive.POLICY\n'
        '            or config["raw_gradient_storage"] != RAW_GRADIENT_STORAGE',
    'record, snapshot = paired.protected_update(': 'record, snapshot = inclusive.protected_update(',
    'paired.check_record(record,': 'inclusive.check_record(record,',
    'endpoint = paired.check_endpoint(': 'endpoint = inclusive.check_endpoint(',
    '"snapshot": error.snapshot, "record": error.record, "gradients": gradients[:len(entries)].cpu(),':
        '"snapshot": error.snapshot, "record": error.record, "gradients": shared_gradients(gradients[:len(entries)], error.snapshot),',
    '"snapshot": snapshot, "record": record, "gradients": gradients[:len(entries)].cpu(),':
        '"snapshot": snapshot, "record": record, "gradients": shared_gradients(gradients[:len(entries)], snapshot),',
    'training.save_tensor({"snapshot": error.snapshot,': 'shared_storage.save({"snapshot": error.snapshot,',
    'digest = training.save_tensor({"snapshot": snapshot,': 'digest = shared_storage.save({"snapshot": snapshot,',
    'endpoint = inclusive.check_endpoint(payload,':
        'hashes.update(shared_storage.audit(payload, output / "raw_update.pt"))\n'
        '    if payload["gradients"].data_ptr() != payload["snapshot"]["task_gradients"].data_ptr():\n'
        '        raise ValueError("Raw gradient aliases must preserve one complete shared tensor")\n'
        '    endpoint = inclusive.check_endpoint(payload,',
    '# Structured-general teacher optimizer preflight': '# Point-inclusive Adam optimizer preflight',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
for function in (baseline.require_predecessor, baseline.expected_role, baseline.check_routing,
                 baseline.replay_targets, baseline.audit_teacher_replay, baseline.register_trial):
    ADAPTED_SOURCE += "\n\n" + inspect.getsource(function)
for old, new in {
    'Require terminal V19 with failed independent qualification and no real campaign':
        'Require terminal V20 with failed independent qualification and no real campaign',
    'Preserve complete independent V19 endpoint and confirmation audits':
        'Preserve complete independent V20 endpoint and confirmation audits',
    'Use only the fresh structured-teacher preflight': 'Use only the fresh point-inclusive proposal preflight',
    '    diagnostic = ROOT / "paired_decoder_v19_teacher_gaps_010"\n'
    '    gaps = read(diagnostic / "result.json")\n'
    '    if not gaps["complete"] or gaps["prediction_performed"] or gaps["selection_performed"]:\n'
    '        raise ValueError("Require the completed saved-development teacher comparison")\n': '',
    'ROOT / "domain_gradient_probe_v1/parameter_inventory.json", diagnostic / "config.json", diagnostic / "result.json",':
        'ROOT / "domain_gradient_probe_v1/parameter_inventory.json", preceding.TEACHERS / "audit.json",',
    'REPO / "tests/test_single_model_structured_teacher.py",\n'
    '               REPO / "tests/test_single_model_structured_teacher_preflight.py"':
        'REPO / "tests/test_single_model_point_inclusive_proposal.py",\n'
        '               REPO / "tests/test_single_model_point_inclusive_preflight.py",\n'
        '               REPO / "tests/test_single_model_shared_tensor_storage.py"',
    '"protection_source_sha256": paired.PROTECTED_SOURCE_SHA256, "audit_source_sha256": paired.AUDIT_SOURCE_SHA256,':
        '"protection_source_sha256": inclusive.PROTECTED_SOURCE_SHA256, "audit_source_sha256": inclusive.AUDIT_SOURCE_SHA256,\n'
        '              "proposal_policy": inclusive.POLICY, "raw_gradient_storage": RAW_GRADIENT_STORAGE,',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run", "replay", "audit"), required=True)
    arguments = parser.parse_args()
    if arguments.stage == "register":
        register(OUTPUT)
    else:
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "replay": replay_targets, "audit": audit}[arguments.stage](OUTPUT)
            except BaseException as error:
                status(OUTPUT, "failed", error=repr(error))
                raise
