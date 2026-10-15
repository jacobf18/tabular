"""Measure the consumed task with all84 training gradient rows resident on the GPU."""

import hashlib
import inspect

import probe_single_model_full_network_memory as template
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = (template.POLICY + " Unlike the preceding isolated-task probe, retain the complete84-row GPU "
    "gradient allocation used by the actual trainer. Only the three rows corresponding to the consumed "
    "task are populated or reported. Bind the completed isolated probe explicitly; it did not include this "
    "allocation and is not training-memory qualification. Preserve four additional GPU vectors and do not "
    "change any scientific forward/backward operation. This remains one measured initializer/task, not a "
    "guarantee for every training shape, state or optimizer step.")
ADAPTED_SOURCE = inspect.getsource(template)
for old, new in {
    'OUTPUT = training.ROOT / "full_network_memory_probe_v1"':
        'OUTPUT = training.ROOT / "full_network_resident_memory_probe_v1"',
    'sources = [Path(__file__).resolve(), training.REPO / "tests/test_single_model_full_network_memory.py"]':
        'sources = [Path(__file__).resolve(), Path(template.__file__).resolve(), '
        'training.REPO / "tests/test_single_model_full_network_memory.py", '
        'training.REPO / "tests/test_single_model_full_network_resident_memory.py"]',
    'paths = [capsule_path, latest_path]':
        'paths = [capsule_path, latest_path]\n'
        '    previous_probe = training.read(template.OUTPUT / "result.json")\n'
        '    if (training.read(template.OUTPUT / "status.json")["state"] != "complete"\n'
        '            or previous_probe["complete"] is not True\n'
        '            or previous_probe["historical_task_and_teacher_digest_equal"] is not True\n'
        '            or previous_probe["training_updates_performed"] != 0\n'
        '            or previous_probe["all_training_tasks_memory_qualified"] is not False\n'
        '            or previous_probe["config_sha256"] != sha256(template.OUTPUT / "config.json")\n'
        '            or training.read(template.OUTPUT / "status.json")["result_sha256"] != sha256(template.OUTPUT / "result.json")):\n'
        '        raise ValueError("Bind the complete isolated diagnostic without overstating its scope")\n'
        '    paths.extend(template.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "result.json"))',
    '"gpu_headroom_vectors": 4, "training_updates_performed": 0':
        '"gpu_gradient_rows": 84, "gpu_headroom_vectors": 4, "training_updates_performed": 0',
    'config["gpu_headroom_vectors"] != 4 or config["parent_sha256"] != training.PARENT_SHA256':
        'config["gpu_headroom_vectors"] != 4 or config["gpu_gradient_rows"] != 84 or config["parent_sha256"] != training.PARENT_SHA256',
    'gradient = gradients[index]': 'gradient = gradients[index].detach().cpu()',
    'gradients = torch.empty((3, training.TRAINABLE_PARAMETERS), dtype=torch.float32, device="cpu")':
        'gradients = torch.empty((84, training.TRAINABLE_PARAMETERS), dtype=torch.float32, device="cuda")',
    '"gradient_records": gradient_records(gradients, entries)':
        '"gpu_gradient_rows": 84, "gpu_gradient_bytes": gradients.numel() * gradients.element_size(),\n'
        '                "gradient_records": gradient_records(gradients[:3], entries)',
}.items():
    ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE, old, new)
ADAPTED_SOURCE = replace_assignment(ADAPTED_SOURCE, "POLICY", POLICY_TEXT)
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())
