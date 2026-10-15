"""Start V36 with a strict allocator-only registration check and fresh synthetic seeds."""

from pathlib import Path

import train_single_matrixformer_full_network_cap90 as predecessor
import single_model_full_network_cap90_evidence as memory_evidence
import single_model_full_network_cap90_recipe as cap90_recipe
from single_model_registered_adaptation import replace_once


ENTRY_POLICY = ("Preserve V35's original failed registration-entry validation without restart or source "
    "mutation. It consumed no training tasks and produced no checkpoint; nevertheless never reuse its "
    "reserved seeds. Register V36 separately with fresh data/backward/confirmation streams from the same "
    "V29 initializer. The verifier requires the exact original base except for the declared.8-to.9 allocator "
    "change, and independently executes the original scientific recipe validator on its unchanged base. "
    "No other base field, scientific comparison, hash check, teacher or benchmark protocol may differ.")


def failed_entry_evidence():
    directory = predecessor.OUTPUT
    current = predecessor.read(directory / "status.json")
    expected = "ValueError('Never change the registered decoder continuation or its original V8 qualification baseline')"
    if (current["state"] != "failed" or current["error"] != expected
            or type(current["pid"]) is not int or current["pid"] <= 0 or Path(f'/proc/{current["pid"]}').exists()):
        raise ValueError("Bind the original stopped V35 entry failure, never a begun scientific trajectory")
    forbidden = [directory / name for name in ("training.jsonl", "validation.jsonl", "validation_0.json",
        "parameter_inventory.json", "training_batches", "data_admission", "latest_checkpoint.pth",
        "selected_checkpoint.pth", "best_validation_checkpoint.pth")]
    forbidden.extend(directory.with_name(directory.name + suffix) for suffix in ("_pipeline", "_audit", "_benchmarks", "_review"))
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise ValueError("V35 must have failed before model setup, any training task or scientific audit")
    for boundary in predecessor.BOUNDARIES:
        path = directory / f"validation_{boundary:03d}"
        if not path.is_symlink() or not path.is_dir() or any(path.iterdir()):
            raise ValueError("Every registered V35 validation location must remain empty")
    config = predecessor.read(directory / "config.json")
    if (predecessor.sha256(directory / "config.json") != predecessor.read(directory / "registration.json")["config_sha256"]
            or config["base"] != cap90_recipe.expected_base(predecessor.read(predecessor.original.alignment.fixed.PRODUCER / "config.json"))):
        raise ValueError("Preserve V35's original, prospectively registered allocator-only recipe")
    predecessor.verify_hashes(config["input_sha256"])
    predecessor.verify_hashes(config["source_sha256"])
    return {str(directory / name): predecessor.sha256(directory / name)
            for name in ("config.json", "registration.json", "status.json", "train_and_audit.log")}


SOURCE = predecessor.SOURCE
for old, new in {
    "full_network_cap90_v35": "full_network_cap90_v36",
    "831100901, 832100901": "851100901, 852100901",
    "841100901, 842100901": "861100901, 862100901",
    "audit_single_model_full_network_cap90.py": "audit_single_model_full_network_cap90_v2.py",
    "run_single_model_full_network_cap90_pipeline.py": "run_single_model_full_network_cap90_v2_pipeline.py",
    "test_single_model_full_network_cap90_trial.py": "test_single_model_full_network_cap90_v2_trial.py",
    "test_single_model_full_network_cap90_audit.py": "test_single_model_full_network_cap90_v2_audit.py",
    "test_single_model_full_network_cap90_pipeline.py": "test_single_model_full_network_cap90_v2_pipeline.py",
}.items():
    if old not in SOURCE:
        raise ValueError(f"Missing explicit V36 adaptation: {old}")
    SOURCE = SOURCE.replace(old, new)
SOURCE = replace_once(SOURCE, 'TRIAL_POLICY += " " + memory_evidence.POLICY',
    'TRIAL_POLICY += " " + memory_evidence.POLICY.replace("V" + "35", "V36") + " " + ENTRY_POLICY')
SOURCE = replace_once(SOURCE, 'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()',
    'ADAPTED_SOURCE = cap90_recipe.adapt_verifier(ADAPTED_SOURCE)\n'
    'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()')
SOURCE = replace_once(SOURCE, '**memory_evidence.require_evidence()}',
    '**memory_evidence.require_evidence(), **failed_entry_evidence()}')
SOURCE = replace_once(SOURCE, 'REPO / "tests/test_single_model_full_network_cap90_evidence.py",',
    'REPO / "tests/test_single_model_full_network_cap90_evidence.py", REPO / "tests/test_single_model_full_network_cap90_recipe.py",')
SOURCE = SOURCE.replace("V35", "V36")
exec(compile(SOURCE, __file__, "exec"), globals())
