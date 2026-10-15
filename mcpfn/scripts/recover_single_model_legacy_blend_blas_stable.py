"""Use stable code identities for the unused V41 BLAS recovery registration."""

import hashlib
import inspect
import json
from pathlib import Path
import types

import recover_single_model_legacy_blend_blas as previous
import recover_single_model_legacy_strata_stable_identity as identities
from single_model_registered_adaptation import replace_once


OUTPUT = previous.original.ROOT / "legacy_blend_teacher_v41_blas_recovery_v2"
REPLAY = previous.original.ROOT / "legacy_blend_teacher_v41_recovered_endpoint_replay_v2"
SCRATCH = previous.SCRATCH.with_name("recovered_endpoint_replay_v2")
IDENTITY_POLICY = ("Preserve the unused first BLAS recovery registration. Its separate launch verifier stopped "
    "before any stage lock, task, inference or confirmation because marshal fingerprints depend on interpreter "
    "object-reference state. Replace only that identity representation with the existing canonical structural "
    "fingerprint over bytecode, recursively typed constants and code metadata. Bind all original and recovery "
    "sources and every unused-registration byte. Keep the exact BLAS diagnosis,24-thread restoration, original "
    "scientific function code, single selected model, unused confirmation and unchanged frozen campaigns.")


def function_inventory():
    result = {}
    for name, function in previous.owned_functions().items():
        payload = json.dumps(identities.canonical_constant(inspect.unwrap(function).__code__),
            sort_keys=True, separators=(",", ":"))
        result[name] = {"code_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "no_grad": function is not inspect.unwrap(function)}
    return result


def require_previous_unused():
    for directory in (previous.OUTPUT, previous.REPLAY):
        if ({path.name for path in directory.iterdir()} != {"config.json", "registration.json", "status.json"}
                or previous.original.read(directory / "status.json")["state"] != "registered"):
            raise ValueError("Preserve the unused first recovery without any scientific output")
        state = previous.original.read(directory / "status.json")
        if previous.original.process_identity(state["pid"]) is not None:
            raise ValueError("The original registration and verifier must be stopped")
    config = previous.original.read(previous.OUTPUT / "config.json")
    if previous.original.read(previous.OUTPUT / "registration.json")["config_sha256"] != previous.sha256(previous.OUTPUT / "config.json"):
        raise ValueError("Preserve the exact first recovery registration")
    for key in ("source_sha256", "input_sha256", "library_sha256"):
        previous.verify_hashes(config[key])
    previous.require_storage(True)


def namespace():
    result = {**vars(previous), "__file__": __file__, "OUTPUT": OUTPUT, "REPLAY": REPLAY, "SCRATCH": SCRATCH,
        "previous": previous, "identities": identities, "IDENTITY_POLICY": IDENTITY_POLICY,
        "require_previous_unused": require_previous_unused}
    for name, function in vars(previous).items():
        if inspect.isfunction(function) and function.__globals__ is vars(previous):
            if function.__closure__ is not None:
                raise ValueError("Do not retain closures into the unused recovery globals")
            cloned = types.FunctionType(function.__code__, result, function.__name__, function.__defaults__)
            cloned.__kwdefaults__ = function.__kwdefaults__
            result[name] = cloned
    result["function_inventory"] = function_inventory
    source = inspect.getsource(previous.register)
    source = replace_once(source, "    require_failed_prefix()", "    require_previous_unused()\n    require_failed_prefix()")
    source = replace_once(source, "    config = {\"policy\": POLICY,", "    inputs.extend(path for directory in (previous.OUTPUT, previous.REPLAY) for path in directory.iterdir())\n"
        "    config = {\"identity_policy\": IDENTITY_POLICY, \"policy\": POLICY,")
    source = replace_once(source, '            REPO / "tests/test_single_model_legacy_blend_blas_recovery.py")},',
        '            REPO / "tests/test_single_model_legacy_blend_blas_recovery.py", Path(previous.__file__).resolve(),\n'
        '            Path(identities.__file__).resolve(), REPO / "tests/test_single_model_legacy_blend_blas_stable.py")},')
    exec(compile(source, __file__, "exec"), result)
    original_verify = result["verify"]

    def verify(output):
        require_previous_unused()
        config = original_verify(output)
        if config["identity_policy"] != IDENTITY_POLICY:
            raise ValueError("Preserve the registered stable-identity scope")
        return config

    result["verify"] = verify
    return result


if __name__ == "__main__":
    namespace()["main"]()
