"""Replace process-dependent marshal identities in an unused V29 operational recovery."""

import hashlib
import inspect
import json
from pathlib import Path
import types

import recover_single_model_legacy_strata_replay as previous
from single_model_registered_adaptation import replace_once


OUTPUT = previous.original.ROOT / "legacy_strata_v29_stable_replay_recovery"
REPLAY = previous.original.ROOT / "legacy_strata_v29_stable_endpoint_replay"
SCRATCH = previous.SCRATCH.with_name("stable_endpoint_replay")
POLICY = previous.POLICY + (" The first recovery registered but its separate run process stopped in initial "
    "verification before opening a stage lock or executing any science: marshal code-object bytes depend on "
    "interpreter reference/interning state. Preserve that unused recovery verbatim. Use deterministic structural "
    "code fingerprints covering bytecode, recursively typed constants and code metadata, plus all original "
    "source hashes. Test real pipeline inventory across fresh interpreters and entry modes before registering "
    "new operational replay storage. No change to scientific execution or the original confirmation reservation.")


def canonical_constant(value):
    if isinstance(value, types.CodeType):
        fields = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize", "co_flags",
                  "co_code", "co_consts", "co_names", "co_varnames", "co_filename", "co_name", "co_firstlineno",
                  "co_lnotab", "co_freevars", "co_cellvars")
        return ["code", {name: canonical_constant(getattr(value, name)) for name in fields}]
    if isinstance(value, tuple):
        return ["tuple", [canonical_constant(item) for item in value]]
    if isinstance(value, frozenset):
        items = [canonical_constant(item) for item in value]
        return ["frozenset", sorted(items, key=lambda item: json.dumps(item, sort_keys=True))]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, complex):
        return ["complex", value.real.hex(), value.imag.hex()]
    if value is Ellipsis:
        return ["ellipsis"]
    if value is None or isinstance(value, (str, int, bool)):
        return [type(value).__name__, value]
    raise TypeError(f"Unsupported code constant: {type(value)}")


def function_inventory():
    result = {}
    for name, function in previous.owned_functions().items():
        payload = json.dumps(canonical_constant(inspect.unwrap(function).__code__), sort_keys=True, separators=(",", ":"))
        result[name] = {"code_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                        "no_grad": function is not inspect.unwrap(function)}
    return result


def require_previous_unused():
    for directory in (previous.OUTPUT, previous.REPLAY):
        if ({path.name for path in directory.iterdir()} != {"config.json", "registration.json", "status.json"}
                or previous.original.read(directory / "status.json")["state"] != "registered"):
            raise ValueError("Preserve the unused first recovery; never continue after any scientific artifact")
        state = previous.original.read(directory / "status.json")
        if previous.original.process_identity(state["pid"]) is not None:
            raise ValueError("The first recovery must be stopped before additive identity repair")
    config = previous.original.read(previous.OUTPUT / "config.json")
    if previous.original.read(previous.OUTPUT / "registration.json")["config_sha256"] != previous.sha256(previous.OUTPUT / "config.json"):
        raise ValueError("Preserve the original unused recovery registration")
    previous.verify_hashes(config["source_sha256"])
    previous.verify_hashes(config["input_sha256"])
    previous.require_failed_prefix()
    previous.require_unused_confirmation()
    previous.require_storage(True)


def namespace():
    result = {**vars(previous), "__file__": __file__, "OUTPUT": OUTPUT, "REPLAY": REPLAY,
              "SCRATCH": SCRATCH, "POLICY": POLICY, "previous": previous,
              "require_previous_unused": require_previous_unused}
    for name, function in vars(previous).items():
        if inspect.isfunction(function) and function.__globals__ is vars(previous):
            if function.__closure__ is not None:
                raise ValueError("Do not retain a closure into the unused recovery")
            cloned = types.FunctionType(function.__code__, result, function.__name__, function.__defaults__)
            cloned.__kwdefaults__ = function.__kwdefaults__
            result[name] = cloned
    result["function_inventory"] = function_inventory
    source = inspect.getsource(previous.register)
    source = replace_once(source, "    require_failed_prefix()", "    require_previous_unused()\n    require_failed_prefix()")
    source = replace_once(source, '    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_legacy_strata_replay_recovery.py"]',
        '    inputs.extend(path for directory in (previous.OUTPUT, previous.REPLAY) for path in directory.iterdir())\n'
        '    sources = [Path(__file__).resolve(), Path(previous.__file__).resolve(),\n'
        '               REPO / "tests/test_single_model_legacy_strata_replay_recovery.py",\n'
        '               REPO / "tests/test_single_model_legacy_strata_stable_identity.py"]')
    exec(compile(source, __file__, "exec"), result)
    return result


if __name__ == "__main__":
    namespace()["main"]()
