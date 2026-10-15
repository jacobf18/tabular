"""Load narrowly declared additive variants without editing immutable experiment sources."""

import ast
import hashlib
import inspect


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Require one exact immutable source expression for each declared adaptation")
    return source.replace(before, after)


def replace_assignment(source, name, value):
    matches = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]
    if len(matches) != 1:
        raise ValueError("Require one top-level assignment for each declared policy")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + f"{name} = {value!r}\n" + "".join(lines[node.end_lineno:])


def install(template, namespace, replacements, assignments):
    description = namespace.get("__doc__")
    source = inspect.getsource(template)
    marker = '\nif __name__ == "__main__":'
    if source.count(marker) != 1:
        raise ValueError("Require a single original CLI boundary")
    source = source.split(marker)[0]
    for before, after in replacements.items():
        source = replace_once(source, before, after)
    for name, value in assignments.items():
        source = replace_assignment(source, name, value)
    namespace["ADAPTED_SOURCE"] = source
    namespace["ADAPTED_SOURCE_SHA256"] = hashlib.sha256(source.encode()).hexdigest()
    exec(compile(source, namespace["__file__"], "exec"), namespace)
    namespace["__doc__"] = description
