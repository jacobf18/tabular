"""Rebind final registered callbacks without retaining predecessor namespaces."""

import ast
import inspect
import types

import torch


def clone_owned(module, namespace):
    for name, value in vars(module).items():
        if not name.startswith("__") and name not in namespace:
            namespace[name] = value
    clones = {}
    for name, function in vars(module).items():
        if not inspect.isfunction(function):
            continue
        primitive = inspect.unwrap(function)
        if primitive.__globals__ is not vars(module):
            continue
        if primitive.__closure__ is not None:
            raise ValueError("Never retain a predecessor's closed callback namespace")
        if function not in clones:
            cloned = types.FunctionType(primitive.__code__, namespace, primitive.__name__, primitive.__defaults__)
            cloned.__kwdefaults__ = primitive.__kwdefaults__
            clones[function] = torch.no_grad()(cloned) if primitive is not function else cloned
        namespace[name] = clones[function]


def function_source(source, name):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1 or nodes[0].decorator_list:
        raise ValueError("Require exactly one explicit undecorated source callback")
    return ast.get_source_segment(source, nodes[0])
