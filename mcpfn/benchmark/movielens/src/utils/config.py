"""Config loading for the benchmark harness."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .paths import CONFIGS_DIR


def _try_import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError:  # pragma: no cover
        return None


def _parse_simple_yaml(text: str) -> Any:
    """Very small YAML parser fallback so this module works without PyYAML.

    It supports nested dict/list/scalars, two-space indentation, lists of
    scalars with leading ``-``, inline [a, b, c] lists, and quoted strings.
    Not a general YAML parser — meant for our own config files only.
    """
    import ast
    import re

    lines = [ln.rstrip() for ln in text.splitlines()]
    # Remove comments and blank lines
    cleaned = []
    for ln in lines:
        # strip full-line comments
        s = ln.split(" #", 1)[0].rstrip()
        if s.lstrip().startswith("#"):
            continue
        if not s.strip():
            continue
        cleaned.append(s)

    def _parse_scalar(val: str) -> Any:
        val = val.strip()
        if val == "" or val.lower() == "null" or val == "~":
            return None
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        # Quoted string
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            return val[1:-1]
        # Inline list
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(",")]
            return [_parse_scalar(p) for p in parts]
        # Try numeric
        try:
            if "." in val or "e" in val.lower():
                return float(val)
            return int(val)
        except ValueError:
            return val

    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def _parse_block(start: int, indent: int):
        i = start
        if i >= len(cleaned):
            return {}, i
        first = cleaned[i]
        if first.lstrip().startswith("- "):
            return _parse_list(start, indent)
        return _parse_mapping(start, indent)

    def _parse_mapping(start: int, indent: int):
        d: Dict[str, Any] = {}
        i = start
        while i < len(cleaned):
            line = cleaned[i]
            cur_indent = _indent(line)
            if cur_indent < indent:
                break
            if cur_indent > indent:
                i += 1
                continue
            stripped = line.strip()
            if stripped.startswith("- "):
                break
            if ":" not in stripped:
                i += 1
                continue
            key, _, rest = stripped.partition(":")
            key = key.strip().strip('"\'')
            rest = rest.strip()
            if rest == "":
                # Nested block
                child, i2 = _parse_block(i + 1, indent + 2)
                d[key] = child
                i = i2
            else:
                d[key] = _parse_scalar(rest)
                i += 1
        return d, i

    def _parse_list(start: int, indent: int):
        lst = []
        i = start
        while i < len(cleaned):
            line = cleaned[i]
            cur_indent = _indent(line)
            if cur_indent < indent:
                break
            stripped = line.strip()
            if not stripped.startswith("- "):
                break
            val = stripped[2:].strip()
            if val == "":
                child, i2 = _parse_block(i + 1, indent + 2)
                lst.append(child)
                i = i2
            else:
                lst.append(_parse_scalar(val))
                i += 1
        return lst, i

    root, _ = _parse_block(0, 0)
    return root


def load_yaml(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text()
    yaml = _try_import_yaml()
    if yaml is not None:
        return yaml.safe_load(text)
    return _parse_simple_yaml(text)


def load_datasets_config() -> Dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "datasets.yaml")


def load_experiments_config() -> Dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "experiments.yaml")


def load_tabimpute_models_config() -> Dict[str, Any]:
    return json.loads((CONFIGS_DIR / "tabimpute_models.json").read_text())
