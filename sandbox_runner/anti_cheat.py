from __future__ import annotations

import ast
from dataclasses import dataclass

FORBIDDEN_IMPORT_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "urllib",
    "http",
    "shutil",
    "ctypes",
    "importlib",
    "multiprocessing",
    "threading",
    "signal",
    "pathlib",
    "pickle",
    "inspect",
}

FORBIDDEN_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "globals",
    "locals",
    "vars",
}

FORBIDDEN_REF_MODULES = (
    "torch.nn.functional",
    "scipy.special",
)

FORBIDDEN_REF_ATTRS = {
    "functional",
    "rms_norm",
    "layer_norm",
    "scaled_dot_product_attention",
    "softmax",
    "silu",
    "gelu",
}


@dataclass(frozen=True)
class CheatReport:
    cheat_detected: bool
    reason: str = ""


def analyze_candidate_source(source_code: str) -> CheatReport:
    """Static AST inspection blocking unsafe syscalls, fallback cheating, and memoization hacks."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        # Let the compiler step report syntax errors cleanly rather than flagging as cheat.
        return CheatReport(cheat_detected=False)

    module_globals: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and isinstance(stmt.value, (ast.Dict, ast.List, ast.Set)):
                    module_globals.add(target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    return CheatReport(True, f"Forbidden import detected: {alias.name}")
                for ref_mod in FORBIDDEN_REF_MODULES:
                    if ref_mod in alias.name:
                        return CheatReport(True, f"Reference fallback import detected: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                return CheatReport(True, f"Forbidden from-import detected: {mod}")
            for ref_mod in FORBIDDEN_REF_MODULES:
                if ref_mod in mod:
                    return CheatReport(True, f"Reference fallback import detected: {mod}")

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                return CheatReport(True, f"Forbidden builtin call detected: {node.func.id}()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_REF_ATTRS:
                return CheatReport(True, f"Reference operator fallback detected: .{node.func.attr}()")

        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_REF_ATTRS:
                return CheatReport(True, f"Reference operator attribute access detected: .{node.attr}")
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in {"__init__"}:
                return CheatReport(True, f"Dunder attribute escape detected: {node.attr}")

        elif isinstance(node, ast.FunctionDef) and node.name == "kernel_forward":
            # Detect global cache mutation inside kernel_forward (memoization cheat)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Global):
                    return CheatReport(True, "Global state mutation inside kernel_forward is forbidden")
                if isinstance(sub, ast.Subscript) and isinstance(sub.ctx, ast.Store):
                    if isinstance(sub.value, ast.Name) and sub.value.id in module_globals:
                        return CheatReport(True, f"Memoization/cache write to global '{sub.value.id}' detected")
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if (
                        isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id in module_globals
                        and sub.func.attr in {"append", "update", "setdefault", "extend", "add"}
                    ):
                        return CheatReport(True, f"Global container mutation on '{sub.func.value.id}' detected")

    return CheatReport(cheat_detected=False)
