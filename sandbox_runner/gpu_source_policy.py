"""Strict static AST policy analyzer for generated OpenAI Triton GPU kernel candidates.

Candid security boundary:
This analyzer is a static reward-hacking and candidate-policy filter—NOT a
runtime security sandbox or OS isolation mechanism. AST inspection cannot prove
runtime isolation, prevent arbitrary side effects in untrusted Python/CUDA code,
or replace external container/VM containment. Use external OS/container isolation
for untrusted code.

What this analyzer enforces:
1. Strict import allow-list: only `torch`, `triton`, `triton.language`, `math`,
   and `__future__` (blocks `torch.nn.functional`, `numpy`, `os`, `subprocess`, etc.).
2. Real `@triton.jit` kernel definition containing actual Triton memory ops
   (`tl.load` and `tl.store` / `tl.atomic_*`)—rejects empty/dummy kernels.
3. Real grid launch (`_kernel[grid](...)`) reachable from `kernel_forward(...)`.
4. Strict host-side PyTorch allow-list (allocation, shape, layout, dtype/metadata
   only)—blocks host-side `torch.matmul`, `torch.mean`, `torch.rsqrt`, tensor
   arithmetic (`a @ b`, `x * gate`, `x + r`), module/function aliases, and nested
   helper fallbacks.
5. Blocks output memoization (global container writes, `global`, `@lru_cache`),
   dynamic execution builtins (`eval`, `exec`, `getattr`, `__import__`, `open`),
   and dunder attribute escapes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

SECURITY_BOUNDARY_NOTE = (
    "static_ast_policy_not_security_isolation: AST checks filter reward-hacking "
    "fallbacks and require real Triton kernel launches, but do NOT provide OS or "
    "runtime security isolation."
)

ALLOWED_IMPORT_MODULES = frozenset({
    "torch",
    "triton",
    "triton.language",
    "math",
    "__future__",
})

FORBIDDEN_BUILTINS = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "input",
})

# Host-side `torch.<attr>` calls allowed strictly for allocation / metadata (NO math)
ALLOWED_HOST_TORCH_CALLS = frozenset({
    "empty",
    "empty_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "is_tensor",
})

# Host-side `torch.<attr>` non-call attribute reads (dtypes / Tensor type annotation)
ALLOWED_HOST_TORCH_ATTRS = frozenset({
    "Tensor",
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "int8",
    "uint8",
    "int16",
    "int32",
    "int64",
    "bool",
}) | ALLOWED_HOST_TORCH_CALLS

# Host-side tensor methods allowed strictly for layout/shape/metadata (NO arithmetic/reductions)
ALLOWED_HOST_TENSOR_METHODS = frozenset({
    "contiguous",
    "reshape",
    "view",
    "stride",
    "size",
    "numel",
    "dim",
    "is_floating_point",
    "is_contiguous",
    "element_size",
    "data_ptr",
    "unsqueeze",
    "squeeze",
    "flatten",
})

ALLOWED_TRITON_HOST_CALLS = frozenset({
    "cdiv",
    "next_power_of_2",
})

TRITON_MEM_LOAD_ATTRS = frozenset({"load"})
TRITON_MEM_STORE_ATTRS = frozenset({
    "store",
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_cas",
    "atomic_xchg",
})


@dataclass(frozen=True)
class GpuSourcePolicyResult:
    allowed: bool
    violations: list[str] = field(default_factory=list)
    jit_kernels: list[str] = field(default_factory=list)
    launched_kernels: list[str] = field(default_factory=list)
    security_boundary: str = SECURITY_BOUNDARY_NOTE


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is not None:
            return f"{parent}.{node.attr}"
    return None


def _is_triton_jit_decorator(dec: ast.AST, triton_aliases: set[str], jit_aliases: set[str]) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Name) and target.id in jit_aliases:
        return True
    if isinstance(target, ast.Attribute) and target.attr == "jit":
        if isinstance(target.value, ast.Name) and target.value.id in triton_aliases:
            return True
    return False


def _is_scalar_param(arg: ast.arg, default_node: ast.AST | None) -> bool:
    if default_node is not None and isinstance(default_node, ast.Constant):
        if isinstance(default_node.value, (int, float, bool)):
            return True
    if arg.annotation is not None:
        ann_str = _dotted_name(arg.annotation)
        if ann_str in {"int", "float", "bool"}:
            return True
    return False


def _inspect_jit_kernel_body(
    fn: ast.FunctionDef,
    torch_aliases: set[str],
    tl_aliases: set[str],
) -> tuple[bool, list[str]]:
    """Verify a @triton.jit kernel has real tl.load + tl.store/atomic and no torch calls."""
    violations: list[str] = []
    has_load = False
    has_store = False

    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                violations.append(f"JIT kernel '{fn.name}' calls forbidden builtin '{node.func.id}()'.")
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                owner = node.func.value.id
                attr = node.func.attr
                if owner in torch_aliases:
                    violations.append(f"JIT kernel '{fn.name}' calls host torch.{attr}() inside @triton.jit.")
                if owner in tl_aliases:
                    if attr in TRITON_MEM_LOAD_ATTRS:
                        has_load = True
                    if attr in TRITON_MEM_STORE_ATTRS:
                        has_store = True
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                violations.append(f"JIT kernel '{fn.name}' accesses dunder attribute '{node.attr}'.")

    # Pure JIT math helpers need no memory operations. Only kernels reachable
    # through a host grid launch must establish the load/store requirement.
    return (has_load and has_store and len(violations) == 0), violations


def check_gpu_source_policy(source_code: str) -> GpuSourcePolicyResult:
    """Run strict static AST policy checks on a candidate Triton GPU kernel source string."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return GpuSourcePolicyResult(allowed=False, violations=[f"SyntaxError: {exc}"])

    violations: list[str] = []
    torch_aliases: set[str] = set()
    triton_aliases: set[str] = set()
    tl_aliases: set[str] = set()
    jit_aliases: set[str] = set()
    forbidden_imported_symbols: dict[str, str] = {}
    allowed_direct_torch_symbols: set[str] = set()
    module_mutable_globals: set[str] = set()

    # 1. Inspect ALL imports anywhere in the AST (top-level and nested)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                as_name = alias.asname or name.split(".")[0]
                if name not in ALLOWED_IMPORT_MODULES:
                    violations.append(f"Forbidden import '{name}'. Only {sorted(ALLOWED_IMPORT_MODULES)} are allowed.")
                    continue
                if name == "torch":
                    torch_aliases.add(as_name)
                elif name == "triton":
                    triton_aliases.add(as_name)
                elif name == "triton.language":
                    tl_aliases.add(as_name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level and node.level > 0:
                violations.append("Relative imports are forbidden in candidate kernels.")
                continue
            if mod not in ALLOWED_IMPORT_MODULES:
                violations.append(f"Forbidden from-import '{mod}'. Only {sorted(ALLOWED_IMPORT_MODULES)} are allowed.")
                continue
            for alias in node.names:
                if alias.name == "*":
                    violations.append(f"Wildcard import from '{mod}' is forbidden.")
                    continue
                as_name = alias.asname or alias.name
                if mod == "triton" and alias.name == "language":
                    tl_aliases.add(as_name)
                elif mod == "triton" and alias.name == "jit":
                    jit_aliases.add(as_name)
                elif mod == "torch":
                    if alias.name in ALLOWED_HOST_TORCH_ATTRS:
                        allowed_direct_torch_symbols.add(as_name)
                    else:
                        forbidden_imported_symbols[as_name] = f"torch.{alias.name}"
                        violations.append(f"Forbidden host torch symbol import: 'from torch import {alias.name}'.")

    # 2. Identify module-level mutable containers (memoization guard) & variable aliases to torch
    for stmt in tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]):
                if isinstance(target, ast.Name):
                    if isinstance(stmt.value, (ast.Dict, ast.List, ast.Set)):
                        module_mutable_globals.add(target.id)
                    if isinstance(stmt.value, ast.Name) and stmt.value.id in torch_aliases:
                        torch_aliases.add(target.id)
                    if isinstance(stmt.value, ast.Attribute) and isinstance(stmt.value.value, ast.Name):
                        if stmt.value.value.id in torch_aliases and stmt.value.attr not in ALLOWED_HOST_TORCH_ATTRS:
                            forbidden_imported_symbols[target.id] = f"torch.{stmt.value.attr}"
                            violations.append(f"Forbidden alias to torch.{stmt.value.attr} via '{target.id}'.")

    # 3. Partition top-level functions into @triton.jit kernels vs host functions
    shadowed = {arg.arg for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                for arg in node.args.args}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            violations.append("Candidate classes and class-held caches are outside the supported source policy.")
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            violations.append("Attribute assignment can persist memoized outputs and is forbidden.")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in FORBIDDEN_BUILTINS and node.id not in shadowed:
                violations.append(f"Reference to forbidden builtin '{node.id}'.")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in torch_aliases and node.attr not in ALLOWED_HOST_TORCH_ATTRS:
                violations.append(f"Disallowed torch attribute '{node.attr}'.")
    jit_kernel_nodes: dict[str, ast.FunctionDef] = {}
    valid_jit_kernels: set[str] = set()
    host_functions: dict[str, ast.FunctionDef] = {}

    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            if any(_is_triton_jit_decorator(d, triton_aliases, jit_aliases) for d in stmt.decorator_list):
                jit_kernel_nodes[stmt.name] = stmt
                ok_jit, jit_v = _inspect_jit_kernel_body(stmt, torch_aliases, tl_aliases)
                violations.extend(jit_v)
                if ok_jit:
                    valid_jit_kernels.add(stmt.name)
            else:
                host_functions[stmt.name] = stmt
                for d in stmt.decorator_list:
                    dname = _dotted_name(d.func if isinstance(d, ast.Call) else d) or ""
                    if "cache" in dname.lower():
                        violations.append(f"Memoization decorator '@{dname}' is forbidden on '{stmt.name}'.")

    if not jit_kernel_nodes:
        violations.append("Candidate must define at least one real `@triton.jit` kernel.")
    if "kernel_forward" not in host_functions:
        violations.append("Candidate must define a top-level host entrypoint `def kernel_forward(...)`.")

    # 4. Inspect all host functions (including nested helper functions) for:
    #    - torch fallback calls / aliases
    #    - host tensor arithmetic / disallowed tensor methods
    #    - memoization / globals / dunder escapes / forbidden builtins
    #    - real Triton grid launches & call graph from kernel_forward
    host_launches: dict[str, set[str]] = {}
    host_calls: dict[str, set[str]] = {}

    def _inspect_host_fn(fn: ast.FunctionDef, inherited_torch_aliases: set[str], inherited_bad_syms: dict[str, str]) -> tuple[set[str], set[str]]:
        if any(isinstance(default, (ast.Dict, ast.List, ast.Set, ast.Call))
               for default in [*fn.args.defaults, *fn.args.kw_defaults] if default is not None):
            violations.append(f"Host function '{fn.name}' has a persistent mutable/default-call argument.")
        local_torch = set(inherited_torch_aliases)
        local_bad = dict(inherited_bad_syms)
        launched: set[str] = set()
        called_hosts: set[str] = set()
        tensor_vars: set[str] = set()

        # Mark non-scalar parameters of kernel_forward / helpers as tensor-tainted
        n_args = len(fn.args.args)
        n_defs = len(fn.args.defaults)
        def_offset = n_args - n_defs
        for idx, arg in enumerate(fn.args.args):
            default_node = fn.args.defaults[idx - def_offset] if idx >= def_offset else None
            if not _is_scalar_param(arg, default_node):
                tensor_vars.add(arg.arg)

        def _is_tensor_expr(expr: ast.AST) -> bool:
            if isinstance(expr, ast.Name):
                return expr.id in tensor_vars
            if isinstance(expr, ast.Call):
                if isinstance(expr.func, ast.Attribute) and _is_tensor_expr(expr.func.value):
                    return expr.func.attr in {"contiguous", "reshape", "view", "unsqueeze", "squeeze", "flatten"}
                if isinstance(expr.func, ast.Attribute) and isinstance(expr.func.value, ast.Name):
                    if expr.func.value.id in local_torch and expr.func.attr in ALLOWED_HOST_TORCH_CALLS:
                        return True
                    if expr.func.value.id in tensor_vars and expr.func.attr in {
                        "contiguous", "reshape", "view", "unsqueeze", "squeeze", "flatten"
                    }:
                        return True
            if isinstance(expr, ast.Attribute):
                return expr.attr not in {"shape", "dtype", "device", "ndim", "is_cuda"} and _is_tensor_expr(expr.value)
            if isinstance(expr, ast.Subscript):
                return _is_tensor_expr(expr.value)
            return False

        def _visit_stmt(stmt: ast.AST) -> None:
            if isinstance(stmt, ast.FunctionDef):
                # Nested helper function (either nested @triton.jit or nested host helper)
                if any(_is_triton_jit_decorator(d, triton_aliases, jit_aliases) for d in stmt.decorator_list):
                    jit_kernel_nodes[stmt.name] = stmt
                    ok_jit, jit_v = _inspect_jit_kernel_body(stmt, local_torch, tl_aliases)
                    violations.extend(jit_v)
                    if ok_jit:
                        valid_jit_kernels.add(stmt.name)
                else:
                    sub_l, sub_c = _inspect_host_fn(stmt, local_torch, local_bad)
                    host_launches[f"{fn.name}.{stmt.name}"] = sub_l
                    host_calls[f"{fn.name}.{stmt.name}"] = sub_c
                    launched.update(sub_l)
                    called_hosts.update(sub_c)
                return

            if isinstance(stmt, ast.Global):
                violations.append(f"Host function '{fn.name}' uses forbidden `global` statement.")
                return

            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                # Track alias assignments like `t = torch` or `mm = torch.matmul` or tensor vars
                rhs = stmt.value
                for tgt in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]):
                    if isinstance(tgt, ast.Name):
                        if isinstance(rhs, ast.Name) and rhs.id in local_torch:
                            local_torch.add(tgt.id)
                        elif isinstance(rhs, ast.Attribute) and isinstance(rhs.value, ast.Name) and rhs.value.id in local_torch:
                            if rhs.attr not in ALLOWED_HOST_TORCH_ATTRS:
                                local_bad[tgt.id] = f"torch.{rhs.attr}"
                                violations.append(f"Host function '{fn.name}' aliases forbidden torch.{rhs.attr} as '{tgt.id}'.")
                        elif _is_tensor_expr(rhs):
                            tensor_vars.add(tgt.id)
                    elif isinstance(tgt, ast.Tuple):
                        # e.g., q3, k3 = q.reshape(...), k.reshape(...)
                        if isinstance(rhs, ast.Tuple) and len(tgt.elts) == len(rhs.elts):
                            for te, re_val in zip(tgt.elts, rhs.elts):
                                if isinstance(te, ast.Name) and _is_tensor_expr(re_val):
                                    tensor_vars.add(te.id)
                    elif isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                        if tgt.value.id in module_mutable_globals:
                            violations.append(f"Host function '{fn.name}' writes to global memoization container '{tgt.value.id}'.")

            # Walk expressions inside this statement (skipping nested FunctionDefs)
            for node in ast.walk(stmt):
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                    violations.append(f"Host function '{fn.name}' performs a subscript write outside Triton.")
                if node is not stmt and isinstance(node, ast.FunctionDef):
                    continue
                if isinstance(node, ast.Attribute):
                    if node.attr.startswith("__") and node.attr.endswith("__"):
                        violations.append(f"Host function '{fn.name}' accesses dunder attribute '{node.attr}'.")
                    if isinstance(node.value, ast.Name) and node.value.id in local_torch:
                        if node.attr not in ALLOWED_HOST_TORCH_ATTRS:
                            violations.append(f"Host function '{fn.name}' accesses disallowed torch.{node.attr}.")
                elif isinstance(node, ast.BinOp):
                    if isinstance(node.op, ast.MatMult):
                        violations.append(f"Host function '{fn.name}' performs forbidden host matrix multiplication (@).")
                    elif _is_tensor_expr(node.left) or _is_tensor_expr(node.right):
                        violations.append(
                            f"Host function '{fn.name}' performs forbidden host tensor arithmetic ({type(node.op).__name__})."
                        )
                elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
                    if _is_tensor_expr(node.operand):
                        violations.append(f"Host function '{fn.name}' performs forbidden unary tensor op.")
                elif isinstance(node, ast.Call):
                    # Check Triton kernel grid launch: `_kernel[grid](...)`
                    if isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name):
                        kname = node.func.value.id
                        if kname in jit_kernel_nodes:
                            launched.add(kname)
                    elif isinstance(node.func, ast.Name):
                        cname = node.func.id
                        if cname in FORBIDDEN_BUILTINS:
                            violations.append(f"Host function '{fn.name}' calls forbidden builtin '{cname}()'.")
                        elif cname in local_bad:
                            violations.append(f"Host function '{fn.name}' calls forbidden aliased op '{cname}' ({local_bad[cname]}).")
                        elif cname in host_functions:
                            called_hosts.add(cname)
                    elif isinstance(node.func, ast.Attribute):
                        attr = node.func.attr
                        if isinstance(node.func.value, ast.Name):
                            owner = node.func.value.id
                            if owner in module_mutable_globals and attr in {"append", "update", "setdefault", "extend", "add", "clear"}:
                                violations.append(f"Host function '{fn.name}' mutates global container '{owner}.{attr}()'.")
                            elif owner in local_torch:
                                if attr not in ALLOWED_HOST_TORCH_CALLS:
                                    violations.append(f"Host function '{fn.name}' calls disallowed host op 'torch.{attr}()'.")
                            elif owner in triton_aliases:
                                if attr not in ALLOWED_TRITON_HOST_CALLS:
                                    violations.append(f"Host function '{fn.name}' calls disallowed triton.{attr}().")
                            elif owner in tensor_vars:
                                if attr not in ALLOWED_HOST_TENSOR_METHODS:
                                    violations.append(
                                        f"Host function '{fn.name}' calls disallowed tensor method '{owner}.{attr}()'."
                                    )
                        elif _is_tensor_expr(node.func.value) and attr not in ALLOWED_HOST_TENSOR_METHODS:
                            violations.append(f"Host function '{fn.name}' calls disallowed chained tensor method '.{attr}()'.")

        for body_stmt in fn.body:
            _visit_stmt(body_stmt)
        return launched, called_hosts

    for fname, fnode in host_functions.items():
        lset, cset = _inspect_host_fn(fnode, torch_aliases, forbidden_imported_symbols)
        host_launches[fname] = lset
        host_calls[fname] = cset

    # Compute kernels transitively launched from `kernel_forward`
    reachable_launches: set[str] = set()
    if "kernel_forward" in host_functions:
        visited: set[str] = set()
        stack = ["kernel_forward"]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            reachable_launches.update(host_launches.get(cur, set()))
            for nxt in host_calls.get(cur, set()):
                if nxt not in visited:
                    stack.append(nxt)

    valid_launched = sorted(reachable_launches & valid_jit_kernels)
    if not valid_launched:
        violations.append(
            "Host `kernel_forward` does not launch a valid load/store Triton kernel (dummy/no-op or missing launch)."
        )

    # Deduplicate violations preserving order
    unique_violations = list(dict.fromkeys(violations))
    return GpuSourcePolicyResult(
        allowed=len(unique_violations) == 0,
        violations=unique_violations,
        jit_kernels=sorted(jit_kernel_nodes.keys()),
        launched_kernels=valid_launched,
    )
