"""Static rules check for a muse_bf16_proj submission. This is not a timing result."""

from __future__ import annotations

import ast
from pathlib import Path

HEADER = "# okc: problem=muse_bf16_proj"

FORBIDDEN_CALLS = {
    "item": "host sync",
    "cpu": "host sync",
    "synchronize": "host sync",
    "cudaDeviceSynchronize": "host sync",
    "cudaStreamSynchronize": "host sync",
}

FORBIDDEN_MODULES = {
    "reference",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "ctypes",
}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _walk_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def check_source(source: str, path: str = "<submission>") -> dict:
    findings = []
    lines = source.splitlines()
    if not lines or lines[0].strip() != HEADER:
        findings.append({"severity": "error", "rule": "header", "detail": f"first line must be {HEADER!r}"})
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return {
            "path": path,
            "ok": False,
            "findings": [{"severity": "error", "rule": "syntax", "detail": str(error)}],
        }

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for name in sorted(imported & FORBIDDEN_MODULES):
        findings.append({"severity": "error", "rule": "import", "detail": f"imports {name}"})

    defines = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    for required in ("prepare", "kernel_fn"):
        if required not in defines:
            findings.append({"severity": "error", "rule": "entrypoint", "detail": f"missing {required}"})

    kernel = _walk_function(tree, "kernel_fn")
    if kernel is not None:
        for node in ast.walk(kernel):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in FORBIDDEN_CALLS:
                    findings.append({
                        "severity": "error",
                        "rule": "graph",
                        "detail": f"kernel_fn calls {name} ({FORBIDDEN_CALLS[name]})",
                    })
                if name == "load_inline":
                    findings.append({
                        "severity": "error",
                        "rule": "graph",
                        "detail": "kernel_fn calls load_inline; compile in prepare",
                    })
                if name == "print":
                    findings.append({
                        "severity": "error",
                        "rule": "graph",
                        "detail": "kernel_fn prints; printing a tensor syncs",
                    })

    # pw24 lesson: wgmma / setmaxnreg / multicast only assemble for sm_90a; a load_inline build without a
    # compute_90a gencode flag fails on the judge and takes the whole submission down.
    arch_only = ("wgmma.", "setmaxnreg", ".multicast::cluster")
    if any(token in source for token in arch_only):
        if "compute_90a" not in source:
            findings.append({"severity": "error", "rule": "sm_90a",
                             "detail": "uses wgmma/setmaxnreg/multicast but no build passes -gencode=arch=compute_90a,code=sm_90a"})
        elif source.count("load_inline(") > 1:
            findings.append({"severity": "warning", "rule": "sm_90a",
                             "detail": "several load_inline builds: confirm the one holding wgmma/setmaxnreg/multicast gets the sm_90a flags"})

    # pw25 lesson: the judge rejects uploads larger than 1,000,000 bytes before any check is consumed.
    size = len(source.encode("utf-8"))
    if size > 1_000_000:
        findings.append({"severity": "error", "rule": "size", "detail": f"{size} bytes > 1,000,000 upload limit"})
    elif size > 950_000:
        findings.append({"severity": "warning", "rule": "size", "detail": f"{size} bytes, within 5% of the 1,000,000 limit"})

    errors = [item for item in findings if item["severity"] == "error"]
    return {"path": path, "ok": not errors, "findings": findings}


def check_file(path: Path) -> dict:
    return check_source(Path(path).read_text(), str(path))
