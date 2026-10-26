"""Isolated router-only scheduling prototype. Import/build never happens on import.

Not a judge submission: the trial loads the hash-pinned shipped source in this repo.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIPPED = HERE.parent / "pw35" / "submission.py"
SHIPPED_SHA256 = "7069218da32202825be1aada89d951cb78c57c166b4f177195c055bbb33632e8"
CONFIGS = {
    200: (512, 6, 128, 128, 1, 4),
    201: (256, 3, 128, 128, 2, 4),
    210: (256, 4, 128, 128, 1, 4),
    300: (512, 6, 128, 128, 1, 132),
    301: (256, 3, 128, 128, 2, 132),
}
PLANS = {
    128: {"baseline": (201, 210), "candidate": 301},
    256: {"baseline": (200, 201), "candidate": 300},
}
ANCHOR = "        rt_tile(dst + (TOKS + 64) * 128, &tl, k0 + 64 * s, n0, fb);\n"
INSERT = (
    "        // rtk_early: one startup gate, then the identical steady-state ring.\n"
    "        if constexpr ((MODE & 128) != 0) {\n"
    "          if (s == 0) rt_mbwait(fb, 0u);\n"
    "        }\n"
)


def shipped_source():
    src = SHIPPED.read_text()
    got = hashlib.sha256(src.encode()).hexdigest()
    if got != SHIPPED_SHA256:
        raise RuntimeError(f"Frozen parent changed: {got}")
    return src


def gate_source(cuda):
    assert cuda.count(ANCHOR) == 1, "Producer anchor changed"
    result = cuda.replace(ANCHOR, ANCHOR + INSERT)
    assert result.replace(INSERT, "") == cuda
    return result


def source_snapshot():
    """Pure Python, no torch import / CUDA initialization / compilation."""
    src = shipped_source()
    tree = ast.parse(src)
    cuda = next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RWCU" for t in n.targets))
    return cuda, gate_source(cuda)


def load():
    shipped_source()
    spec = importlib.util.spec_from_file_location("rtk_early_frozen_pw35", SHIPPED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for v in (200, 201, 210):
        assert mod.RTV[v] == CONFIGS[v]
    mod.RWCU = gate_source(mod.RWCU)
    mod.RWV = {}  # no old rwk instance; exactly this trial's five rtk instances
    mod.RTV = dict(CONFIGS)
    mod.RTV.update({v + 20000: c[:5] + (c[5] | 16,) for v, c in CONFIGS.items()})
    mod.RSPALL = [(f"rtk{v}", v, False, (128, 256)) for v in CONFIGS]
    mod.RSP = list(mod.RSPALL)
    mod.RWALL = True
    return mod
