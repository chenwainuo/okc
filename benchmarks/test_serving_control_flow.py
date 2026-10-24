"""CPU tests of the actual Marlin forward control flow, with GPU calls replaced.

Extracts the method AST so these tests do not import CUDA-only SGLang modules.
These establish dispatch/initialization safety, not numerical GPU correctness.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "serving/sglang/python/sglang/srt/layers"
          / "muse_spark_fp4_moe.py")


def forward_function(events):
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "MuseSpark1xFp4Experts")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_forward_marlin")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), method], type_ignores=[])
    ast.fix_missing_locations(module)

    def fused(*args, **kwargs):
        events.append(("fused", args, kwargs))
        return "fused-output"

    def fallback(**kwargs):
        events.append(("fallback", kwargs))
        return "fallback-output"

    namespace = {"fused_post_expert_norm_weighted_sum": fused,
                 "finalize_identity_with_post_expert_norm": fallback}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["_forward_marlin"]


class MarlinForwardDispatchTest(unittest.TestCase):
    def run_case(self, effective_weight, *, stale_weight=None, pre_norm=None):
        events = []
        norm = object()
        layer = SimpleNamespace(_post_expert_norm=None,
                                _post_expert_norm_weight=stale_weight,
                                _post_expert_norm_eps=1e-8,
                                _post_expert_norm_weight_first=True)

        def bind(value):
            events.append(("bind", value))
            layer._post_expert_norm = value
            layer._post_expert_norm_weight = effective_weight

        def experts(hidden, scores, ids, *, skip_nonlocal=False):
            events.append(("experts", skip_nonlocal, hidden))
            return "expert-outputs", "local-ids"

        layer.set_post_expert_norm = bind
        layer._marlin_expert_outputs = experts
        hidden = SimpleNamespace(dtype="bf16")
        result = forward_function(events)(layer, hidden, "scores", "ids", norm, pre_norm)
        return events, result

    def test_original_expert_path_uses_fused_finalize(self):
        events, result = self.run_case("weight")
        self.assertEqual([e[0] for e in events], ["experts", "bind", "fused"])
        self.assertFalse(events[0][1])
        self.assertEqual(events[2][2]["slot_expert_ids"], "local-ids")
        self.assertTrue(events[2][2]["fuse_small_tokens"])
        self.assertEqual(result, "fused-output")

    def test_unmasked_fallback_keeps_nonlocal_blocks(self):
        events, result = self.run_case(None)
        self.assertFalse(events[0][1])
        self.assertEqual(result, "fallback-output")

    def test_changed_norm_does_not_use_stale_skip_decision(self):
        events, result = self.run_case(None, stale_weight="old-weight")
        self.assertFalse(events[0][1])
        self.assertEqual(result, "fallback-output")

    def test_pre_norm_result_reaches_experts(self):
        transformed = SimpleNamespace(dtype="bf16")
        events, _ = self.run_case("weight", pre_norm=lambda x: transformed)
        self.assertIs(events[0][2], transformed)

    def test_missing_post_norm_is_rejected(self):
        with self.assertRaises(ValueError):
            forward_function([])(SimpleNamespace(), None, None, None, None)


if __name__ == "__main__":
    unittest.main()
