import unittest
import torch
from diagnostics import routing_summary


class RoutingDiagnosticsTest(unittest.TestCase):
    def test_global_ids_are_mapped_to_local_range(self):
        result = routing_summary({"topk_ids": torch.tensor([[1, 2, 7], [7, 8, 9]]),
                                  "config": {"local_start": 1, "local_count": 2}})
        self.assertEqual(result["routes_per_local_expert"], [1, 1])
        self.assertEqual(result["tokens_without_local_routes"], 1)
        self.assertEqual(result["local_routes"], 2)
        self.assertAlmostEqual(result["local_fraction"], 1 / 3)

    def test_no_local_routes(self):
        result = routing_summary({"topk_ids": torch.tensor([[8, 9]]),
                                  "config": {"local_start": 0, "local_count": 2}})
        self.assertEqual(result["distinct_local_experts"], 0)
        self.assertEqual(result["routes_per_local_expert"], [0, 0])


if __name__ == "__main__":
    unittest.main()
