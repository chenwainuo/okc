import unittest
from measurement import summarize


class MeasurementTest(unittest.TestCase):
    def test_insufficient_repetitions_do_not_imply_confidence(self):
        self.assertIsNone(summarize([2, 2], [1, 1])["bootstrap_95"])

    def test_alternating_order_effect_cancels(self):
        result = summarize([4, 1] * 10, [1, 1] * 10)
        self.assertAlmostEqual(result["balanced_speedup"], 2)
        self.assertEqual(result["assessment"], "promising")

    def test_neutral_and_regression(self):
        self.assertEqual(summarize([1] * 20, [1] * 20)["assessment"], "inconclusive")
        self.assertEqual(summarize([1] * 20, [2] * 20)["assessment"], "regression")

    def test_invalid_measurements(self):
        for a, b in (([1], [1]), ([1, 2], [1]), ([0, 1], [1, 1]),
                     ([float("nan"), 1], [1, 1])):
            with self.assertRaises(ValueError):
                summarize(a, b)


if __name__ == "__main__":
    unittest.main()
