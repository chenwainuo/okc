import unittest
from optimize_loop import gate


class GateTests(unittest.TestCase):
    def report(self, intervals, status='complete', pairs=20):
        return dict(status=status, pairs=pairs, results=[
            dict(bs=bs, bootstrap_95=ci) for bs, ci in intervals.items()])

    def test_requires_complete_coverage_and_samples(self):
        for report in [self.report({1: [1.04, 1.1]}, status='measuring'),
                       self.report({1: [1.04, 1.1]}, pairs=2),
                       self.report({1: [1.04, 1.1]})]:
            self.assertFalse(gate(report, [1, 2], [1]))

    def test_regression_or_uncertainty_blocks_gain(self):
        self.assertFalse(gate(self.report({1: [1.04, 1.1], 2: [.95, 1.05]}), [1, 2], [1]))
        self.assertFalse(gate(self.report({1: None}), [1], [1]))

    def test_gain_must_be_in_target(self):
        report = self.report({1: [.99, 1.01], 2: [1.04, 1.1]})
        self.assertFalse(gate(report, [1, 2], [1]))
        self.assertTrue(gate(report, [1, 2], [2]))
        self.assertTrue(gate(report, [1, 2], [1], require_gain=False))


if __name__ == '__main__':
    unittest.main()
