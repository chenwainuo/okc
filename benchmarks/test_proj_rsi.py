import tempfile
import unittest
from pathlib import Path

from proj_rsi.correctness import check_submission
from proj_rsi.ledger import Ledger
from proj_rsi.score import (
    FLOORS_US,
    LAYERS,
    floor_layer_us,
    floor_load_us,
    geomean,
    score_from_layer_times,
    speedup_from_load_times,
)
from proj_rsi.static_check import check_source


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "muse_bf16_proj" / "submission.py"


class ScoreTest(unittest.TestCase):
    def test_published_floors_match_the_byte_and_flop_model(self):
        loads = {
            "decode 1": 1,
            "decode 16": 16,
            "decode 32": 32,
            "verify 1": 8,
            "verify 16": 128,
            "verify 32": 256,
        }
        for load, m in loads.items():
            for layer, published in zip(LAYERS, FLOORS_US[load]):
                self.assertAlmostEqual(floor_layer_us(m, layer), published, delta=0.15)

    def test_speedup_is_the_geomean_of_load_ratios(self):
        baseline = {load: 2.0 for load in FLOORS_US}
        candidate = {load: 1.0 for load in FLOORS_US}
        self.assertAlmostEqual(speedup_from_load_times(candidate, baseline), 2.0)
        self.assertAlmostEqual(geomean([1.0, 4.0]), 2.0)

    def test_score_covers_every_load(self):
        by_load = {
            load: {layer: floor_layer_us(m, layer) for layer in LAYERS}
            for load, m in (
                ("decode 1", 1),
                ("decode 16", 16),
                ("decode 32", 32),
                ("verify 1", 8),
                ("verify 16", 128),
                ("verify 32", 256),
            )
        }
        scored = score_from_layer_times(by_load)
        self.assertAlmostEqual(scored["fraction_of_floor"], 1.0, places=6)
        self.assertAlmostEqual(scored["geomean_us"], scored["floor_geomean_us"], places=3)
        # Weight 62 lifts one call into the load. Published per-layer floors are
        # rounded to 0.1 us, so the load sum is a few microseconds off 28.9*62.
        self.assertAlmostEqual(floor_load_us("decode 1"), 28.9 * 62, delta=5.0)


class StaticCheckTest(unittest.TestCase):
    def test_starter_passes(self):
        report = check_source(STARTER.read_text(), str(STARTER))
        self.assertTrue(report["ok"], report)

    def test_graph_sync_in_kernel_fails(self):
        source = STARTER.read_text().replace(
            "return data[\"x\"] @ w.T",
            "torch.cuda.synchronize()\n    return data[\"x\"] @ w.T",
        )
        report = check_source(source)
        self.assertFalse(report["ok"])
        self.assertTrue(any(item["rule"] == "graph" for item in report["findings"]))


class CorrectnessTest(unittest.TestCase):
    def test_starter_smoke_on_cpu(self):
        report = check_submission(STARTER)
        self.assertEqual(report["status"], "passed", report)


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root / "ledger.json")
        self.hypothesis = {"claim": "packed tensor-core GEMM", "target_shapes": ["verify 16"], "kill_if": "pooler m=128 >= 20.3"}

    def test_critic_must_differ_from_author_and_jobs_carry_the_numbers(self):
        self.ledger.propose("pw1", "implementer", "proj_rsi/template/submission.py", self.hypothesis)
        with self.assertRaises(ValueError):
            self.ledger.review("pw1", "implementer", True, "self-approval")
        self.ledger.review("pw1", "critic", True, "ok")
        self.ledger.attach_job("pw1", "job1", "check")
        summary = {"status": "ok", "passed": True, "score_us": 3300.0, "speedup": 1.46, "speedup_ci": [1.44, 1.48], "tests": 9, "tests_passed": 9}
        self.ledger.record_job("pw1", "job1", summary, {"decode 1": 38.0}, "check")
        status = self.ledger.status()
        self.assertEqual(status["experiments"][0]["state"], "measured")
        self.assertEqual(status["jobs"]["job1"]["score_us"], 3300.0)

    def test_failed_job_rejects_the_experiment(self):
        self.ledger.propose("pw1", "implementer", "proj_rsi/template/submission.py", self.hypothesis)
        self.ledger.review("pw1", "critic", True, "ok")
        self.ledger.record_job("pw1", "job2", {"status": "error", "passed": False, "error": "timing fail"}, {}, "check")
        self.assertEqual(self.ledger.status()["experiments"][0]["state"], "rejected")

    def test_hypothesis_needs_claim_targets_and_kill_bar(self):
        with self.assertRaises(ValueError):
            self.ledger.propose("pw1", "implementer", "proj_rsi/template/submission.py", {"claim": "x"})


class BoardMathTest(unittest.TestCase):
    def test_r6_job_reproduces_the_published_score(self):
        from proj_rsi.okc_result import load_job, shapes, speedup_from_table, score_us, ln_gain_needed
        job = load_job("23cecdef1bb54c9e")
        speed = speedup_from_table(shapes(job))
        self.assertAlmostEqual(speed, 1.4092, places=3)
        self.assertAlmostEqual(score_us(speed), 3428.0, delta=0.5)
        self.assertAlmostEqual(ln_gain_needed(speed), 0.80, delta=0.01)


if __name__ == "__main__":
    unittest.main()
