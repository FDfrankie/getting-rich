import unittest

from crypto_trading.backtest_engine import FeeSchedule
from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.brain_job_queue import BrainJobQueue


class TestPhase4ThroughputAndFees(unittest.TestCase):
    def test_multi_worker_brain_job_queue(self):
        c = BrainClient(email=None, password=None)
        q = BrainJobQueue(c, autostart=True, num_workers=3)
        try:
            self.assertEqual(q.num_workers, 3)
            self.assertTrue(q.worker_alive())
        finally:
            q.stop()

    def test_alpha_miner_101_and_settings_rotation(self):
        miner = ContinuousAlphaMiner()
        s = miner.generate_candidate_settings()
        self.assertIn("neutralization", s)
        self.assertIn("decay", s)
        self.assertIn("truncation", s)
        exprs = [miner.generate_candidate_expression()["expression"] for _ in range(100)]
        self.assertTrue(any("vwap" in e or "high - low" in e for e in exprs))

    def test_venue_accurate_fee_schedules(self):
        b_usdc = FeeSchedule.for_venue("BINANCE_USDC_FUTURES")
        self.assertEqual(b_usdc.maker_fee_rate, 0.0)
        self.assertEqual(b_usdc.taker_fee_rate, 0.0004)
        bitunix = FeeSchedule.for_venue("BITUNIX_FUTURES")
        self.assertEqual(bitunix.maker_fee_rate, 0.0002)
        self.assertEqual(bitunix.taker_fee_rate, 0.0006)


if __name__ == "__main__":
    unittest.main()
