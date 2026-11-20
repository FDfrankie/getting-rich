"""Unit tests for Crypto Cost-Aware Filter, Timeout-Safe Order FSM, Central Rate-Limit Budget, and Datafield Profiler."""
from __future__ import annotations

import unittest

from crypto_trading.execution_cost_filter import (
    CentralRateLimitBudget,
    OrderRecord,
    OrderState,
    evaluate_cost_barrier,
)
from worldquant_brain.datafield_profiler import DatafieldProfile, qualify_datafield


class TestP0CostFSMAndProfiler(unittest.TestCase):
    def test_cost_barrier_blocks_gross_positive_but_net_negative_trade(self) -> None:
        # 10 bps gross forecast vs 12 bps fee + 2 bps slippage + 1 bp spread -> blocked!
        dec = evaluate_cost_barrier(expected_return=0.0010, explicit_fee=0.0012, spread_cost=0.0001, slippage=0.0002)
        self.assertFalse(dec.trade_allowed)
        self.assertLess(dec.cost_estimate.net_edge, 0.0)

        # 45 bps gross forecast clearly clears 1.25x cost barrier -> allowed!
        dec_ok = evaluate_cost_barrier(expected_return=0.0045, explicit_fee=0.0012, spread_cost=0.0001, slippage=0.0002, funding=0.0001, uncertainty_buffer=0.0002, minimum_edge_multiple=1.25)
        self.assertTrue(dec_ok.trade_allowed)

    def test_order_fsm_timeout_prevents_duplicate_order_until_reconciled(self) -> None:
        order = OrderRecord(client_order_id="cid_1", symbol="BTC/USDT", side="BUY", qty=0.01)
        order.transition(OrderState.SENT, "post_order")
        order.on_timeout("-1007")
        self.assertEqual(order.state, OrderState.UNKNOWN)
        self.assertFalse(order.can_create_replacement_order)

        order.begin_reconciliation()
        self.assertEqual(order.state, OrderState.RECONCILING)
        self.assertFalse(order.can_create_replacement_order)

        # Only after exchange confirms CANCELLED/REJECTED can a replacement be created
        order.transition(OrderState.CANCELLED, "confirmed_absent_on_exchange")
        self.assertTrue(order.can_create_replacement_order)

    def test_central_rate_limit_budget_enforces_429_and_418(self) -> None:
        budget = CentralRateLimitBudget("BINANCE", weight_limit_per_min=1200)
        ok, _ = budget.can_spend(10, now=100.0)
        self.assertTrue(ok)

        budget.record_response(status_code=429, retry_after_seconds=45.0, now=100.0)
        ok_429, wait_429 = budget.can_spend(1, now=110.0)
        self.assertFalse(ok_429)
        self.assertAlmostEqual(wait_429, 35.0)

        budget.record_response(status_code=418, retry_after_seconds=300.0, now=200.0)
        ok_418, wait_418 = budget.can_spend(1, now=250.0)
        self.assertFalse(ok_418)
        self.assertAlmostEqual(wait_418, 250.0)

    def test_datafield_profiler_routes_windows_and_blocks_unverified_pit(self) -> None:
        slow_fund = DatafieldProfile(
            field_id="cashflow_op", dataset_id="fundamental6", update_interval_median_days=90.0, coverage=0.85, point_in_time_verified=True
        )
        res_f = qualify_datafield(slow_fund)
        self.assertTrue(res_f.promotion_eligible)
        self.assertEqual(res_f.recommended_windows, (63, 126, 252))
        self.assertFalse(res_f.allow_short_backfill)

        vec_unverified = DatafieldProfile(
            field_id="news_sent_vec", dataset_id="news12", field_type="VECTOR", coverage=0.45, point_in_time_verified=False
        )
        res_v = qualify_datafield(vec_unverified)
        self.assertTrue(res_v.requires_vector_reducer)
        self.assertFalse(res_v.promotion_eligible)
        self.assertEqual(res_v.sub_universe_risk, "HIGH")


if __name__ == "__main__":
    unittest.main()
