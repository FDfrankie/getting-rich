"""Regression tests for P27 (BRAIN Bronze Guardrails + Remaining Delta Fixes)."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_bronze_neutralization_sanitization_and_operator_whitelist(monkeypatch):
    monkeypatch.delenv("BRAIN_CONSULTANT_MODE", raising=False)
    from worldquant_brain.multi_sim_client import (
        BRONZE_UNLOCKED_OPERATORS,
        build_sim_payload,
        find_locked_bronze_operators,
        sanitize_sim_settings,
    )
    from worldquant_brain.two_stage_funnel import STAGE1_PROBES, STAGE2_NEUTRALIZATIONS

    assert len(BRONZE_UNLOCKED_OPERATORS) == 60  # 66 total minus binary symbol operators
    for bad in ("CROWDING", "STATISTICAL", "SLOW", "FAST", "REVERSION_AND_MOMENTUM"):
        payload = build_sim_payload("rank(close)", {"neutralization": bad})
        assert payload["settings"]["neutralization"] in {"NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"}

    assert sanitize_sim_settings({"neutralization": "CROWDING"}, dataset_id="news18")["neutralization"] == "SUBINDUSTRY"
    assert sanitize_sim_settings({"neutralization": "STATISTICAL"}, dataset_id="fundamental6")["neutralization"] == "INDUSTRY"
    assert sanitize_sim_settings({"neutralization": "SLOW"}, dataset_id="option9")["neutralization"] == "SECTOR"

    # Every Stage-1 probe must use ONLY Bronze-unlocked operators
    for name, tmpl in STAGE1_PROBES:
        expr = tmpl.format(f="close")
        locked = find_locked_bronze_operators(expr)
        assert not locked, f"Stage1 probe {name} uses locked Bronze operators: {locked}"

    assert "SLOW" not in STAGE2_NEUTRALIZATIONS
    assert "FAST" not in STAGE2_NEUTRALIZATIONS


def test_canonical_ast_commutative_sorting():
    from worldquant_brain.canonical_ast import parse_fastexpr

    a = parse_fastexpr("rank(close) + rank(volume)")
    b = parse_fastexpr("rank(volume) + rank(close)")
    assert a.canonical_expression == b.canonical_expression

    c = parse_fastexpr("add(close, open)")
    d = parse_fastexpr("add(open, close)")
    assert c.canonical_expression == d.canonical_expression


def test_alpha_lifecycle_submit_failed_retry_and_rejected_terminal():
    from worldquant_brain.alpha_lifecycle import AlphaState, can_transition

    assert can_transition(AlphaState.SUBMITTING, AlphaState.SUBMIT_FAILED)
    assert can_transition(AlphaState.SUBMIT_FAILED, AlphaState.ELIGIBLE)
    assert not can_transition(AlphaState.REJECTED, AlphaState.SIMULATION_QUEUED)
    assert can_transition(AlphaState.REJECTED, AlphaState.ARCHIVED)


def test_trial_registry_dsr_correlated_neff(tmp_path):
    from worldquant_brain.trial_registry_dsr import TrialRegistry, deflated_sharpe_ratio

    uncorr = deflated_sharpe_ratio(1.8, 50, avg_trial_corr=0.0)
    corr = deflated_sharpe_ratio(1.8, 50, avg_trial_corr=0.8)
    assert corr.expected_max_sharpe < uncorr.expected_max_sharpe
    reg = TrialRegistry(tmp_path / "trials.db")
    assert Path(reg.db_path).exists()


def test_brain_feature_bridge_causal_ceil(tmp_path):
    from crypto_trading.brain_feature_bridge import BrainFeatureBridge

    db = tmp_path / "exp.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE alpha_experiments (
            completed_at TEXT, sharpe REAL, fitness REAL, turnover REAL,
            margin REAL, self_correlation REAL, simulation_status TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO alpha_experiments VALUES ('2026-01-01T22:00:00Z', 1.6, 1.2, 0.15, 0.002, 0.2, 'SUBMITTED')"
    )
    conn.commit()
    conn.close()

    bridge = BrainFeatureBridge(db, lag="1D")
    ctx = bridge.daily_context()
    assert pd.Timestamp(ctx["date"].iloc[0]) == pd.Timestamp("2026-01-03T00:00:00Z")


def test_lgbm_ic_wrapper_single_row_and_all_nan():
    from crypto_trading.lgbm_validation_gate import ICStabilityLinearWrapper

    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))
    X[:, 2] = np.nan
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.1, size=80)

    class _DummyBase:
        def fit(self, X, y):
            self.w = np.ones(X.shape[1]) * 0.1
        def predict(self, X):
            return X @ self.w

    wrap = ICStabilityLinearWrapper(base_factory=_DummyBase, blend_alpha=0.5).fit(X, y)
    single = wrap.predict(np.array([[1.0, -0.5, np.nan, 0.2]]))
    assert single.shape == (1,)
    assert np.isfinite(single[0]) and abs(single[0]) < 100.0


def test_meta_labeling_exact_dsr_and_correlated_neff():
    from crypto_trading.meta_labeling import compute_deflated_sharpe_and_pbo

    rng = np.random.default_rng(7)
    base = rng.normal(0.001, 0.01, size=(120, 1))
    # 3 nearly identical trials -> N_eff ~ 1.0
    mat_corr = np.hstack([base, base + rng.normal(0, 1e-4, size=(120, 1)), base + rng.normal(0, 1e-4, size=(120, 1))])
    res = compute_deflated_sharpe_and_pbo(mat_corr)
    assert 0.0 <= res["dsr_prob"] <= 1.0


def test_rl_agent_nan_state_and_single_sample_train():
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"torch unavailable: {exc}")
    from crypto_trading.rl_agent import PPOAgent

    agent = PPOAgent(state_dim=4, action_dim=1)
    act, lp, val = agent.select_action(np.array([np.nan, np.inf, -np.inf, 0.5]))
    assert math.isfinite(act) and -1.0 <= act <= 1.0
    assert math.isfinite(lp) and math.isfinite(val)

    losses = agent.train_step(
        states=[np.zeros(4, dtype=np.float32)],
        actions=[0.1],
        old_log_probs=[-0.5],
        advantages=torch.tensor([1.0]),
        returns=torch.tensor([1.0]),
        k_epochs=1,
        batch_size=1,
    )
    assert all(math.isfinite(v) for v in losses.values())


def test_risk_allocator_states_and_reduce_only_cap():
    from portfolio.risk_allocator import PortfolioRiskAllocator

    halted = PortfolioRiskAllocator(total_equity_usdt=10000.0, trading_state="HALTED")
    assert halted.evaluate_allocation("BTCUSDT", 500.0, {"BTCUSDT": 1000.0}).reason == "TRADING_STATE_HALTED"

    active = PortfolioRiskAllocator(total_equity_usdt=10000.0, soft_drawdown_pct=0.08, hard_drawdown_pct=0.15)
    assert active.evaluate_allocation("BTCUSDT", 500.0, {}, is_reduce_only=True).reason == "REDUCE_ONLY_NO_POSITION"
    red = active.evaluate_allocation("BTCUSDT", 2000.0, {"BTCUSDT": 800.0}, is_reduce_only=True)
    assert red.allowed and red.approved_notional_usdt == 800.0
    assert active.evaluate_allocation("BTCUSDT", 500.0, {}, current_drawdown_pct=0.16).reason == "DRAWDOWN_OR_REDUCING_BLOCK_NEW_RISK"
    soft = active.evaluate_allocation("BTCUSDT", 4000.0, {}, current_drawdown_pct=0.10)
    assert soft.allowed and soft.approved_notional_usdt == 2500.0  # 0.5 * 0.50 * 10000


def test_taiwan_ranker_empty_rows_and_daily_ic():
    from taiwan_stock.ml_ranker import TWSE_STANDARD_ROUND_TRIP_COST, TaiwanStockMLRanker

    assert math.isclose(TWSE_STANDARD_ROUND_TRIP_COST, 0.00585, rel_tol=1e-9)
    ranker = TaiwanStockMLRanker()
    ranker.model = object()
    ranker.feature_cols = ["mom_5d"]
    empty_df = ranker.predict_rankings({})
    assert empty_df.empty and list(empty_df.columns) == [
        "symbol", "current_price", "predicted_fwd_5d_ret_pct", "bias_ma20", "rsi_14", "rank"
    ]
