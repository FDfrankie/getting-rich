"""Comprehensive offline test matrix for Antigravity Quant Batches A through J."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from crypto_trading.brain_feature_bridge import BrainFeatureBridge
from crypto_trading.hypothesis_runner import evaluate_video_hypothesis, verify_indicator_causality
from crypto_trading.indicator_library import INDICATOR_CATALOG, add_indicators
from crypto_trading.math_models import (
    HierarchicalEmpiricalBayesModel,
    estimate_hurst_with_ci,
    estimate_ou_parameters,
    fractional_kelly_size,
)
from crypto_trading.shadow_trading import ShadowTradingService
from data_persistence.migrations import REQUIRED_TABLES
from data_persistence.repositories import PostgresRepository, SQLiteRepository
from observability.telemetry import build_platform_observability_report
from taiwan_stock.universe_registry import check_fundamental_publication_lag
from worldquant_brain.alpha_lifecycle import AlphaState
from worldquant_brain.alpha_repository import AlphaRepository, compute_ast_family_hash
from worldquant_brain.candidate_utility import evaluate_candidate, turnover_quality_score
from worldquant_brain.daily_candidate_selector import select_daily_candidates, should_use_exploration_slot
from worldquant_brain.direct_submit_service import DirectSubmitService
from worldquant_brain.meta_model import split_family_chronological
from worldquant_brain.score_attribution import (
    build_batch_attribution,
    estimate_days_to_10k,
    score_refresh_at_for_taipei_date,
)
from worldquant_brain.submission_service import SubmissionService


def test_batch_a_and_b_repository_inventory_and_20_tables_and_locks(tmp_path: Path) -> None:
    inv_path = BASE_DIR / "docs" / "repository_inventory.json"
    if inv_path.exists():
        data = json.loads(inv_path.read_text(encoding="utf-8"))
        assert data["total_files"] >= 150
        assert "SOURCE" in data["by_category"]

    repo = SQLiteRepository(tmp_path / "gov.db")
    with repo.connect() as conn:
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    for tbl in REQUIRED_TABLES:
        assert tbl in existing, f"Missing required table: {tbl}"

    # Two concurrent schedulers cannot claim the same daily run
    ok1, run1 = repo.claim_daily_scheduler_run("2026-11-18", "worker_1", "2026-11-18T22:00:00+08:00")
    ok2, run2 = repo.claim_daily_scheduler_run("2026-11-18", "worker_2", "2026-11-18T22:00:00+08:00")
    assert ok1 is True and run1 is not None
    assert ok2 is False and run2 is None

    # Idempotent duplicate expression insertion
    a1 = repo.upsert_alpha({"name": "A1", "expression": "rank(close - open)"})
    a2 = repo.upsert_alpha({"name": "A2", "expression": "rank(close - open)"})
    assert a1["idempotent_duplicate"] is False
    assert a2["idempotent_duplicate"] is True
    assert a1["alpha_id"] == a2["alpha_id"]

    pg = PostgresRepository(fallback_sqlite_path=tmp_path / "pg_fallback.db")
    sql, lock_id = pg.pg_advisory_lock_sql("daily_submit", "2026-11-18")
    assert "pg_try_advisory_lock" in sql and isinstance(lock_id, int)


def test_batch_c_utility_v1_daily_top2_80_20_and_score_attribution(tmp_path: Path) -> None:
    # Bell-shaped turnover quality peaks at 0.20
    q_peak = turnover_quality_score(0.20)
    q_low = turnover_quality_score(0.03)
    q_high = turnover_quality_score(0.65)
    assert q_peak > 0.99 and q_peak > q_low and q_peak > q_high

    # Missing correlation & margin apply 0.30 score and -5 penalties
    res_missing = evaluate_candidate({
        "expression_hash": "h1",
        "sharpe": 1.8,
        "fitness": 1.3,
        "turnover": 0.20,
        "is_check_status": "PASS",
    })
    assert res_missing.novelty_score == 0.30
    assert res_missing.margin_score == 0.30
    assert res_missing.missing_evidence_penalty == 10.0

    # High correlation penalty
    res_high_corr = evaluate_candidate({
        "expression_hash": "h2",
        "sharpe": 2.1,
        "fitness": 1.6,
        "turnover": 0.20,
        "margin_bps": 18.0,
        "self_correlation": 0.92,
        "is_check_status": "PASS",
    })
    assert res_high_corr.correlation_penalty == 20.0
    assert res_high_corr.decision == "NEAR_MISS_CORRELATION"

    # Rolling 80/20: 2 out of 10 formal submissions use exploration slot
    exp_flags = [should_use_exploration_slot(i) for i in range(10)]
    assert sum(exp_flags) == 2

    # Daily Top-2 selection retains candidate 3 as pending and rejects duplicates
    cands = [
        {"alpha_id": "A_best", "expression_hash": "e1", "sharpe": 2.0, "fitness": 1.5, "turnover": 0.2, "margin_bps": 20.0, "self_correlation": 0.15, "is_check_status": "PASS", "brain_simulation_id": "s1"},
        {"alpha_id": "B_second", "expression_hash": "e2", "sharpe": 1.7, "fitness": 1.3, "turnover": 0.22, "margin_bps": 16.0, "self_correlation": 0.20, "is_check_status": "PASS", "brain_simulation_id": "s2"},
        {"alpha_id": "C_third", "expression_hash": "e3", "sharpe": 1.4, "fitness": 1.1, "turnover": 0.25, "margin_bps": 12.0, "self_correlation": 0.25, "is_check_status": "PASS", "brain_simulation_id": "s3"},
        {"alpha_id": "D_dup", "expression_hash": "e4", "duplicate_expression": True, "sharpe": 2.4, "fitness": 1.9},
    ]
    batch = select_daily_candidates(cands, top_k=2, formal_submission_count=0)
    assert [x.alpha_id for x in batch.selected] == ["A_best", "B_second"]
    assert batch.retained_pending_ids == ("C_third",)
    assert batch.rejected_ids == ("D_dup",)

    # Submission service & DirectSubmitService retain records without deleting
    sub_svc = SubmissionService(tmp_path / "sub.db", top_k=2)
    out = sub_svc.execute_daily_top_k(cands, lambda aid: {"success": True, "status": "SUBMITTED", "brain_alpha_id": aid})
    assert out["submitted_ids"] == ["A_best", "B_second"]
    assert out["retained_pending_ids"] == ["C_third"]

    ds = DirectSubmitService(lambda aid: {"success": True, "status": "SUBMITTED"})
    d_res = ds.submit_existing_alpha({"brain_alpha_id": "WpX123", "status": "BRAIN_COMPLETE"})
    assert d_res["success"] is True and d_res["retained_in_vault"] is True

    # Score attribution & DST-aware New York refresh
    winter_refresh = score_refresh_at_for_taipei_date("2026-01-15", 3)
    summer_refresh = score_refresh_at_for_taipei_date("2026-07-15", 3)
    assert winter_refresh.hour != summer_refresh.hour  # EST (16:00) vs EDT (15:00)

    attr = build_batch_attribution(
        [{"brain_alpha_id": "A1", "batch_date": "2026-11-17"}, {"brain_alpha_id": "A2", "batch_date": "2026-11-17"}],
        score_before=1200.0,
        score_after=1450.0,
        observed_at=datetime.now(timezone.utc),
    )
    assert attr.attribution_status == "BATCH_ATTRIBUTION"
    assert attr.score_delta == 250.0

    # Require >= 7 observations before estimating completion date to 10,000
    est_short = estimate_days_to_10k([{"total_score": 1000 + i * 100, "daily_score": 100} for i in range(5)])
    est_ready = estimate_days_to_10k([{"total_score": 1000 + i * 150, "daily_score": 150} for i in range(8)])
    assert est_short["ready"] is False
    assert est_ready["ready"] is True and est_ready["estimated_days"] > 0


def test_batch_d_e_f_g_h_i_j_indicators_math_bridge_shadow_and_observability(tmp_path: Path) -> None:
    # AST family clustering groups window variants together
    h3 = compute_ast_family_hash("ts_decay_linear(rank(close - open), 3)")
    h5 = compute_ast_family_hash("ts_decay_linear(rank(close - open), 5)")
    h8 = compute_ast_family_hash("ts_decay_linear(rank(close - open), 8)")
    assert h3 == h5 == h8

    tr_s, te_s = split_family_chronological([
        {"expression": "ts_decay_linear(close, 3)", "completed_at": "2026-11-01T00:00:00Z"},
        {"expression": "ts_decay_linear(close, 8)", "completed_at": "2026-11-10T00:00:00Z"},
        {"expression": "group_neutralize(rank(volume), sector)", "completed_at": "2026-11-15T00:00:00Z"},
    ], train_frac=0.5)
    tr_fams = {r["ast_family_hash"] for r in tr_s}
    te_fams = {r["ast_family_hash"] for r in te_s}
    assert tr_fams.isdisjoint(te_fams)

    # 26+ causal indicators & video hypothesis runner
    assert len(INDICATOR_CATALOG) >= 26
    idx = pd.date_range("2026-01-01", periods=140, freq="1h", tz="UTC")
    c = 90000.0 + np.cumsum(np.sin(np.linspace(0, 15, len(idx))) * 120.0)
    df = pd.DataFrame({"Open": c - 20, "High": c + 60, "Low": c - 50, "Close": c, "Volume": np.full(len(idx), 500.0)}, index=idx)
    ind_df = add_indicators(df)
    for col in ("wma_20", "dema_20", "tema_20", "hma_20", "ppo", "cci_20", "mfi_14", "donchian_high_20", "aroon_up_25", "keltner_upper_20", "roc_12", "williams_r_14"):
        assert col in ind_df.columns
    assert verify_indicator_causality(df, "rsi_14", check_idx=80) is True
    hyp_res = evaluate_video_hypothesis(df, {"task_id": "video-rsi-trend-001", "topic": "RSI + Trend"})
    assert hyp_res.causal_passed is True

    # Phase 12 Mathematical Models
    eb = HierarchicalEmpiricalBayesModel(min_ess=5.0)
    for _ in range(12):
        eb.update_realized("UP|BULL|MID|HV", "UP|BULL", True)
    post = eb.evaluate("UP|BULL|MID|HV", "UP|BULL")
    assert post.credible_lower_90 < post.posterior_mean < post.credible_upper_90
    assert post.effective_sample_size >= 10.0

    ou = estimate_ou_parameters(100.0 + np.sin(np.linspace(0, 30, 120)) * 2.0)
    assert hasattr(ou, "adf_t_stat") and hasattr(ou, "kpss_stat") and hasattr(ou, "half_life_cv")
    h_ci = estimate_hurst_with_ci(c)
    assert h_ci.ci_low <= h_ci.hurst <= h_ci.ci_high
    assert h_ci.regime in {"TREND", "MEAN_REVERTING", "UNCERTAIN"}
    assert abs(fractional_kelly_size(0.01, 0.02, kelly_fraction=0.50, portfolio_cap=0.20)) <= 0.20

    # Taiwan publication lag check
    assert check_fundamental_publication_lag("2026-05-20", "2026-03-31") is True
    assert check_fundamental_publication_lag("2026-04-10", "2026-03-31") is False

    # Shadow trading OrderIntent gates
    shadow = ShadowTradingService(tmp_path / "shadow.db")
    blocked = shadow.create_order_intent(
        strategy="perp_carry", model_version="v1", symbol="BTC/USDT",
        features={"z": 2.1}, signal=1.0, confidence=0.8, target_position=0.2,
        current_price=90000.0, data_stale=True,
    )
    assert blocked.risk_decision == "BLOCKED_STALE_DATA" and blocked.target_position == 0.0

    ok_intent = shadow.create_order_intent(
        strategy="perp_carry", model_version="v1", symbol="BTC/USDT",
        features={"z": 2.2}, signal=1.0, confidence=0.8, target_position=0.2,
        current_price=90000.0, next_realized_price=90900.0,
    )
    dup_intent = shadow.create_order_intent(
        strategy="perp_carry", model_version="v1", symbol="BTC/USDT",
        features={"z": 2.2}, signal=1.0, confidence=0.8, target_position=0.2,
        current_price=90000.0,
    )
    assert ok_intent.risk_decision == "SHADOW_APPROVED" and ok_intent.counterfactual_pnl > 0
    assert dup_intent.risk_decision == "BLOCKED_DUPLICATE_INTENT"

    # Observability degraded component detection
    obs = build_platform_observability_report(brain_metrics={"unknown_submission_count": 1})
    assert obs["overall_status"] == "DEGRADED"
    assert obs["degraded_components"][0]["component"] == "BRAIN_SUBMISSION"


def test_gates_u9_to_u16_and_taiwan_microstructure_execution(tmp_path: Path) -> None:
    from datetime import date
    from taiwan_stock.universe_registry import (
        TaiwanInstrument,
        TaiwanUniverseRegistry,
        build_universe_cache_key,
        check_universe_continuity_diff,
        classify_6layer_tradability_funnel,
        evaluate_delisted_price_coverage,
        executable_taiwan_price,
        reconcile_dual_sources,
    )

    reg = TaiwanUniverseRegistry(db_path=tmp_path / "u9.db")
    reg.write_snapshot(
        [
            TaiwanInstrument("2330", "台積電", "TWSE", "COMMON_STOCK"),
            TaiwanInstrument("1234", "停牌測試股", "TWSE", "COMMON_STOCK"),
        ],
        as_of=date(2026, 1, 10),
    )
    reg.record_instrument_event("1234", "SUSPENDED", "2026-01-12", end_date="2026-01-20")

    listed_jan15 = {r["symbol"] for r in reg.historical_equities("2026-01-15", tradable_only=False)}
    tradable_jan15 = {r["symbol"] for r in reg.historical_equities("2026-01-15", tradable_only=True)}
    assert "1234" in listed_jan15
    assert "1234" not in tradable_jan15
    assert "2330" in tradable_jan15

    # U10 dual-source reconciliation
    rec_ok = reconcile_dual_sources(
        [{"symbol": "2330", "market": "TWSE", "name": "台積電"}, {"symbol": "1234", "market": "TWSE", "name": "停牌測試股"}],
        [{"symbol": "2330", "market": "TWSE", "name": "台積電"}],
        suspended_symbols={"1234"},
    )
    assert rec_ok["status"] == "PASS"
    rec_blk = reconcile_dual_sources(
        [{"symbol": "2330", "market": "TWSE", "name": "台積電"}],
        [{"symbol": "2330", "market": "TWSE", "name": "台積電"}, {"symbol": "9999", "market": "TWSE", "name": "幽靈股"}],
    )
    assert rec_blk["status"] == "BLOCKED"

    # U12 delisted price history coverage
    cov_low = evaluate_delisted_price_coverage(["2330", "2317", "1101"], ["2330", "2317"])
    assert cov_low["survivorship_status"] == "RESEARCH_NOT_SURVIVORSHIP_SAFE"
    cov_ok = evaluate_delisted_price_coverage([f"S{i}" for i in range(100)], [f"S{i}" for i in range(99)])
    assert cov_ok["survivorship_status"] == "SURVIVORSHIP_SAFE"

    # U14 cache key penetration
    k1 = build_universe_cache_key(feature_schema_hash="f1", universe_sha256="u224", as_of_date="2026-11-18")
    k2 = build_universe_cache_key(feature_schema_hash="f1", universe_sha256="u1950", as_of_date="2026-11-18")
    assert k1 != k2

    # U15 discontinuity quarantine
    prev_u = [{"symbol": f"{1000+i}", "market": "TWSE", "security_type": "COMMON_STOCK"} for i in range(200)]
    curr_drop = prev_u[:120]
    diff_res = check_universe_continuity_diff(prev_u, curr_drop)
    assert diff_res["quarantined"] is True and diff_res["reason"] == "UNIVERSE_DISCONTINUITY"

    # U16 6-layer funnel
    funnel = classify_6layer_tradability_funnel([
        {"symbol": "2330", "security_type": "COMMON_STOCK", "last_close": 1000.0, "volume": 30_000_000, "history_bars": 200},
        {"symbol": "1234", "security_type": "COMMON_STOCK", "suspended": True, "last_close": 50.0, "volume": 100_000},
        {"symbol": "5678", "security_type": "COMMON_STOCK", "last_close": 80.0, "volume": 500_000, "price_limit_locked": True},
    ])
    assert funnel["OFFICIAL_COMMON_EQUITY_UNIVERSE"] == 3
    assert funnel["TRADABLE_TODAY_UNIVERSE"] == 2
    assert funnel["PORTFOLIO_ELIGIBLE_UNIVERSE"] == 1

    # Taiwan microstructure execution model (limit-up lock, pre-2015 7% vs post-2015 10%, partial fill)
    unfilled = executable_taiwan_price("BUY", prev_close=100.0, next_open=110.0, as_of_date="2026-11-18")
    assert unfilled["status"] == "UNFILLED" and unfilled["reason"] == "LIMIT_UP_LOCKED"
    pre2015 = executable_taiwan_price("BUY", prev_close=100.0, next_open=107.0, as_of_date="2014-05-01")
    assert pre2015["status"] == "UNFILLED" and pre2015["limit_up"] == 107.0
    part = executable_taiwan_price("BUY", prev_close=100.0, next_open=102.0, available_volume_shares=5000.0, order_shares=1000.0, max_participation_rate=0.10)
    assert part["status"] == "PARTIAL_FILL" and part["filled_shares"] == 500


def test_gates_m1_to_m10_triple_barrier_purging_oof_lambdarank_and_dsr_pbo(tmp_path: Path) -> None:
    from crypto_trading.meta_labeling import (
        ResearchTrialRegistry,
        build_cross_sectional_ranking_groups,
        compute_deflated_sharpe_and_pbo,
        compute_triple_barrier_labels,
        generate_oof_meta_labels,
        purge_overlapping_events,
    )

    idx = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    close = np.full(80, 100.0)
    high = close + 0.2
    low = close - 0.2
    # Force an intrabar dual breach on bar 35
    high[35] = 115.0
    low[35] = 85.0
    df = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close}, index=idx)
    tb = compute_triple_barrier_labels(df, pt_mult=1.5, sl_mult=1.5, max_hold_bars=5)
    assert "AMBIGUOUS_INTRABAR" in set(tb["first_touch_type"])
    amb_rows = tb[tb["first_touch_type"] == "AMBIGUOUS_INTRABAR"]
    assert all(amb_rows["tb_label"] == -1)  # Loss-first conservative rule

    # M3 Purging overlapping event intervals
    tr_idx = np.arange(0, 40, dtype=int)
    te_idx = np.arange(40, 50, dtype=int)
    purged = purge_overlapping_events(
        tr_idx, te_idx, tb["event_start_idx"].to_numpy(), tb["event_end_idx"].to_numpy(), embargo_bars=3
    )
    assert len(purged) < len(tr_idx)
    assert all(tb["event_end_idx"].iloc[i] < 40 for i in purged)

    # M2 OOF Meta-labels without in-sample leakage
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, size=(80, 3))
    y = 0.01 * X[:, 0] + rng.normal(0, 0.002, size=80)

    class _SimpleOLS:
        def fit(self, X_tr, y_tr):
            self.w = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
            return self
        def predict(self, X_te):
            return X_te @ self.w

    oof = generate_oof_meta_labels(
        X, y, y, tb["event_start_idx"].to_numpy(), tb["event_end_idx"].to_numpy(), _SimpleOLS
    )
    assert set(np.unique(oof["meta_labels"])).issubset({0, 1})

    # M6 & M7 Cross-sectional ranking groups & within-date quantiles
    rows = []
    for d in pd.date_range("2026-01-01", periods=6, freq="1D", tz="UTC"):
        for s_i in range(10):
            rows.append({"timestamp": d, "symbol": f"S{s_i:02d}", "fwd_net_ret": float(s_i - 5) * 0.01})
    ranked_df, groups = build_cross_sectional_ranking_groups(pd.DataFrame(rows), n_bins=5)
    assert groups == [10, 10, 10, 10, 10, 10]
    assert set(ranked_df["cs_rank_label"].unique()) == {0, 1, 2, 3, 4}

    # M9 & M10 Trial registry + DSR/PBO
    reg = ResearchTrialRegistry(tmp_path / "trials.db")
    reg.record_trial(config={"lr": 0.03}, data_hash="d1", universe_hash="u1", oos_metrics={"sharpe": 1.4}, accepted=True)
    assert reg.count_trials() == 1

    good_trial = np.full((300, 1), 0.002) + rng.normal(0, 0.005, size=(300, 1))
    noise_trials = rng.normal(0, 0.01, size=(300, 4))
    dsr_res = compute_deflated_sharpe_and_pbo(np.hstack([good_trial, noise_trials]))
    assert dsr_res["dsr_prob"] >= 0.90 and dsr_res["pbo"] <= 0.30


def test_brain_challenge_score_fetch_and_10k_progress(tmp_path: Path) -> None:
    from worldquant_brain.brain_client import WorldQuantBrainClient
    from worldquant_brain.score_tracker import ScoreTracker, parse_score_payload
    from worldquant_brain.score_attribution import estimate_days_to_10k

    # Offline mode never sends real network calls
    client = WorldQuantBrainClient(email="", password="")
    res_off = client.fetch_challenge_score()
    assert res_off["success"] is False and res_off["status"] == "OFFLINE"

    tracker = ScoreTracker(tmp_path / "challenge.db")
    snap = parse_score_payload(
        {
            "totalScore": 3650.0,
            "dailyScore": 240.0,
            "dailyScoreCap": 2000.0,
            "rank": 412,
            "daysOfSubmission": 18,
            "uniqueness": 0.84,
            "isScore": 1920.0,
        },
        source="AUTHENTICATED_CHALLENGE_API",
    )
    tracker.record(snap)
    hist = tracker.history(limit=10)
    assert len(hist) == 1
    assert hist[0]["total_score"] == 3650.0
    assert round(10000.0 - hist[0]["total_score"], 2) == 6350.0
