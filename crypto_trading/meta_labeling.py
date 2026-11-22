"""Strict Triple-Barrier Labeling, OOF Meta-Labeling, Cross-Sectional LambdaRank Group Builder,
and Overfitting Gates (DSR / PBO / CPCV / Trial Registry) — Gates M1–M10.

References:
- Lopez de Prado (2018), Advances in Financial Machine Learning (Triple-Barrier, Meta-Labeling, Purging).
- Bailey, Borwein, Lopez de Prado & Zhu (2014/2017), Deflated Sharpe Ratio (DSR) & Probability of Backtest Overfitting (PBO).
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import closing
import sqlite3
from statistics import NormalDist
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from worldquant_brain.sqlite_resilience import ensure_schema_from_ddl


def compute_triple_barrier_labels(
    df_or_close: pd.DataFrame | pd.Series,
    *,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    max_hold_bars: int = 12,
    vol_window: int = 24,
    roundtrip_cost_bps: float = 6.0,
    intrabar_tie_rule: str = "AMBIGUOUS_LOSS_FIRST",
) -> pd.DataFrame:
    """Gate M1, M3 & M5: Event-driven Triple-Barrier labels using strictly trailing volatility sigma_t (:t),
    recording event_start, event_end, first_touch_type, first_touch_price, and net-of-cost return.
    """
    if isinstance(df_or_close, pd.Series):
        c = pd.to_numeric(df_or_close, errors="coerce").to_numpy(dtype=float)
        h = c.copy()
        l = c.copy()
        o = c.copy()
        idx = df_or_close.index
    else:
        c = pd.to_numeric(df_or_close["Close"], errors="coerce").to_numpy(dtype=float)
        h = pd.to_numeric(df_or_close.get("High", df_or_close["Close"]), errors="coerce").to_numpy(dtype=float)
        l = pd.to_numeric(df_or_close.get("Low", df_or_close["Close"]), errors="coerce").to_numpy(dtype=float)
        o = pd.to_numeric(df_or_close.get("Open", df_or_close["Close"]), errors="coerce").to_numpy(dtype=float)
        idx = df_or_close.index

    # Strictly causal trailing volatility at bar t (never full-sample volatility)
    log_ret = pd.Series(np.log(np.maximum(c, 1e-12))).diff().fillna(0.0)
    sigma_t = (
        log_ret.rolling(vol_window, min_periods=max(5, vol_window // 2))
        .std(ddof=1)
        .fillna(0.005)
        .clip(lower=0.001)
        .to_numpy()
    )

    n = len(c)
    cost_rate = float(roundtrip_cost_bps) / 10000.0
    labels = np.zeros(n, dtype=int)
    tb_gross_rets = np.zeros(n, dtype=float)
    tb_net_rets = np.zeros(n, dtype=float)
    hold_bars = np.zeros(n, dtype=int)
    event_end_idx = np.arange(n, dtype=int)
    first_touch_types: list[str] = ["VERTICAL_TIME"] * n
    first_touch_prices = c.copy()

    for i in range(n - 1):
        entry = c[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        sig_i = sigma_t[i]
        upper = entry * math.exp(pt_mult * sig_i)
        lower = entry * math.exp(-sl_mult * sig_i)
        end_j = min(n - 1, i + max_hold_bars)

        hit = 0
        exit_j = end_j
        touch_type = "VERTICAL_TIME"
        touch_price = c[end_j]

        for j in range(i + 1, end_j + 1):
            hit_pt = h[j] >= upper
            hit_sl = l[j] <= lower
            if hit_pt and hit_sl:
                # Gate M1: Same-bar dual breach is AMBIGUOUS_INTRABAR -> conservative loss-first
                exit_j = j
                touch_type = "AMBIGUOUS_INTRABAR"
                if intrabar_tie_rule == "AMBIGUOUS_NEUTRAL":
                    hit = 0
                    touch_price = c[j]
                else:
                    hit = -1
                    touch_price = min(o[j], lower) if o[j] < lower else lower
                break
            elif hit_sl:
                hit = -1
                exit_j = j
                touch_type = "STOP_LOSS"
                # Respect gap-down open beyond barrier
                touch_price = o[j] if o[j] < lower else lower
                break
            elif hit_pt:
                hit = 1
                exit_j = j
                touch_type = "PROFIT_TAKE"
                # Respect gap-up open beyond barrier
                touch_price = o[j] if o[j] > upper else upper
                break

        gross_r = (touch_price - entry) / entry
        net_r = gross_r - cost_rate
        labels[i] = hit
        tb_gross_rets[i] = gross_r
        tb_net_rets[i] = net_r
        hold_bars[i] = exit_j - i
        event_end_idx[i] = exit_j
        first_touch_types[i] = touch_type
        first_touch_prices[i] = touch_price

    return pd.DataFrame(
        {
            "event_start_idx": np.arange(n, dtype=int),
            "event_end_idx": event_end_idx,
            "sigma_t": sigma_t,
            "tb_label": labels,
            "tb_return": tb_gross_rets,
            "tb_net_return": tb_net_rets,
            "tb_hold_bars": hold_bars,
            "first_touch_type": first_touch_types,
            "first_touch_price": first_touch_prices,
        },
        index=idx,
    )


def purge_overlapping_events(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    event_start_idx: np.ndarray,
    event_end_idx: np.ndarray,
    *,
    embargo_bars: int = 5,
) -> np.ndarray:
    """Gate M3: Purge any training event whose [event_start, event_end] overlaps with any test event window + embargo."""
    if len(train_indices) == 0 or len(test_indices) == 0:
        return train_indices
    t_min_start = int(np.min(event_start_idx[test_indices]))
    t_max_end = int(np.max(event_end_idx[test_indices])) + max(0, int(embargo_bars))

    keep_mask = np.ones(len(train_indices), dtype=bool)
    for k, tr_i in enumerate(train_indices):
        s_i = int(event_start_idx[tr_i])
        e_i = int(event_end_idx[tr_i])
        if s_i <= t_max_end and e_i >= t_min_start:
            keep_mask[k] = False
    return train_indices[keep_mask]


def generate_oof_meta_labels(
    X: np.ndarray,
    primary_targets: np.ndarray,
    realized_fwd_returns: np.ndarray,
    event_start_idx: np.ndarray,
    event_end_idx: np.ndarray,
    primary_model_factory: Callable[[], Any],
    *,
    n_splits: int = 4,
    roundtrip_cost: float = 0.0006,
    embargo_bars: int = 5,
) -> dict[str, Any]:
    """Gate M2 & M5: Build strictly Out-Of-Fold (OOF) primary predictions with event-interval purging,
    then define cost-adjusted Meta-Labels (1 iff side * realized_return - cost > 0).
    """
    n = len(X)
    oof_pred = np.zeros(n, dtype=float)
    oof_side = np.zeros(n, dtype=float)
    oof_valid = np.zeros(n, dtype=bool)
    fold_size = n // (n_splits + 1)

    for fold in range(1, n_splits + 1):
        test_start = fold * fold_size
        test_end = min(n, (fold + 1) * fold_size)
        test_idx = np.arange(test_start, test_end, dtype=int)
        raw_train_idx = np.arange(0, test_start, dtype=int)
        purged_train_idx = purge_overlapping_events(
            raw_train_idx, test_idx, event_start_idx, event_end_idx, embargo_bars=embargo_bars
        )
        if len(purged_train_idx) < 15:
            continue
        m = primary_model_factory()
        m.fit(X[purged_train_idx], primary_targets[purged_train_idx])
        preds = np.asarray(m.predict(X[test_idx]), dtype=float)
        oof_pred[test_idx] = preds
        oof_side[test_idx] = np.sign(preds)
        oof_valid[test_idx] = True

    net_edge = oof_side * realized_fwd_returns - np.abs(oof_side) * float(roundtrip_cost)
    meta_labels = (net_edge > 0.0).astype(int)
    return {
        "oof_predictions": oof_pred,
        "oof_side": oof_side,
        "oof_valid_mask": oof_valid,
        "meta_labels": meta_labels,
        "oof_net_returns": net_edge,
    }


def apply_meta_label_filter(
    primary_signals: np.ndarray,
    features: pd.DataFrame,
    *,
    min_adx: float = 18.0,
    max_vol_z: float = 3.0,
    target_ann_vol: float = 0.35,
    bars_per_year: float = 24.0 * 365.0,
    roundtrip_cost: float = 0.0006,
) -> dict[str, Any]:
    """Gate M4: Secondary Meta-Labeling & Volatility-Targeting filter with 4-way baseline comparison."""
    sig = np.asarray(primary_signals, dtype=float)
    n = len(sig)
    close = pd.to_numeric(features["Close"], errors="coerce") if "Close" in features.columns else pd.Series(np.ones(n))
    ret1 = close.pct_change().fillna(0.0)
    fwd1 = close.pct_change().shift(-1).fillna(0.0).to_numpy()
    realized_ann_vol = (ret1.rolling(24, min_periods=6).std(ddof=1).fillna(0.01) * np.sqrt(bars_per_year)).clip(lower=0.05).to_numpy()
    vol_scalar = np.clip(target_ann_vol / realized_ann_vol, 0.25, 1.5)

    adx_ok = np.ones(n, dtype=bool)
    if "adx_14" in features.columns:
        adx_ok = features["adx_14"].fillna(20.0).to_numpy() >= min_adx

    vol_ok = np.ones(n, dtype=bool)
    if "volume_zscore_24" in features.columns:
        vol_ok = features["volume_zscore_24"].fillna(0.0).abs().to_numpy() <= max_vol_z

    meta_prob = np.where(adx_ok & vol_ok, 0.65, 0.35)
    filtered_weights = np.clip(np.where(meta_prob >= 0.5, sig * vol_scalar, 0.0), -1.0, 1.0)

    def _eval_net(w: np.ndarray) -> dict[str, float]:
        turn = np.abs(np.diff(np.concatenate([[0.0], w])))
        net = w * fwd1 - turn * (roundtrip_cost * 0.5)
        sh = float((np.mean(net) / (np.std(net, ddof=1) + 1e-12)) * np.sqrt(bars_per_year)) if len(net) > 5 else 0.0
        gains = float(np.sum(net[net > 0]))
        losses = float(abs(np.sum(net[net < 0]))) + 1e-12
        return {"net_sharpe": round(sh, 4), "profit_factor": round(gains / losses, 4), "trades": int(np.sum(turn > 0))}

    rng = np.random.default_rng(42)
    rand_mask = rng.uniform(0, 1, size=n) >= 0.3
    abs_sig = np.abs(sig)
    causal_med = pd.Series(abs_sig).expanding(min_periods=1).median().to_numpy()
    conf_mask = abs_sig >= causal_med

    comparison = {
        "primary_only": _eval_net(np.sign(sig)),
        "primary_plus_meta": _eval_net(filtered_weights),
        "primary_plus_random": _eval_net(np.where(rand_mask, np.sign(sig), 0.0)),
        "primary_plus_confidence": _eval_net(np.where(conf_mask, np.sign(sig), 0.0)),
    }

    return {
        "filtered_weights": filtered_weights,
        "meta_probability": meta_prob,
        "vol_scalar": vol_scalar,
        "pass_rate": round(float(np.mean(meta_prob >= 0.5)), 4) if n > 0 else 0.0,
        "ablation_comparison": comparison,
    }


def build_cross_sectional_ranking_groups(
    panel_df: pd.DataFrame,
    *,
    date_col: str = "timestamp",
    symbol_col: str = "symbol",
    net_return_col: str = "fwd_net_ret",
    n_bins: int = 5,
) -> tuple[pd.DataFrame, list[int]]:
    """Gate M6 & M7: Sort contiguously by (date, symbol), compute within-date cross-sectional quantile labels (0..n_bins-1),
    and return contiguous group sizes for LGBMRanker(objective='lambdarank').
    """
    df = panel_df.sort_values([date_col, symbol_col]).reset_index(drop=True).copy()

    def _within_date_qcut(s: pd.Series) -> pd.Series:
        if len(s) < n_bins or s.nunique() < 2:
            return pd.Series(np.zeros(len(s), dtype=int), index=s.index)
        ranks = s.rank(method="first")
        return pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop").fillna(0).astype(int)

    df["cs_rank_label"] = df.groupby(date_col, sort=False)[net_return_col].transform(_within_date_qcut)
    group_sizes = [int(sz) for sz in df.groupby(date_col, sort=False).size().tolist()]
    assert sum(group_sizes) == len(df)
    return df, group_sizes


def risk_adjusted_momentum_rank(panel_df: pd.DataFrame, *, lookback: int = 24) -> pd.Series:
    """Compute cross-sectional Sharpe-adjusted momentum score (return / realized_vol) per symbol."""
    close = pd.to_numeric(panel_df["Close"], errors="coerce")
    ret = close.pct_change(lookback)
    vol = close.pct_change(1).rolling(lookback, min_periods=max(5, lookback // 2)).std(ddof=1).replace(0, np.nan)
    return (ret / (vol * np.sqrt(lookback))).fillna(0.0)


def compute_deflated_sharpe_and_pbo(
    trial_returns_matrix: np.ndarray,
    *,
    annualization_factor: float = 252.0,
) -> dict[str, Any]:
    """Gate M9 & M10: Compute Deflated Sharpe Ratio (DSR) and CSCV Probability of Backtest Overfitting (PBO)."""
    mat = np.asarray(trial_returns_matrix, dtype=float)
    if mat.ndim == 1:
        mat = mat[:, None]
    n_obs, n_trials = mat.shape
    if n_obs < 10:
        return {"dsr_prob": 0.0, "pbo": 1.0, "promotion_passed": False}

    means = np.mean(mat, axis=0)
    stds = np.std(mat, axis=0, ddof=1) + 1e-12
    sharpes = (means / stds) * np.sqrt(annualization_factor)
    best_idx = int(np.argmax(sharpes))
    best_sr = float(sharpes[best_idx])
    best_r = mat[:, best_idx]

    # Exact Bailey & Lopez de Prado (2014) expected max Sharpe with correlated effective trials N_eff
    nd = NormalDist()
    euler_gamma = 0.5772156649015329
    if n_trials <= 1:
        sr_benchmark = 0.0
        n_eff = 1.0
    else:
        with np.errstate(all="ignore"):
            corr_mat = np.corrcoef(mat, rowvar=False)
        if np.ndim(corr_mat) == 2 and corr_mat.shape[0] == n_trials:
            off_diag = corr_mat[~np.eye(n_trials, dtype=bool)]
            finite_corr = off_diag[np.isfinite(off_diag)]
            avg_corr = float(np.clip(np.mean(finite_corr), 0.0, 1.0)) if finite_corr.size else 0.0
        else:
            avg_corr = 0.0
        n_eff = max(1.0, avg_corr + (1.0 - avg_corr) * float(n_trials))
        if n_eff <= 1.0 + 1e-9:
            sr_benchmark = 0.0
        else:
            null_sr_std = math.sqrt(annualization_factor / max(1.0, float(n_obs)))
            emp_sr_std = float(np.std(sharpes, ddof=1)) if n_trials >= 2 else 0.0
            sr_std = emp_sr_std if np.isfinite(emp_sr_std) and emp_sr_std > 1e-9 else null_sr_std
            p1 = min(1.0 - 1e-12, max(1e-12, 1.0 - 1.0 / n_eff))
            p2 = min(1.0 - 1e-12, max(1e-12, 1.0 - 1.0 / (n_eff * math.e)))
            sr_benchmark = max(0.0, float(sr_std * ((1.0 - euler_gamma) * nd.inv_cdf(p1) + euler_gamma * nd.inv_cdf(p2))))

    # Deflated Sharpe test statistic
    r_centered = (best_r - np.mean(best_r)) / (np.std(best_r, ddof=1) + 1e-12)
    skew = float(np.mean(r_centered ** 3))
    kurt = float(np.mean(r_centered ** 4))
    sr_non_ann = best_sr / np.sqrt(annualization_factor)
    sr_bench_non_ann = sr_benchmark / np.sqrt(annualization_factor)
    denom = math.sqrt(max(1e-9, 1.0 - skew * sr_non_ann + ((kurt - 1.0) / 4.0) * (sr_non_ann ** 2)))
    z_dsr = ((sr_non_ann - sr_bench_non_ann) * math.sqrt(max(1, n_obs - 1))) / denom
    dsr_prob = 0.5 * (1.0 + math.erf(z_dsr / math.sqrt(2.0)))

    # CSCV / PBO across 4 temporal sub-blocks (6 2-block IS/OOS splits)
    if n_trials >= 2 and n_obs >= 40:
        from itertools import combinations
        blocks = np.array_split(mat, 4, axis=0)
        underperformed = 0
        total_splits = 0
        for is_idx in combinations(range(4), 2):
            oos_idx = [k for k in range(4) if k not in is_idx]
            is_mat = np.vstack([blocks[k] for k in is_idx])
            oos_mat = np.vstack([blocks[k] for k in oos_idx])
            is_sr = np.mean(is_mat, axis=0) / (np.std(is_mat, axis=0, ddof=1) + 1e-12)
            oos_sr = np.mean(oos_mat, axis=0) / (np.std(oos_mat, axis=0, ddof=1) + 1e-12)
            w_star = int(np.argmax(is_sr))
            if oos_sr[w_star] < float(np.median(oos_sr)):
                underperformed += 1
            total_splits += 1
        pbo = round(float(underperformed) / max(1, total_splits), 4)
    else:
        pbo = 1.0  # Fail-closed when n_trials < 2 or n_obs < 40

    # Check fold dominance across 4 quartiles
    q_size = max(1, n_obs // 4)
    fold_pnls = [float(np.sum(best_r[i * q_size : (i + 1) * q_size])) for i in range(4)]
    tot_pos_pnl = sum(max(0.0, p) for p in fold_pnls) + 1e-12
    max_fold_share = max(max(0.0, p) / tot_pos_pnl for p in fold_pnls)

    passed = (
        dsr_prob >= 0.95
        and pbo <= 0.20
        and best_sr > 0.0
        and max_fold_share <= 0.60
    )
    return {
        "best_trial_idx": best_idx,
        "best_sharpe": round(best_sr, 4),
        "sr_benchmark": round(sr_benchmark, 4),
        "dsr_prob": round(dsr_prob, 4),
        "pbo": round(pbo, 4),
        "max_fold_pnl_share": round(max_fold_share, 4),
        "n_trials": n_trials,
        "n_observations": n_obs,
        "promotion_passed": bool(passed),
    }


class ResearchTrialRegistry:
    """Gate M9: Persist every strategy/parameter trial (not just winners) so DSR/PBO accounts for all attempts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_trials (
                    trial_id TEXT PRIMARY KEY,
                    parent_trial_id TEXT,
                    config_hash TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    universe_hash TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    oos_metrics_json TEXT NOT NULL,
                    accepted INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            ensure_schema_from_ddl(conn, '\n                CREATE TABLE IF NOT EXISTS research_trials (\n                    trial_id TEXT PRIMARY KEY,\n                    parent_trial_id TEXT,\n                    config_hash TEXT NOT NULL,\n                    data_hash TEXT NOT NULL,\n                    universe_hash TEXT NOT NULL,\n                    started_at TEXT NOT NULL,\n                    oos_metrics_json TEXT NOT NULL,\n                    accepted INTEGER NOT NULL DEFAULT 0\n                )\n                ')

    def record_trial(
        self,
        *,
        config: Mapping[str, Any],
        data_hash: str,
        universe_hash: str,
        oos_metrics: Mapping[str, Any],
        accepted: bool,
        parent_trial_id: str | None = None,
    ) -> str:
        trial_id = uuid.uuid4().hex
        cfg_hash = hashlib.sha256(json.dumps(dict(config), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO research_trials (
                    trial_id, parent_trial_id, config_hash, data_hash, universe_hash,
                    started_at, oos_metrics_json, accepted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id,
                    parent_trial_id,
                    cfg_hash,
                    data_hash,
                    universe_hash,
                    now,
                    json.dumps(dict(oos_metrics), sort_keys=True),
                    1 if accepted else 0,
                ),
            )
        return trial_id

    def count_trials(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM research_trials").fetchone()[0])
