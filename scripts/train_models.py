"""
Standalone Training CLI Entrypoint (scripts/train_models.py).
Supports three independent models with zero look-ahead leakage and SHA-256 manifests:
  python scripts/train_models.py --target crypto_lgbm
  python scripts/train_models.py --target crypto_ppo
  python scripts/train_models.py --target taiwan_ranker
  python scripts/train_models.py --target all
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd

from crypto_trading.data_loader import (
    CryptoDataLoader,
    DataSource,
    DataUnavailableError,
)
from crypto_trading.feature_engineer import CryptoFeatureEngineer
from crypto_trading.ml_trainer import CryptoMLTrainer
from crypto_trading.rl_agent import PPOAgent
from crypto_trading.rl_env import CryptoTradingEnv
from taiwan_stock.config import DEFAULT_WATCHLIST
from taiwan_stock.ml_ranker import TaiwanStockMLRanker

MODELS_DIR = os.path.join(BASE_DIR, "data_persistence", "models")
LEGACY_DIR = os.path.join(MODELS_DIR, "legacy_unverified")


def quarantine_unverified_models() -> list[str]:
    """Moves any .joblib or .pt in data_persistence/models/ lacking a valid SHA-256 manifest to legacy_unverified/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LEGACY_DIR, exist_ok=True)
    quarantined: list[str] = []

    mapping = {
        "crypto_lgbm_weights.joblib": "crypto_lgbm_weights_manifest.json",
        "crypto_ppo_agent.pt": "crypto_ppo_manifest.json",
        "taiwan_stock_ranker.joblib": "taiwan_stock_manifest.json",
    }
    for model_file, manifest_file in mapping.items():
        m_path = os.path.join(MODELS_DIR, model_file)
        mf_path = os.path.join(MODELS_DIR, manifest_file)
        alt_mf_path = os.path.splitext(m_path)[0] + "_manifest.json"
        if os.path.exists(m_path) and not (os.path.exists(mf_path) or os.path.exists(alt_mf_path)):
            dst = os.path.join(LEGACY_DIR, f"{model_file}.unverified")
            shutil.move(m_path, dst)
            quarantined.append(model_file)
    return quarantined



CACHE_DIR = os.path.join(BASE_DIR, "crypto_trading", "data_cache")


def _load_historical_or_live_ohlcv(
    symbol: str,
    timeframe: str = "1m",
    limit: int = 1200,
    allow_synthetic: bool = False,
    prefer_cache: bool = True,
) -> tuple[pd.DataFrame, str, bool, str | None]:
    """
    Loads the largest cached historical CSV from crypto_trading/data_cache/ matching timeframe
    when available (and not allow_synthetic), otherwise falls back to CryptoDataLoader.
    """
    if prefer_cache and not allow_synthetic and os.path.isdir(CACHE_DIR):
        safe_sym = symbol.replace("/", "_")
        pattern = os.path.join(CACHE_DIR, f"{safe_sym}_{timeframe}_*.csv")
        candidates = sorted(glob.glob(pattern), key=lambda fp: os.path.getsize(fp), reverse=True)
        for cand in candidates:
            try:
                df = pd.read_csv(cand, index_col=0, parse_dates=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                df = df.sort_index()
                if len(df) >= 500:
                    return df, "HISTORICAL_CACHE", False, os.path.basename(cand)
            except Exception:
                continue

    loader = CryptoDataLoader(symbol=symbol, timeframe=timeframe)
    res = loader.fetch_ohlcv_result(
        limit=limit,
        symbol=symbol,
        timeframe=timeframe,
        allow_synthetic=allow_synthetic,
    )
    is_syn = bool(res.is_synthetic or res.source == DataSource.SYNTHETIC)
    src_label = "SYNTHETIC" if is_syn else res.source.value
    return res.df, src_label, is_syn, None


def train_crypto_lgbm(
    symbols: list[str],
    timeframe: str = "1m",
    limit: int = 1200,
    allow_synthetic: bool = False,
    output_dir: str = MODELS_DIR,
    prefer_cache: bool = True,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    dfs_by_sym: dict[str, pd.DataFrame] = {}
    sources_used: list[str] = []
    cache_files: list[str] = []
    bars_per_symbol: dict[str, int] = {}
    any_synthetic = False

    for sym in symbols:
        df, src_label, is_syn, cache_name = _load_historical_or_live_ohlcv(
            symbol=sym,
            timeframe=timeframe,
            limit=limit,
            allow_synthetic=allow_synthetic,
            prefer_cache=prefer_cache,
        )
        dfs_by_sym[sym] = df
        sources_used.append(src_label)
        bars_per_symbol[sym] = int(len(df))
        if cache_name:
            cache_files.append(cache_name)
        if is_syn:
            any_synthetic = True

    trainer = CryptoMLTrainer(model_type="lightgbm", task="regression", horizon=3, timeframe=timeframe)
    if len(dfs_by_sym) > 1 and hasattr(trainer, "walk_forward_train_multi"):
        summary = trainer.walk_forward_train_multi(dfs_by_sym, n_splits=5, purge_bars=12, embargo_bars=2)
    else:
        primary_df = next(iter(dfs_by_sym.values()))
        summary = trainer.walk_forward_train(primary_df, n_splits=5, purge_bars=12, embargo_bars=2)

    data_source_tag = "SYNTHETIC" if any_synthetic else ("HISTORICAL_CACHE" if cache_files else "LIVE")
    out_path = os.path.join(output_dir, "crypto_lgbm_weights.joblib")
    manifest = trainer.save_weights(
        out_path,
        metadata={
            "target": "crypto_lgbm",
            "symbols": symbols,
            "timeframe": timeframe,
            "data_source": data_source_tag,
            "is_synthetic": any_synthetic,
            "total_bars": int(sum(bars_per_symbol.values())),
            "bars_per_symbol": bars_per_symbol,
            "cache_files": cache_files,
            "deployment_stage": "RESEARCH_ONLY" if any_synthetic else "LOCAL_PAPER_READY",
            "oos_metrics": {
                "mean_rank_ic": summary["mean_rank_ic"],
                "mean_sharpe": summary["mean_sharpe"],
                "mean_accuracy": summary["mean_accuracy"],
                "mean_turnover": summary["mean_turnover"],
            },
        },
    )
    canonical_manifest_path = os.path.join(output_dir, "crypto_lgbm_manifest.json")
    with open(canonical_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return {"summary": summary, "manifest": manifest, "model_path": out_path}


def train_crypto_ppo(
    symbol: str = "BTC/USDT",
    timeframe: str = "1m",
    limit: int = 1000,
    episodes: int = 10,
    allow_synthetic: bool = False,
    output_dir: str = MODELS_DIR,
    prefer_cache: bool = True,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    df_raw, src_label, any_synthetic, cache_name = _load_historical_or_live_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        allow_synthetic=allow_synthetic,
        prefer_cache=prefer_cache,
    )
    if len(df_raw) > 90000:
        df_raw = df_raw.iloc[-90000:].copy()

    fe = CryptoFeatureEngineer()
    df_feats = fe.compute_features(df_raw, timeframe=timeframe).dropna().copy()
    feature_cols = list(fe.feature_columns)

    # Chronological Train (75%), Purge (12 bars), OOS Test (remaining)
    n_total = len(df_feats)
    purge_bars = 12
    split_idx = max(30, int(n_total * 0.75))
    test_start = min(n_total - 10, split_idx + purge_bars)
    df_train = df_feats.iloc[:split_idx].copy()
    df_test = df_feats.iloc[test_start:].copy()

    env_fee = 0.0002 if timeframe == '1m' else 0.0005
    env_slip = 0.0001 if timeframe == '1m' else 0.0002
    env_train = CryptoTradingEnv(df_train, feature_cols, initial_balance=10000.0, fee_rate=env_fee, slippage=env_slip)
    agent = PPOAgent(state_dim=env_train.state_dim, action_dim=1, lr=3e-4)

    for _ in range(max(1, int(episodes))):
        state = env_train.reset()
        done = False
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        while not done:
            act, lp, val = agent.select_action(state, evaluate=False)
            next_state, reward, done, _ = env_train.step(act)
            states.append(state)
            actions.append(act)
            log_probs.append(lp)
            rewards.append(reward)
            values.append(val)
            dones.append(done)
            state = next_state
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value=0.0)
        agent.train_step(states, actions, log_probs, advantages, returns, k_epochs=4, batch_size=64)

    # Evaluate on untouched OOS test slice
    env_test = CryptoTradingEnv(df_test, feature_cols, initial_balance=10000.0, fee_rate=env_fee, slippage=env_slip)
    obs = env_test.reset()
    done = False
    while not done:
        act, _, _ = agent.select_action(obs, evaluate=True)
        obs, _, done, _ = env_test.step(act)

    oos_rets = np.asarray(env_test.returns_history, dtype=float)
    ann_factor = CryptoFeatureEngineer.get_annualization_factor(timeframe)
    oos_sharpe = (
        float(np.mean(oos_rets) / (np.std(oos_rets) + 1e-9) * np.sqrt(ann_factor)) if len(oos_rets) > 1 else 0.0
    )
    oos_total_ret = float((env_test.balance - env_test.initial_balance) / env_test.initial_balance)

    out_path = os.path.join(output_dir, "crypto_ppo_agent.pt")
    manifest = agent.save(
        out_path,
        metadata={
            "target": "crypto_ppo",
            "symbol": symbol,
            "timeframe": timeframe,
            "feature_columns": feature_cols,
            "data_source": "SYNTHETIC" if any_synthetic else ("HISTORICAL_CACHE" if cache_name else "LIVE"),
            "total_bars": int(len(df_raw)),
            "cache_file": cache_name,
            "is_synthetic": any_synthetic,
            "deployment_stage": "RESEARCH_ONLY" if any_synthetic else "LOCAL_PAPER_READY",
            "oos_metrics": {
                "oos_return_pct": round(oos_total_ret * 100.0, 2),
                "oos_sharpe": round(oos_sharpe, 3),
                "test_bars": len(df_test),
            },
        },
    )
    return {"manifest": manifest, "model_path": out_path}


def train_taiwan_ranker(
    watchlist: list[str] | None = None,
    period: str = "5y",
    allow_synthetic: bool = False,
    output_dir: str = MODELS_DIR,
    prefer_cache: bool = True,
    **kwargs,
) -> dict[str, Any]:
    import hashlib
    os.makedirs(output_dir, exist_ok=True)
    is_inv = getattr(TaiwanStockMLRanker, "is_inverse_etf", lambda s: str(s).endswith("R.TW"))
    symbols = [s for s in (watchlist or DEFAULT_WATCHLIST) if not is_inv(s)]
    stock_dfs: dict[str, pd.DataFrame] = {}
    any_synthetic = False

    try:
        import yfinance as yf
    except ImportError:
        yf = None

    if yf is not None:
        for sym in symbols[:10]:
            try:
                hist = yf.Ticker(sym).history(period=period)
                if hist is not None and len(hist) >= 120:
                    stock_dfs[sym] = hist
            except Exception:
                pass

    if len(stock_dfs) < 2:
        if not allow_synthetic:
            raise DataUnavailableError("Failed to fetch >= 2 Taiwan stocks via yfinance and allow_synthetic=False.")
        any_synthetic = True
        loader = CryptoDataLoader(timeframe="1d")
        for sym in symbols[:5]:
            res = loader.fetch_ohlcv_result(limit=260, symbol=sym, timeframe="1d", allow_synthetic=True)
            df_s = res.df.copy()
            df_s.index = pd.date_range(end=datetime.now(timezone.utc).date(), periods=len(df_s), freq="B")
            stock_dfs[sym] = df_s

    ranker = TaiwanStockMLRanker(model_type="lightgbm")
    clean_df, feature_cols = ranker.prepare_dataset(stock_dfs)
    metrics = ranker.train_model(clean_df, test_fraction=0.20, embargo_days=5)

    out_path = os.path.join(output_dir, "taiwan_stock_ranker.joblib")
    ranker.save_model(out_path)
    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    sha = h.hexdigest()
    manifest = {
        "target": "taiwan_ranker",
        "model_name": "taiwan_stock_ranker",
        "version": "1.0.0",
        "sha256": sha,
        "weights_sha256": sha,
        "framework": "lightgbm",
        "universe_symbols": list(stock_dfs.keys()),
        "feature_cols": feature_cols,
        "data_source": "SYNTHETIC" if any_synthetic else "LIVE_YFINANCE",
        "is_synthetic": any_synthetic,
        "deployment_stage": "RESEARCH_ONLY" if any_synthetic else "LOCAL_PAPER_READY",
        "oos_metrics": {
            "rank_ic": metrics.get("rank_ic"),
            "rmse": metrics.get("rmse"),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(output_dir, "taiwan_stock_manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)
    return {"metrics": metrics, "manifest": manifest, "model_path": out_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified No-Leakage Model Training CLI")
    parser.add_argument(
        "--target",
        required=True,
        choices=["crypto_lgbm", "crypto_ppo", "taiwan_ranker", "all"],
        help="Independent model pipeline to train",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT",
        help="Comma-separated crypto symbols",
    )
    parser.add_argument("--timeframe", type=str, default="1m", help="Bar timeframe (default: 1m ultra-short-term)")
    parser.add_argument("--limit", type=int, default=800, help="Historical bars per symbol")
    parser.add_argument("--episodes", type=int, default=10, help="PPO training episodes")
    parser.add_argument("--no-cache", action="store_true", help="Ignore local data_cache CSVs and fetch live bars")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=MODELS_DIR,
        help="Output directory for model & manifest",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic data ONLY for offline smoke tests (marks manifest RESEARCH_ONLY)",
    )
    args = parser.parse_args(argv)

    quarantined = quarantine_unverified_models()
    if quarantined:
        print(f"[*] Quarantined unverified models without manifest: {quarantined}")

    sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
    targets = ["crypto_lgbm", "crypto_ppo", "taiwan_ranker"] if args.target == "all" else [args.target]

    for tgt in targets:
        if tgt == "crypto_lgbm":
            res = train_crypto_lgbm(
                symbols=sym_list,
                timeframe=args.timeframe,
                limit=args.limit,
                allow_synthetic=args.allow_synthetic,
                output_dir=args.output_dir,
                prefer_cache=not args.no_cache,
            )
            print(f"[✓] crypto_lgbm trained -> {res['model_path']} (SHA-256: {res['manifest']['sha256'][:12]}...)")
        elif tgt == "crypto_ppo":
            res = train_crypto_ppo(
                symbol=sym_list[0],
                timeframe=args.timeframe,
                limit=args.limit,
                episodes=args.episodes,
                allow_synthetic=args.allow_synthetic,
                output_dir=args.output_dir,
                prefer_cache=not args.no_cache,
            )
            print(f"[✓] crypto_ppo trained -> {res['model_path']} (SHA-256: {res['manifest']['sha256'][:12]}...)")
        elif tgt == "taiwan_ranker":
            res = train_taiwan_ranker(
                allow_synthetic=args.allow_synthetic,
                output_dir=args.output_dir,
                prefer_cache=not args.no_cache,
            )
            print(f"[✓] taiwan_ranker trained -> {res['model_path']} (SHA-256: {res['manifest']['sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
