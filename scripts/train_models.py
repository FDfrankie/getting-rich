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
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd

from crypto_trading.data_loader import CryptoDataLoader, DataSource, DataUnavailableError
from crypto_trading.feature_engineer import CryptoFeatureEngineer
from crypto_trading.ml_trainer import CryptoMLTrainer
from crypto_trading.rl_env import CryptoTradingEnv
from crypto_trading.rl_agent import PPOAgent
from taiwan_stock.config import DEFAULT_WATCHLIST
from taiwan_stock.ml_ranker import TaiwanStockMLRanker

MODELS_DIR = os.path.join(BASE_DIR, "data_persistence", "models")
LEGACY_DIR = os.path.join(MODELS_DIR, "legacy_unverified")


def quarantine_unverified_models() -> List[str]:
    """Moves any .joblib or .pt in data_persistence/models/ lacking a valid SHA-256 manifest to legacy_unverified/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LEGACY_DIR, exist_ok=True)
    quarantined: List[str] = []

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


def train_crypto_lgbm(
    symbols: List[str],
    timeframe: str = "1h",
    limit: int = 1200,
    allow_synthetic: bool = False,
    output_dir: str = MODELS_DIR,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    loader = CryptoDataLoader(timeframe=timeframe)
    frames: List[pd.DataFrame] = []
    sources_used: List[str] = []
    any_synthetic = False

    for sym in symbols:
        res = loader.fetch_ohlcv_result(
            limit=limit, symbol=sym, timeframe=timeframe, allow_synthetic=allow_synthetic
        )
        frames.append(res.df)
        sources_used.append(res.source.value)
        if res.is_synthetic or res.source == DataSource.SYNTHETIC:
            any_synthetic = True

    primary_df = frames[0]
    trainer = CryptoMLTrainer(model_type="lightgbm", task="regression", horizon=3, timeframe=timeframe)
    summary = trainer.walk_forward_train(primary_df, n_splits=5, purge_bars=12, embargo_bars=2)

    out_path = os.path.join(output_dir, "crypto_lgbm_weights.joblib")
    manifest = trainer.save_weights(
        out_path,
        metadata={
            "target": "crypto_lgbm",
            "symbols": symbols,
            "timeframe": timeframe,
            "data_source": "SYNTHETIC" if any_synthetic else "LIVE",
            "is_synthetic": any_synthetic,
            "deployment_stage": "RESEARCH_ONLY" if any_synthetic else "LOCAL_PAPER_READY",
            "oos_metrics": {
                "mean_rank_ic": summary["mean_rank_ic"],
                "mean_sharpe": summary["mean_sharpe"],
                "mean_accuracy": summary["mean_accuracy"],
                "mean_turnover": summary["mean_turnover"],
            },
        },
    )
    # Also write canonical crypto_lgbm_manifest.json
    canonical_manifest_path = os.path.join(output_dir, "crypto_lgbm_manifest.json")
    with open(canonical_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return {"summary": summary, "manifest": manifest, "model_path": out_path}


def train_crypto_ppo(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1000,
    episodes: int = 12,
    allow_synthetic: bool = False,
    output_dir: str = MODELS_DIR,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    loader = CryptoDataLoader(symbol=symbol, timeframe=timeframe)
    res = loader.fetch_ohlcv_result(
        limit=limit, symbol=symbol, timeframe=timeframe, allow_synthetic=allow_synthetic
    )
    any_synthetic = bool(res.is_synthetic or res.source == DataSource.SYNTHETIC)

    fe = CryptoFeatureEngineer()
    df_feats = fe.compute_features(res.df, timeframe=timeframe).dropna().copy()
    feature_cols = list(fe.feature_columns)

    # Chronological Train (75%), Purge (12 bars), OOS Test (remaining)
    n_total = len(df_feats)
    purge_bars = 12
    split_idx = max(30, int(n_total * 0.75))
    test_start = min(n_total - 10, split_idx + purge_bars)
    df_train = df_feats.iloc[:split_idx].copy()
    df_test = df_feats.iloc[test_start:].copy()

    env_train = CryptoTradingEnv(df_train, feature_cols, initial_balance=10000.0)
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
    env_test = CryptoTradingEnv(df_test, feature_cols, initial_balance=10000.0)
    obs = env_test.reset()
    done = False
    while not done:
        act, _, _ = agent.select_action(obs, evaluate=True)
        obs, _, done, _ = env_test.step(act)

    oos_rets = np.asarray(env_test.returns_history, dtype=float)
    ann_factor = CryptoFeatureEngineer.get_annualization_factor(timeframe)
    oos_sharpe = float(np.mean(oos_rets) / (np.std(oos_rets) + 1e-9) * np.sqrt(ann_factor)) if len(oos_rets) > 1 else 0.0
    oos_total_ret = float((env_test.balance - env_test.initial_balance) / env_test.initial_balance)

    out_path = os.path.join(output_dir, "crypto_ppo_agent.pt")
    manifest = agent.save(
        out_path,
        metadata={
            "target": "crypto_ppo",
            "symbol": symbol,
            "timeframe": timeframe,
            "feature_columns": feature_cols,
            "data_source": "SYNTHETIC" if any_synthetic else "LIVE",
            "is_synthetic": any_synthetic,
            "deployment_stage": "RESEARCH_ONLY" if any_synthetic else "LOCAL_PAPER_READY",
            "oos_metrics": {
                "oos_return_pct": round(oos_total_ret * 100.0, 2),
                "oos_sharpe": round(oos_sharpe, 3),
                "test_bars": int(len(df_test)),
            },
        },
    )
    return {"manifest": manifest, "model_path": out_path}


def train_taiwan_ranker(
    watchlist: List[str] | None = None,
    period: str = "2y",
    allow_synthetic: bool = False,
    output_dir: str = MODELS_DIR,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    symbols = [s for s in (watchlist or DEFAULT_WATCHLIST) if not TaiwanStockMLRanker.is_inverse_etf(s)]
    stock_dfs: Dict[str, pd.DataFrame] = {}
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
            raise DataUnavailableError(
                "Failed to fetch >= 2 Taiwan stocks via yfinance and allow_synthetic=False."
            )
        any_synthetic = True
        loader = CryptoDataLoader(timeframe="1d")
        for sym in symbols[:5]:
            res = loader.fetch_ohlcv_result(limit=260, symbol=sym, timeframe="1d", allow_synthetic=True)
            df_s = res.df.copy()
            df_s.index = pd.date_range(end=datetime.now(timezone.utc).date(), periods=len(df_s), freq="B")
            stock_dfs[sym] = df_s

    ranker = TaiwanStockMLRanker(model_type="lightgbm", target_mode="RAW_RETURN")
    clean_df, feature_cols = ranker.prepare_dataset(stock_dfs, exclude_inverse_etfs=True)
    metrics = ranker.train_model(clean_df, test_fraction=0.20, embargo_days=5)

    out_path = os.path.join(output_dir, "taiwan_stock_ranker.joblib")
    manifest = ranker.save_model(
        out_path,
        metadata={
            "target": "taiwan_ranker",
            "universe_symbols": list(stock_dfs.keys()),
            "feature_cols": feature_cols,
            "data_source": "SYNTHETIC" if any_synthetic else "LIVE_YFINANCE",
            "is_synthetic": any_synthetic,
            "deployment_stage": "RESEARCH_ONLY" if any_synthetic else "LOCAL_PAPER_READY",
            "oos_metrics": {
                "rank_ic": metrics["rank_ic"],
                "daily_mean_ic": metrics["daily_mean_ic"],
                "icir": metrics["icir"],
                "mae": metrics["mae"],
            },
        },
    )
    return {"metrics": metrics, "manifest": manifest, "model_path": out_path}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified No-Leakage Model Training CLI")
    parser.add_argument(
        "--target",
        required=True,
        choices=["crypto_lgbm", "crypto_ppo", "taiwan_ranker", "all"],
        help="Independent model pipeline to train",
    )
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT", help="Comma-separated crypto symbols")
    parser.add_argument("--timeframe", type=str, default="1h", help="Bar timeframe")
    parser.add_argument("--limit", type=int, default=800, help="Historical bars per symbol")
    parser.add_argument("--episodes", type=int, default=10, help="PPO training episodes")
    parser.add_argument("--output-dir", type=str, default=MODELS_DIR, help="Output directory for model & manifest")
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
            )
            print(f"[✓] crypto_ppo trained -> {res['model_path']} (SHA-256: {res['manifest']['sha256'][:12]}...)")
        elif tgt == "taiwan_ranker":
            res = train_taiwan_ranker(
                allow_synthetic=args.allow_synthetic,
                output_dir=args.output_dir,
            )
            print(f"[✓] taiwan_ranker trained -> {res['model_path']} (SHA-256: {res['manifest']['sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
