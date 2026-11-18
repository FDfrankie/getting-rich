#!/usr/bin/env python3
"""5-seed PPO V2 retraining & strict OOS promotion gate using hardened CryptoTradingEnv."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from crypto_trading.feature_engineer import CryptoFeatureEngineer
from crypto_trading.rl_agent import PPOAgent
from crypto_trading.rl_env import CryptoTradingEnv


def load_btc_1h(max_bars: int = 4500) -> tuple[pd.DataFrame, list[str]]:
    cache_dir = BASE_DIR / "crypto_trading" / "data_cache"
    candidates = sorted(cache_dir.glob("BTC_USDT_1h_*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        candidates = sorted(cache_dir.glob("BTC_USDT_1m_*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Missing BTC_USDT_*.csv in {cache_dir}")
    raw = pd.read_csv(candidates[0], index_col=0, parse_dates=True).sort_index()
    if len(raw) > max_bars:
        raw = raw.iloc[-max_bars:]
    fe = CryptoFeatureEngineer()
    feat_df = fe.compute_features(raw, timeframe="1h").dropna()
    feature_cols = [c for c in feat_df.columns if not c.startswith("target_") and c not in {"Open", "High", "Low", "Close", "Volume"}]
    return feat_df, feature_cols


def evaluate_agent_oos(agent: PPOAgent, test_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    env = CryptoTradingEnv(
        test_df,
        feature_cols=feature_cols,
        fee_rate=0.0002,
        slippage=0.0001,
        position_deadband=0.25,
        min_holding_bars=4,
        rebalance_cooldown=2,
        funding_rate_per_bar=0.0001 / 8.0,
        turnover_penalty_rate=0.05,
        drawdown_penalty_rate=1.2,
    )
    state = env.reset()
    done = False
    rets: list[float] = []
    prev_val = env.balance
    while not done:
        action, _, _ = agent.select_action(state, evaluate=True)
        state, _, done, _ = env.step(action)
        rets.append((env.balance / max(prev_val, 1e-9)) - 1.0)
        prev_val = env.balance

    ret_arr = np.asarray(rets, dtype=float)
    oos_ret_pct = (env.balance / env.initial_balance - 1.0) * 100.0
    sharpe = float((np.mean(ret_arr) / (np.std(ret_arr, ddof=1) + 1e-12)) * np.sqrt(24.0 * 365.0)) if len(ret_arr) > 10 else 0.0
    return {
        "oos_return_pct": round(float(oos_ret_pct), 4),
        "oos_sharpe": round(sharpe, 4),
        "test_bars": len(test_df),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="5-seed PPO V2 retraining & OOS gate.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-bars", type=int, default=4500)
    parser.add_argument("--seeds", default="42,101,202,303,404")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    df, feature_cols = load_btc_1h(max_bars=args.max_bars)
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    seed_results: list[dict[str, Any]] = []
    best_agent: PPOAgent | None = None
    best_sharpe = -1e9

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = CryptoTradingEnv(
            train_df,
            feature_cols=feature_cols,
            fee_rate=0.0002,
            slippage=0.0001,
            position_deadband=0.25,
            min_holding_bars=4,
            rebalance_cooldown=2,
            funding_rate_per_bar=0.0001 / 8.0,
            turnover_penalty_rate=0.05,
            drawdown_penalty_rate=1.2,
        )
        state_dim = env.state_dim
        agent = PPOAgent(state_dim=state_dim, action_dim=1)
        for _ in range(args.episodes):
            state = env.reset()
            done = False
            states_buf: list[np.ndarray] = []
            actions_buf: list[float] = []
            logprobs_buf: list[float] = []
            rewards_buf: list[float] = []
            vals_buf: list[float] = []
            dones_buf: list[bool] = []
            while not done:
                action, log_prob, val = agent.select_action(state, evaluate=False)
                next_state, reward, done, _ = env.step(action)
                states_buf.append(state)
                actions_buf.append(action)
                logprobs_buf.append(log_prob)
                rewards_buf.append(reward)
                vals_buf.append(val)
                dones_buf.append(done)
                state = next_state
                if len(states_buf) >= 512 or done:
                    _, _, next_val = agent.select_action(state, evaluate=True) if not done else (0.0, 0.0, 0.0)
                    adv, ret_t = agent.compute_gae(rewards_buf, vals_buf, dones_buf, next_val)
                    agent.train_step(states_buf, actions_buf, logprobs_buf, adv, ret_t)
                    states_buf.clear()
                    actions_buf.clear()
                    logprobs_buf.clear()
                    rewards_buf.clear()
                    vals_buf.clear()
                    dones_buf.clear()
        metrics = evaluate_agent_oos(agent, test_df, feature_cols)
        metrics["seed"] = seed
        seed_results.append(metrics)
        if metrics["oos_sharpe"] > best_sharpe:
            best_sharpe = metrics["oos_sharpe"]
            best_agent = agent

    med_ret = float(np.median([r["oos_return_pct"] for r in seed_results]))
    med_sharpe = float(np.median([r["oos_sharpe"] for r in seed_results]))
    promoted = med_ret > 0.0 and med_sharpe > 0.5

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "episodes_per_seed": args.episodes,
        "median_oos_return_pct": round(med_ret, 4),
        "median_oos_sharpe": round(med_sharpe, 4),
        "promoted_from_quarantine": promoted,
        "seed_results": seed_results,
    }
    out_dir = BASE_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ppo_v2_5seed_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    models_dir = BASE_DIR / "data_persistence" / "models"
    model_path = models_dir / "crypto_ppo_agent.pt"
    manifest_path = models_dir / "crypto_ppo_agent_manifest.json"
    if best_agent is not None and promoted:
        best_agent.save(str(model_path))
        sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest.update({
            "sha256": sha,
            "weights_sha256": sha,
            "deployment_stage": "SHADOW",
            "approved_for_inference": True,
            "approved_for_order_entry": False,
            "rejection_reason": "PENDING_ORDER_ENTRY_APPROVAL",
            "oos_metrics": {"median_oos_return_pct": med_ret, "median_oos_sharpe": med_sharpe, "seeds": len(seeds)},
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elif manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["deployment_stage"] = "QUARANTINED"
        manifest["approved_for_inference"] = False
        manifest["approved_for_order_entry"] = False
        manifest["rejection_reason"] = "FAILED_5SEED_OOS_GATE"
        manifest["oos_metrics_5seed"] = {"median_oos_return_pct": med_ret, "median_oos_sharpe": med_sharpe}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[✓] PPO V2 5-seed median return={med_ret:.2f}%, median Sharpe={med_sharpe:.2f}, promoted={promoted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
