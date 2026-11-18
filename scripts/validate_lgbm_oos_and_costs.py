#!/usr/bin/env python3
"""Run Global-Date Split, Purged Walk-Forward, LOAO, and 3-Tier Cost Stress Gate on crypto_lgbm."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from crypto_trading.feature_engineer import CryptoFeatureEngineer
from crypto_trading.lgbm_validation_gate import ICStabilityLinearWrapper, run_validation_gate


def load_panel(symbols: list[str], timeframe: str = "1m", horizon: int = 15) -> tuple[pd.DataFrame, list[str]]:
    cache_dir = BASE_DIR / "crypto_trading" / "data_cache"
    fe = CryptoFeatureEngineer(forward_horizons=[1, horizon])
    frames: list[pd.DataFrame] = []
    feature_cols: list[str] = []

    for sym in symbols:
        safe = sym.replace("/", "_")
        candidates = sorted(cache_dir.glob(f"{safe}_{timeframe}_*.csv"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"Missing cache CSV for {sym} ({timeframe}) in {cache_dir}")
        raw = pd.read_csv(candidates[0], index_col=0, parse_dates=True)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")
        raw = raw.sort_index()

        feat_df = fe.compute_features(raw, timeframe=timeframe)
        if not feature_cols:
            feature_cols = [c for c in feat_df.columns if not c.startswith("target_") and c not in {"Open", "High", "Low", "Close", "Volume"}]

        feat_df["target_ret"] = raw["Close"].pct_change(horizon).shift(-horizon)
        feat_df["fwd_ret_1"] = raw["Close"].pct_change(1).shift(-1)
        feat_df["symbol"] = sym
        feat_df["timestamp"] = feat_df.index
        feat_df = feat_df.dropna(subset=feature_cols + ["target_ret", "fwd_ret_1"])
        frames.append(feat_df.reset_index(drop=True))

    panel = pd.concat(frames, ignore_index=True)
    return panel, feature_cols


def update_manifests(report: dict[str, Any]) -> None:
    models_dir = BASE_DIR / "data_persistence" / "models"
    for fname in ("crypto_lgbm_manifest.json", "crypto_lgbm_weights_manifest.json"):
        mpath = models_dir / fname
        if not mpath.exists():
            continue
        data = json.loads(mpath.read_text(encoding="utf-8"))
        data["deployment_stage"] = report["recommended_stage"]
        data["approved_for_inference"] = True
        data["approved_for_order_entry"] = False
        data["validation_gate_version"] = "model-governance-v2-oos-stress"
        data["rejection_reason"] = (
            "NONE"
            if report["gate_passed"]
            else ",".join(k for k, v in report["gate_checks"].items() if not v)
        )
        data["oos_validation_gate"] = {
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "gate_passed": report["gate_passed"],
            "global_test_rank_ic": report["global_split"]["test_rank_ic"],
            "cost_scenarios": {
                k: {
                    "avg_net_return_pct": v["avg_net_return_pct"],
                    "avg_daily_sharpe": v["avg_daily_sharpe"],
                    "avg_max_drawdown_pct": v["avg_max_drawdown_pct"],
                }
                for k, v in report["global_split"]["cost_scenarios"].items()
            },
            "leave_one_asset_out": report["leave_one_asset_out"],
        }
        mpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate crypto_lgbm with Global Split, Purged WF, LOAO, and Cost Stress.")
    parser.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--report-only", action="store_true", default=True, help="Read-only validation (default)")
    parser.add_argument("--update-manifests", action="store_true", help="Explicitly update manifests after review")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    panel, feature_cols = load_panel(symbols, timeframe=args.timeframe, horizon=args.horizon)

    def make_lgbm():
        return ICStabilityLinearWrapper(
            base_factory=lambda: lgb.LGBMRegressor(
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            ),
            min_ic=0.015,
            blend_alpha=1.0,
        )

    bars_8h = 8.0 if args.timeframe == "1h" else (32.0 if args.timeframe == "15m" else (96.0 if args.timeframe == "5m" else 480.0))
    embargo = 15 if args.timeframe == "1h" else max(60, 3 * args.horizon)
    report = run_validation_gate(
        panel,
        feature_cols,
        make_lgbm,
        purge_bars=args.horizon,
        embargo_bars=embargo,
        use_hysteresis=True,
        entry_z=1.8,
        exit_z=-0.5,
        max_holding_bars=4,
        bars_per_8h=bars_8h,
    )

    out_dir = BASE_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "lgbm_validation_gate_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    gs = report["global_split"]
    lines = [
        "# LightGBM Strict OOS & Cost Stress Validation Report",
        f"- Gate Passed: **{report['gate_passed']}** | Recommended Stage: **{report['recommended_stage']}**",
        f"- Global Split Test Rank IC: **{gs['test_rank_ic']}** (Val Rank IC: {gs['val_rank_ic']})",
        "",
        "## Cost Stress Scenarios (Global Test)",
        "| Scenario | Avg Net Return % | Avg Daily Sharpe | Avg Max DD % | Total Trades |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for sc_name, sc in gs["cost_scenarios"].items():
        lines.append(
            f"| {sc_name} | {sc['avg_net_return_pct']}% | {sc['avg_daily_sharpe']} | "
            f"{sc['avg_max_drawdown_pct']}% | {sc['total_trades']} |"
        )
    lines.extend(["", "## Leave-One-Asset-Out (LOAO)", "| Held-Out Symbol | Rank IC | Base Daily Sharpe | Base Net Return % |", "| :--- | :--- | :--- | :--- |"])
    for sym, res in report["leave_one_asset_out"].items():
        lines.append(f"| {sym} | {res['rank_ic']} | {res['base_daily_sharpe']} | {res['base_net_return_pct']}% |")

    md_path = out_dir / "lgbm_validation_gate_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.update_manifests:
        update_manifests(report)

    print(f"[✓] LightGBM Gate Passed={report['gate_passed']} | Stage={report['recommended_stage']} | Report={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
