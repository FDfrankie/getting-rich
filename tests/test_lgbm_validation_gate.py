from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from crypto_trading.lgbm_validation_gate import (
    DEFAULT_COST_SCENARIOS,
    evaluate_positions_with_costs,
    make_global_timestamp_splits,
    run_validation_gate,
)
from crypto_trading.model_governance import sha256_file, validate_model_manifest


class _DummyLinearModel:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DummyLinearModel":
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        if X.shape[1] > 0:
            self.coef_[0] = 1.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


def test_taiwan_ranker_manifest_alias_resolves_and_blocks_orders(tmp_path: Path) -> None:
    model_path = tmp_path / "taiwan_stock_ranker.joblib"
    model_path.write_bytes(b"taiwan-ranker-bytes")
    digest = sha256_file(model_path)
    manifest_path = tmp_path / "taiwan_stock_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "weights_sha256": digest,
                "deployment_stage": "REPORTING_ONLY",
                "approved_for_inference": True,
                "approved_for_order_entry": False,
            }
        ),
        encoding="utf-8",
    )

    inf = validate_model_manifest(model_path, require_inference_approval=True, require_order_approval=False)
    ord_dec = validate_model_manifest(model_path, require_inference_approval=True, require_order_approval=True)
    assert inf.allowed is True
    assert ord_dec.allowed is False
    assert ord_dec.status == "ORDER_ENTRY_NOT_APPROVED"


def test_global_timestamp_split_has_zero_overlap_and_respects_embargo() -> None:
    ts = pd.date_range("2026-01-01", periods=1000, freq="1min", tz="UTC")
    rows = []
    for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        for t in ts:
            rows.append({"timestamp": t, "symbol": sym, "f1": 0.1, "target_ret": 0.01, "fwd_ret_1": 0.001})
    df = pd.DataFrame(rows)

    tr, va, te = make_global_timestamp_splits(df, purge_bars=15, embargo_bars=60)
    tr_ts = set(tr["timestamp"])
    va_ts = set(va["timestamp"])
    te_ts = set(te["timestamp"])
    assert tr_ts.isdisjoint(va_ts)
    assert va_ts.isdisjoint(te_ts)
    assert tr_ts.isdisjoint(te_ts)
    assert (va["timestamp"].min() - tr["timestamp"].max()).total_seconds() >= 75 * 60


def test_cost_stress_monotonically_reduces_net_return() -> None:
    ts = pd.date_range("2026-01-01", periods=2000, freq="1min", tz="UTC")
    rng = np.random.default_rng(7)
    preds = rng.normal(0, 1, size=len(ts))
    fwd = rng.normal(0.0001, 0.001, size=len(ts))

    res = [evaluate_positions_with_costs(ts, preds, fwd, scenario=sc) for sc in DEFAULT_COST_SCENARIOS]
    assert res[0]["net_return_pct"] >= res[1]["net_return_pct"] >= res[2]["net_return_pct"]


def test_run_validation_gate_end_to_end_synthetic() -> None:
    ts = pd.date_range("2026-01-01", periods=400, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    rows = []
    for sym in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        sig = rng.normal(0, 0.01, size=len(ts))
        for i, t in enumerate(ts):
            rows.append(
                {
                    "timestamp": t,
                    "symbol": sym,
                    "f1": float(sig[i]),
                    "f2": float(rng.normal(0, 1)),
                    "target_ret": float(sig[i] + rng.normal(0, 0.002)),
                    "fwd_ret_1": float(sig[i] * 0.2),
                }
            )
    panel = pd.DataFrame(rows)
    report = run_validation_gate(panel, ["f1", "f2"], _DummyLinearModel, purge_bars=5, embargo_bars=10)
    assert set(report["leave_one_asset_out"].keys()) == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}
    assert len(report["purged_walk_forward"]) == 5
    assert "severe_2_0x_funding" in report["global_split"]["cost_scenarios"]
