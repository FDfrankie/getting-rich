import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_trading.dataset_manifest import build_dataset_manifest
from crypto_trading.model_governance import validate_model_manifest
from crypto_trading.rl_env import CryptoTradingEnv


def test_manifest_missing_explicit_approval_is_quarantined(tmp_path):
    model = tmp_path / "m.bin"
    model.write_bytes(b"model")
    sha = hashlib.sha256(b"model").hexdigest()
    (tmp_path / "m_manifest.json").write_text(json.dumps({"weights_sha256": sha}), encoding="utf-8")
    decision = validate_model_manifest(model)
    assert not decision.allowed
    assert decision.status == "INFERENCE_NOT_APPROVED"


def test_manifest_hash_and_approval_are_required(tmp_path):
    model = tmp_path / "m.bin"
    model.write_bytes(b"model")
    sha = hashlib.sha256(b"model").hexdigest()
    (tmp_path / "m_manifest.json").write_text(json.dumps({
        "weights_sha256": sha,
        "approved_for_inference": True,
        "approved_for_order_entry": False,
        "deployment_stage": "SHADOW",
    }), encoding="utf-8")
    assert validate_model_manifest(model).allowed
    assert not validate_model_manifest(model, require_order_approval=True).allowed


def test_dataset_manifest_finds_missing_bar(tmp_path):
    idx = pd.date_range("2026-01-01", periods=4, freq="1min", tz="UTC").delete(2)
    df = pd.DataFrame({"Open": 1, "High": 2, "Low": .5, "Close": 1.5, "Volume": 10}, index=idx)
    p = tmp_path / "x.csv"
    df.to_csv(p)
    manifest = build_dataset_manifest(p, symbol="X/USDT", timeframe="1m", source="test")
    assert manifest["missing_intervals"] == 1
    assert not manifest["quality_passed"]


def test_rl_deadband_and_min_holding_block_churn():
    n = 80
    df = pd.DataFrame({"Close": np.linspace(100, 101, n), "f": np.zeros(n)})
    env = CryptoTradingEnv(df, ["f"], position_deadband=.15, min_holding_bars=5, rebalance_cooldown=2)
    env.reset()
    _, _, _, first = env.step(.5)
    assert first["executed_position"] == .5
    _, _, _, second = env.step(.55)
    assert second["executed_position"] == .5
    assert second["action_blocked_reason"] == "DEADBAND"
    _, _, _, third = env.step(-.5)
    assert third["executed_position"] == .5
    assert third["action_blocked_reason"] == "MIN_HOLDING"
