"""
Tests for testnet CLI ordering guard.
"""

import os
import subprocess
import sys


def test_testnet_cli_no_order_by_default():
    env = os.environ.copy()
    env.pop("ALLOW_MANUAL_TESTNET_ORDER", None)
    res = subprocess.run(
        [sys.executable, "-m", "crypto_trading.testnet_client"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0
    assert "下單回報" not in res.stdout


def test_testnet_cli_order_requires_env_and_quantity():
    # Calling with --create-order but without ALLOW_MANUAL_TESTNET_ORDER should exit 1
    env = os.environ.copy()
    env["ALLOW_MANUAL_TESTNET_ORDER"] = ""
    res = subprocess.run(
        [sys.executable, "-m", "crypto_trading.testnet_client", "--create-order"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode != 0
    assert "ALLOW_MANUAL_TESTNET_ORDER=YES" in res.stderr

    # Calling with ALLOW_MANUAL_TESTNET_ORDER=YES but quantity=0 should exit 1
    env["ALLOW_MANUAL_TESTNET_ORDER"] = "YES"
    res2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "crypto_trading.testnet_client",
            "--create-order",
            "--quantity",
            "0",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res2.returncode != 0
    assert "--quantity 必須大於 0" in res2.stderr
