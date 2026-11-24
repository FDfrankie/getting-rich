"""P42 verification suite: FE01-FE15 linter, option decay floor, high-TO gem rescue, KB v78 seeds, and full paginated backfill."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from worldquant_brain.brain_client import BrainClient
from worldquant_brain.fastexpr_linter import lint_fastexpr
from worldquant_brain.multi_sim_client import (
    DATASET_SIM_PROFILES,
    find_locked_bronze_operators,
    min_decay_for_expression,
)
from worldquant_brain.two_stage_funnel import (
    STAGE1_PROBES,
    V8_ECONOMIC_HYPOTHESIS_SEEDS,
    classify_verdict,
)


def test_p42_fe01_to_fe15_linter_blocks_invalid_and_passes_clean() -> None:
    clean = lint_fastexpr(
        "group_rank(ts_scale(ts_backfill(cashflow_op, 60) / cap, 120), industry)",
        {"neutralization": "INDUSTRY", "truncation": 0.08, "decay": 2},
        consultant_mode=False,
    )
    assert clean["ok"] is True
    assert clean["errors"] == []

    # FE01: positional winsorize arg
    r_fe01 = lint_fastexpr("winsorize(rank(close), 4)", consultant_mode=False)
    assert r_fe01["ok"] is False and "FE01" in r_fe01["rules_triggered"]

    # FE02: 2-arg group_mean
    r_fe02 = lint_fastexpr("group_mean(returns, subindustry)", consultant_mode=False)
    assert r_fe02["ok"] is False and "FE02" in r_fe02["rules_triggered"]

    # FE03: un-densified pv13 grouping
    r_fe03 = lint_fastexpr("group_rank(returns, pv13_h2_sector)", consultant_mode=False)
    assert r_fe03["ok"] is False and "FE03" in r_fe03["rules_triggered"]

    # FE04: bare vec_* field
    r_fe04 = lint_fastexpr("rank(vec_nws18_score)", consultant_mode=False)
    assert r_fe04["ok"] is False and "FE04" in r_fe04["rules_triggered"]

    # FE09: Genius-only operator blocked in Bronze/Silver mode
    r_fe09 = lint_fastexpr("group_cartesian_product(sector, industry)", consultant_mode=False)
    assert r_fe09["ok"] is False and "FE09" in r_fe09["rules_triggered"]

    # FE10: invalid non-positive window
    r_fe10 = lint_fastexpr("ts_rank(close, 0)", consultant_mode=False)
    assert r_fe10["ok"] is False and "FE10" in r_fe10["rules_triggered"]

    # FE15: placeholder token
    r_fe15 = lint_fastexpr("group_rank(ts_backfill({f}, 20), industry)", consultant_mode=False)
    assert r_fe15["ok"] is False and "FE15" in r_fe15["rules_triggered"]


def test_p42_option_decay_floor_and_stabilized_probes() -> None:
    assert DATASET_SIM_PROFILES["option"]["decay"] >= 6
    assert min_decay_for_expression("group_rank(ts_delta(parkinson_volatility_10, 2), industry)", "option8") >= 6

    probes = dict(STAGE1_PROBES)
    assert "ts_decay_linear(" in probes["cap_delta2_zscore20_sub"]
    assert "ts_decay_linear(" in probes["delta4_sector_neut"]


def test_p42_high_sharpe_high_turnover_gem_rescue() -> None:
    # Matches live 2026-11-24 Page 1 gem j2V5zpGk (Sharpe=1.77, TO=135%, Fit=0.25)
    verdict, _ = classify_verdict(
        {
            "sharpe": 1.77,
            "fitness": 0.25,
            "turnover": 1.35,
            "margin": 0.00008,
            "checks_passed": False,
            "failed_checks": ["LOW_FITNESS", "HIGH_TURNOVER"],
            "settings": {"delay": 1},
        }
    )
    assert verdict == "NEAR_MISS"

    # Matches Sharpe=1.42, TO=0.95, Fit=0.28 -> also rescuable via decay/hump
    verdict2, _ = classify_verdict(
        {
            "sharpe": 1.42,
            "fitness": 0.28,
            "turnover": 0.95,
            "margin": 0.00012,
            "checks_passed": False,
            "failed_checks": ["LOW_FITNESS", "HIGH_TURNOVER"],
            "settings": {"delay": 1},
        }
    )
    assert verdict2 == "NEAR_MISS"


def test_p42_kb78_orthogonal_seeds_are_bronze_safe_and_lint_clean() -> None:
    p42_seeds = [s for s in V8_ECONOMIC_HYPOTHESIS_SEEDS if str(s.get("name", "")).startswith("p42_")]
    assert len(p42_seeds) == 8
    for seed in p42_seeds:
        expr = str(seed["expression"])
        assert find_locked_bronze_operators(expr) == [], f"Locked op in {seed['name']}"
        lint = lint_fastexpr(expr, {"neutralization": seed["neutralization"], "decay": seed["decay"], "truncation": 0.08}, consultant_mode=False)
        assert lint["ok"] is True, f"Lint failed for {seed['name']}: {lint['errors']}"


def test_p42_backfill_all_passed_alphas_paginates_active_and_unsubmitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "alpha_vault.db"
    vault_json = tmp_path / "qualified_alpha_formulas_vault.json"
    monkeypatch.setenv("ALPHA_VAULT_DB", str(db_file))

    client = BrainClient(email="test@example.com", password="secret")
    client.authenticated = True
    monkeypatch.setattr(client, "_quota_db", lambda: str(db_file))

    active_page = {
        "results": [
            {
                "id": "VkNrZJa8",
                "regular": {"code": "signed_power(ts_rank(close, 5) - 0.6, 2) * -1"},
                "settings": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                "is": {"sharpe": 1.85, "fitness": 1.00, "turnover": 0.5712, "margin": 0.00058, "returns": 0.1655, "drawdown": 0.063, "checks": []},
            }
        ]
    }
    unsub_page_1 = {
        "results": [
            {
                "id": "P05pjYgE",
                "regular": {"code": "ts_decay_linear(group_neutralize(rank(liabilities / (abs(assets) + 0.001)), subindustry), 3)"},
                "settings": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "SUBINDUSTRY", "decay": 4, "truncation": 0.08},
                "is": {
                    "sharpe": 1.50, "fitness": 1.25, "turnover": 0.0153, "margin": 0.011278, "returns": 0.0863, "drawdown": 0.056,
                    "checks": [{"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "value": 0.9}],
                },
            },
            {
                "id": "FAIL0001",
                "regular": {"code": "rank(close)"},
                "settings": {"region": "USA", "universe": "TOP3000", "delay": 1},
                "is": {
                    "sharpe": 1.40, "fitness": 1.10, "turnover": 0.20, "margin": 0.001,
                    "checks": [{"name": "CONCENTRATED_WEIGHT", "result": "FAIL"}],
                },
            },
        ]
    }
    unsub_page_2 = {
        "results": [
            {
                "id": "9qle3Nb9",
                "regular": {"code": "group_rank(ts_av_diff(ts_backfill(common_stock_repurchase_payment, 60) / cap, 60), subindustry)"},
                "settings": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "SUBINDUSTRY", "decay": 0, "truncation": 0.08},
                "is": {"sharpe": 1.39, "fitness": 1.03, "turnover": 0.1174, "margin": 0.001158, "returns": 0.072, "drawdown": 0.05, "checks": []},
            }
        ]
    }

    def fake_get(url, params=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"{}"
        st = (params or {}).get("status")
        off = (params or {}).get("offset", 0)
        if st == "ACTIVE":
            resp.json.return_value = active_page if off == 0 else {"results": []}
        else:
            resp.json.return_value = unsub_page_1 if off == 0 else (unsub_page_2 if off == 2 else {"results": []})
        return resp

    mock_sess = MagicMock()
    mock_sess.get.side_effect = fake_get
    client.session = mock_sess

    res = client.backfill_all_passed_alphas(max_pages=5, page_size=2, vault_json_path=vault_json)
    assert res["ok"] is True
    assert res["submitted_archived"] == 1
    assert res["unsubmitted_archived"] == 2
    assert res["skipped_failed_checks"] == 1
    assert res["total_archived"] == 3
    assert res["submitted_count"] == 1
    assert res["consultant_reserve_count"] == 1
    assert res["bronze_submittable_count"] == 1
    exported = json.loads(vault_json.read_text(encoding="utf-8"))
    assert [x["alpha_id"] for x in exported["submitted"]] == ["VkNrZJa8"]
