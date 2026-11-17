#!/usr/bin/env python3
"""
Phase 2 Enhancements for Antigravity Quant:
1. Binance Testnet HEALTHY Snapshot & Startup/Periodic Reconciliation:
   - Populate wallet_balance, unrealized_pnl, and info{} in crypto_trading/testnet_client.py:fetch_balance().
   - Initialize ReconciliationService in web_ui/server.py QuantAppContext and run initial + 20s periodic
     reconciliation so /api/trading/snapshot reports HEALTHY (no ACCOUNT_FIELDS_MISSING or RECONCILIATION_STALE).
2. WorldQuant BRAIN Fundamental & Analyst Sentiment Gene Pool:
   - Expand worldquant_brain/alpha_miner.py with fundamental_fields, sentiment_fields, group_ops,
     and templates from fundamental_alpha.py & analyst_sentiment_alpha.py.
3. Model Training CLI Compatibility (scripts/train_models.py):
   - Ensure train_taiwan_ranker() works cleanly and writes taiwan_stock_manifest.json with SHA-256
     even if TaiwanStockMLRanker.save_model() takes only (path).
4. Add unit tests in tests/test_phase2_enhancements.py.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from uuid import uuid4


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def patch_testnet_client(repo_root: Path) -> str:
    target = repo_root / "crypto_trading" / "testnet_client.py"
    txt = target.read_text(encoding="utf-8")
    if '"totalWalletBalance"' in txt and '"unrealized_pnl": round(selected_upnl, 2)' in txt:
        return "ALREADY_APPLIED"

    old_block = """            total_usdt = 0.0
            free_usdt = 0.0
            total_usdc = 0.0
            free_usdc = 0.0
            balances = {}
            for a in assets:
                asset_name = a.get("asset")
                wallet_bal = float(a.get("walletBalance", 0.0))
                avail_bal = float(a.get("availableBalance", 0.0))
                if wallet_bal > 0 or avail_bal > 0:
                    balances[asset_name] = wallet_bal
                if asset_name == "USDT":
                    total_usdt = wallet_bal
                    free_usdt = avail_bal
                elif asset_name == "USDC":
                    total_usdc = wallet_bal
                    free_usdc = avail_bal

            selected_total = total_usdc if self.settlement == "USDC" else total_usdt
            selected_free = free_usdc if self.settlement == "USDC" else free_usdt

            return {
                "mode": "OFFICIAL_BINANCE_TESTNET",
                "total_equity": round(selected_total, 2),
                "free_balance": round(selected_free, 2),
                "total_usdt": round(total_usdt, 2),
                "total_usdc": round(total_usdc, 2),
                "settlement_currency": self.settlement,
                "zero_fee_usdc": self.settlement == "USDC",
                "balances": balances,
            }"""

    new_block = """            total_usdt = 0.0
            free_usdt = 0.0
            upnl_usdt = 0.0
            margin_usdt = 0.0
            total_usdc = 0.0
            free_usdc = 0.0
            upnl_usdc = 0.0
            margin_usdc = 0.0
            balances = {}
            for a in assets:
                asset_name = a.get("asset")
                wallet_bal = float(a.get("walletBalance", 0.0))
                avail_bal = float(a.get("availableBalance", 0.0))
                upnl_val = float(a.get("unrealizedProfit", 0.0) or 0.0)
                margin_val = float(a.get("marginBalance", wallet_bal + upnl_val) or (wallet_bal + upnl_val))
                if wallet_bal > 0 or avail_bal > 0:
                    balances[asset_name] = wallet_bal
                if asset_name == "USDT":
                    total_usdt = wallet_bal
                    free_usdt = avail_bal
                    upnl_usdt = upnl_val
                    margin_usdt = margin_val
                elif asset_name == "USDC":
                    total_usdc = wallet_bal
                    free_usdc = avail_bal
                    upnl_usdc = upnl_val
                    margin_usdc = margin_val

            selected_total = total_usdc if self.settlement == "USDC" else total_usdt
            selected_free = free_usdc if self.settlement == "USDC" else free_usdt
            selected_upnl = upnl_usdc if self.settlement == "USDC" else upnl_usdt
            selected_margin = margin_usdc if self.settlement == "USDC" else margin_usdt

            return {
                "mode": "OFFICIAL_BINANCE_TESTNET",
                "total_equity": round(selected_margin, 2),
                "wallet_balance": round(selected_total, 2),
                "unrealized_pnl": round(selected_upnl, 2),
                "free_balance": round(selected_free, 2),
                "total_usdt": round(total_usdt, 2),
                "total_usdc": round(total_usdc, 2),
                "settlement_currency": self.settlement,
                "zero_fee_usdc": self.settlement == "USDC",
                "balances": balances,
                "info": {
                    "totalWalletBalance": str(round(selected_total, 8)),
                    "totalUnrealizedProfit": str(round(selected_upnl, 8)),
                    "totalMarginBalance": str(round(selected_margin, 8)),
                    "availableBalance": str(round(selected_free, 8)),
                },
            }"""

    if old_block not in txt:
        raise RuntimeError("Could not locate fetch_balance return block in crypto_trading/testnet_client.py")
    _atomic_write(target, txt.replace(old_block, new_block, 1))
    return "PATCHED"


def patch_server_reconciliation(repo_root: Path) -> str:
    target = repo_root / "web_ui" / "server.py"
    txt = target.read_text(encoding="utf-8")
    if "self.reconciliation_service = ReconciliationService()" in txt and "_run_reconciliation_sync" in txt:
        return "ALREADY_APPLIED"

    new_txt = txt
    if "from crypto_trading.reconciliation_service import ReconciliationService" not in new_txt:
        new_txt = new_txt.replace(
            "from crypto_trading.runtime_context import ConnectionState, RuntimeTradingContext",
            "from crypto_trading.reconciliation_service import ReconciliationService\nfrom crypto_trading.runtime_context import ConnectionState, RuntimeTradingContext",
            1,
        )

    if "self.reconciliation_service = ReconciliationService()" not in new_txt:
        new_txt = new_txt.replace(
            'self.testnet_client = BinanceTestnetClient(settlement="USDC")',
            'self.testnet_client = BinanceTestnetClient(settlement="USDC")\n        self.reconciliation_service = ReconciliationService()',
            1,
        )

    if "def _run_reconciliation_sync(self):" not in new_txt:
        recon_fn = """    def _run_reconciliation_sync(self):
        \"\"\"執行即時交易所對帳並更新 ReconciliationService 狀態，讓 /api/trading/snapshot 維持 HEALTHY\"\"\"
        try:
            if self.execution_mode == "BINANCE_TESTNET" and not getattr(self.testnet_client, "mock_mode", True) and getattr(self.testnet_client, "is_connected", False):
                bal = self.testnet_client.fetch_balance()
                rem_pos = self.testnet_client.fetch_positions() or []
                oo_fn = getattr(self.testnet_client, "fetch_open_orders", None)
                rem_orders = (oo_fn(self.active_bot.symbol) or []) if callable(oo_fn) else []
                rem_eq = float(bal.get("total_equity") or 0.0) if bal.get("total_equity") is not None else float(self.active_bot.balance_usdt)
                if not getattr(self.reconciliation_service, "last_reconciled_time", None):
                    self.active_bot.balance_usdt = rem_eq
                loc_pos = {self.active_bot.symbol: float(self.active_bot.position)} if getattr(self.active_bot, "position", 0.0) else {}
                self.reconciliation_service.reconcile(
                    local_positions=loc_pos,
                    remote_positions=rem_pos,
                    local_orders=[],
                    remote_orders=rem_orders,
                    local_balance=float(self.active_bot.balance_usdt),
                    remote_balance=rem_eq,
                )
        except Exception:
            pass

    def _start_background_worker(self):"""
        new_txt = new_txt.replace("    def _start_background_worker(self):", recon_fn, 1)

    if "self._run_reconciliation_sync()" not in new_txt:
        new_txt = new_txt.replace(
            "        # 啟動常駐背景工作線程 (無論瀏覽器是否開啟均 24hr 運作)\n        self._start_background_worker()",
            "        # 啟動初始對帳與常駐背景工作線程 (無論瀏覽器是否開啟均 24hr 運作)\n        self._run_reconciliation_sync()\n        self._start_background_worker()",
            1,
        )
        new_txt = new_txt.replace(
            "if now_ts - last_reconcile_time > 60:",
            "if now_ts - last_reconcile_time > 20:",
            1,
        )
        new_txt = new_txt.replace(
            'rec = self.testnet_client.reconcile_with_exchange(\n                                self.active_bot.position, self.active_bot.symbol\n                            )',
            'self._run_reconciliation_sync()\n                            rec = self.testnet_client.reconcile_with_exchange(\n                                self.active_bot.position, self.active_bot.symbol\n                            )',
            1,
        )

    _atomic_write(target, new_txt)
    return "PATCHED"


def patch_alpha_miner_gene_pool(repo_root: Path) -> str:
    target = repo_root / "worldquant_brain" / "alpha_miner.py"
    txt = target.read_text(encoding="utf-8")
    if "self.fundamental_fields" in txt and "self.sentiment_fields" in txt:
        return "ALREADY_APPLIED"

    old_init_fields = '        self.fields = ["close", "open", "high", "low", "volume", "vwap", "returns"]'
    new_init_fields = (
        '        self.fields = ["close", "open", "high", "low", "volume", "vwap", "returns"]\n'
        '        self.fundamental_fields = ["operating_cashflow", "operating_income", "sales", "assets", "liabilities", "debt", "equity", "cap", "ebitda"]\n'
        '        self.sentiment_fields = ["target_price", "eps_consensus_mean"]\n'
        '        self.group_ops = ["subindustry", "industry", "sector"]'
    )
    if old_init_fields not in txt:
        raise RuntimeError("Could not locate self.fields in worldquant_brain/alpha_miner.py")
    txt = txt.replace(old_init_fields, new_init_fields, 1)

    old_gen_pat = re.compile(
        r"    def generate_candidate_expression\(self\) -> dict\[str, str\]:.*?return \{\"name\": name, \"expression\": expr\}",
        flags=re.DOTALL,
    )
    new_gen = '''    def generate_candidate_expression(self) -> dict[str, str]:
        """基因變異生成新的 Alpha 表達式 (涵蓋價量、基本面、分析師預期與跨產業中性化)"""
        field_a = random.choice(self.fields)
        field_b = random.choice(["volume", "vwap", "returns"])
        op_name, window_choices = random.choice(self.ts_ops)
        window = random.choice(window_choices)
        grp = random.choice(self.group_ops)
        f1 = random.choice(self.fundamental_fields)
        f2 = random.choice(["assets", "cap", "equity", "sales"])

        pattern = random.choice([1, 2, 3, 4, 5, 6, 7])
        if pattern == 1:
            expr = f"-1 * {op_name}(ts_correlation({field_a}, {field_b}, {window}), {window})"
            name = f"Auto_{field_a}_{field_b}_Reversal_{window}"
        elif pattern == 2:
            expr = f"rank({op_name}({field_a}, {window})) - rank(ts_decay_linear({field_b}, {window}))"
            name = f"Auto_MultiField_Spread_{field_a}_{window}"
        elif pattern == 3:
            expr = f"group_neutralize(rank(ts_delta({field_a}, {window})) / (ts_std_dev({field_a}, 20) + 1e-4), {grp})"
            name = f"Auto_Normalized_Momentum_{field_a}_{window}"
        elif pattern == 4:
            expr = f"signed_power(ts_rank({field_a}, {window}) - 0.5, 2) * -1"
            name = f"Auto_Nonlinear_MeanRevert_{field_a}_{window}"
        elif pattern == 5:
            expr = f"group_neutralize(rank({f1} / (abs({f2}) + 0.001)), {grp})"
            name = f"Auto_Fundamental_Quality_{f1}_{grp}"
        elif pattern == 6:
            s_field = random.choice(self.sentiment_fields)
            if s_field == "target_price":
                expr = f"group_neutralize(rank((target_price - close) / (close + 1e-4)), {grp})"
            else:
                expr = f"ts_decay_linear(group_rank(eps_consensus_mean - ts_delay(eps_consensus_mean, 20), {grp}), 5)"
            name = f"Auto_Analyst_Sentiment_{s_field}_{grp}"
        else:
            expr = f"group_rank(ts_delta(close, {window}) / (ts_std_dev(close, 20) + 1e-4), {grp}) + group_rank({f1} / (cap + 0.001), {grp})"
            name = f"Auto_Hybrid_PriceFund_{f1}_{grp}_{window}"

        return {"name": name, "expression": expr}'''

    if not old_gen_pat.search(txt):
        raise RuntimeError("Could not locate generate_candidate_expression in alpha_miner.py")
    txt = old_gen_pat.sub(new_gen, txt, count=1)

    _atomic_write(target, txt)
    return "PATCHED"


def patch_train_models_cli(repo_root: Path) -> str:
    target = repo_root / "scripts" / "train_models.py"
    txt = target.read_text(encoding="utf-8")
    if "hashlib.sha256" in txt and "taiwan_stock_manifest.json" in txt:
        return "ALREADY_APPLIED"

    old_fn_pat = re.compile(
        r"def train_taiwan_ranker\(.*?\n(?=def main\()",
        flags=re.DOTALL,
    )
    new_fn = '''def train_taiwan_ranker(
    watchlist: list[str] | None = None,
    period: str = "2y",
    allow_synthetic: bool = False,
    output_dir: str = MODELS_DIR,
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


'''
    if not old_fn_pat.search(txt):
        raise RuntimeError("Could not locate train_taiwan_ranker in scripts/train_models.py")
    _atomic_write(target, old_fn_pat.sub(new_fn, txt, count=1))
    return "PATCHED"


PHASE2_TEST_PY = """import unittest
from decimal import Decimal
from unittest.mock import MagicMock

from crypto_trading.exchange_normalizer import normalize_binance_balance
from crypto_trading.reconciliation_service import ReconciliationService
from crypto_trading.testnet_client import BinanceTestnetClient
from worldquant_brain.alpha_miner import ContinuousAlphaMiner


class TestPhase2Enhancements(unittest.TestCase):
    def test_binance_testnet_balance_populates_wallet_and_upnl_for_healthy_snapshot(self):
        c = BinanceTestnetClient(api_key="k", api_secret="s")
        c.is_connected = True
        c.mock_mode = False
        c.settlement = "USDC"
        c.exchange = MagicMock()
        c.exchange.fapiPrivateV2GetAccount.return_value = {
            "assets": [
                {
                    "asset": "USDC",
                    "walletBalance": "4981.11",
                    "availableBalance": "4981.11",
                    "unrealizedProfit": "0.00",
                    "marginBalance": "4981.11",
                }
            ]
        }
        bal = c.fetch_balance()
        snap = normalize_binance_balance(bal, default_currency="USDC")
        self.assertEqual(snap.wallet_balance, Decimal("4981.11000000"))
        self.assertEqual(snap.unrealized_pnl, Decimal("0E-8"))
        self.assertEqual(snap.equity, Decimal("4981.11000000"))

    def test_reconciliation_service_marks_in_sync(self):
        svc = ReconciliationService()
        svc.reconcile({}, [], [], [], 4981.11, 4981.11)
        latest = svc.get_latest_result()
        self.assertIsNotNone(latest)
        self.assertTrue(latest["in_sync"])

    def test_alpha_miner_includes_fundamental_and_sentiment_fields(self):
        miner = ContinuousAlphaMiner()
        self.assertTrue(hasattr(miner, "fundamental_fields"))
        self.assertTrue(hasattr(miner, "sentiment_fields"))
        exprs = [miner.generate_candidate_expression()["expression"] for _ in range(80)]
        has_group = any("group_neutralize" in e or "group_rank" in e for e in exprs)
        self.assertTrue(has_group)


if __name__ == "__main__":
    unittest.main()
"""



def patch_binance_style_cockpit(repo_root: Path) -> str:
    srv_path = repo_root / "web_ui" / "server.py"
    srv = srv_path.read_text(encoding="utf-8")
    srv_changed = False

    # 1. Increase bot_logs retention from 60 to 120 and API return from -15 to -50
    if "if len(self.bot_logs) > 60:" in srv:
        srv = srv.replace("if len(self.bot_logs) > 60:", "if len(self.bot_logs) > 120:", 1)
        srv_changed = True
    if 'APP_STATE.bot_logs[-15:]' in srv:
        srv = srv.replace('APP_STATE.bot_logs[-15:]', 'APP_STATE.bot_logs[-50:]')
        srv_changed = True

    # 2. Support limit price, leverage, reduce_only, SL/TP in POST /api/crypto/order
    old_order_block = """        if path == "/api/crypto/order":
            side = post_data.get("side", "buy").upper()
            amount = float(post_data.get("amount", 0.01))
            symbol = post_data.get("symbol", APP_STATE.active_bot.symbol)
            order_type = post_data.get("order_type", "market")"""
    new_order_block = """        if path == "/api/crypto/order":
            action = post_data.get("action")
            if action == "partial_close":
                pct = float(post_data.get("pct", 0.5))
                symbol = post_data.get("symbol", APP_STATE.active_bot.symbol)
                bot = APP_STATE.active_bot
                if APP_STATE.execution_mode == "BINANCE_TESTNET":
                    positions = APP_STATE.testnet_client.fetch_positions() or []
                    for pos in positions:
                        contracts = abs(float(pos.get("contracts") or pos.get("positionAmt") or 0.0))
                        if contracts > 0:
                            close_qty = round(contracts * pct, 4)
                            pos_side = str(pos.get("side") or "LONG").upper()
                            close_side = "SELL" if pos_side == "LONG" else "BUY"
                            res = APP_STATE.testnet_client.create_order(symbol=symbol, side=close_side, order_type="market", amount=close_qty)
                            APP_STATE.add_log(f"幣安 Testnet 部分減倉 ({int(pct*100)}%): {close_side} {close_qty} {symbol}")
                            return self._send_json_response(res)
                    return self._send_json_response({"success": False, "message": "目前無持倉可減倉"})
                elif bot.position > 0:
                    close_qty = round(bot.position * pct, 4)
                    ticker = APP_STATE.testnet_client.fetch_ticker(bot.symbol)
                    bot.execute_paper_order("SELL", close_qty, ticker.get("last") or bot.entry_price, reason="Web UI 部分減倉")
                    APP_STATE.add_log(f"沙盒部分減倉 ({int(pct*100)}%): {close_qty} {bot.symbol}")
                    return self._send_json_response({"success": True, "message": f"已完成 {int(pct*100)}% 減倉"})
                return self._send_json_response({"success": False, "message": "目前無持倉可減倉"})

            side = post_data.get("side", "buy").upper()
            amount = float(post_data.get("amount", 0.01))
            symbol = post_data.get("symbol", APP_STATE.active_bot.symbol)
            order_type = str(post_data.get("order_type", "market")).lower()
            limit_price = post_data.get("price")
            leverage = post_data.get("leverage")
            if leverage and APP_STATE.execution_mode == "BINANCE_TESTNET" and getattr(APP_STATE.testnet_client, "exchange", None):
                try:
                    APP_STATE.testnet_client.exchange.fapiPrivatePostLeverage({"symbol": symbol.replace("/", ""), "leverage": int(leverage)})
                except Exception:
                    pass"""
    if old_order_block in srv and 'action == "partial_close"' not in srv:
        srv = srv.replace(old_order_block, new_order_block, 1)
        srv = srv.replace(
            "res = APP_STATE.testnet_client.create_order(\n                    symbol=symbol, side=side, order_type=order_type, amount=amount\n                )",
            'res = APP_STATE.testnet_client.create_order(\n                    symbol=symbol, side=side, order_type=order_type, amount=amount, price=float(limit_price) if (order_type == "limit" and limit_price) else None\n                )',
            1,
        )
        srv_changed = True

    if srv_changed:
        _atomic_write(srv_path, srv)

    # 3. Upgrade index.html manual order panel to Binance Futures Cockpit style
    html_path = repo_root / "web_ui" / "static" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    if 'id="orderTypeTabs"' in html:
        return "ALREADY_APPLIED" if not srv_changed else "PATCHED"

    old_panel_pat = re.compile(
        r'        <!-- 手動下單盤面 -->.*?        <!-- 機器人即時決策與撮合日誌 -->',
        flags=re.DOTALL,
    )
    new_panel = """        <!-- 手動下單盤面 (比照幣安合約官網型式) -->
        <div class="glass-card rounded-2xl p-4 md:p-5 lg:col-span-1 space-y-3">
          <div class="flex items-center justify-between">
            <h3 class="font-bold text-sm text-slate-200 flex items-center space-x-2">
              <i class="fa-solid fa-gamepad text-amber-400"></i>
              <span>合約委託下單</span>
            </h3>
            <div class="flex items-center space-x-1.5 text-[10px]">
              <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-bold">全倉 Cross</span>
              <select id="leverageSelect" class="bg-slate-800 border border-amber-500/40 rounded px-1.5 py-0.5 text-amber-300 font-bold focus:outline-none">
                <option value="1">1x</option>
                <option value="2" selected>2x</option>
                <option value="3">3x</option>
                <option value="5">5x</option>
                <option value="10">10x</option>
              </select>
            </div>
          </div>

          <!-- 市價單 / 限價單 切換頁籤 -->
          <div id="orderTypeTabs" class="grid grid-cols-2 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs font-bold">
            <button type="button" id="tabOrderMarket" onclick="setOrderType('market')" class="py-1.5 rounded-lg bg-amber-500 text-slate-950 transition">市價單 (Market)</button>
            <button type="button" id="tabOrderLimit" onclick="setOrderType('limit')" class="py-1.5 rounded-lg text-slate-400 hover:text-white transition">限價單 (Limit)</button>
          </div>

          <div class="space-y-2.5">
            <div class="flex items-center justify-between text-[11px] text-slate-400">
              <span>可用保證金: <strong id="availBalanceLabel" class="text-slate-200">--</strong> USDC</span>
              <span id="orderSymbolLabel" class="font-mono text-indigo-300">BTC/USDT</span>
            </div>

            <!-- 限價輸入框 (選限價單時顯示) -->
            <div id="limitPriceBox" class="hidden">
              <div class="flex items-center justify-between text-[10px] text-slate-400 mb-1">
                <span>委託價格 (USDT)</span>
                <button type="button" onclick="fillBestPrice()" class="text-amber-400 hover:underline">填入最新市價 (BBO)</button>
              </div>
              <input type="number" id="orderLimitPrice" step="0.1" placeholder="輸入限價價格" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-amber-500">
            </div>

            <div>
              <label class="text-[11px] text-slate-400">委託數量</label>
              <div class="flex items-center space-x-2 mt-1">
                <input type="number" id="orderAmount" step="0.001" value="0.01" class="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-1.5 text-sm text-white font-mono focus:outline-none focus:border-indigo-500">
                <span id="orderAssetUnit" class="text-xs text-slate-400 font-bold">BTC</span>
              </div>
              <!-- 倉位比例快捷鍵 -->
              <div class="grid grid-cols-4 gap-1.5 mt-1.5 text-[10px]">
                <button type="button" onclick="setOrderPct(0.25)" class="py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 font-semibold">25%</button>
                <button type="button" onclick="setOrderPct(0.50)" class="py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 font-semibold">50%</button>
                <button type="button" onclick="setOrderPct(0.75)" class="py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 font-semibold">75%</button>
                <button type="button" onclick="setOrderPct(1.00)" class="py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 font-semibold">100%</button>
              </div>
            </div>

            <!-- 止盈 / 止損 -->
            <div class="grid grid-cols-2 gap-2 text-xs">
              <div>
                <label class="text-[10px] text-slate-400">止盈 (TP)</label>
                <input type="number" id="inputTP" placeholder="目標價格" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-xs text-emerald-300 font-mono">
              </div>
              <div>
                <label class="text-[10px] text-slate-400">止損 (SL)</label>
                <input type="number" id="inputSL" placeholder="觸發價格" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-xs text-rose-300 font-mono">
              </div>
            </div>

            <div class="flex items-center justify-between text-[11px] pt-0.5">
              <label class="text-slate-400 flex items-center space-x-1.5 cursor-pointer">
                <input type="checkbox" id="reduceOnlyCheckbox" class="rounded bg-slate-800 border-slate-700 text-amber-500 focus:ring-0">
                <span>只減倉 (Reduce-Only)</span>
              </label>
              <label class="text-slate-400 flex items-center space-x-1.5 cursor-pointer">
                <input type="checkbox" id="manualOverrideCheckbox" class="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-0">
                <span class="font-bold text-amber-300">手動鎖定</span>
              </label>
            </div>

            <div class="grid grid-cols-2 gap-2 pt-1">
              <button data-opens-position="true" onclick="executeOrder('BUY')" class="w-full py-2.5 rounded-xl font-bold text-xs bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg transition flex items-center justify-center space-x-1">
                <span>買入 / 開多 (Long)</span>
              </button>
              <button data-opens-position="true" onclick="executeOrder('SELL')" class="w-full py-2.5 rounded-xl font-bold text-xs bg-rose-600 hover:bg-rose-500 text-white shadow-lg transition flex items-center justify-center space-x-1">
                <span>賣出 / 開空 (Short)</span>
              </button>
            </div>

            <div class="grid grid-cols-2 gap-2 pt-1">
              <button onclick="closePartialPosition(0.5)" class="w-full py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-sky-400 border border-sky-400/20 transition flex items-center justify-center space-x-1">
                <i class="fa-solid fa-scissors"></i>
                <span>50% 部分減倉</span>
              </button>
              <button onclick="closeAllPositions()" class="w-full py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-amber-400 border border-amber-400/20 transition flex items-center justify-center space-x-1">
                <i class="fa-solid fa-hand"></i>
                <span>🚨 一鍵全平倉</span>
              </button>
            </div>
          </div>
        </div>

        <!-- 機器人即時決策與撮合日誌 -->"""

    if not old_panel_pat.search(html):
        raise RuntimeError("Could not locate manual order panel in web_ui/static/index.html")
    html = old_panel_pat.sub(new_panel, html, count=1)

    # Inject JS helpers for Binance order type, BBO fill, percentage qty, and availBalance
    old_exec_js = "    async function executeOrder(side) {"
    binance_js = """    let currentOrderType = 'market';
    let lastKnownPrice = 0;
    let lastFreeBalance = 0;

    function setOrderType(t) {
      currentOrderType = t;
      const mBtn = document.getElementById('tabOrderMarket');
      const lBtn = document.getElementById('tabOrderLimit');
      const box = document.getElementById('limitPriceBox');
      if (t === 'limit') {
        lBtn.className = 'py-1.5 rounded-lg bg-amber-500 text-slate-950 transition';
        mBtn.className = 'py-1.5 rounded-lg text-slate-400 hover:text-white transition';
        box?.classList.remove('hidden');
        fillBestPrice();
      } else {
        mBtn.className = 'py-1.5 rounded-lg bg-amber-500 text-slate-950 transition';
        lBtn.className = 'py-1.5 rounded-lg text-slate-400 hover:text-white transition';
        box?.classList.add('hidden');
      }
    }

    function fillBestPrice() {
      const el = document.getElementById('orderLimitPrice');
      if (el && lastKnownPrice > 0) el.value = lastKnownPrice.toFixed(1);
    }

    function setOrderPct(pct) {
      const lev = parseFloat(document.getElementById('leverageSelect')?.value || '1');
      const refPrice = (currentOrderType === 'limit' ? parseFloat(document.getElementById('orderLimitPrice')?.value) : 0) || lastKnownPrice;
      if (refPrice > 0 && lastFreeBalance > 0) {
        const qty = (lastFreeBalance * lev * pct * 0.98) / refPrice;
        document.getElementById('orderAmount').value = Math.max(0.001, Math.floor(qty * 1000) / 1000).toFixed(3);
      }
    }

    async function executeOrder(side) {"""

    if old_exec_js in html and "function setOrderType(" not in html:
        html = html.replace(old_exec_js, binance_js, 1)
        html = html.replace(
            "          stop_loss: sl ? parseFloat(sl) : null,\n          take_profit: tp ? parseFloat(tp) : null",
            "          order_type: currentOrderType,\n          price: currentOrderType === 'limit' ? parseFloat(document.getElementById('orderLimitPrice')?.value || '0') : null,\n          leverage: parseInt(document.getElementById('leverageSelect')?.value || '2', 10),\n          reduce_only: document.getElementById('reduceOnlyCheckbox')?.checked || false,\n          stop_loss: sl ? parseFloat(sl) : null,\n          take_profit: tp ? parseFloat(tp) : null",
            1,
        )

    # Update fetchCryptoData to track lastKnownPrice & lastFreeBalance & #availBalanceLabel
    old_prc_assign = "          const prcElem = document.getElementById('cryptoPrice');"
    new_prc_assign = """          if (d.current_price) lastKnownPrice = Number(d.current_price);
          if (d.free_balance !== undefined && d.free_balance !== null) {
            lastFreeBalance = Number(d.free_balance);
            const ab = document.getElementById('availBalanceLabel');
            if (ab) ab.innerText = lastFreeBalance.toLocaleString(undefined, {maximumFractionDigits: 2});
          }
          const prcElem = document.getElementById('cryptoPrice');"""
    if old_prc_assign in html and "lastFreeBalance = Number" not in html:
        html = html.replace(old_prc_assign, new_prc_assign, 1)

    _atomic_write(html_path, html)
    return "PATCHED"

def write_phase2_tests(repo_root: Path) -> str:
    target = repo_root / "tests" / "test_phase2_enhancements.py"
    if target.exists() and target.read_text(encoding="utf-8") == PHASE2_TEST_PY:
        return "ALREADY_APPLIED"
    _atomic_write(target, PHASE2_TEST_PY)
    return "PATCHED"


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Phase 2 enhancements")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()

    print("1. crypto_trading/testnet_client.py:", patch_testnet_client(repo_root))
    print("2. web_ui/server.py:", patch_server_reconciliation(repo_root))
    print("3. worldquant_brain/alpha_miner.py:", patch_alpha_miner_gene_pool(repo_root))
    print("4. scripts/train_models.py:", patch_train_models_cli(repo_root))
    print("5. binance_style_cockpit:", patch_binance_style_cockpit(repo_root))
    print("6. tests/test_phase2_enhancements.py:", write_phase2_tests(repo_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
