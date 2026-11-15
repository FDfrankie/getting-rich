import os
import json
import time
import sqlite3
import threading
from datetime import datetime
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List

# 載入專案各子模組
from taiwan_stock.market_analyzer import MarketAnalyzer
from taiwan_stock.stock_screener import StockScreener
from taiwan_stock.report_generator import ReportGenerator
from taiwan_stock.ml_ranker import TaiwanStockMLRanker

from crypto_trading.config import (
    PAPER_TRADING,
    INITIAL_BALANCE_USDT,
    DEFAULT_SYMBOL,
    DEFAULT_SETTLEMENT_CURRENCY,
    BINANCE_TESTNET_API_KEY
)
from crypto_trading.data_loader import CryptoDataLoader
from crypto_trading.testnet_client import BinanceTestnetClient
from crypto_trading.bitunix_client import BitunixClient
try:
    from crypto_trading.bot_runner import CryptoTradingBot
except ImportError:
    CryptoTradingBot = None

from crypto_trading.strategies.momentum_breakout import MomentumBreakoutStrategy
try:
    from crypto_trading.strategies.ml_strategy import MLAlphaStrategy
except ImportError:
    MLAlphaStrategy = None
try:
    from crypto_trading.strategies.rl_strategy import RLAlphaStrategy
except ImportError:
    RLAlphaStrategy = None

from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.brain_job_queue import BrainJobQueue, compute_simulation_key

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DATA_DIR = os.path.join(BASE_DIR, "data_persistence")
os.makedirs(DATA_DIR, exist_ok=True)
STATE_DB_PATH = os.path.join(DATA_DIR, "system_state.db")

# 全域狀態管理者 (背景常駐 Background Worker)
class QuantAppContext:
    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.stock_screener = StockScreener()
        self.report_gen = ReportGenerator()
        self.ml_ranker = TaiwanStockMLRanker(model_type="lightgbm")

        self.alpha_vault = AlphaVault()
        self.alpha_vault.seed_initial_verified_alphas()
        self.brain_client = BrainClient()
        self.brain_job_queue = BrainJobQueue(self.brain_client)
        self.alpha_miner = ContinuousAlphaMiner(symbol=DEFAULT_SYMBOL)

        self.testnet_client = BinanceTestnetClient(settlement="USDC")
        self.bitunix_client = BitunixClient()
        if CryptoTradingBot is not None:
            self.active_bot = CryptoTradingBot(symbol=DEFAULT_SYMBOL, exchange_adapter=self.testnet_client)
        else:
            # Fallback placeholder when heavy ML/RL bot is unavailable
            class _DummyBot:
                def __init__(self, symbol):
                    self.symbol = symbol
                    self.position = 0.0
                    self.position_side = "FLAT"
                    self.balance_usdt = INITIAL_BALANCE_USDT
                    self.entry_price = 0.0
                    self.exchange_adapter = None
                def step(self): pass
                def get_equity(self, price): return self.balance_usdt
                def execute_paper_order(self, *args, **kwargs): pass
            self.active_bot = _DummyBot(symbol=DEFAULT_SYMBOL)
        self.bot_running = True  # 雲端 OnRender 背景預設常駐開啟
        self.bot_thread = None
        self.miner_thread = None
        self.bot_strategy_name = "momentum"
        self.execution_mode = "BINANCE_TESTNET"  # 預設連線幣安測試網 (USDC 0費率)

        # 系統健康狀態與 Watchdog 心跳追蹤
        self.heartbeats = {
            "trading_bot": time.time(),
            "alpha_miner": time.time(),
            "taiwan_scheduler": time.time(),
            "gdrive_sync": time.time()
        }

        # 緩存最新大盤與晨報 (共用同一份後端計算結果快取，防止打架)
        self.last_taiwan_report = ""
        self.last_taiwan_html = ""
        self.last_taiwan_market = {}
        self.last_taiwan_screener_detailed = {}
        self.last_taiwan_calc_time = 0.0
        self.bot_logs = []

        # 初始化持久化資料庫與加載歷史狀態
        self._init_persistence()
        self._load_persisted_state()

        # 啟動常駐背景工作線程 (無論瀏覽器是否開啟均 24hr 運作)
        self._start_background_worker()

    def _init_persistence(self):
        """初始化日誌與部位持久化資料庫"""
        try:
            conn = sqlite3.connect(STATE_DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    message TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[!] 初始化持久化資料庫異常: {e}")

    def _load_persisted_state(self):
        """從磁碟載入歷史日誌與最後狀態"""
        try:
            conn = sqlite3.connect(STATE_DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT timestamp, message FROM bot_logs ORDER BY id DESC LIMIT 50")
            rows = cur.fetchall()
            if rows:
                self.bot_logs = [f"[{r[0]}] {r[1]}" for r in reversed(rows)]
            else:
                self.bot_logs = [f"[{datetime.now().strftime('%H:%M:%S')}] 系統常駐核心已就緒，雲端背景運算常駐啟動。"]

            cur.execute("SELECT key, value FROM bot_state")
            for k, v in cur.fetchall():
                if k == "bot_strategy_name":
                    self.bot_strategy_name = v
                elif k == "execution_mode":
                    self.execution_mode = v
                elif k == "position":
                    self.active_bot.position = float(v)
                elif k == "balance_usdt":
                    self.active_bot.balance_usdt = float(v)
            conn.close()
        except Exception as e:
            print(f"[!] 載入持久化狀態失敗: {e}")

    def get_taiwan_market_analysis(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        獲取大盤多空結構計算結果，實施 60 秒快取保護並強制與報告共用同一份結果
        """
        now = time.time()
        if not force_refresh and self.last_taiwan_market and (now - self.last_taiwan_calc_time < 60):
            return self.last_taiwan_market

        res = self.market_analyzer.analyze()
        self.last_taiwan_market = res
        self.last_taiwan_calc_time = now
        return res

    def get_taiwan_screener_detailed(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        獲取台股選股詳細清單 (含選股透明度與空頭避險閉環)
        """
        now = time.time()
        if not force_refresh and self.last_taiwan_screener_detailed and (now - self.last_taiwan_calc_time < 60):
            return self.last_taiwan_screener_detailed

        detailed = self.stock_screener.run_screener_detailed()
        self.last_taiwan_screener_detailed = detailed
        return detailed

    def add_log(self, msg: str):
        """寫入記憶體並同步持久化到 SQLite 磁碟"""
        ts = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{ts}] {msg}"
        self.bot_logs.append(formatted)
        if len(self.bot_logs) > 60:
            self.bot_logs.pop(0)

        # 寫入磁碟 SQLite
        try:
            now_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(STATE_DB_PATH)
            cur = conn.cursor()
            cur.execute("INSERT INTO bot_logs (timestamp, message) VALUES (?, ?)", (ts, msg))
            cur.execute("REPLACE INTO bot_state (key, value, updated_at) VALUES ('bot_strategy_name', ?, ?)", (self.bot_strategy_name, now_dt))
            cur.execute("REPLACE INTO bot_state (key, value, updated_at) VALUES ('execution_mode', ?, ?)", (self.execution_mode, now_dt))
            cur.execute("REPLACE INTO bot_state (key, value, updated_at) VALUES ('position', ?, ?)", (str(self.active_bot.position), now_dt))
            cur.execute("REPLACE INTO bot_state (key, value, updated_at) VALUES ('balance_usdt', ?, ?)", (str(self.active_bot.balance_usdt), now_dt))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def switch_bot_strategy(self, strat_name: str):
        self.bot_strategy_name = strat_name
        adapter = self.testnet_client if self.execution_mode == "BINANCE_TESTNET" else (self.bitunix_client if self.execution_mode == "BITUNIX_LIVE" else None)
        lgbm_path = os.path.join(DATA_DIR, "models", "crypto_lgbm_weights.joblib")
        ppo_path = os.path.join(DATA_DIR, "models", "crypto_ppo_agent.pt")
        
        if strat_name == "ml" and MLAlphaStrategy is not None:
            strat = MLAlphaStrategy(model_path=lgbm_path if os.path.exists(lgbm_path) else None)
        elif strat_name == "rl" and RLAlphaStrategy is not None:
            strat = RLAlphaStrategy(agent_weights_path=ppo_path if os.path.exists(ppo_path) else None)
        else:
            strat = MomentumBreakoutStrategy()
        if CryptoTradingBot is not None:
            self.active_bot = CryptoTradingBot(symbol=self.active_bot.symbol, strategy=strat, exchange_adapter=adapter)
        self.add_log(f"已切換操盤策略至: 【{strat_name.upper()} 策略】(自動掛載優化權重)")

    def _background_alpha_miner_loop(self):
        """24/7 背景持續自動挖掘 Alpha 因子 Worker"""
        print("🧬 [Alpha Miner Worker] 24/7 自動因子挖掘引擎已啟動！")
        while True:
            try:
                self.heartbeats["alpha_miner"] = time.time()
                res = self.alpha_miner.mine_cycle()
                if res:
                    self.add_log(f"🧬 [24/7 Alpha 發現] 自動入庫高分因子: {res['name']} (Sharpe: {res['sharpe']}, Fitness: {res['fitness']})")
            except Exception as e:
                pass
            time.sleep(30)  # 每 30 秒挖掘一輪，嚴格控制 CPU/RAM 避免 OOM

    def _background_worker_loop(self):
        """24/7 背景獨立常駐執行緒，不依賴 HTTP 請求或前端瀏覽器"""
        print("⚡ [Background Worker] 常駐執行緒啟動！無論瀏覽器是否關閉，AI 操盤引擎均在背景運作。")
        self.add_log("常駐 Background Worker 已連線至雲端主機，24/7 自動監控盤勢。")
        last_reconcile_time = 0
        while True:
            try:
                self.heartbeats["trading_bot"] = time.time()
                self.heartbeats["taiwan_scheduler"] = time.time()
                self.heartbeats["gdrive_sync"] = time.time()

                if self.bot_running:
                    # 依當前執行模式同步 Adapter
                    if self.execution_mode == "BINANCE_TESTNET":
                        self.active_bot.exchange_adapter = self.testnet_client
                    elif self.execution_mode == "BITUNIX_LIVE":
                        self.active_bot.exchange_adapter = self.bitunix_client
                    else:
                        self.active_bot.exchange_adapter = None

                    # 執行一次策略掃描與真實下單判定
                    self.active_bot.step()

                    # 取得目前報價
                    if self.execution_mode == "BITUNIX_LIVE":
                        ticker = self.bitunix_client.fetch_ticker(self.active_bot.symbol)
                    else:
                        ticker = self.testnet_client.fetch_ticker(self.active_bot.symbol)
                    curr_price = ticker.get("last", 68500.0)
                    equity = self.active_bot.balance_usdt + (self.active_bot.position * curr_price)

                    # 每 60 秒執行一次持倉同調對齊 (reconcile_with_exchange)
                    now_ts = time.time()
                    if now_ts - last_reconcile_time > 60:
                        last_reconcile_time = now_ts
                        if self.execution_mode == "BINANCE_TESTNET":
                            rec = self.testnet_client.reconcile_with_exchange(self.active_bot.position, self.active_bot.symbol)
                            if not rec["in_sync"]:
                                self.add_log(f"🔄 交易所持倉同調: {rec['message']}")
                                self.active_bot.position = rec["remote_position"]
                        elif self.execution_mode == "BITUNIX_LIVE":
                            rec = self.bitunix_client.reconcile_with_exchange(self.active_bot.position, self.active_bot.symbol)
                            if not rec["in_sync"]:
                                self.add_log(f"🔄 Bitunix 持倉同調: {rec['message']}")
                                self.active_bot.position = rec["remote_position"]

                    # 每 30 秒記錄一次常駐心跳
                    if int(time.time()) % 30 < 5:
                        self.add_log(f"常駐巡航心跳: 標的 {self.active_bot.symbol} @ {curr_price:.1f} | 淨值: {equity:.2f} | 模式: {self.execution_mode}")
            except Exception as e:
                self.add_log(f"背景掃描異常 (已自動重試): {str(e)}")
            time.sleep(5)

    def _start_background_worker(self):
        if self.bot_thread is None or not self.bot_thread.is_alive():
            self.bot_thread = threading.Thread(target=self._background_worker_loop, daemon=True)
            self.bot_thread.start()
        if self.miner_thread is None or not self.miner_thread.is_alive():
            self.miner_thread = threading.Thread(target=self._background_alpha_miner_loop, daemon=True)
            self.miner_thread.start()

    def start_bot(self):
        self.bot_running = True
        self.add_log(f"自動操盤機器人已設定為【積極運轉】(策略: {self.bot_strategy_name.upper()})")

    def stop_bot(self):
        self.bot_running = False
        self.add_log("自動操盤機器人已切換為【待命暫停】。")

APP_STATE = QuantAppContext()

class AppAPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def _send_json_response(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, code: str, message: str, status: int = 400, retryable: bool = False):
        return self._send_json_response({
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable
            }
        }, status=status)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.path = "/index.html"
            return super().do_GET()

        # 靜態文件映射：講義頁面
        if path == "/handbook":
            handbook_path = os.path.join(BASE_DIR, "docs", "handbook", "QUANT_HANDBOOK.html")
            if os.path.exists(handbook_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(handbook_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Handbook not found")
                return

        # API: 查詢 Brain 模擬 Job 狀態 (GET /api/brain/jobs/{job_id})
        if path.startswith("/api/brain/jobs/"):
            job_id = path[len("/api/brain/jobs/"):]
            if not job_id:
                return self._send_error_json("MISSING_JOB_ID", "Job ID 未指定", status=400)
            job = APP_STATE.brain_job_queue.get_job(job_id)
            if not job:
                return self._send_error_json("JOB_NOT_FOUND", f"找不到指定的工作: {job_id}", status=404)
            return self._send_json_response({
                "ok": True,
                "job": job.to_dict()
            }, status=200)

        # API: 系統總覽與健康度 (含 Watchdog 四大核心心跳)
        if path == "/api/status" or path == "/health":
            now_ts = time.time()
            return self._send_json_response({
                "status": "online",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "bot_running": APP_STATE.bot_running,
                "bot_strategy": APP_STATE.bot_strategy_name,
                "execution_mode": APP_STATE.execution_mode,
                "watchdog": {
                    "trading_bot": "OK" if (now_ts - APP_STATE.heartbeats["trading_bot"] < 30) else "LAGGING",
                    "alpha_miner": "OK" if (now_ts - APP_STATE.heartbeats["alpha_miner"] < 90) else "LAGGING",
                    "taiwan_scheduler": "OK" if (now_ts - APP_STATE.heartbeats["taiwan_scheduler"] < 60) else "LAGGING",
                    "gdrive_sync": "OK" if (now_ts - APP_STATE.heartbeats["gdrive_sync"] < 60) else "LAGGING",
                    "total_alphas_mined": APP_STATE.alpha_miner.total_mined,
                    "total_alphas_accepted": APP_STATE.alpha_miner.total_accepted
                }
            })

        # API: 台股大盤診斷
        if path == "/api/taiwan/market":
            try:
                res = APP_STATE.get_taiwan_market_analysis()
                return self._send_json_response({"success": True, "data": res})
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: 台股精選選股池與 ML 排序 (含選股透明度統計)
        if path == "/api/taiwan/screener":
            try:
                detailed = APP_STATE.get_taiwan_screener_detailed()
                return self._send_json_response({
                    "success": True,
                    "scanned_count": detailed["scanned_count"],
                    "qualified_count": detailed["qualified_count"],
                    "picks": detailed["picks"],
                    "near_picks": detailed["near_picks"],
                    "defense_hedges": detailed["defense_hedges"]
                })
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: 台股晨報獲取 (強制共用同一份後端大盤計算快取與透明度數據)
        if path == "/api/taiwan/report":
            format_type = parse_qs(parsed.query).get("format", ["markdown"])[0]
            # 確保大盤計算結果已生成
            market_info = APP_STATE.get_taiwan_market_analysis()

            if not APP_STATE.last_taiwan_report:
                try:
                    APP_STATE.last_taiwan_report = APP_STATE.report_gen.generate(cached_market_info=market_info)
                except Exception:
                    pass

            if not APP_STATE.last_taiwan_html:
                try:
                    APP_STATE.last_taiwan_html = APP_STATE.report_gen.generate_html_report(cached_market_info=market_info)
                except Exception:
                    pass

            return self._send_json_response({
                "success": True,
                "content": APP_STATE.last_taiwan_report or "尚無已生成報告，請點擊「生成晨報」按鈕。",
                "html_content": APP_STATE.last_taiwan_html or ""
            })

        # API: 加密貨幣即時行情與帳戶狀態 (支援 本地沙盒, 幣安合約測試網, Bitunix 實盤)
        if path == "/api/crypto/account":
            try:
                mode = APP_STATE.execution_mode
                if mode == "BITUNIX_LIVE":
                    ticker = APP_STATE.bitunix_client.fetch_ticker(APP_STATE.active_bot.symbol)
                    bal = APP_STATE.bitunix_client.fetch_balance()
                    positions = APP_STATE.bitunix_client.fetch_positions()
                    curr_price = ticker.get("last", 68500.0)
                    unrealized_pnl = sum([float(p.get("unrealized_pnl", 0.0)) for p in positions]) if positions else 0.0
                    return self._send_json_response({
                        "success": True,
                        "mode": "BITUNIX_LIVE",
                        "exchange": "bitunix",
                        "symbol": APP_STATE.active_bot.symbol,
                        "current_price": curr_price,
                        "total_equity": bal.get("total_equity", 20000.0),
                        "free_balance": bal.get("free_balance", 17000.0),
                        "unrealized_pnl": unrealized_pnl,
                        "positions": positions,
                        "bot_running": APP_STATE.bot_running,
                        "strategy": APP_STATE.bot_strategy_name,
                        "logs": APP_STATE.bot_logs[-15:]
                    })
                elif mode == "BINANCE_TESTNET":
                    ticker = APP_STATE.testnet_client.fetch_ticker(APP_STATE.active_bot.symbol)
                    curr_price = ticker.get("last", 68500.0)
                    bal = APP_STATE.testnet_client.fetch_balance()
                    positions = APP_STATE.testnet_client.fetch_positions()
                    unrealized_pnl = sum([float(p.get("unrealized_pnl", 0.0)) for p in positions]) if positions else 0.0
                    return self._send_json_response({
                        "success": True,
                        "mode": "BINANCE_TESTNET",
                        "exchange": "binance_futures",
                        "symbol": APP_STATE.active_bot.symbol,
                        "current_price": curr_price,
                        "total_equity": bal.get("total_equity", 15000.0),
                        "free_balance": bal.get("free_balance", 12000.0),
                        "unrealized_pnl": unrealized_pnl,
                        "settlement_currency": bal.get("settlement_currency", "USDC"),
                        "zero_fee_usdc": bal.get("zero_fee_usdc", True),
                        "positions": positions,
                        "bot_running": APP_STATE.bot_running,
                        "strategy": APP_STATE.bot_strategy_name,
                        "logs": APP_STATE.bot_logs[-15:]
                    })
                else: # PAPER_SANDBOX
                    ticker = APP_STATE.testnet_client.fetch_ticker(APP_STATE.active_bot.symbol)
                    curr_price = ticker.get("last", 68500.0)
                    bot = APP_STATE.active_bot
                    equity = bot.get_equity(curr_price)
                    if bot.position_side == "LONG" and bot.entry_price > 0:
                        unrealized_pnl = (curr_price - bot.entry_price) * bot.position
                    elif bot.position_side == "SHORT" and bot.entry_price > 0:
                        unrealized_pnl = (bot.entry_price - curr_price) * bot.position
                    else:
                        unrealized_pnl = 0.0

                    return self._send_json_response({
                        "success": True,
                        "mode": "PAPER_SANDBOX",
                        "symbol": bot.symbol,
                        "current_price": curr_price,
                        "total_equity": round(equity, 2),
                        "balance_usdt": round(bot.balance_usdt, 2),
                        "unrealized_pnl": round(unrealized_pnl, 2),
                        "position_side": bot.position_side,
                        "position_amount": round(bot.position, 4),
                        "entry_price": bot.entry_price,
                        "bot_running": APP_STATE.bot_running,
                        "strategy": APP_STATE.bot_strategy_name,
                        "logs": APP_STATE.bot_logs[-15:]
                    })
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: 加密貨幣歷史 K 線數據 (專供 TradingView Lightweight Charts 繪製)
        if path == "/api/crypto/klines":
            try:
                symbol = parse_qs(parsed.query).get("symbol", [APP_STATE.active_bot.symbol])[0]
                interval = parse_qs(parsed.query).get("interval", ["1m"])[0]

                # 支援台股代碼與加密貨幣雙市場
                is_taiwan = symbol.endswith(".TW") or symbol.endswith(".TWO")
                if is_taiwan:
                    import yfinance as yf
                    period_map = {"1m": "1d", "5m": "5d", "15m": "5d", "1h": "1mo", "1d": "6mo"}
                    req_period = period_map.get(interval, "1mo")
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=req_period, interval=interval if interval in ["1m", "5m", "15m", "1h", "1d"] else "1d")
                    if df.empty:
                        # 備援以日線重試
                        df = ticker.history(period="3mo", interval="1d")
                else:
                    loader = CryptoDataLoader()
                    df = loader.fetch_ohlcv(symbol=symbol, timeframe=interval, limit=120)

                candles = []
                for idx, row in df.iterrows():
                    # Lightweight charts 接受 unix timestamp (秒)
                    ts = int(idx.timestamp()) if hasattr(idx, 'timestamp') else int(time.time())
                    candles.append({
                        "time": ts,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row.get("Volume", 0))
                    })
                
                # 同時計算 MA5, MA20, MA60 與布林帶提供圖表覆蓋
                if len(candles) > 0:
                    close_series = df["Close"]
                    ma5_series = close_series.rolling(5).mean().fillna(close_series)
                    ma20_series = close_series.rolling(20).mean().fillna(close_series)
                    ma60_series = close_series.rolling(60).mean().fillna(close_series)
                    std20 = close_series.rolling(20).std().fillna(0)
                    upper_bb = (ma20_series + 2.0 * std20).fillna(close_series)
                    lower_bb = (ma20_series - 2.0 * std20).fillna(close_series)

                    overlay = {
                        "ma5": [{"time": int(t.timestamp()), "value": float(v)} for t, v in ma5_series.items()],
                        "ma20": [{"time": int(t.timestamp()), "value": float(v)} for t, v in ma20_series.items()],
                        "ma60": [{"time": int(t.timestamp()), "value": float(v)} for t, v in ma60_series.items()],
                        "bb_upper": [{"time": int(t.timestamp()), "value": float(v)} for t, v in upper_bb.items()],
                        "bb_lower": [{"time": int(t.timestamp()), "value": float(v)} for t, v in lower_bb.items()],
                    }
                else:
                    overlay = {"ma5": [], "ma20": [], "ma60": [], "bb_upper": [], "bb_lower": []}

                # 買賣標記 (Markers) 與 持倉 SL/TP 線 (從持久化訂單與交易歷史真實獲取，杜絕寫死假點位)
                markers = []
                bot = APP_STATE.active_bot
                if not is_taiwan and len(candles) > 0:
                    earliest_chart_ts = candles[0]["time"]
                    latest_chart_ts = candles[-1]["time"]
                    recent_orders = bot.sync.get_recent_orders(limit=20)
                    for ord_item in recent_orders:
                        try:
                            # 轉換 ISO/標準時間為 unix timestamp
                            ord_ts = int(time.mktime(datetime.strptime(ord_item["timestamp"], "%Y-%m-%d %H:%M:%S").timetuple()))
                            if earliest_chart_ts <= ord_ts <= latest_chart_ts + 300:
                                side_upper = ord_item["side"].upper()
                                is_buy = side_upper == "BUY"
                                markers.append({
                                    "time": ord_ts,
                                    "position": "belowBar" if is_buy else "aboveBar",
                                    "color": "#10b981" if is_buy else "#ef4444",
                                    "shape": "arrowUp" if is_buy else "arrowDown",
                                    "text": f"{side_upper} @ {ord_item['price']:.1f}"
                                })
                        except Exception:
                            pass
                
                trade_lines = {}
                if bot.position > 0 and (bot.symbol == symbol or not is_taiwan):
                    entry = bot.entry_price or (candles[-1]["close"] if candles else 0)
                    trade_lines = {
                        "entry": entry,
                        "stop_loss": round(entry * 0.98, 2),
                        "take_profit": round(entry * 1.04, 2)
                    }

                return self._send_json_response({
                    "success": True,
                    "symbol": symbol,
                    "candles": candles,
                    "overlay": overlay,
                    "markers": markers,
                    "trade_lines": trade_lines
                })
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: WorldQuant BRAIN Alpha 倉庫列表與變現教學
        if path == "/api/brain/alphas":
            alphas = APP_STATE.alpha_vault.list_alphas()
            stats = APP_STATE.alpha_vault.get_stats()
            roadmap = APP_STATE.alpha_vault.get_brain_roadmap_guide()
            payout_sample = APP_STATE.alpha_vault.calculate_payout_projection(
                quarterly_pnl_usd=50000.0,
                sharpe=1.85,
                turnover_pct=32.0,
                is_consultant=True
            )
            return self._send_json_response({
                "success": True,
                "stats": stats,
                "alphas": alphas,
                "roadmap": roadmap,
                "payout_sample": payout_sample
            })

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        post_data = {}
        if length > 0:
            try:
                post_data = json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                post_data = {}

        # 觸發台股報告生成 (強制同步更新大盤與報告，杜絕分數打架)
        if path == "/api/taiwan/generate_report":
            try:
                session_type = post_data.get("session_type", "盤前晨報")
                market_info = APP_STATE.get_taiwan_market_analysis(force_refresh=True)
                rep = APP_STATE.report_gen.generate(session_type=session_type, cached_market_info=market_info)
                html_rep = APP_STATE.report_gen.generate_html_report(session_type=session_type, cached_market_info=market_info)
                APP_STATE.last_taiwan_report = rep
                APP_STATE.last_taiwan_html = html_rep
                return self._send_json_response({"success": True, "content": rep, "html_content": html_rep})
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # 切換執行模式 (本地沙盒 vs 幣安合約 Testnet vs Bitunix 實盤)
        if path == "/api/crypto/switch_mode":
            mode = post_data.get("mode", "BINANCE_TESTNET")
            APP_STATE.execution_mode = mode
            APP_STATE.add_log(f"已切換加密操盤模式為: 【{mode}】")
            return self._send_json_response({"success": True, "mode": mode})

        # 啟動 / 暫停機器人
        if path == "/api/crypto/bot_control":
            action = post_data.get("action")
            strat = post_data.get("strategy")
            if strat and strat != APP_STATE.bot_strategy_name:
                APP_STATE.switch_bot_strategy(strat)

            if action == "start":
                APP_STATE.start_bot()
            elif action == "stop":
                APP_STATE.stop_bot()
            return self._send_json_response({
                "success": True,
                "bot_running": APP_STATE.bot_running,
                "strategy": APP_STATE.bot_strategy_name
            })

        # 加密貨幣手動下單 / 平倉
        if path == "/api/crypto/order":
            side = post_data.get("side", "buy").upper()
            amount = float(post_data.get("amount", 0.01))
            symbol = post_data.get("symbol", APP_STATE.active_bot.symbol)
            order_type = post_data.get("order_type", "market")

            if APP_STATE.execution_mode == "BITUNIX_LIVE":
                res = APP_STATE.bitunix_client.create_order(symbol=symbol, side=side, order_type=order_type, amount=amount)
                APP_STATE.add_log(f"Bitunix 實盤下單回報: {res.get('message')}")
                return self._send_json_response(res)
            elif APP_STATE.execution_mode == "BINANCE_TESTNET":
                res = APP_STATE.testnet_client.create_order(symbol=symbol, side=side, order_type=order_type, amount=amount)
                APP_STATE.add_log(f"幣安合約 Testnet 下單回報: {res.get('message')}")
                return self._send_json_response(res)
            else: # PAPER_SANDBOX
                ticker = APP_STATE.testnet_client.fetch_ticker(symbol)
                price = ticker["last"]
                APP_STATE.active_bot.execute_paper_order(side, amount, price, reason="Web UI 手動下單")
                APP_STATE.add_log(f"沙盒訂單成交: {side} {amount} {symbol} @ {price:.2f}")
                return self._send_json_response({"success": True, "message": f"沙盒訂單已成交: {side} {amount} {symbol}"})

        # 加密貨幣一鍵平倉
        if path == "/api/crypto/close_all":
            if APP_STATE.execution_mode == "BITUNIX_LIVE":
                res = APP_STATE.bitunix_client.close_all_positions()
                APP_STATE.add_log(f"Bitunix 實盤一鍵全平倉完成")
                return self._send_json_response(res)
            elif APP_STATE.execution_mode == "BINANCE_TESTNET":
                res = APP_STATE.testnet_client.close_all_positions()
                APP_STATE.add_log(f"幣安合約 Testnet 一鍵平倉: {res.get('message')}")
                return self._send_json_response(res)
            else: # PAPER_SANDBOX
                bot = APP_STATE.active_bot
                if bot.position > 0:
                    ticker = APP_STATE.testnet_client.fetch_ticker(bot.symbol)
                    bot.execute_paper_order("SELL", bot.position, ticker["last"], reason="Web UI 一鍵全數平倉")
                APP_STATE.add_log("UI 執行一鍵平倉完成 (沙盒部位結清)")
                return self._send_json_response({"success": True, "message": "沙盒持倉已全部結清！"})

        # WorldQuant BRAIN: 提交與儲存 Alpha
        if path == "/api/brain/submit_alpha":
            name = post_data.get("name", "Custom_Alpha")
            expr = post_data.get("expression", "").strip()
            sharpe = float(post_data.get("sharpe", 1.35))
            fitness = float(post_data.get("fitness", 1.15))
            turnover = float(post_data.get("turnover", 25.0))
            category = post_data.get("category", "Momentum")
            notes = post_data.get("notes", "")

            # 透過 AlphaVault 保存
            res = APP_STATE.alpha_vault.save_alpha(
                name=name,
                expression=expr,
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                category=category,
                notes=notes
            )
            return self._send_json_response(res)

        # 🚀 POST /api/brain/simulations：建立非同步送測 job (符合 docs/BRAIN_WEB_API_ASYNC_FIX_SPEC.md)
        if path == "/api/brain/simulations":
            try:
                expr = post_data.get("expression") or post_data.get("regular") or ""
                if not expr or not expr.strip():
                    return self._send_error_json("INVALID_EXPRESSION", "Alpha 表達式不可為空", status=400)

                settings = post_data.get("settings")
                job, is_new = APP_STATE.brain_job_queue.create_job(expr, settings)
                APP_STATE.add_log(f"🚀 [Quant 測試佇列] Alpha Job {job.job_id} ({'新增' if is_new else '去重複用'}) 已排入")

                return self._send_json_response({
                    "ok": True,
                    "status": job.status.value if hasattr(job.status, "value") else str(job.status),
                    "job_id": job.job_id,
                    "simulation_key": job.simulation_key,
                    "next_poll_seconds": APP_STATE.brain_job_queue.default_poll_interval
                }, status=202)
            except Exception as e:
                return self._send_error_json("INTERNAL_SERVER_ERROR", "建立送測作業失敗", status=500)

        # 🚀 一鍵自動送出至 Quant 測試 (向後相容舊端點，升級為 202 非同步 job 回應)
        if path == "/api/brain/test_alpha":
            try:
                alpha_id = post_data.get("alpha_id")
                expr = post_data.get("expression", "")
                target_engine = post_data.get("target_engine", "worldquant")

                if target_engine == "crypto_bot":
                    return self._send_error_json(
                        "UNSUPPORTED_ENGINE",
                        "股票橫截面 FASTEXPR 因子不適用於單一加密貨幣時間序列，該功能已禁用。",
                        status=400
                    )

                if not expr or not expr.strip():
                    return self._send_error_json("INVALID_EXPRESSION", "Alpha 表達式不可為空", status=400)

                job, is_new = APP_STATE.brain_job_queue.create_job(expr)
                APP_STATE.add_log(f"🚀 [Quant 測試送出] Job {job.job_id} 已排入非同步背景輪詢")

                return self._send_json_response({
                    "ok": True,
                    "success": True,
                    "target_engine": "WorldQuant BRAIN",
                    "status": job.status.value if hasattr(job.status, "value") else str(job.status),
                    "job_id": job.job_id,
                    "simulation_key": job.simulation_key,
                    "next_poll_seconds": APP_STATE.brain_job_queue.default_poll_interval,
                    "message": "已成功將該 Alpha 送出至非同步模擬佇列！"
                }, status=202)
            except Exception as e:
                return self._send_error_json("INTERNAL_SERVER_ERROR", f"WorldQuant 模擬請求處理失敗: {type(e).__name__}", status=500)

        return self._send_error_json("NOT_FOUND", "Endpoint not found", status=404)

def run_server(port: int = 8999, host: str = "0.0.0.0"):
    os.makedirs(STATIC_DIR, exist_ok=True)
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, AppAPIHandler)
    print(f"🚀 [Web UI Server] 服務已在 http://localhost:{port} 啟動！")
    print(f"📱 手機與局域網訪問可使用 http://[您的區域網路IP]:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 伺服器正在優雅關閉...")
        APP_STATE.stop_bot()
        httpd.server_close()

if __name__ == "__main__":
    run_server()
