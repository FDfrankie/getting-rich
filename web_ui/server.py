import os
import json
import time
import threading
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

# 載入專案各子模組
from taiwan_stock.market_analyzer import MarketAnalyzer
from taiwan_stock.stock_screener import StockScreener
from taiwan_stock.report_generator import ReportGenerator
from taiwan_stock.ml_ranker import TaiwanStockMLRanker

from crypto_trading.config import PAPER_TRADING, INITIAL_BALANCE_USDT, DEFAULT_SYMBOL
from crypto_trading.testnet_client import BinanceTestnetClient
from crypto_trading.bot_runner import CryptoTradingBot
from crypto_trading.strategies.momentum_breakout import MomentumBreakoutStrategy
from crypto_trading.strategies.ml_strategy import MLAlphaStrategy
from crypto_trading.strategies.rl_strategy import RLAlphaStrategy

from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_client import BrainClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# 全域狀態管理者
class QuantAppContext:
    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.stock_screener = StockScreener()
        self.report_gen = ReportGenerator()
        self.ml_ranker = TaiwanStockMLRanker(model_type="lightgbm")

        self.alpha_vault = AlphaVault()
        self.alpha_vault.seed_initial_verified_alphas()
        self.brain_client = BrainClient()

        self.testnet_client = BinanceTestnetClient()
        self.active_bot = CryptoTradingBot(symbol=DEFAULT_SYMBOL)
        self.bot_running = False
        self.bot_thread = None
        self.bot_strategy_name = "momentum"
        self.execution_mode = "PAPER_SANDBOX"  # PAPER_SANDBOX or BINANCE_TESTNET

        # 緩存最新大盤與晨報
        self.last_taiwan_report = ""
        self.last_taiwan_market = {}
        self.bot_logs = [
            f"[{datetime.now().strftime('%H:%M:%S')}] 系統核心已初始化完畢，準備接受各項量化操作指示。"
        ]

    def add_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.bot_logs.append(f"[{ts}] {msg}")
        if len(self.bot_logs) > 50:
            self.bot_logs.pop(0)

    def switch_bot_strategy(self, strat_name: str):
        self.bot_strategy_name = strat_name
        if strat_name == "ml":
            strat = MLAlphaStrategy()
        elif strat_name == "rl":
            strat = RLAlphaStrategy()
        else:
            strat = MomentumBreakoutStrategy()
        self.active_bot = CryptoTradingBot(symbol=self.active_bot.symbol, strategy=strat)
        self.add_log(f"已切換操盤策略至: 【{strat_name.upper()} 策略】")

    def run_bot_step_loop(self):
        while self.bot_running:
            try:
                self.active_bot.step()
                curr_price = self.active_bot.loader.fetch_ohlcv(limit=2)['Close'].iloc[-1]
                equity = self.active_bot.balance_usdt + (self.active_bot.position * curr_price)
                self.add_log(f"機器人心跳自檢: 淨值 {equity:.2f} USDT | 持倉: {self.active_bot.position:.4f} {self.active_bot.symbol}")
            except Exception as e:
                self.add_log(f"機器人執行例外: {str(e)}")
            time.sleep(5)

    def start_bot(self):
        if not self.bot_running:
            self.bot_running = True
            self.bot_thread = threading.Thread(target=self.run_bot_step_loop, daemon=True)
            self.bot_thread.start()
            self.add_log(f"自動操盤機器人已啟動運行！(監控: {self.active_bot.symbol}, 策略: {self.bot_strategy_name.upper()})")

    def stop_bot(self):
        if self.bot_running:
            self.bot_running = False
            self.add_log("自動操盤機器人已安全暫停。")

APP_STATE = QuantAppContext()

class AppAPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def _send_json_response(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

        # API: 系統總覽與健康度
        if path == "/api/status":
            return self._send_json_response({
                "status": "online",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "bot_running": APP_STATE.bot_running,
                "bot_strategy": APP_STATE.bot_strategy_name,
                "execution_mode": APP_STATE.execution_mode
            })

        # API: 台股大盤診斷
        if path == "/api/taiwan/market":
            try:
                res = APP_STATE.market_analyzer.analyze()
                APP_STATE.last_taiwan_market = res
                return self._send_json_response({"success": True, "data": res})
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: 台股精選選股池與 ML 排序
        if path == "/api/taiwan/screener":
            try:
                picks = APP_STATE.stock_screener.run_screener()
                return self._send_json_response({"success": True, "picks": picks})
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: 台股晨報獲取
        if path == "/api/taiwan/report":
            if not APP_STATE.last_taiwan_report:
                # 檢查 reports 資料夾是否有現成報告
                rep_dir = os.path.join(BASE_DIR, "taiwan_stock", "reports")
                if os.path.exists(rep_dir):
                    files = [f for f in os.listdir(rep_dir) if f.endswith(".md")]
                    if files:
                        files.sort(reverse=True)
                        with open(os.path.join(rep_dir, files[0]), "r", encoding="utf-8") as f:
                            APP_STATE.last_taiwan_report = f.read()

            return self._send_json_response({
                "success": True,
                "content": APP_STATE.last_taiwan_report or "尚無已生成晨報，請點擊「生成晨報」按鈕。"
            })

        # API: 加密貨幣即時行情與帳戶狀態
        if path == "/api/crypto/account":
            try:
                mode = APP_STATE.execution_mode
                ticker = APP_STATE.testnet_client.fetch_ticker(APP_STATE.active_bot.symbol)
                curr_price = ticker.get("last", 68500.0)

                if mode == "PAPER_SANDBOX":
                    bot = APP_STATE.active_bot
                    equity = bot.balance_usdt + (bot.position * curr_price)
                    return self._send_json_response({
                        "success": True,
                        "mode": "PAPER_SANDBOX",
                        "symbol": bot.symbol,
                        "current_price": curr_price,
                        "total_equity": round(equity, 2),
                        "balance_usdt": round(bot.balance_usdt, 2),
                        "position_side": bot.position_side,
                        "position_amount": round(bot.position, 4),
                        "entry_price": bot.entry_price,
                        "bot_running": APP_STATE.bot_running,
                        "strategy": APP_STATE.bot_strategy_name,
                        "logs": APP_STATE.bot_logs[-12:]
                    })
                else:
                    # 幣安官方 Testnet
                    bal = APP_STATE.testnet_client.fetch_balance()
                    positions = APP_STATE.testnet_client.fetch_positions()
                    return self._send_json_response({
                        "success": True,
                        "mode": "BINANCE_TESTNET",
                        "symbol": APP_STATE.active_bot.symbol,
                        "current_price": curr_price,
                        "total_equity": bal.get("total_usdt", 15000.0),
                        "free_usdt": bal.get("free_usdt", 12000.0),
                        "positions": positions,
                        "bot_running": APP_STATE.bot_running,
                        "strategy": APP_STATE.bot_strategy_name,
                        "logs": APP_STATE.bot_logs[-12:]
                    })
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: WorldQuant BRAIN Alpha 倉庫列表
        if path == "/api/brain/alphas":
            alphas = APP_STATE.alpha_vault.list_alphas()
            stats = APP_STATE.alpha_vault.get_stats()
            return self._send_json_response({"success": True, "stats": stats, "alphas": alphas})

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

        # 觸發台股晨報生成
        if path == "/api/taiwan/generate_report":
            try:
                rep = APP_STATE.report_gen.generate()
                APP_STATE.last_taiwan_report = rep
                return self._send_json_response({"success": True, "content": rep})
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # 切換執行模式 (本地沙盒 vs 幣安 Testnet)
        if path == "/api/crypto/switch_mode":
            mode = post_data.get("mode", "PAPER_SANDBOX")
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

            if APP_STATE.execution_mode == "PAPER_SANDBOX":
                ticker = APP_STATE.testnet_client.fetch_ticker(symbol)
                price = ticker["last"]
                APP_STATE.active_bot.execute_paper_order(side, amount, price, reason="Web UI 手動下單")
                APP_STATE.add_log(f"UI 手動執行沙盒訂單: {side} {amount} {symbol} @ {price:.2f}")
                return self._send_json_response({"success": True, "message": f"沙盒訂單已成交: {side} {amount} {symbol}"})
            else:
                res = APP_STATE.testnet_client.create_order(symbol=symbol, side=side, order_type=order_type, amount=amount)
                APP_STATE.add_log(f"UI 幣安 Testnet 下單結果: {res.get('message')}")
                return self._send_json_response(res)

        # 加密貨幣一鍵平倉
        if path == "/api/crypto/close_all":
            if APP_STATE.execution_mode == "PAPER_SANDBOX":
                bot = APP_STATE.active_bot
                if bot.position > 0:
                    ticker = APP_STATE.testnet_client.fetch_ticker(bot.symbol)
                    bot.execute_paper_order("SELL", bot.position, ticker["last"], reason="Web UI 一鍵全數平倉")
                APP_STATE.add_log("UI 執行一鍵平倉完成 (沙盒部位結清)")
                return self._send_json_response({"success": True, "message": "沙盒持倉已全部結清！"})
            else:
                res = APP_STATE.testnet_client.close_all_positions()
                APP_STATE.add_log(f"UI 幣安 Testnet 一鍵平倉: {res.get('message')}")
                return self._send_json_response(res)

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

        return self._send_json_response({"error": "Endpoint not found"}, status=404)

def run_server(port: int = 8000, host: str = "0.0.0.0"):
    os.makedirs(STATIC_DIR, exist_ok=True)
    server_address = (host, port)
    httpd = HTTPServer(server_address, AppAPIHandler)
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
