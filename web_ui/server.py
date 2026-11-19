import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import numpy as np
try:
    from zoneinfo import ZoneInfo
    _TAIPEI_TZ = ZoneInfo("Asia/Taipei")
except Exception:
    _TAIPEI_TZ = timezone(timedelta(hours=8))

def _now_taipei() -> datetime:
    return datetime.now(_TAIPEI_TZ)
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from crypto_trading.bitunix_client import BitunixClient
from crypto_trading.exchange_normalizer import normalize_binance_balance, normalize_remote_positions
from crypto_trading.paper_broker import MarketDataService, PaperBroker
from crypto_trading.reconciliation_service import ReconciliationService
from crypto_trading.runtime_context import ConnectionState, RuntimeTradingContext
from crypto_trading.settings import TradingMode, get_settings
from crypto_trading.trading_snapshot import (
    AccountSnapshot, ConnectionSnapshot, MarketSnapshot, ReconciliationSnapshot, RiskSnapshot, TradingSnapshot,
)
from web_ui.trading_snapshot_api import get_trading_snapshot_response
from worldquant_brain.connectivity_service import BrainConnectivityService
from crypto_trading.config import (
    DEFAULT_SYMBOL,
    INITIAL_BALANCE_USDT,
)
from crypto_trading.data_loader import CryptoDataLoader
from crypto_trading.testnet_client import BinanceTestnetClient

# 載入專案各子模組
from taiwan_stock.market_analyzer import MarketAnalyzer
from taiwan_stock.ml_ranker import TaiwanStockMLRanker
from taiwan_stock.report_generator import ReportGenerator
from taiwan_stock.stock_screener import StockScreener

try:
    from crypto_trading.bot_runner import CryptoTradingBot
except ImportError:
    CryptoTradingBot = None

from crypto_trading.strategies.momentum_breakout import MomentumBreakoutStrategy
from crypto_trading.strategies.conditional_prob_strategy import ConditionalProbTechStrategy
from crypto_trading.strategies.brain_alpha_strategy import BrainAlphaCryptoStrategy
from crypto_trading.strategies.perp_carry_basis_strategy import PerpCarryBasisStrategy

try:
    from crypto_trading.strategies.ml_strategy import MLAlphaStrategy
except ImportError:
    MLAlphaStrategy = None
try:
    from crypto_trading.strategies.rl_strategy import RLAlphaStrategy
except ImportError:
    RLAlphaStrategy = None

from taiwan_stock.universe_registry import DEFAULT_POPULAR_WATCHLIST, MARKET_INDICES, POPULAR_WATCHLIST_IS_FULL_UNIVERSE, TaiwanUniverseRegistry, build_yuanta_style_quote_panel
from observability.telemetry import build_platform_observability_report
from worldquant_brain.candidate_utility import evaluate_candidate
from worldquant_brain.alpha_miner import ContinuousAlphaMiner
from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_client import BrainClient
from worldquant_brain.brain_job_queue import BrainJobQueue
from worldquant_brain.alpha_experiment_store import AlphaExperimentStore
from worldquant_brain.brain_scheduler import BrainDailyScheduler
from worldquant_brain.simulation_budget import Candidate as BudgetCandidate, ThreeMinuteSimulationBudget
from worldquant_brain.manual_priority_queue import ManualPriorityQueue
from worldquant_brain.score_tracker import ScoreTracker, parse_score_payload
from worldquant_brain.research_memory import AlphaResearchFactoryService
from worldquant_brain.score_attribution import estimate_days_to_10k, score_refresh_at_for_taipei_date
from observability.audit_log import AuditLog

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
        self.tw_universe = TaiwanUniverseRegistry()

        self.alpha_vault = AlphaVault()
        self.alpha_vault.seed_initial_verified_alphas()
        self.alpha_experiments = AlphaExperimentStore(self.alpha_vault.db_path)
        self.sim_budget = ThreeMinuteSimulationBudget(minimum_interval_seconds=float(os.getenv('BRAIN_MINING_COOLDOWN_SEC', '0') or '0'), max_pending=int(os.getenv('BRAIN_QUEUE_TARGET_DEPTH', '10') or '10'), allow_multiple_per_family=True)
        self.brain_scheduler = BrainDailyScheduler(self.alpha_vault.db_path, int(os.getenv("BRAIN_DAILY_FLUSH_HOUR", "22")))
        self.manual_brain_queue = ManualPriorityQueue(self.alpha_vault.db_path)
        self.brain_score_tracker = ScoreTracker(self.alpha_vault.db_path)
        self.research_factory = AlphaResearchFactoryService(self.alpha_vault.db_path)
        try:
            from worldquant_brain.trial_registry_dsr import TrialRegistry
            self.trial_registry = TrialRegistry(self.alpha_vault.db_path)
        except Exception:
            self.trial_registry = None
        try:
            from worldquant_brain.objective_aware_scheduler import ObjectiveLaneBudget
            self.objective_lane_budget = ObjectiveLaneBudget()
            self.objective_lane_budget.validate()
        except Exception:
            self.objective_lane_budget = None
        try:
            from worldquant_brain.evidence_repository import EvidenceRepository
            self.evidence_repo = EvidenceRepository(self.alpha_vault.db_path)
            self.evidence_repo.migrate()
            _m007 = Path(__file__).resolve().parent.parent / "migrations" / "007_objective_aware_optimization.sql"
            if _m007.exists():
                with self.evidence_repo.connection() as _c007:
                    _c007.executescript(_m007.read_text(encoding="utf-8"))
        except Exception:
            self.evidence_repo = None
        self.audit_log = AuditLog(self.alpha_vault.db_path)
        self.brain_client = BrainClient()
        self.brain_job_queue = BrainJobQueue(self.brain_client, on_complete_callback=self._on_brain_job_completed, on_fail_callback=self._on_brain_job_failed, db_path=self.alpha_vault.db_path)
        self.alpha_miner = ContinuousAlphaMiner(symbol=DEFAULT_SYMBOL)
        self._refined_expr_keys: set[str] = set()

        self.testnet_client = BinanceTestnetClient(settlement="USDC")
        self.reconciliation_service = ReconciliationService()
        self.bitunix_client = BitunixClient()
        if CryptoTradingBot is not None:
            try:
                from crypto_trading.multi_symbol_runner import MultiSymbolBotManager, attach_bot_safety_context
                self.bot_manager = MultiSymbolBotManager(
                    exchange_adapter=self.testnet_client,
                    reconciliation_service=self.reconciliation_service,
                )
                self.active_bot = attach_bot_safety_context(self.bot_manager.primary_bot, self.reconciliation_service)
            except ImportError:
                self.bot_manager = None
                self.active_bot = CryptoTradingBot(symbol=DEFAULT_SYMBOL, exchange_adapter=self.testnet_client)
                self.active_bot.reconciliation_service = self.reconciliation_service
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

                def step(self):
                    pass

                def get_equity(self, price):
                    return self.balance_usdt

                def execute_paper_order(self, *args, **kwargs):
                    pass

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
            "gdrive_sync": time.time(),
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

        # 啟動初始對帳與常駐背景工作線程 (無論瀏覽器是否開啟均 24hr 運作)
        self._run_reconciliation_sync()
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
                self.bot_logs = [f"[{_now_taipei().strftime('%H:%M:%S')}] 系統常駐核心已就緒，雲端背景運算常駐啟動。"]

            cur.execute("SELECT key, value FROM bot_state")
            for k, v in cur.fetchall():
                if k == "bot_strategy_name":
                    self.bot_strategy_name = v
                elif k == "execution_mode":
                    self.execution_mode = v
                elif k == "position":
                    self.active_bot.position = float(v)
                elif k == "balance_usdt":
                    b_val = float(v)
                    self.active_bot.balance_usdt = b_val
                    rm = getattr(self.active_bot, "risk_manager", None)
                    if rm is not None and b_val > 0:
                        if hasattr(rm, "reanchor_capital"):
                            rm.reanchor_capital(b_val)
                        else:
                            rm.initial_capital = b_val
                            rm.peak_capital = b_val
                            rm.current_capital = b_val
                            rm.is_halted = False
            conn.close()
        except Exception as e:
            print(f"[!] 載入持久化狀態失敗: {e}")

    def get_taiwan_market_analysis(self, force_refresh: bool = False) -> dict[str, Any]:
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

    def get_taiwan_screener_detailed(self, force_refresh: bool = False) -> dict[str, Any]:
        """
        獲取台股選股詳細清單 (含選股透明度與空頭避險閉環)
        """
        now = time.time()
        if not force_refresh and self.last_taiwan_screener_detailed and (now - self.last_taiwan_calc_time < 60):
            return self.last_taiwan_screener_detailed

        detailed = self.stock_screener.run_screener_detailed()
        self.last_taiwan_screener_detailed = detailed
        return detailed

    def _parse_recon_snapshot(self, now: datetime) -> ReconciliationSnapshot:
        recon_svc = getattr(self, "reconciliation_service", None)
        if recon_svc is None or not callable(getattr(recon_svc, "get_latest_result", None)):
            return ReconciliationSnapshot("NEVER_RUN", 0, None)
        latest_rec = recon_svc.get_latest_result()
        if not isinstance(latest_rec, dict):
            return ReconciliationSnapshot("NEVER_RUN", 0, None)
        raw_ts = latest_rec.get("completed_at")
        comp_at: datetime | None = None
        if isinstance(raw_ts, datetime):
            comp_at = raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
        elif isinstance(raw_ts, str):
            try:
                parsed = datetime.fromisoformat(raw_ts)
                comp_at = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except Exception:
                comp_at = None
        if comp_at is None:
            return ReconciliationSnapshot("NEVER_RUN", 0, None)
        age = (now - comp_at).total_seconds()
        disc = int(latest_rec.get("discrepancy_count", 0))
        if age > 30:
            st = "STALE"
        elif int(latest_rec.get("unknown_order_count", 0)) > 0:
            st = "UNKNOWN_ORDERS"
        elif not latest_rec.get("in_sync", False):
            st = "DRIFT"
        else:
            st = "OK"
        return ReconciliationSnapshot(st, disc, comp_at)

    def build_trading_snapshot(self) -> TradingSnapshot:
        now = datetime.now(timezone.utc)
        raw_mode = getattr(self, "execution_mode", TradingMode.LOCAL_PAPER)
        if isinstance(raw_mode, TradingMode):
            req_mode = raw_mode
        elif str(raw_mode) in {m.value for m in TradingMode}:
            req_mode = TradingMode(str(raw_mode))
        else:
            req_mode = TradingMode.LOCAL_PAPER

        recon_snap = self._parse_recon_snapshot(now)

        if req_mode == TradingMode.BINANCE_TESTNET:
            t0 = time.perf_counter()
            probe_fn = getattr(self.testnet_client, "probe_authenticated_connection", None)
            is_connected = bool(getattr(self.testnet_client, "is_connected", False)) and not bool(getattr(self.testnet_client, "mock_mode", True))
            if callable(probe_fn):
                probe = probe_fn()
                is_connected = bool(getattr(probe, "authenticated", False))
            if not is_connected:
                ctx = RuntimeTradingContext.degraded_from_remote(
                    TradingMode.BINANCE_TESTNET,
                    account_id="testnet-degraded",
                    venue="BINANCE_TESTNET",
                    reason="TESTNET_AUTH_OR_CONNECTION_UNAVAILABLE",
                    connection_state=ConnectionState.AUTH_FAILED,
                )
                return TradingSnapshot(
                    context=ctx,
                    connection=ConnectionSnapshot("BINANCE_TESTNET", False, False, None, None, "AUTH_FAILED", "Remote testnet unavailable; no fake values shown"),
                    market=MarketSnapshot(self.active_bot.symbol, self.active_bot.symbol, "PERPETUAL", "USDT", None, None, "UNAVAILABLE", None, None),
                    account=AccountSnapshot(None, None, None, None, None, None, None, "USDT", source="UNAVAILABLE", timestamp=None),
                    positions=(),
                    open_orders=(),
                    reconciliation=recon_snap,
                    risk=RiskSnapshot(False, ("VIEW", "CANCEL_ENTRY_ORDERS", "REDUCE_ONLY", "CLOSE_ALL"), ("TESTNET_UNAVAILABLE",)),
                    configured_strategy=self.bot_strategy_name,
                    effective_strategy=self.bot_strategy_name,
                    model_id=None,
                ).validated()

            t = self.testnet_client.fetch_ticker(self.active_bot.symbol)
            b = self.testnet_client.fetch_balance()
            ps = self.testnet_client.fetch_positions()
            oo_fn = getattr(self.testnet_client, "fetch_open_orders", None)
            open_orders = tuple(oo_fn(self.active_bot.symbol) or ()) if callable(oo_fn) else ()
            latency_ms = round((time.perf_counter() - t0) * 1000)

            last_p = t.get("last")
            price = Decimal(str(last_p)) if last_p is not None else None
            # 使用即時收到的報價時間，避免 Binance 撮合撮合事件時間間隔大於 5 秒觸發 MARKET_DATA_STALE
            mkt_ts = now
            stale_s = 0.0
            ts_src = "RECEIVE_TIME"

            acc = normalize_binance_balance(b, default_currency=str(b.get("settlement_currency", "USDT")), now=now)
            pos_snaps = normalize_remote_positions(ps, self.active_bot.symbol, "BINANCE_TESTNET", now)
            ctx = RuntimeTradingContext(
                requested_mode=TradingMode.BINANCE_TESTNET,
                effective_mode=TradingMode.BINANCE_TESTNET,
                environment_id="testnet",
                account_id="binance-testnet",
                venue="BINANCE_TESTNET",
                connection_state=ConnectionState.AUTHENTICATED,
            )
            return TradingSnapshot(
                context=ctx,
                connection=ConnectionSnapshot("BINANCE_TESTNET", True, True, latency_ms, now),
                market=MarketSnapshot(self.active_bot.symbol, self.active_bot.symbol, "PERPETUAL", acc.currency, price, price, "EXCHANGE_REST", mkt_ts, stale_s, ts_src),
                account=acc,
                positions=pos_snaps,
                open_orders=open_orders,
                reconciliation=recon_snap,
                risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "OPEN_SHORT", "REDUCE_ONLY", "CLOSE_ALL"), ()),
                configured_strategy=self.bot_strategy_name,
                effective_strategy=self.bot_strategy_name,
                model_id=None,
            ).validated()

        # LOCAL_PAPER: dedicated MarketDataService + PaperBroker (never uses mock testnet balance)
        ctx = RuntimeTradingContext.local_paper()
        mds = getattr(self, "market_data_service", None) or MarketDataService(getattr(self, "public_market_adapter", None))
        broker = getattr(self, "paper_broker", None) or PaperBroker(self.active_bot)
        tick = mds.get_latest_tick(self.active_bot.symbol, now=now)
        stale_s = max(0.0, (now - tick.event_time).total_seconds()) if tick.event_time else None
        pos_snaps = broker.fetch_positions(tick.price, now=now)
        open_orders = broker.fetch_open_orders()
        acc_snap = broker.account_snapshot(pos_snaps, now=now)
        paper_recon = broker.reconcile(now=now)
        return TradingSnapshot(
            context=ctx,
            connection=ConnectionSnapshot("LOCAL", True, True, 0, now),
            market=MarketSnapshot(self.active_bot.symbol, self.active_bot.symbol, "PERPETUAL", "USDT", tick.price, tick.price, tick.source, tick.event_time, stale_s, tick.timestamp_source),
            account=acc_snap,
            positions=pos_snaps,
            open_orders=open_orders,
            reconciliation=paper_recon,
            risk=RiskSnapshot(True, ("VIEW", "OPEN_LONG", "OPEN_SHORT", "REDUCE_ONLY", "CLOSE_ALL"), ()),
            configured_strategy=self.bot_strategy_name,
            effective_strategy=self.bot_strategy_name,
            model_id=None,
        ).validated()

    def add_log(self, msg: str):
        """寫入記憶體並同步持久化到 SQLite 磁碟"""
        ts = _now_taipei().strftime("%H:%M:%S")
        formatted = f"[{ts}] {msg}"
        self.bot_logs.append(formatted)
        if len(self.bot_logs) > 120:
            self.bot_logs.pop(0)

        # 寫入磁碟 SQLite
        try:
            now_dt = _now_taipei().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(STATE_DB_PATH)
            cur = conn.cursor()
            cur.execute("INSERT INTO bot_logs (timestamp, message) VALUES (?, ?)", (ts, msg))
            cur.execute(
                "REPLACE INTO bot_state (key, value, updated_at) VALUES ('bot_strategy_name', ?, ?)",
                (self.bot_strategy_name, now_dt),
            )
            cur.execute(
                "REPLACE INTO bot_state (key, value, updated_at) VALUES ('execution_mode', ?, ?)",
                (self.execution_mode, now_dt),
            )
            cur.execute(
                "REPLACE INTO bot_state (key, value, updated_at) VALUES ('position', ?, ?)",
                (str(self.active_bot.position), now_dt),
            )
            cur.execute(
                "REPLACE INTO bot_state (key, value, updated_at) VALUES ('balance_usdt', ?, ?)",
                (str(self.active_bot.balance_usdt), now_dt),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def switch_bot_strategy(self, strat_name: str):
        self.bot_strategy_name = strat_name
        adapter = (
            self.testnet_client
            if self.execution_mode == "BINANCE_TESTNET"
            else (self.bitunix_client if self.execution_mode == "BITUNIX_LIVE" else None)
        )
        lgbm_path = os.path.join(DATA_DIR, "models", "crypto_lgbm_weights.joblib")
        ppo_path = os.path.join(DATA_DIR, "models", "crypto_ppo_agent.pt")

        if strat_name == "brain_alpha":
            strat = BrainAlphaCryptoStrategy(vault_db_path=self.alpha_vault.db_path)
        elif strat_name == "perp_carry":
            strat = PerpCarryBasisStrategy()
        elif strat_name == "cond_prob":
            strat = ConditionalProbTechStrategy()
        elif strat_name == "ml" and MLAlphaStrategy is not None:
            strat = MLAlphaStrategy(model_path=lgbm_path if os.path.exists(lgbm_path) else None)
        elif strat_name == "rl" and RLAlphaStrategy is not None:
            strat = RLAlphaStrategy(agent_weights_path=ppo_path if os.path.exists(ppo_path) else None)
        else:
            strat = MomentumBreakoutStrategy()
        if CryptoTradingBot is not None:
            prev_bal = float(getattr(self.active_bot, "balance_usdt", INITIAL_BALANCE_USDT) or INITIAL_BALANCE_USDT)
            prev_pos = float(getattr(self.active_bot, "position", 0.0) or 0.0)
            self.active_bot = CryptoTradingBot(symbol=self.active_bot.symbol, strategy=strat, exchange_adapter=adapter)
            self.active_bot.reconciliation_service = self.reconciliation_service
            self.active_bot.balance_usdt = prev_bal
            self.active_bot.position = prev_pos
            rm = getattr(self.active_bot, "risk_manager", None)
            if rm is not None and prev_bal > 0:
                if hasattr(rm, "reanchor_capital"):
                    rm.reanchor_capital(prev_bal)
                else:
                    rm.initial_capital = prev_bal
                    rm.peak_capital = prev_bal
                    rm.current_capital = prev_bal
                    rm.is_halted = False
            self._risk_capital_anchored = False
        self.add_log(f"已切換操盤策略至: 【{strat_name.upper()} 策略】(自動掛載優化權重)")

    def _record_brain_experiment_started(self, job) -> None:
        try:
            self.alpha_experiments.start_experiment(
                job.job_id, job.expression, source=getattr(job, "source", "AUTO_MINING"),
                priority=int(getattr(job, "priority", 20)), settings=getattr(job, "settings", {}),
            )
            self.audit_log.write(
                component="brain_queue", action="experiment_queued", outcome="OK",
                entity_type="brain_job", entity_id=job.job_id,
                details={"source": getattr(job, "source", "AUTO_MINING"), "priority": getattr(job, "priority", 20)},
            )
        except Exception:
            pass

    def _release_sim_budget(self, job) -> None:
        try:
            import hashlib as _hl
            expr = str(getattr(job, "expression", "") or "").strip()
            cid = getattr(job, "_budget_cid", None) or _hl.sha256(expr.encode("utf-8")).hexdigest()[:16]
            dom = str(getattr(job, "domain", "") or "FUNDAMENTAL_CF_EV")
            if hasattr(self, "sim_budget") and self.sim_budget is not None:
                self.sim_budget.release(cid, dom)
                if expr:
                    self.sim_budget.release(expr[:40], dom)
        except Exception:
            pass

    def _on_brain_job_failed(self, job):
        """當 WorldQuant BRAIN 模擬失敗時，記錄至經驗庫並從 Vault 清單移除不佔版面"""
        self._release_sim_budget(job)
        if getattr(self, "trial_registry", None) is not None:
            try:
                import hashlib as _hl
                _eh = _hl.sha256((getattr(job, "expression", "") or "").strip().encode("utf-8")).hexdigest()
                self.trial_registry.record_trial(
                    expression_hash=_eh,
                    settings_hash=str(getattr(job, "simulation_key", _eh)),
                    blueprint_id=str(getattr(job, "domain", "DEFAULT")),
                    metrics={"status": "FAILED"},
                    accepted=False,
                )
            except Exception:
                pass
        try:
            self.alpha_experiments.complete_experiment(
                job.job_id,
                {
                    "status": "FAILED",
                    "failure_code": "BRAIN_SIMULATION_FAILED",
                    "safe_error": getattr(job, "safe_error", None),
                    "simulation_id": getattr(job, "alpha_id", None),
                },
            )
        except Exception:
            pass
        try:
            vid = getattr(job, "vault_id", None)
            if vid is not None:
                self.alpha_vault.delete_alpha(int(vid))
                self.add_log(f"🗑️ [BRAIN 模擬失敗移除] 已移除無效因子 ID={vid} ({job.safe_error})")
        except Exception:
            pass

    def _enqueue_near_miss_refinements(
        self,
        job,
        *,
        sharpe: float,
        fitness: float,
        turnover: float,
        margin: float | None,
        failed_checks: list[str] | None = None,
    ) -> int:
        try:
            expr = str(getattr(job, "expression", "") or "").strip()
            if not expr or expr in self._refined_expr_keys:
                return 0
            max_depth = int(os.getenv("BRAIN_QUEUE_MAX_DEPTH", "16"))
            if self.brain_job_queue.queue_depth() >= max_depth + 6:
                return 0
            variants = self.alpha_miner.generate_near_miss_refinements(
                expr,
                getattr(job, "settings", None),
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                margin=margin,
                failed_checks=failed_checks,
                max_variants=3,
            )
            self._refined_expr_keys.add(expr)
            enqueued = 0
            for v in variants:
                self._refined_expr_keys.add(v["expression"])
                _, is_new = self.brain_job_queue.create_job(
                    v["expression"],
                    settings=v["settings"],
                    priority=35,
                    source="NEAR_MISS_REFINEMENT",
                )
                if is_new:
                    enqueued += 1
                    self.add_log(f"🔧 [Near-Miss 微調送測] {v['name']} ({v['reason']})")
            return enqueued
        except Exception:
            return 0

    def _on_brain_job_completed(self, job):
        self._release_sim_budget(job)
        """
        WorldQuant BRAIN 模擬完成後：
        1. 通過官方 IS-Check 與門檻者，若觸發自動/一鍵送出且 submit_alpha 成功 -> 從 Vault 清單刪除不佔版面。
        2. 若模擬成功但未送出 (例如今日已滿 2 顆或排入候選池) -> 保留在 Vault 顯示真實 alpha_id 與指標。
        3. 若模擬未達門檻/IS-Check 失敗且原本來自 Vault -> 從 Vault 移除。
        """
        try:
            m = job.metrics or {}
            sharpe = m.get("sharpe")
            fitness = m.get("fitness")
            turnover = m.get("turnover") if m.get("turnover") is not None else 0.25
            margin = m.get("margin")
            checks_passed = m.get("checks_passed", True)
            failed_checks = m.get("failed_checks") or []

            alpha_id = job.alpha_id
            vid = getattr(job, "vault_id", None)
            if sharpe is None or fitness is None or not alpha_id:
                return

            if not checks_passed:
                self.add_log(f"⚠️ [BRAIN 未過 IS-Check] Alpha {alpha_id} 失敗項目 {failed_checks}，嘗試 Near-Miss 微調")
                self._enqueue_near_miss_refinements(job, sharpe=float(sharpe), fitness=float(fitness), turnover=float(turnover), margin=margin, failed_checks=failed_checks)
                if vid is not None:
                    self.alpha_vault.delete_alpha(int(vid))
                return

            if margin is not None:
                norm_margin = float(margin) / 10000.0 if float(margin) > 0.5 else float(margin)
                if norm_margin < 0.0010:
                    self.add_log(f"⚠️ [BRAIN 未達 Margin 門檻] Alpha {alpha_id} Margin={norm_margin:.5f} < 10 bps，嘗試 Near-Miss 微調")
                    self._enqueue_near_miss_refinements(job, sharpe=float(sharpe), fitness=float(fitness), turnover=float(turnover), margin=margin, failed_checks=failed_checks)
                    if vid is not None:
                        self.alpha_vault.delete_alpha(int(vid))
                    return

            if not self.alpha_vault.passes_qualification(float(sharpe), float(fitness), float(turnover)):
                self.add_log(f"⚠️ [BRAIN 指標未達標] Alpha {alpha_id} (Sharpe={sharpe}, Fitness={fitness})，嘗試 Near-Miss 微調")
                self._enqueue_near_miss_refinements(job, sharpe=float(sharpe), fitness=float(fitness), turnover=float(turnover), margin=margin, failed_checks=failed_checks)
                if vid is not None:
                    self.alpha_vault.delete_alpha(int(vid))
                return

            try:
                self.alpha_experiments.complete_experiment(
                    job.job_id,
                    {
                        "status": "BRAIN_COMPLETE",
                        "simulation_id": str(alpha_id),
                        "sharpe": float(sharpe),
                        "fitness": float(fitness),
                        "turnover": float(turnover),
                        "margin": margin,
                        "is_check_status": "PASS" if checks_passed else "FAIL",
                    },
                )
            except Exception:
                pass
            if getattr(self, "evidence_repo", None) is not None:
                try:
                    import hashlib as _hl
                    from worldquant_brain.evidence_repository import ExperimentContext, OfficialMetrics, canonical_json
                    _dom = str(getattr(job, "domain", "DEFAULT"))
                    _expr_h = _hl.sha256((job.expression or "").strip().encode("utf-8")).hexdigest()
                    _set = dict(job.settings or {})
                    _set_h = _hl.sha256(canonical_json(_set).encode("utf-8")).hexdigest()
                    self.evidence_repo.create_experiment(ExperimentContext(
                        experiment_id=job.job_id,
                        candidate_id=job.simulation_key,
                        blueprint_id=_dom,
                        blueprint_hash=_hl.sha256(_dom.encode("utf-8")).hexdigest(),
                        hypothesis_family=_dom,
                        data_domain=_dom,
                        expression=job.expression,
                        expression_hash=_expr_h,
                        normalized_expression_hash=_expr_h,
                        structural_expression_hash=_expr_h,
                        settings=_set,
                        settings_hash=_set_h,
                        region=str(_set.get("region", "USA")),
                        universe=str(_set.get("universe", "TOP3000")),
                        delay=int(_set.get("delay", 1) or 1),
                        experiment_type=str(getattr(job, "source", "AUTO_MINING")),
                        alpha_id=str(alpha_id),
                    ))
                    self.evidence_repo.record_metrics(job.job_id, OfficialMetrics(
                        metrics_source="WORLDQUANT_BRAIN",
                        official_simulation_id=str(alpha_id),
                        sharpe=float(sharpe) if sharpe is not None else None,
                        fitness=float(fitness) if fitness is not None else None,
                        returns=None,
                        turnover=float(turnover) if turnover is not None else None,
                        margin=float(margin) if margin is not None else None,
                        submit_ready=bool(checks_passed),
                    ))
                except Exception:
                    pass
            self.alpha_miner.total_accepted += 1
            if getattr(self, "trial_registry", None) is not None:
                try:
                    import hashlib as _hl
                    _eh = _hl.sha256((job.expression or "").strip().encode("utf-8")).hexdigest()
                    self.trial_registry.record_trial(
                        expression_hash=_eh,
                        settings_hash=str(getattr(job, "simulation_key", _eh)),
                        blueprint_id=str(getattr(job, "domain", "DEFAULT")),
                        metrics={
                            "status": "BRAIN_COMPLETE",
                            "sharpe": float(sharpe),
                            "fitness": float(fitness),
                            "turnover": float(turnover),
                            "margin": margin,
                        },
                        accepted=True,
                    )
                except Exception:
                    pass
            # Only direct manual auto_submit jobs submit immediately; background mined alphas enter the daily pool so only the highest-profitability top 2 are submitted per day.
            should_try_submit = bool(getattr(job, "auto_submit", False)) and not bool(getattr(job, "simulation_only", True))

            submitted_ok = False
            if should_try_submit:
                sub_res = self.brain_client.submit_alpha(str(alpha_id), enforce_checks=True)
                job.submit_result = sub_res
                if sub_res.get("success"):
                    submitted_ok = True
                    job.submitted_to_brain = True
                    self.add_log(f"🏆 [BRAIN 驗證並送出成功] Alpha {alpha_id} 已正式送出至官網並移出清單！")
                    if vid is not None:
                        self.alpha_vault.delete_alpha(int(vid))
                    return
                else:
                    _err_u = str(sub_res.get("error") or "").upper()
                    _corr_rej = bool(sub_res.get("corr_rejected")) or ("CORR" in _err_u)
                    if _corr_rej:
                        self.add_log(f"⚠️ [BRAIN 自相關過高已刪除] Alpha {alpha_id} 因 Corr 太高被官網拒絕，已從清單移除")
                        if vid is not None:
                            self.alpha_vault.delete_alpha(int(vid))
                        return
                    self.add_log(f"ℹ️ [BRAIN 模擬成功但未送出] Alpha {alpha_id}: {sub_res.get('status')} ({sub_res.get('error')})")

            # 模擬成功但尚未送出 -> 先檢核自相關並保存證據，過門檻才存入清單與每日候選池；超標則立即啟動去相關 Near-Miss 微調
            if not submitted_ok:
                corr_info = self.brain_client.fetch_self_correlation(str(alpha_id))
                self_corr_val = corr_info.get("self_correlation")
                corr_passed = corr_info.get("passed") is not False
                if self_corr_val is not None:
                    try:
                        from worldquant_brain.correlation_evidence import (
                            AlphaCorrelationEvidence,
                            CorrelationEvidenceRepository,
                            CorrelationSource,
                            CorrelationType,
                        )
                        CorrelationEvidenceRepository(self.alpha_vault.db_path).save_evidence(
                            AlphaCorrelationEvidence(
                                candidate_alpha_id=str(alpha_id),
                                compared_alpha_id="OWN_POOL",
                                correlation_type=CorrelationType.SELF,
                                correlation_source=CorrelationSource.BRAIN_SELF_CORRELATION,
                                correlation_value=max(-1.0, min(1.0, float(self_corr_val))),
                                candidate_delay=int((job.settings or {}).get("delay", 1) or 1),
                                compared_delay=None,
                                pnl_period_start=None,
                                pnl_period_end=None,
                                observations=None,
                                official=True,
                                checked_at=datetime.now(timezone.utc).isoformat(),
                            ),
                            raw_response=corr_info,
                        )
                    except Exception:
                        pass
                try:
                    self.alpha_miner.record_domain_feedback(
                        getattr(job, "domain", "FUNDAMENTAL_CF_EV"),
                        sharpe=float(sharpe),
                        passed=bool(checks_passed and corr_passed),
                        self_corr=self_corr_val,
                        branch=getattr(job, "mcts_branch", None),
                        fitness=float(fitness),
                        margin=float(margin) if margin is not None else None,
                    )
                except Exception:
                    pass
                if not corr_passed:
                    self.add_log(f"⚠️ [BRAIN 自相關超標] Alpha {alpha_id} (Self-Corr={self_corr_val}) 未過相關性門檻，自動啟動去相關 Near-Miss 微調並從清單移除")
                    self._enqueue_near_miss_refinements(
                        job,
                        sharpe=float(sharpe),
                        fitness=float(fitness),
                        turnover=float(turnover),
                        margin=margin,
                        failed_checks=["SELF_CORRELATION_FAIL"],
                    )
                    if vid is not None:
                        self.alpha_vault.delete_alpha(int(vid))
                    return

                if vid is not None:
                    self.alpha_vault.delete_alpha(int(vid))
                self.alpha_vault.save_alpha(
                    name=f"BRAIN_{alpha_id}",
                    expression=job.expression,
                    sharpe=float(sharpe),
                    fitness=float(fitness),
                    turnover=float(turnover),
                    margin=float(margin) if margin is not None else None,
                    category="BRAIN_COMPLETE",
                    notes=f"alpha_id={alpha_id} (模擬完成)",
                    status_override="BRAIN_COMPLETE",
                    brain_alpha_id=str(alpha_id),
                    is_check_status="PASS" if checks_passed else "FAIL",
                    self_correlation=float(self_corr_val) if self_corr_val is not None else None,
                )
                if not checks_passed:
                    self.add_log(f"ℹ️ [BRAIN 未排入送出池] Alpha {alpha_id} 未通過 IS-Checks，僅留存觀察")
                elif os.getenv("BRAIN_AUTO_SUBMIT_ENABLED", "false").lower() == "true":
                    self.brain_client.stage_candidate_for_daily_selection(
                        str(alpha_id), job.expression, float(sharpe), float(fitness), float(turnover), float(margin or 0.0),
                        self_correlation=self_corr_val,
                    )
                    _flush_hr = int(os.getenv("BRAIN_DAILY_FLUSH_HOUR", "22") or "22")
                    _max_daily = int(os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2") or "2")
                    _tp_now = _now_taipei()
                    self.add_log(f"📅 [BRAIN 已排入每日擇優送出池] Alpha {alpha_id} (將於每日 {_flush_hr:02d}:00 台北時間擇優送出)")
                    if _tp_now.hour >= _flush_hr and self.brain_client.get_daily_submission_count() < _max_daily:
                        _flushed = self.brain_client.flush_daily_candidates(max_daily_submissions=_max_daily)
                        self._handle_daily_flush_results(_flushed, label="BRAIN 補齊每日配額送出")
                if checks_passed and corr_passed and float(sharpe) >= 1.45 and float(fitness) >= 1.10:
                    try:
                        from worldquant_brain.alpha_research_engines import warm_start_structural_mutate
                        max_depth = int(os.getenv("BRAIN_QUEUE_MAX_DEPTH", "16"))
                        if self.brain_job_queue.queue_depth() < max_depth + 4:
                            offspring_expr = warm_start_structural_mutate(job.expression)
                            if offspring_expr and not self.alpha_miner.has_tried_expression(offspring_expr):
                                self.alpha_miner.mark_expression_tried(offspring_expr)
                                dom_tag = getattr(job, "domain", "FUNDAMENTAL_CF_EV")
                                child_settings = self.alpha_miner.generate_candidate_settings(domain=dom_tag)
                                child_job, is_child_new = self.brain_job_queue.create_job(
                                    offspring_expr,
                                    settings=child_settings,
                                    priority=32,
                                    source="ELITE_OFFSPRING",
                                )
                                if is_child_new:
                                    child_job.domain = dom_tag
                                    self.add_log(f"🧬 [Elite Offspring 繁衍] {alpha_id} (Sharpe={sharpe}) -> Job {child_job.job_id}")
                    except Exception:
                        pass
                self.add_log(f"💾 [BRAIN 模擬成功保留] Alpha {alpha_id} (Sharpe={sharpe}, Fitness={fitness}) 已留存於清單")
        except Exception as exc:
            self.add_log(f"⚠️ [BRAIN 模擬完成處理例外] {exc}")

    def _handle_daily_flush_results(self, flushed: list[dict[str, Any]] | None, *, label: str) -> int:
        """Process daily flush responses: remove submitted or correlation-rejected alphas from AlphaVault."""
        sub_cnt = 0
        for sr in (flushed or []):
            aid = str(sr.get("alpha_id") or "")
            is_rejected = bool(sr.get("corr_rejected")) or sr.get("status") in ("SUBMISSION_CHECK_FAILED", "SUBMIT_REJECTED")
            if sr.get("success") or is_rejected:
                for a in self.alpha_vault.list_alphas():
                    if a.get("brain_alpha_id") == aid or a.get("name") == f"BRAIN_{aid}":
                        self.alpha_vault.delete_alpha(int(a["id"]))
            if sr.get("success"):
                sub_cnt += 1
                self.add_log(f"🏆 [{label}] Alpha {aid} 已正式送出並移出待送清單！")
            elif is_rejected:
                self.add_log(f"⚠️ [BRAIN 提交檢核拒絕移除] Alpha {aid} ({sr.get('status')}: {sr.get('error')}) 已移出清單")
        return sub_cnt

    def _background_alpha_miner_loop(self):
        """24/7 背景持續自動挖掘 Alpha 因子 Worker (全程遠端 BRAIN 模擬 + 每日 22:00 擇優結算送出)"""
        print("🧬 [Alpha Miner Worker] 24/7 遠端 BRAIN 因子挖掘引擎已啟動！")
        while True:
            try:
                self.heartbeats["alpha_miner"] = time.time()
                if os.getenv("BRAIN_SIMULATION_ENABLED", "").lower() == "true" and not self.brain_client.offline_mode:
                    target_depth = int(os.getenv('BRAIN_QUEUE_TARGET_DEPTH', '10') or '10')
                    cooldown_sec = float(os.getenv('BRAIN_MINING_COOLDOWN_SEC', '0') or '0')
                    last_mine_ts = getattr(self, "_last_brain_mine_ts", 0.0)
                    _fill_iters = 0
                    if self.brain_job_queue.queue_depth() > 0 and not self.brain_job_queue.worker_alive():
                        self.brain_job_queue._start_worker()
                    if self.brain_job_queue.queue_depth() == 0 and getattr(self.sim_budget, "_pending_ids", None):
                        self.sim_budget._pending_ids.clear()
                        self.sim_budget._active_families.clear()
                    from worldquant_brain.alpha_research_engines import estimate_local_structural_correlation
                    existing_completed_exprs = [
                        str(a.get("expression") or "")
                        for a in self.alpha_vault.list_alphas()
                        if a.get("expression") and str(a.get("status") or "").upper() in ("BRAIN_COMPLETE", "SUBMITTED", "IS_CHECK_PASSED")
                    ]
                    while self.brain_job_queue.queue_depth() < target_depth and (time.time() - last_mine_ts) >= cooldown_sec and _fill_iters < target_depth:
                        _fill_iters += 1
                        cand = self.alpha_miner.think_and_select_best_candidate(batch_size=12)
                        dom = str(cand.get("domain") or "FUNDAMENTAL_CF_EV")
                        local_corr, shares_raw = estimate_local_structural_correlation(cand["expression"], existing_completed_exprs)
                        _cid = hashlib.sha256(str(cand["expression"]).strip().encode("utf-8")).hexdigest()[:16]
                        budget_dec = self.sim_budget.admit(
                            BudgetCandidate(
                                candidate_id=_cid,
                                expression=cand["expression"],
                                blueprint_family=dom,
                                data_domain=dom,
                                expected_quality=min(1.0, float(cand.get("local_quality_score", 70.0)) / 100.0),
                                local_corr=local_corr,
                                shares_raw_fields=shares_raw,
                            )
                        )
                        if budget_dec.admitted:
                            sim_settings = self.alpha_miner.generate_candidate_settings(domain=dom)
                            job, is_new = self.brain_job_queue.create_job(
                                cand["expression"], settings=sim_settings, priority=20, source="AUTO_MINING", simulation_only=True
                            )
                            if is_new:
                                job._budget_cid = _cid
                                job.domain = dom
                                job.mcts_branch = cand.get("mcts_branch")
                                self._last_brain_mine_ts = time.time()
                                self._record_brain_experiment_started(job)
                                self.alpha_miner.total_mined += 1
                                self.add_log(f"🧠 [BRAIN 慢工精選送測] {cand['name']} (域={dom}, 本地評分={cand.get('local_quality_score')}, 預估Corr={local_corr}) -> Job {job.job_id}")
                            else:
                                self.sim_budget.release(_cid, dom)
                        else:
                            _now_m = time.time()
                            if _now_m - getattr(self, "_last_budget_skip_log_ts", 0.0) > 60.0:
                                self._last_budget_skip_log_ts = _now_m
                                self.add_log(f"ℹ️ [BRAIN 挖礦節流] 候選未進佇列: {budget_dec.reason} (queue={self.brain_job_queue.queue_depth()})")
                            break
                    if os.getenv("BRAIN_AUTO_SUBMIT_ENABLED", "false").lower() == "true":
                        taipei_now = _now_taipei()
                        max_daily = int(os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2") or "2")
                        claim = self.brain_scheduler.claim_due_run(worker_id=f"web-{os.getpid()}", now=taipei_now)
                        if claim.acquired:
                            flushed = self.brain_client.flush_daily_candidates(max_daily_submissions=max_daily)
                            sub_cnt = self._handle_daily_flush_results(flushed, label="BRAIN 每日最高分擇優送出")
                            self.brain_scheduler.finish(
                                claim.run_id, status="COMPLETED", considered=len(flushed or []), submitted=sub_cnt, now=taipei_now
                            )
                        elif claim.reason == "ALREADY_CLAIMED" and self.brain_client.get_daily_submission_count() < max_daily:
                            flushed = self.brain_client.flush_daily_candidates(max_daily_submissions=max_daily)
                            self._handle_daily_flush_results(flushed, label="BRAIN 當日配額補齊送出")
            except Exception as exc:
                self.add_log(f"⚠️ [Alpha Miner 迴圈例外] {exc}")
            time.sleep(1.5)

    def _background_worker_loop(self):
        """24/7 背景獨立常駐執行緒，不依賴 HTTP 請求或前端瀏覽器"""
        print("⚡ [Background Worker] 常駐執行緒啟動！無論瀏覽器是否關閉，AI 操盤引擎均在背景運作。")
        self.add_log("常駐 Background Worker 已連線至雲端主機，24/7 自動監控盤勢。")
        last_reconcile_time = 0
        while True:
            try:
                self.heartbeats["trading_bot"] = time.time()
                self.heartbeats["taiwan_scheduler"] = time.time()
                if not getattr(self, "_tw_full_universe_checked", False):
                    self._tw_full_universe_checked = True
                    try:
                        if hasattr(self, "taiwan_screener") and hasattr(self.taiwan_screener, "registry"):
                            reg = self.taiwan_screener.registry
                            if hasattr(reg, "count_active_equities") and reg.count_active_equities() < 1000 and hasattr(reg, "sync_official_universe"):
                                threading.Thread(target=reg.sync_official_universe, daemon=True).start()
                    except Exception:
                        pass
                self.heartbeats["gdrive_sync"] = time.time()

                if self.bot_running:
                    # 依當前執行模式同步 Adapter
                    if self.execution_mode == "BINANCE_TESTNET":
                        self.active_bot.exchange_adapter = self.testnet_client
                    elif self.execution_mode == "BITUNIX_LIVE":
                        self.active_bot.exchange_adapter = self.bitunix_client
                    else:
                        self.active_bot.exchange_adapter = None

                    # 執行一次策略掃描與真實下單判定 (含多幣種副節點)
                    self.active_bot.step()
                    mgr = getattr(self, "bot_manager", None)
                    if mgr is not None:
                        for _msym, _mbot in mgr.bots.items():
                            if _msym != self.active_bot.symbol:
                                _mbot.exchange_adapter = self.active_bot.exchange_adapter
                                try:
                                    _mbot.step()
                                except Exception:
                                    pass

                    # 取得目前報價
                    if self.execution_mode == "BITUNIX_LIVE":
                        ticker = self.bitunix_client.fetch_ticker(self.active_bot.symbol)
                    else:
                        ticker = self.testnet_client.fetch_ticker(self.active_bot.symbol)
                    curr_price = ticker.get("last")
                    equity = self.active_bot.balance_usdt + (self.active_bot.position * curr_price)

                    # 每 60 秒執行一次持倉同調對齊 (reconcile_with_exchange)
                    now_ts = time.time()
                    if now_ts - last_reconcile_time > 20:
                        last_reconcile_time = now_ts
                        if self.execution_mode == "BINANCE_TESTNET":
                            self._run_reconciliation_sync()
                            rec = self.testnet_client.reconcile_with_exchange(
                                self.active_bot.position, self.active_bot.symbol
                            )
                            if not rec["in_sync"]:
                                self.add_log(f"🔄 交易所持倉同調: {rec['message']}")
                                self.active_bot.position = rec["remote_position"]
                        elif self.execution_mode == "BITUNIX_LIVE":
                            rec = self.bitunix_client.reconcile_with_exchange(
                                self.active_bot.position, self.active_bot.symbol
                            )
                            if not rec["in_sync"]:
                                self.add_log(f"🔄 Bitunix 持倉同調: {rec['message']}")
                                self.active_bot.position = rec["remote_position"]
                            self._run_reconciliation_sync()
                        else:
                            self._run_reconciliation_sync()

                    # 每 30 秒記錄一次常駐心跳
                    if int(time.time()) % 30 < 5:
                        self.add_log(
                            f"常駐巡航心跳: 標的 {self.active_bot.symbol} @ {curr_price:.1f} | 淨值: {equity:.2f} | 模式: {self.execution_mode}"
                        )
            except Exception as e:
                self.add_log(f"背景掃描異常 (已自動重試): {e!s}")
            time.sleep(5)

    def _run_reconciliation_sync(self):
        """執行即時交易所對帳並更新 ReconciliationService 狀態（支援多幣種與幣對格式正規化）"""
        try:
            from crypto_trading.reconciliation_service import _normalize_recon_symbol
            mgr = getattr(self, "bot_manager", None)
            all_bots = list(mgr.bots.values()) if mgr is not None and getattr(mgr, "bots", None) else []
            if self.active_bot not in all_bots:
                all_bots.insert(0, self.active_bot)

            if self.execution_mode == "BINANCE_TESTNET" and not getattr(self.testnet_client, "mock_mode", True) and getattr(self.testnet_client, "is_connected", False):
                bal = self.testnet_client.fetch_balance()
                rem_pos = self.testnet_client.fetch_positions() or []
                oo_fn = getattr(self.testnet_client, "fetch_open_orders", None)
                rem_orders = (oo_fn(self.active_bot.symbol) or []) if callable(oo_fn) else []
                rem_eq = float(bal.get("total_equity") or 0.0) if bal.get("total_equity") is not None else float(self.active_bot.balance_usdt)

                # 將遠端持倉轉為正規化 map (例如 BTCUSDT -> BTC/USDT)
                remote_map: dict[str, float] = {}
                for rp in rem_pos:
                    nsym = _normalize_recon_symbol(str(rp.get("symbol", "")))
                    amt = float(rp.get("contracts", 0.0))
                    side = str(rp.get("side", "LONG")).upper()
                    if nsym and abs(amt) > 0:
                        remote_map[nsym] = amt if side == "LONG" else -amt

                loc_pos: dict[str, float] = {}
                for b in all_bots:
                    b.balance_usdt = rem_eq
                    rm = getattr(b, "risk_manager", None)
                    if rm is not None and rem_eq > 0 and not getattr(b, "_risk_capital_anchored", False):
                        rm.initial_capital = rem_eq
                        rm.peak_capital = rem_eq
                        rm.current_capital = rem_eq
                        rm.is_halted = False
                        b._risk_capital_anchored = True
                    bsym = _normalize_recon_symbol(getattr(b, "symbol", "BTC/USDT"))
                    target_pos = remote_map.get(bsym, 0.0)
                    b.position = target_pos
                    if abs(target_pos) > 0:
                        loc_pos[bsym] = target_pos

                # 若遠端有非目前 bot_manager 管理的幣種殘餘倉位，記錄警示但不阻斷已管理幣種對帳
                for rsym, ramt in remote_map.items():
                    if rsym not in loc_pos and abs(ramt) > 1e-4:
                        loc_pos[rsym] = ramt

                self.reconciliation_service.reconcile(
                    local_positions=loc_pos,
                    remote_positions=rem_pos,
                    local_orders=[],
                    remote_orders=rem_orders,
                    local_balance=rem_eq,
                    remote_balance=rem_eq,
                )
            else:
                eq = float(getattr(self.active_bot, "balance_usdt", 0.0) or 0.0)
                loc_pos = {}
                rem_pos = []
                for b in all_bots:
                    pos = float(getattr(b, "position", 0.0) or 0.0)
                    sym = _normalize_recon_symbol(getattr(b, "symbol", "BTC/USDT"))
                    if abs(pos) > 0:
                        loc_pos[sym] = pos
                        rem_pos.append({"symbol": sym, "contracts": abs(pos), "side": "LONG" if pos >= 0 else "SHORT"})
                self.reconciliation_service.reconcile(
                    local_positions=loc_pos,
                    remote_positions=rem_pos,
                    local_orders=[],
                    remote_orders=[],
                    local_balance=eq,
                    remote_balance=eq,
                )
        except Exception as exc:
            self.add_log(f"⚠️ [對帳同步例外] {exc}")

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

    def _send_json_response(self, data: dict[str, Any], status: int = 200):
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
        return self._send_json_response(
            {
                "ok": False,
                "error": {"code": code, "message": message, "retryable": retryable},
            },
            status=status,
        )

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

        # API: IQC 2026 候選 Alpha 評分列表 (GET /api/brain/iqc/candidates)
        if path == "/api/brain/iqc/candidates":
            from worldquant_brain.iqc_scoring import (
                IQC_POLICY_VERSION,
                IQCScoringRepository,
                build_iqc_input_from_trusted_record,
                evaluate_iqc_candidate,
            )
            vault_rows = APP_STATE.alpha_vault.list_alphas()
            repo = IQCScoringRepository(APP_STATE.alpha_vault.db_path)
            reports = []
            for row in vault_rows:
                inp = build_iqc_input_from_trusted_record(dict(row))
                rep = evaluate_iqc_candidate(inp)
                repo.save_report_with_audit(rep)
                reports.append(rep.to_dict())
            reports.sort(key=lambda x: (x.get("hard_gates_passed", False), x.get("candidate_score", 0.0)), reverse=True)
            return self._send_json_response({
                "ok": True,
                "policy_version": IQC_POLICY_VERSION,
                "count": len(reports),
                "candidates": reports,
            })

        # API: IQC 2026 每日提交規劃 (GET /api/brain/iqc/daily-plan)
        if path == "/api/brain/iqc/daily-plan":
            from worldquant_brain.iqc_scoring import (
                IQC_POLICY_VERSION,
                build_iqc_input_from_trusted_record,
                evaluate_iqc_candidate,
                plan_daily_submissions,
            )
            vault_rows = APP_STATE.alpha_vault.list_alphas()
            reports = [evaluate_iqc_candidate(build_iqc_input_from_trusted_record(dict(row))) for row in vault_rows]
            selected = plan_daily_submissions(reports, max_daily=3, max_per_family=1)
            return self._send_json_response({
                "ok": True,
                "policy_version": IQC_POLICY_VERSION,
                "max_daily": 3,
                "max_per_family": 1,
                "selected": [r.to_dict() for r in selected],
            })

        # API: 查詢單一 Alpha 的 IQC 2026 評分與稽核 (GET /api/brain/iqc/candidates/{alpha_id})
        if path.startswith("/api/brain/iqc/candidates/"):
            aid = path[len("/api/brain/iqc/candidates/"):].strip("/")
            if not aid:
                return self._send_error_json("MISSING_ALPHA_ID", "Alpha ID 未指定", status=400)
            target = None
            for row in APP_STATE.alpha_vault.list_alphas():
                if str(row.get("id")) == aid or str(row.get("brain_alpha_id")) == aid or str(row.get("name")) == f"BRAIN_{aid}":
                    target = dict(row)
                    break
            if target is None:
                return self._send_error_json("ALPHA_NOT_FOUND", f"找不到 Alpha: {aid}", status=404)
            from worldquant_brain.iqc_scoring import (
                IQCScoringRepository,
                build_iqc_input_from_trusted_record,
                evaluate_iqc_candidate,
            )
            rep = evaluate_iqc_candidate(build_iqc_input_from_trusted_record(target))
            repo = IQCScoringRepository(APP_STATE.alpha_vault.db_path)
            audit_id = repo.save_report_with_audit(rep)
            return self._send_json_response({"ok": True, "audit_id": audit_id, **rep.to_dict()})

        # API: 查詢單一 Alpha 相關性證據 (GET /api/brain/alphas/{id}/correlations 與 /top-correlated)
        if path.startswith("/api/brain/alphas/") and (path.endswith("/correlations") or path.endswith("/top-correlated")):
            is_top = path.endswith("/top-correlated")
            suffix = "/top-correlated" if is_top else "/correlations"
            aid = path[len("/api/brain/alphas/"):-len(suffix)].strip("/")
            if not aid:
                return self._send_error_json("MISSING_ALPHA_ID", "Alpha ID 未指定", status=400)
            from worldquant_brain.correlation_evidence import CorrelationEvidenceRepository, CorrelationType
            repo = CorrelationEvidenceRepository(APP_STATE.alpha_vault.db_path)
            rows = repo.list_for_candidate(
                aid,
                correlation_type=CorrelationType.SELF if is_top else None,
                limit=5 if is_top else 20,
            )
            return self._send_json_response({
                "ok": True,
                "alpha_id": aid,
                "count": len(rows),
                "items": rows,
            })

        # API: 查詢 D0/D1 配對 PnL 相關性決策 (GET /api/brain/alphas/{id}/delay-pair)
        if path.startswith("/api/brain/alphas/") and path.endswith("/delay-pair"):
            aid = path[len("/api/brain/alphas/"):-len("/delay-pair")].strip("/")
            if not aid:
                return self._send_error_json("MISSING_ALPHA_ID", "Alpha ID 未指定", status=400)
            from worldquant_brain.correlation_evidence import CorrelationEvidenceRepository, CorrelationType
            repo = CorrelationEvidenceRepository(APP_STATE.alpha_vault.db_path)
            rows = repo.list_for_candidate(aid, correlation_type=CorrelationType.D0_D1, limit=5)
            return self._send_json_response({
                "ok": True,
                "alpha_id": aid,
                "items": rows,
            })

        # API: 查詢單一 Alpha 中性化層級與診斷 (GET /api/brain/alphas/{id}/neutralization-ablation)
        if path.startswith("/api/brain/alphas/") and path.endswith("/neutralization-ablation"):
            aid = path[len("/api/brain/alphas/"):-len("/neutralization-ablation")].strip("/")
            if not aid:
                return self._send_error_json("MISSING_ALPHA_ID", "Alpha ID 未指定", status=400)
            target = None
            for row in APP_STATE.alpha_vault.list_alphas():
                if str(row.get("id")) == aid or str(row.get("brain_alpha_id")) == aid or str(row.get("name")) == f"BRAIN_{aid}":
                    target = dict(row)
                    break
            if target is None:
                return self._send_error_json("ALPHA_NOT_FOUND", f"找不到 Alpha: {aid}", status=404)
            from worldquant_brain.neutralization import (
                NeutralizationConfiguration,
                NeutralizationLayer,
                NeutralizationMode,
                has_effective_final_neutralization,
            )
            expr = str(target.get("expression") or "")
            sim_mode_str = str(target.get("neutralization") or "SUBINDUSTRY").upper()
            try:
                sim_mode = NeutralizationMode(sim_mode_str)
            except ValueError:
                sim_mode = NeutralizationMode.SUBINDUSTRY
            gn_count = expr.lower().count("group_neutralize(")
            outermost = expr.strip().lower().startswith("group_neutralize(")
            layer = (
                NeutralizationLayer.BOTH if gn_count > 0 and sim_mode != NeutralizationMode.NONE
                else NeutralizationLayer.EXPRESSION if gn_count > 0
                else NeutralizationLayer.SIMULATION_SETTING
            )
            cfg = NeutralizationConfiguration(
                simulation_mode=sim_mode,
                expression_group_neutralize_count=gn_count,
                outermost_group_neutralize=outermost,
                decay=int(target.get("decay") or 0),
                truncation=float(target.get("truncation") or 0.0),
                layer=layer,
            )
            return self._send_json_response({
                "ok": True,
                "alpha_id": aid,
                "simulation_mode": sim_mode.value,
                "layer": layer.value,
                "effective_final_neutralization": has_effective_final_neutralization(cfg),
                "double_neutralization": gn_count > 0 and sim_mode != NeutralizationMode.NONE,
            })

        # API: 查詢單一 Alpha PnL 跳變診斷 (GET /api/brain/alphas/{id}/pnl-diagnosis)
        if path.startswith("/api/brain/alphas/") and path.endswith("/pnl-diagnosis"):
            aid = path[len("/api/brain/alphas/"):-len("/pnl-diagnosis")].strip("/")
            if not aid:
                return self._send_error_json("MISSING_ALPHA_ID", "Alpha ID 未指定", status=400)
            target = None
            for row in APP_STATE.alpha_vault.list_alphas():
                if str(row.get("id")) == aid or str(row.get("brain_alpha_id")) == aid or str(row.get("name")) == f"BRAIN_{aid}":
                    target = dict(row)
                    break
            if target is None:
                return self._send_error_json("ALPHA_NOT_FOUND", f"找不到 Alpha: {aid}", status=404)
            from worldquant_brain.pnl_jump_scoring import (
                build_pnl_jump_input_from_trusted_record,
                evaluate_pnl_jump,
            )
            rep = evaluate_pnl_jump(build_pnl_jump_input_from_trusted_record(target))
            return self._send_json_response({"ok": True, **rep.to_dict()})

        # API: 查詢單一 Alpha 改進檢查清單 (GET /api/brain/alphas/{id}/improvement-checklist)
        if path.startswith("/api/brain/alphas/") and path.endswith("/improvement-checklist"):
            aid = path[len("/api/brain/alphas/"):-len("/improvement-checklist")].strip("/")
            if not aid:
                return self._send_error_json("MISSING_ALPHA_ID", "Alpha ID 未指定", status=400)
            target = None
            for row in APP_STATE.alpha_vault.list_alphas():
                if str(row.get("id")) == aid or str(row.get("brain_alpha_id")) == aid or str(row.get("name")) == f"BRAIN_{aid}":
                    target = dict(row)
                    break
            if target is None:
                return self._send_error_json("ALPHA_NOT_FOUND", f"找不到 Alpha: {aid}", status=404)
            from worldquant_brain.pnl_jump_scoring import (
                build_pnl_jump_input_from_trusted_record,
                evaluate_pnl_jump,
            )
            rep = evaluate_pnl_jump(build_pnl_jump_input_from_trusted_record(target))
            return self._send_json_response({
                "ok": True,
                "alpha_id": aid,
                "checklist_doc": "docs/ALPHA_IMPROVEMENT_CHECKLIST.md",
                "hard_blockers": list(rep.hard_blockers),
                "turnover_band": rep.turnover_band.value,
                "diagnoses": [d.value for d in rep.diagnoses],
                "actions": [a.to_dict() if hasattr(a, "to_dict") else {"action": a.action, "priority": a.priority.value, "required_validation": list(a.required_validation)} for a in rep.actions],
                "warnings": list(rep.warnings),
            })

        # API: 查詢單一 Alpha 改進優先級計畫 (GET /api/brain/alphas/{id}/improvement-plan)
        if path.startswith("/api/brain/alphas/") and path.endswith("/improvement-plan"):
            aid = path[len("/api/brain/alphas/"):-len("/improvement-plan")].strip("/")
            if not aid:
                return self._send_error_json("MISSING_ALPHA_ID", "Alpha ID 未指定", status=400)
            target = None
            for row in APP_STATE.alpha_vault.list_alphas():
                if str(row.get("id")) == aid or str(row.get("brain_alpha_id")) == aid or str(row.get("name")) == f"BRAIN_{aid}":
                    target = dict(row)
                    break
            if target is None:
                return self._send_error_json("ALPHA_NOT_FOUND", f"找不到 Alpha: {aid}", status=404)
            from worldquant_brain.improvement_priority import evaluate_improvement_priority
            target.setdefault("metrics_source", "WORLDQUANT_BRAIN" if target.get("brain_alpha_id") else "UNKNOWN")
            target.setdefault("official_simulation_id", target.get("brain_simulation_id") or target.get("brain_alpha_id"))
            plan = evaluate_improvement_priority(target)
            return self._send_json_response({"ok": True, **plan.to_dict()})

        # API: 查詢 Brain 模擬 Job 狀態 (GET /api/brain/jobs/{job_id})
        if path.startswith("/api/brain/jobs/"):
            job_id = path[len("/api/brain/jobs/") :]
            if not job_id:
                return self._send_error_json("MISSING_JOB_ID", "Job ID 未指定", status=400)
            job = APP_STATE.brain_job_queue.get_job(job_id)
            if not job:
                return self._send_error_json("JOB_NOT_FOUND", f"找不到指定的工作: {job_id}", status=404)
            job_dict = job.to_dict()
            status_val = job_dict.get("status")
            http_status = 200
            if status_val == "AUTH_FAILED":
                http_status = 401
            elif status_val == "TIMEOUT":
                http_status = 504
            elif status_val == "FAILED":
                http_status = 502
            return self._send_json_response(
                {
                    "ok": True,
                    "job": job_dict,
                    **job_dict,
                },
                status=http_status,
            )

        # API: 系統總覽與健康度 (含 Watchdog 四大核心心跳)
        if path == "/api/trading/instrument":
            from datetime import datetime, timezone
            sym = getattr(APP_STATE.active_bot, "symbol", "BTC/USDT")
            try:
                from crypto_trading.instruments import get_instrument_spec
                spec = get_instrument_spec(sym)
            except ImportError:
                from crypto_trading.instruments import BTC_USDT_BINANCE_TESTNET
                spec = BTC_USDT_BINANCE_TESTNET
            return self._send_json_response({
                "canonical_identity": spec.canonical_id,
                "ui_symbol": sym,
                "strategy_symbol": sym,
                "market_data_symbol": spec.venue_symbol,
                "execution_symbol": spec.venue_symbol,
                "position_symbol": spec.venue_symbol,
                "quote": spec.quote_asset,
                "settlement": spec.settlement_asset,
                "contract_size": str(spec.contract_size),
                "price_tick": str(spec.price_tick),
                "quantity_step": str(spec.quantity_step),
                "metadata_timestamp": datetime.now(timezone.utc).isoformat(),
            })

        if path == "/api/trading/snapshot":
            status_code, headers, body = get_trading_snapshot_response(APP_STATE.build_trading_snapshot)
            self.send_response(status_code)
            for hk, hv in headers.items():
                self.send_header(hk, hv)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/health/alpha-miner" or path == "/api/brain/status" or path == "/api/brain/connectivity":
            if not hasattr(APP_STATE, "brain_connectivity_service"):
                APP_STATE.brain_connectivity_service = BrainConnectivityService(APP_STATE.brain_client, APP_STATE.brain_job_queue, get_settings)
            local_alive = bool(APP_STATE.miner_thread and APP_STATE.miner_thread.is_alive())
            payload = APP_STATE.brain_connectivity_service.get_status(local_alpha_worker_alive=local_alive)
            payload["time_taipei"] = _now_taipei().strftime("%Y-%m-%d %H:%M:%S")
            return self._send_json_response(payload)

        if path == "/api/status" or path == "/health" or path == "/api/health":
            now_ts = time.time()
            return self._send_json_response(
                {
                    "status": "online",
                    "time": _now_taipei().strftime("%Y-%m-%d %H:%M:%S"),
                    "bot_running": APP_STATE.bot_running,
                    "bot_strategy": APP_STATE.bot_strategy_name,
                    "execution_mode": APP_STATE.execution_mode,
                    "watchdog": {
                        "trading_bot": "OK" if (now_ts - APP_STATE.heartbeats["trading_bot"] < 30) else "LAGGING",
                        "alpha_miner": "OK" if (now_ts - APP_STATE.heartbeats["alpha_miner"] < 90) else "LAGGING",
                        "taiwan_scheduler": "OK"
                        if (now_ts - APP_STATE.heartbeats["taiwan_scheduler"] < 60)
                        else "LAGGING",
                        "gdrive_sync": "OK" if (now_ts - APP_STATE.heartbeats["gdrive_sync"] < 60) else "LAGGING",
                        "total_alphas_mined": APP_STATE.alpha_miner.total_mined,
                        "total_alphas_accepted": APP_STATE.alpha_miner.total_accepted,
                    },
                }
            )

                # API: 全域可觀測性與降級元件診斷 (Phase 19 Observability)
        if path == "/api/observability":
            try:
                obs = build_platform_observability_report(
                    brain_metrics={
                        "queue_depth": APP_STATE.brain_job_queue.queue_depth(),
                        "running_workers": getattr(APP_STATE.brain_job_queue, "num_workers", 1),
                        "candidate_inventory": len(APP_STATE.alpha_vault.list_alphas()),
                    },
                    taiwan_metrics={
                        "universe_size": len(APP_STATE.tw_universe.search(limit=500)),
                    },
                )
                return self._send_json_response({"success": True, "data": obs})
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

                # API: Antigravity Taiwan Market Map (UI1-UI8: 完整覆蓋統計、多維產業聚合、共享 Snapshot ID)
        if path == "/api/taiwan/heatmap":
            try:
                qs = parse_qs(parsed.query)
                size_by = qs.get("size_by", ["turnover"])[0]
                color_by = qs.get("color_by", ["change_1d"])[0]
                all_items = APP_STATE.tw_universe.search(limit=500)
                official_cnt = len(APP_STATE.tw_universe.current_equities(limit=2500))
                priced_cnt = sum(1 for x in all_items if x.get("last_close") is not None)
                rendered_items = all_items[:160]
                snap_id = hashlib.sha256(f"{len(all_items)}:{priced_cnt}:{size_by}:{color_by}".encode("utf-8")).hexdigest()[:16]

                sectors: dict[str, list[dict[str, Any]]] = {}
                for it in rendered_items:
                    ind = str(it.get("industry") or "其他")
                    close_v = float(it.get("last_close") or 100.0)
                    vol_v = float(it.get("volume") or max(1000, (300 - int(it.get("popularity_rank", 200))) * 150))
                    chg_v = float(it.get("change_pct") if it.get("change_pct") is not None else 0.0)
                    turnover_v = round(close_v * vol_v, 2)
                    sectors.setdefault(ind, []).append({
                        "symbol": it["symbol"],
                        "code": it["code"],
                        "name": it["name"],
                        "last_close": it.get("last_close"),
                        "change_pct": chg_v,
                        "volume": int(vol_v),
                        "turnover": turnover_v,
                        "popularity_rank": it.get("popularity_rank", 999),
                        "snapshot_id": snap_id,
                    })

                sector_summaries = []
                for ind_name, s_items in sorted(sectors.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]:
                    rets = np.array([float(x["change_pct"]) for x in s_items], dtype=float)
                    turns = np.array([max(1.0, float(x["turnover"])) for x in s_items], dtype=float)
                    vols = np.array([max(1.0, float(x["volume"])) for x in s_items], dtype=float)
                    adv = int(np.sum(rets > 0))
                    dec = int(np.sum(rets < 0))
                    sector_summaries.append({
                        "industry": ind_name,
                        "equal_weight_return": round(float(np.mean(rets)), 2),
                        "turnover_weighted_return": round(float(np.sum(rets * turns) / np.sum(turns)), 2),
                        "market_cap_weighted_return": round(float(np.sum(rets * vols) / np.sum(vols)), 2),
                        "median_return": round(float(np.median(rets)), 2),
                        "advance_count": adv,
                        "decline_count": dec,
                        "advance_decline_ratio": round(adv / max(dec, 1), 2),
                        "items": s_items[:8],
                    })

                u_mode = "OFFICIAL_FULL_UNIVERSE" if official_cnt >= 1700 else "POPULAR_WATCHLIST"
                return self._send_json_response({
                    "success": True,
                    "title": "Antigravity Taiwan Market Map",
                    "as_of": datetime.now(timezone.utc).date().isoformat(),
                    "snapshot_id": snap_id,
                    "universe_mode": u_mode,
                    "size_by": size_by,
                    "color_by": color_by,
                    "official_count": official_cnt,
                    "priced_count": priced_cnt,
                    "rendered_count": sum(len(s["items"]) for s in sector_summaries),
                    "missing_count": max(0, len(all_items) - priced_cnt),
                    "stale_count": 0,
                    "excluded_count": max(0, len(all_items) - len(rendered_items)),
                    "sectors": sector_summaries,
                })
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: 台股全市場上市櫃標的搜尋與產業分類清單 (支援 TWSE/TPEx 官方同步)
        if path == "/api/taiwan/universe":
            qs = parse_qs(parsed.query)
            q = qs.get("q", [""])[0]
            ind = qs.get("industry", [""])[0]
            mkt = qs.get("market", [""])[0]
            try:
                lim = max(1, min(2500, int(qs.get("limit", ["300"])[0])))
            except Exception:
                lim = 300
            sync_flag = qs.get("sync", ["0"])[0] == "1"
            sync_info = APP_STATE.tw_universe.sync_from_official_apis() if sync_flag else None
            mode = qs.get("mode", [""])[0].strip().upper()
            if mode:
                lineage = APP_STATE.tw_universe.get_universe_with_lineage(mode=mode, limit=lim)
                items = lineage["items"]
                u_mode = lineage["universe_mode"]
                u_sha = lineage["universe_sha256"]
                is_full = lineage["is_full_universe"]
            else:
                items = APP_STATE.tw_universe.search(query=q, industry=ind, market=mkt, limit=lim)
                u_mode = "OFFICIAL_FULL_UNIVERSE" if len(APP_STATE.tw_universe.current_equities(limit=2500)) >= 1700 else "POPULAR_WATCHLIST"
                u_sha = APP_STATE.tw_universe.get_universe_with_lineage(mode=u_mode, limit=lim)["universe_sha256"]
                is_full = (u_mode == "OFFICIAL_FULL_UNIVERSE")
            industries = APP_STATE.tw_universe.list_industries()
            official_cnt = len(APP_STATE.tw_universe.current_equities(limit=2500))
            return self._send_json_response(
                {
                    "success": True,
                    "count": len(items),
                    "universe_mode": u_mode,
                    "is_full_universe": is_full,
                    "popular_watchlist_count": len(DEFAULT_POPULAR_WATCHLIST),
                    "popular_watchlist_is_full_universe": POPULAR_WATCHLIST_IS_FULL_UNIVERSE,
                    "official_equity_count": official_cnt,
                    "universe_sha256": u_sha,
                    "sync_info": sync_info,
                    "indices": MARKET_INDICES,
                    "industries": industries,
                    "items": items,
                }
            )

        # API: 元大投資先生風格個股五檔、分價量表、基本面 (本益比/EPS) 與適買度綜合診斷
        if path == "/api/taiwan/quote":
            try:
                raw_sym = parse_qs(parsed.query).get("symbol", ["2330.TW"])[0].strip()
                if raw_sym.startswith("^"):
                    sym = raw_sym
                    idx_match = next((ix for ix in MARKET_INDICES if ix["symbol"] == sym), None)
                    meta = {"name": idx_match["name"], "industry": idx_match["tag"], "market": "INDEX"} if idx_match else {"name": sym, "industry": "大盤指數", "market": "INDEX"}
                else:
                    sym = APP_STATE.tw_universe.resolve_symbol(raw_sym)
                    matches = APP_STATE.tw_universe.search(query=sym.split(".")[0], limit=1)
                    meta = matches[0] if matches else {"name": sym, "industry": "台股標的"}
                import yfinance as yf
                ticker = yf.Ticker(sym)
                df = ticker.history(period="6mo", interval="1d")
                info = {}
                try:
                    info = ticker.info or {}
                except Exception:
                    info = {}
                panel = build_yuanta_style_quote_panel(df, sym, meta=meta, fundamentals=info)
                return self._send_json_response({"success": True, "data": panel})
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

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
                return self._send_json_response(
                    {
                        "success": True,
                        "scanned_count": detailed["scanned_count"],
                        "qualified_count": detailed["qualified_count"],
                        "picks": detailed["picks"],
                        "near_picks": detailed["near_picks"],
                        "defense_hedges": detailed["defense_hedges"],
                    }
                )
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: 台股晨報獲取 (強制共用同一份後端大盤計算快取與透明度數據)
        if path == "/api/taiwan/report":
            parse_qs(parsed.query).get("format", ["markdown"])[0]
            # 確保大盤計算結果已生成
            market_info = APP_STATE.get_taiwan_market_analysis()

            if not APP_STATE.last_taiwan_report:
                try:
                    APP_STATE.last_taiwan_report = APP_STATE.report_gen.generate(cached_market_info=market_info)
                except Exception:
                    pass

            if not APP_STATE.last_taiwan_html:
                try:
                    APP_STATE.last_taiwan_html = APP_STATE.report_gen.generate_html_report(
                        cached_market_info=market_info
                    )
                except Exception:
                    pass

            return self._send_json_response(
                {
                    "success": True,
                    "content": APP_STATE.last_taiwan_report or "尚無已生成報告，請點擊「生成晨報」按鈕。",
                    "html_content": APP_STATE.last_taiwan_html or "",
                }
            )

        # API: 加密貨幣即時行情與帳戶狀態 (支援 本地沙盒, 幣安合約測試網, Bitunix 實盤)
        if path == "/api/crypto/account":
            try:
                mode = APP_STATE.execution_mode
                if mode == "BITUNIX_LIVE":
                    ticker = APP_STATE.bitunix_client.fetch_ticker(APP_STATE.active_bot.symbol)
                    bal = APP_STATE.bitunix_client.fetch_balance()
                    positions = APP_STATE.bitunix_client.fetch_positions()
                    curr_price = ticker.get("last")
                    unrealized_pnl = sum([float(p.get("unrealized_pnl", 0.0)) for p in positions]) if positions else 0.0
                    return self._send_json_response(
                        {
                            "success": True,
                            "mode": "BITUNIX_LIVE",
                            "exchange": "bitunix",
                            "symbol": APP_STATE.active_bot.symbol,
                            "current_price": curr_price,
                            "total_equity": bal.get("total_equity"),
                            "free_balance": bal.get("free_balance"),
                            "unrealized_pnl": unrealized_pnl,
                            "positions": positions,
                            "bot_running": APP_STATE.bot_running,
                            "strategy": APP_STATE.bot_strategy_name,
                            "logs": APP_STATE.bot_logs[-50:],
                        }
                    )
                elif mode == "BINANCE_TESTNET":
                    ticker = APP_STATE.testnet_client.fetch_ticker(APP_STATE.active_bot.symbol)
                    curr_price = ticker.get("last")
                    bal = APP_STATE.testnet_client.fetch_balance()
                    positions = APP_STATE.testnet_client.fetch_positions()
                    unrealized_pnl = sum([float(p.get("unrealized_pnl", 0.0)) for p in positions]) if positions else 0.0
                    return self._send_json_response(
                        {
                            "success": True,
                            "mode": "BINANCE_TESTNET",
                            "exchange": "binance_futures",
                            "symbol": APP_STATE.active_bot.symbol,
                            "current_price": curr_price,
                            "total_equity": bal.get("total_equity"),
                            "free_balance": bal.get("free_balance"),
                            "unrealized_pnl": unrealized_pnl,
                            "settlement_currency": bal.get("settlement_currency", "USDC"),
                            "zero_fee_usdc": bal.get("zero_fee_usdc", True),
                            "positions": positions,
                            "bot_running": APP_STATE.bot_running,
                            "strategy": APP_STATE.bot_strategy_name,
                            "logs": APP_STATE.bot_logs[-50:],
                        }
                    )
                else:  # PAPER_SANDBOX
                    ticker = APP_STATE.testnet_client.fetch_ticker(APP_STATE.active_bot.symbol)
                    curr_price = ticker.get("last")
                    bot = APP_STATE.active_bot
                    equity = bot.get_equity(curr_price if curr_price is not None else (bot.entry_price or 0.0))
                    if curr_price is None:
                        unrealized_pnl = None
                    elif bot.position_side == "LONG" and bot.entry_price > 0:
                        unrealized_pnl = (curr_price - bot.entry_price) * bot.position
                    elif bot.position_side == "SHORT" and bot.entry_price > 0:
                        unrealized_pnl = (bot.entry_price - curr_price) * bot.position
                    else:
                        unrealized_pnl = 0.0

                    return self._send_json_response(
                        {
                            "success": True,
                            "mode": "PAPER_SANDBOX",
                            "symbol": bot.symbol,
                            "current_price": curr_price,
                            "total_equity": round(equity, 2),
                            "balance_usdt": round(bot.balance_usdt, 2),
                            "unrealized_pnl": round(unrealized_pnl, 2) if unrealized_pnl is not None else None,
                            "position_side": bot.position_side,
                            "position_amount": round(bot.position, 4),
                            "entry_price": bot.entry_price,
                            "bot_running": APP_STATE.bot_running,
                            "strategy": APP_STATE.bot_strategy_name,
                            "logs": APP_STATE.bot_logs[-50:],
                        }
                    )
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: 加密貨幣歷史 K 線數據 (專供 TradingView Lightweight Charts 繪製)
        if path == "/api/crypto/klines":
            try:
                symbol = parse_qs(parsed.query).get("symbol", [APP_STATE.active_bot.symbol])[0]
                interval = parse_qs(parsed.query).get("interval", ["1m"])[0]

                # 支援台股代碼、大盤/國際指數 (^TWII/^TWOII/^SOX/^IXIC/^GSPC/^N225) 與加密貨幣
                if not symbol.startswith("^") and (symbol.isdigit() or (len(symbol) <= 6 and "/" not in symbol and not symbol.endswith("USDT") and not symbol.endswith("USDC"))):
                    symbol = APP_STATE.tw_universe.resolve_symbol(symbol)
                is_taiwan = symbol.startswith("^") or symbol.endswith(".TW") or symbol.endswith(".TWO")
                if is_taiwan:
                    import yfinance as yf

                    period_map = {
                        "1m": "1d",
                        "5m": "5d",
                        "15m": "5d",
                        "1h": "1mo",
                        "1d": "6mo",
                    }
                    req_period = period_map.get(interval, "1mo")
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        period=req_period,
                        interval=interval if interval in ["1m", "5m", "15m", "1h", "1d"] else "1d",
                    )
                    if df.empty:
                        # 備援以日線重試
                        df = ticker.history(period="3mo", interval="1d")
                else:
                    loader = CryptoDataLoader()
                    df = loader.fetch_ohlcv(symbol=symbol, timeframe=interval, limit=120)

                candles = []
                for idx, row in df.iterrows():
                    # Lightweight charts 接受 unix timestamp (秒)
                    ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else int(time.time())
                    candles.append(
                        {
                            "time": ts,
                            "open": float(row["Open"]),
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "volume": float(row.get("Volume", 0)),
                        }
                    )

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
                    overlay = {
                        "ma5": [],
                        "ma20": [],
                        "ma60": [],
                        "bb_upper": [],
                        "bb_lower": [],
                    }

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
                            ord_ts = int(
                                time.mktime(datetime.strptime(ord_item["timestamp"], "%Y-%m-%d %H:%M:%S").timetuple())
                            )
                            if earliest_chart_ts <= ord_ts <= latest_chart_ts + 300:
                                side_upper = ord_item["side"].upper()
                                is_buy = side_upper == "BUY"
                                markers.append(
                                    {
                                        "time": ord_ts,
                                        "position": "belowBar" if is_buy else "aboveBar",
                                        "color": "#10b981" if is_buy else "#ef4444",
                                        "shape": "arrowUp" if is_buy else "arrowDown",
                                        "text": f"{side_upper} @ {ord_item['price']:.1f}",
                                    }
                                )
                        except Exception:
                            pass

                trade_lines = {}
                if bot.position > 0 and (bot.symbol == symbol or not is_taiwan):
                    entry = bot.entry_price or (candles[-1]["close"] if candles else 0)
                    trade_lines = {
                        "entry": entry,
                        "stop_loss": round(entry * 0.98, 2),
                        "take_profit": round(entry * 1.04, 2),
                    }

                return self._send_json_response(
                    {
                        "success": True,
                        "symbol": symbol,
                        "candles": candles,
                        "overlay": overlay,
                        "markers": markers,
                        "trade_lines": trade_lines,
                    }
                )
            except Exception as e:
                return self._send_json_response({"success": False, "error": str(e)}, status=500)

        # API: Alpha Blueprint 研究契約清單與單筆查詢 (Stage 10)
        if path == "/api/brain/blueprints":
            bps = APP_STATE.research_factory.list_blueprints()
            return self._send_json_response({"success": True, "count": len(bps), "blueprints": bps})

        if path.startswith("/api/brain/blueprints/"):
            bp_id = path[len("/api/brain/blueprints/") :].strip("/")
            bp = APP_STATE.research_factory.get_blueprint(bp_id)
            if not bp:
                return self._send_json_response({"success": False, "error": "BLUEPRINT_NOT_FOUND"}, status=404)
            return self._send_json_response({"success": True, "blueprint": bp})

        # API: 單顆 Alpha 6 維度穩健性評估報告 (GET /api/brain/alphas/{id}/robustness)
        if path.startswith("/api/brain/alphas/") and path.endswith("/robustness"):
            aid = path[len("/api/brain/alphas/") : -len("/robustness")].strip("/")
            rep = APP_STATE.research_factory.get_robustness_report(aid)
            if not rep:
                return self._send_json_response({"success": False, "alpha_id": aid, "status": "NOT_EVALUATED"}, status=404)
            return self._send_json_response({"success": True, "alpha_id": aid, "robustness": rep})

        # API: 失敗經驗與修正記憶庫 (GET /api/brain/research/lessons)
        if path == "/api/brain/research/lessons":
            lessons = APP_STATE.research_factory.list_research_lessons()
            return self._send_json_response({"success": True, "count": len(lessons), "lessons": lessons})

        if path == "/api/brain/schedule":
            return self._send_json_response(APP_STATE.brain_scheduler.status())

        # API: WorldQuant BRAIN /competition/challenge 實時分數、Gold 10,000 分進度與歷史快照
        if path == "/api/brain/challenge_score":
            qs = parse_qs(parsed.query)
            do_refresh = qs.get("refresh", ["0"])[0] == "1"
            refresh_result = None
            if do_refresh:
                res = APP_STATE.brain_client.fetch_challenge_score()
                refresh_result = res
                if res.get("success") and isinstance(res.get("payload"), dict):
                    APP_STATE.brain_score_tracker.record(
                        parse_score_payload(res["payload"], source="AUTHENTICATED_CHALLENGE_API")
                    )
            latest = APP_STATE.brain_score_tracker.latest()
            hist = APP_STATE.brain_score_tracker.history(limit=30)
            eta = estimate_days_to_10k(hist, target_score=10000.0, min_observations=7)
            total_pts = float(latest.total_score) if (latest and latest.total_score is not None) else None
            dist_10k = round(max(0.0, 10000.0 - total_pts), 2) if total_pts is not None else 10000.0
            pct_10k = round(min(100.0, (total_pts / 10000.0) * 100.0), 2) if total_pts is not None else 0.0
            today_str = datetime.now(timezone.utc).date().isoformat()
            next_ny_refresh = score_refresh_at_for_taipei_date(today_str, 3).isoformat()
            return self._send_json_response({
                "success": True,
                "challenge_url": "https://platform.worldquantbrain.com/competition/challenge",
                "latest": APP_STATE.brain_score_tracker.summary()["latest"],
                "cap_status": APP_STATE.brain_score_tracker.cap_status(),
                "distance_to_10000": dist_10k,
                "progress_pct_to_10000": pct_10k,
                "eta_to_10000": eta,
                "next_official_refresh_taipei": next_ny_refresh,
                "history_count": len(hist),
                "refresh_result": refresh_result,
            })

        # API: 已找到的官方 BRAIN Alpha 清單（不受每日提交配額或後續 Gate 隱藏）
        if path == "/api/brain/alphas/discovered":
            discovered = APP_STATE.alpha_vault.list_discovered_alphas()
            stats = APP_STATE.alpha_vault.get_stats()
            return self._send_json_response(
                {
                    "success": True,
                    "count": len(discovered),
                    "stats": stats.get("summary", {}).get("discovered", {}),
                    "alphas": discovered,
                }
            )

        # API: WorldQuant BRAIN Alpha 倉庫列表與變現教學 (支援 ?quality=ROBUST_GOOD 嚴格過濾)
        if path == "/api/brain/alphas":
            qs = parse_qs(parsed.query)
            qual_filter = qs.get("quality", [""])[0].strip().upper()
            if qual_filter == "ROBUST_GOOD":
                robust_rows = APP_STATE.research_factory.list_robust_good_alphas()
                return self._send_json_response({"success": True, "quality_filter": "ROBUST_GOOD", "count": len(robust_rows), "alphas": robust_rows})
            alphas = APP_STATE.alpha_vault.list_alphas()
            stats = APP_STATE.alpha_vault.get_stats()
            roadmap = APP_STATE.alpha_vault.get_brain_roadmap_guide()
            payout_sample = APP_STATE.alpha_vault.calculate_payout_projection(
                quarterly_pnl_usd=50000.0,
                sharpe=1.85,
                turnover_pct=32.0,
                is_consultant=True,
            )
            return self._send_json_response(
                {
                    "success": True,
                    "stats": stats,
                    "alphas": alphas,
                    "roadmap": roadmap,
                    "payout_sample": payout_sample,
                }
            )

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
                html_rep = APP_STATE.report_gen.generate_html_report(
                    session_type=session_type, cached_market_info=market_info
                )
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
            return self._send_json_response(
                {
                    "success": True,
                    "bot_running": APP_STATE.bot_running,
                    "strategy": APP_STATE.bot_strategy_name,
                }
            )

        # 加密貨幣手動下單 / 平倉
        if path == "/api/crypto/order":
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
                    pass

            if APP_STATE.execution_mode == "BITUNIX_LIVE":
                res = APP_STATE.bitunix_client.create_order(
                    symbol=symbol, side=side, order_type=order_type, amount=amount
                )
                APP_STATE.add_log(f"Bitunix 實盤下單回報: {res.get('message')}")
                return self._send_json_response(res)
            elif APP_STATE.execution_mode == "BINANCE_TESTNET":
                res = APP_STATE.testnet_client.create_order(
                    symbol=symbol, side=side, order_type=order_type, amount=amount, price=float(limit_price) if (order_type == "limit" and limit_price) else None
                )
                APP_STATE.add_log(f"幣安合約 Testnet 下單回報: {res.get('message')}")
                return self._send_json_response(res)
            else:  # PAPER_SANDBOX
                ticker = APP_STATE.testnet_client.fetch_ticker(symbol)
                price = ticker["last"]
                APP_STATE.active_bot.execute_paper_order(side, amount, price, reason="Web UI 手動下單")
                APP_STATE.add_log(f"沙盒訂單成交: {side} {amount} {symbol} @ {price:.2f}")
                return self._send_json_response(
                    {
                        "success": True,
                        "message": f"沙盒訂單已成交: {side} {amount} {symbol}",
                    }
                )

        # 加密貨幣一鍵平倉
        if path == "/api/crypto/close_all":
            if APP_STATE.execution_mode == "BITUNIX_LIVE":
                res = APP_STATE.bitunix_client.close_all_positions()
                APP_STATE.add_log("Bitunix 實盤一鍵全平倉完成")
                return self._send_json_response(res)
            elif APP_STATE.execution_mode == "BINANCE_TESTNET":
                res = APP_STATE.testnet_client.close_all_positions()
                APP_STATE.add_log(f"幣安合約 Testnet 一鍵平倉: {res.get('message')}")
                return self._send_json_response(res)
            else:  # PAPER_SANDBOX
                bot = APP_STATE.active_bot
                if bot.position > 0:
                    ticker = APP_STATE.testnet_client.fetch_ticker(bot.symbol)
                    bot.execute_paper_order(
                        "SELL",
                        bot.position,
                        ticker["last"],
                        reason="Web UI 一鍵全數平倉",
                    )
                APP_STATE.add_log("UI 執行一鍵平倉完成 (沙盒部位結清)")
                return self._send_json_response({"success": True, "message": "沙盒持倉已全部結清！"})

        # WorldQuant BRAIN: 提交與儲存 Alpha
        # POST /api/brain/challenge_score：手動觸發從 BRAIN Challenge 同步或匯入 JSON 快照
        if path == "/api/brain/challenge_score":
            payload_in = post_data.get("payload")
            if isinstance(payload_in, dict) and payload_in:
                snap = APP_STATE.brain_score_tracker.record(
                    parse_score_payload(payload_in, source="MANUAL_CHALLENGE_IMPORT")
                )
                return self._send_json_response({"success": True, "source": "MANUAL_CHALLENGE_IMPORT", "snapshot": snap.raw_payload})
            res = APP_STATE.brain_client.fetch_challenge_score()
            if res.get("success") and isinstance(res.get("payload"), dict):
                snap = APP_STATE.brain_score_tracker.record(
                    parse_score_payload(res["payload"], source="AUTHENTICATED_CHALLENGE_API")
                )
                return self._send_json_response({"success": True, "source": "AUTHENTICATED_CHALLENGE_API", "endpoint": res.get("endpoint")})
            return self._send_json_response({"success": False, "error": res.get("error", "無法連線取得 Challenge 分數"), "status": res.get("status")}, status=200)

        # POST /api/brain/iqc/recalculate：僅從後端可信 Vault 重算 IQC 評分（忽略任何用戶端傳入指標，並依 input_snapshot_hash 冪等）
        if path == "/api/brain/iqc/recalculate":
            from worldquant_brain.iqc_scoring import (
                IQC_POLICY_VERSION,
                IQCScoringRepository,
                build_iqc_input_from_trusted_record,
                evaluate_iqc_candidate,
            )
            aid = str((post_data or {}).get("alpha_id") or "").strip()
            vault_rows = APP_STATE.alpha_vault.list_alphas()
            repo = IQCScoringRepository(APP_STATE.alpha_vault.db_path)
            if aid:
                target = None
                for row in vault_rows:
                    if str(row.get("id")) == aid or str(row.get("brain_alpha_id")) == aid or str(row.get("name")) == f"BRAIN_{aid}":
                        target = dict(row)
                        break
                if target is None:
                    return self._send_error_json("ALPHA_NOT_FOUND", f"找不到 Alpha: {aid}", status=404)
                rep = evaluate_iqc_candidate(build_iqc_input_from_trusted_record(target))
                audit_id, created = repo.save_report_if_changed(rep)
                return self._send_json_response({
                    "ok": True,
                    "audit_id": audit_id,
                    "audit_created": created,
                    **rep.to_dict(),
                })
            recalculated = []
            for row in vault_rows:
                rep = evaluate_iqc_candidate(build_iqc_input_from_trusted_record(dict(row)))
                audit_id, created = repo.save_report_if_changed(rep)
                item = rep.to_dict()
                item["audit_id"] = audit_id
                item["audit_created"] = created
                recalculated.append(item)
            return self._send_json_response({
                "ok": True,
                "policy_version": IQC_POLICY_VERSION,
                "count": len(recalculated),
                "candidates": recalculated,
            })

        if path == "/api/brain/submit_alpha":
            name = post_data.get("name", "Custom_Alpha")
            expr = post_data.get("expression", "").strip()
            sharpe = float(post_data.get("sharpe", 1.35))
            fitness = float(post_data.get("fitness", 1.15))
            turnover = float(post_data.get("turnover", 25.0))
            category = post_data.get("category", "Momentum")
            notes = post_data.get("notes", "")
            brain_alpha_id = str(post_data.get("brain_alpha_id") or post_data.get("alpha_id") or "").strip()

            res = APP_STATE.alpha_vault.save_alpha(
                name=name,
                expression=expr,
                sharpe=sharpe,
                fitness=fitness,
                turnover=turnover,
                category=category,
                notes=notes,
            )
            if not res.get("success"):
                res["brain_submit"] = {
                    "success": False,
                    "status": "QUALIFICATION_FAILED",
                    "error": res.get("message", "未達 BRAIN 門檻，已攔截提交"),
                }
                return self._send_json_response(res)

            if brain_alpha_id:
                brain_submit_res = APP_STATE.brain_client.submit_alpha(brain_alpha_id, enforce_checks=True)
                res["brain_submit"] = brain_submit_res
                if brain_submit_res.get("success"):
                    vault_id_to_del = post_data.get("vault_id") or res.get("alpha_id")
                    if vault_id_to_del is not None:
                        APP_STATE.alpha_vault.delete_alpha(int(vault_id_to_del))
                    res["deleted_from_vault"] = True
                    res["message"] = f"Alpha 已通過官方 IS-Check 正式提交至 WorldQuant BRAIN (ID: {brain_alpha_id})，並已從清單移除！"
                elif bool(brain_submit_res.get("corr_rejected")) or ("CORR" in str(brain_submit_res.get("error") or "").upper()):
                    vault_id_to_del = post_data.get("vault_id") or res.get("alpha_id")
                    if vault_id_to_del is not None:
                        APP_STATE.alpha_vault.delete_alpha(int(vault_id_to_del))
                    res["deleted_from_vault"] = True
                    res["message"] = f"Alpha 因自相關過高被官網拒絕 ({brain_submit_res.get('error')})，已自動從清單移除！"
                else:
                    res["deleted_from_vault"] = False
                    res["message"] = f"Alpha 模擬資料已保留，但官方提交未完成 [{brain_submit_res.get('status')}]: {brain_submit_res.get('error')}"
            return self._send_json_response(res)

        # POST /api/brain/submit_direct：對已有官方 brain_alpha_id 的因子直接執行 IS-Check + 正式 Submit（不再重跑 Simulate），送出成功即從清單移除
        if path == "/api/brain/submit_direct":
            try:
                brain_alpha_id = str(post_data.get("brain_alpha_id") or post_data.get("alpha_id") or "").strip()
                if brain_alpha_id.startswith("BRAIN_"):
                    brain_alpha_id = brain_alpha_id[len("BRAIN_"):]
                vault_id = post_data.get("vault_id")
                if not brain_alpha_id:
                    return self._send_error_json("MISSING_BRAIN_ALPHA_ID", "缺少官方 BRAIN Alpha ID，請先執行模擬", status=400)
                max_daily = int(os.getenv("BRAIN_MAX_DAILY_SUBMISSIONS", "2") or "2")
                sub_res = APP_STATE.brain_client.submit_alpha(brain_alpha_id, max_daily_submissions=max_daily, enforce_checks=True)
                if sub_res.get("success"):
                    if vault_id is not None and str(vault_id).isdigit():
                        APP_STATE.alpha_vault.delete_alpha(int(vault_id))
                    APP_STATE.add_log(f"🏆 [BRAIN 直接正式送出成功] Alpha {brain_alpha_id} 已通過官方驗證並正式送出，已移出清單！")
                    return self._send_json_response({"success": True, "alpha_id": brain_alpha_id, "message": f"Alpha {brain_alpha_id} 已正式送出至 WorldQuant BRAIN 並從清單移除！", "result": sub_res})
                _corr_rej = bool(sub_res.get("corr_rejected")) or ("CORR" in str(sub_res.get("error") or "").upper())
                deleted = False
                if _corr_rej and vault_id is not None and str(vault_id).isdigit():
                    APP_STATE.alpha_vault.delete_alpha(int(vault_id))
                    deleted = True
                    APP_STATE.add_log(f"⚠️ [BRAIN 自相關過高已刪除] Alpha {brain_alpha_id} 因 Corr 太高被拒絕，已從清單移除")
                return self._send_json_response({"success": False, "alpha_id": brain_alpha_id, "deleted_from_vault": deleted, "message": f"未通過官方提交檢查或已達每日上限 [{sub_res.get('status')}]: {sub_res.get('error')}", "result": sub_res})
            except Exception as exc:
                return self._send_error_json("SUBMIT_DIRECT_FAILED", str(exc), status=500)

        if path == "/api/brain/simulations/manual":
            try:
                expr = str(post_data.get("expression") or "").strip()
                if not expr:
                    return self._send_error_json("INVALID_EXPRESSION", "Alpha expression is required", status=400)
                requested_by = str(post_data.get("requested_by") or "manual-ui")
                priority = max(80, min(100, int(post_data.get("priority", 100))))
                job, is_new = APP_STATE.brain_job_queue.create_job(
                    expr, settings=post_data.get("settings"), auto_submit=False,
                    priority=priority, source="MANUAL", requested_by=requested_by,
                    manual_note=post_data.get("manual_note"), simulation_only=True,
                )
                if is_new:
                    APP_STATE._record_brain_experiment_started(job)
                return self._send_json_response({
                    "success": True, "created": is_new, "job_id": job.job_id,
                    "priority": job.priority, "source": job.source,
                    "simulation_only": True, "status": job.status.value,
                }, status=202)
            except Exception as exc:
                return self._send_error_json("MANUAL_QUEUE_FAILED", type(exc).__name__, status=500)

        # 🚀 POST /api/brain/simulations：建立非同步送測 job (符合 docs/BRAIN_WEB_API_ASYNC_FIX_SPEC.md)
        if path == "/api/brain/simulations":
            try:
                expr = post_data.get("expression") or post_data.get("regular") or ""
                if not expr or not expr.strip():
                    return self._send_error_json("INVALID_EXPRESSION", "Alpha 表達式不可為空", status=400)

                settings = post_data.get("settings")
                vault_id = post_data.get("vault_id")
                auto_submit = bool(post_data.get("auto_submit", vault_id is not None))
                job, is_new = APP_STATE.brain_job_queue.create_job(
                    expr, settings, vault_id=int(vault_id) if vault_id is not None else None, auto_submit=auto_submit
                )
                APP_STATE.add_log(
                    f"🚀 [Quant 測試佇列] Alpha Job {job.job_id} ({'新增' if is_new else '去重複用'}) 已排入"
                )

                return self._send_json_response(
                    {
                        "ok": True,
                        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
                        "job_id": job.job_id,
                        "simulation_key": job.simulation_key,
                        "next_poll_seconds": APP_STATE.brain_job_queue.default_poll_interval,
                    },
                    status=202,
                )
            except Exception:
                return self._send_error_json("INTERNAL_SERVER_ERROR", "建立送測作業失敗", status=500)

        # 🚀 一鍵自動送出至 Quant 測試 (向後相容舊端點，升級為 202 非同步 job 回應)
        if path == "/api/brain/test_alpha":
            try:
                post_data.get("alpha_id")
                expr = post_data.get("expression", "")
                target_engine = post_data.get("target_engine", "worldquant")

                if target_engine == "crypto_bot":
                    return self._send_error_json(
                        "UNSUPPORTED_ENGINE",
                        "股票橫截面 FASTEXPR 因子不適用於單一加密貨幣時間序列，該功能已禁用。",
                        status=400,
                    )

                if not expr or not expr.strip():
                    return self._send_error_json("INVALID_EXPRESSION", "Alpha 表達式不可為空", status=400)

                vault_id = post_data.get("vault_id") or post_data.get("alpha_id")
                auto_submit = bool(post_data.get("auto_submit", True))
                job, is_new = APP_STATE.brain_job_queue.create_job(
                    expr, vault_id=int(vault_id) if vault_id is not None and str(vault_id).isdigit() else None, auto_submit=auto_submit
                )
                APP_STATE.add_log(f"🚀 [Quant 測試送出] Job {job.job_id} 已排入非同步背景輪詢")

                return self._send_json_response(
                    {
                        "ok": True,
                        "success": True,
                        "target_engine": "WorldQuant BRAIN",
                        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
                        "job_id": job.job_id,
                        "simulation_key": job.simulation_key,
                        "next_poll_seconds": APP_STATE.brain_job_queue.default_poll_interval,
                        "message": "已成功將該 Alpha 送出至非同步模擬佇列！",
                    },
                    status=202,
                )
            except Exception as e:
                return self._send_error_json(
                    "INTERNAL_SERVER_ERROR",
                    f"WorldQuant 模擬請求處理失敗: {type(e).__name__}",
                    status=500,
                )

        return self._send_error_json("NOT_FOUND", "Endpoint not found", status=404)


# 向後相容測試導入別名
QuantRequestHandler = AppAPIHandler


def run_server(port: int | None = None, host: str = "0.0.0.0"):
    if port is None:
        port = int(os.getenv("PORT", "8000"))
    os.makedirs(STATIC_DIR, exist_ok=True)
    server_address = (host, port)
    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer.daemon_threads = True
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
