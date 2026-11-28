import time
from datetime import datetime, timezone
from typing import Any

from crypto_trading.config import DEFAULT_SYMBOL, INITIAL_BALANCE_USDT
from crypto_trading.settings import RISK_PARAMS, USDT_M_MAKER_FEE_PCT, USDT_M_TAKER_FEE_PCT
from crypto_trading.settings import TradingMode, get_settings
from crypto_trading.data_loader import CryptoDataLoader
from crypto_trading.strategies.conditional_prob_strategy import ConditionalProbTechStrategy
try:
    from crypto_trading.strategies.native_ensemble_strategy import NativeCryptoEnsembleStrategy
except ImportError:
    NativeCryptoEnsembleStrategy = None
from crypto_trading.strategies.momentum_breakout import (
    BaseStrategy,
    MomentumBreakoutStrategy,
)

try:
    from crypto_trading.strategies.ml_strategy import MLAlphaStrategy
except ImportError:
    MLAlphaStrategy = None
try:
    from crypto_trading.strategies.rl_strategy import RLAlphaStrategy
except ImportError:
    RLAlphaStrategy = None
from crypto_trading.manual_controller import ManualPositionController
from crypto_trading.order_manager import OrderManager
from crypto_trading.regime_filter import HierarchicalRegimeFilter
from crypto_trading.risk_manager import RiskManager
from data_persistence.gdrive_sync import LocalPersistenceService


def _resolve_default_timeframe() -> str:
    import json
    from pathlib import Path
    rep = Path(__file__).resolve().parent.parent / "reports" / "mtf_conditional_prob_benchmark.json"
    if rep.exists():
        try:
            data = json.loads(rep.read_text(encoding="utf-8"))
            tf = str(data.get("recommended_timeframe", "")).strip()
            if tf in {"1m", "3m", "5m", "15m", "1h", "1d"}:
                return tf
        except Exception:
            pass
    return "1h"


class CryptoTradingBot:
    """
    虛擬貨幣自動操盤機器人核心
    支援：
    1. Paper Trading (高擬真模擬搓合帳戶)
    2. 即時價格監聽與技術/ML/RL 指標計算
    3. 自動風險管控與動態停損停利更新
    4. 支援經典動量、LightGBM/XGBoost ML Alpha、以及 PyTorch PPO RL 策略無縫切換
    5. 真實對接 Binance Demo Testnet API 與 Bitunix Adapter，自動獲取遠端真實 Order ID
    """

    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        strategy: BaseStrategy | None = None,
        exchange_adapter: Any | None = None,
        exec_timeframe: str | None = None,
        execution_mode: str | None = None,
    ):
        self.symbol = symbol
        self.execution_mode = execution_mode
        self.is_paper = get_settings().trading_mode == TradingMode.LOCAL_PAPER
        self.loader = CryptoDataLoader(symbol=symbol)
        if strategy is not None:
            self.strategy = strategy
        else:
            import os as _os
            _prof = _os.getenv("CRYPTO_STRATEGY_PROFILE", "condprob_3m").strip().lower()
            _resolved_strat = None
            if _prof in ("fib_condprob_3m", "fib_condprob_5m", "fib_condprob_15m", "fib_3m", "elite_fib_3m"):
                try:
                    from crypto_trading.strategies.fib_condprob_kline_strategy import FibCondProbKLineStrategy
                    _tf_map = {
                        "fib_condprob_3m": "3m", "fib_3m": "3m", "elite_fib_3m": "3m",
                        "fib_condprob_5m": "5m", "fib_condprob_15m": "15m",
                    }
                    _resolved_strat = FibCondProbKLineStrategy(timeframe=_tf_map.get(_prof, "3m"), maker_mode=True)
                except Exception:
                    _resolved_strat = None
            elif _prof in ("institutional_3m", "router_3m"):
                try:
                    from crypto_trading.strategies.institutional_alpha_router import InstitutionalAlphaRouter
                    _resolved_strat = InstitutionalAlphaRouter(timeframe="3m", maker_mode=True)
                except Exception:
                    _resolved_strat = None
            elif _prof in ("condprob_3m", "default", ""):
                try:
                    from crypto_trading.strategies.conditional_prob_rl_strategy import ConditionalProbRLStrategy
                    _resolved_strat = ConditionalProbRLStrategy(timeframe="3m", maker_mode=True)
                except Exception:
                    _resolved_strat = None
            if _resolved_strat is None:
                _resolved_strat = NativeCryptoEnsembleStrategy() if NativeCryptoEnsembleStrategy is not None else ConditionalProbTechStrategy()
            self.strategy = _resolved_strat
        self.exec_timeframe = exec_timeframe or getattr(self.strategy, "preferred_timeframe", None) or _resolve_default_timeframe()
        self.risk_manager = RiskManager(initial_capital=INITIAL_BALANCE_USDT)
        self.order_manager = OrderManager()
        self.manual_controller = ManualPositionController()
        self.regime_filter = HierarchicalRegimeFilter(high_tf="1h", low_tf="1m")
        self.sync = LocalPersistenceService()
        self.exchange_adapter = exchange_adapter  # BinanceTestnetClient or BitunixClient

        # 帳戶狀態
        self.balance_usdt = INITIAL_BALANCE_USDT
        self.position = 0.0  # 持倉標的數量
        self.position_side = "FLAT"  # FLAT, LONG, SHORT
        self.entry_price = 0.0
        self.entry_fee_usdt = 0.0
        self.active_stop_loss = 0.0
        self.active_take_profit = 0.0
        self.max_stop_loss_pct = float(RISK_PARAMS.get("max_stop_loss_pct", 0.035))
        self.maker_fee_pct = float(RISK_PARAMS.get("maker_fee_pct", USDT_M_MAKER_FEE_PCT))
        self.taker_fee_pct = float(RISK_PARAMS.get("taker_fee_pct", USDT_M_TAKER_FEE_PCT))
        self.trade_logs = []
        self.last_order_info = {}

    def log(self, msg: str):
        line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self.trade_logs.append(line)
        if len(self.trade_logs) > 1000:
            self.trade_logs = self.trade_logs[-500:]
        print(line)

    def _compute_order_fee(self, notional: float, *, order_type: str, post_only: bool) -> float:
        """Calculate USDT-M Futures fee: 0.02% Maker (LIMIT+post_only) vs 0.05% Taker (MARKET)."""
        if "USDC" in self.symbol.upper():
            return 0.0
        rate = self.maker_fee_pct if (order_type.upper() == "LIMIT" and post_only) else self.taker_fee_pct
        return float(notional) * rate

    def _init_or_ratchet_stops(self, side: str, price: float, atr: float) -> None:
        """Anchor stop-loss at entry (capped at max_stop_loss_pct) and only trail favorably."""
        pos_risk = self.risk_manager.calculate_position_size(price, atr)
        max_dist = price * self.max_stop_loss_pct
        if side == "LONG":
            raw_sl = float(pos_risk["stop_loss_long"])
            capped_sl = max(raw_sl, price - max_dist)
            if self.active_stop_loss <= 0.0:
                self.active_stop_loss = capped_sl
                self.active_take_profit = float(pos_risk["take_profit_long"])
            else:
                # Ratchet UP only — never widen stop-loss downward when ATR spikes
                self.active_stop_loss = max(self.active_stop_loss, capped_sl)
                if self.entry_price > 0 and atr > 0:
                    gain = price - self.entry_price
                    if gain >= 1.6 * atr:
                        # Stage 1: Break-Even Lock covering full Maker+Taker fees (+6 bps)
                        self.active_stop_loss = max(self.active_stop_loss, self.entry_price * 1.0006)
                    if gain >= 2.8 * atr:
                        # Stage 2: Chandelier Profit Ratchet (lock trend runner)
                        self.active_stop_loss = max(self.active_stop_loss, price - 1.5 * atr)
        elif side == "SHORT":
            raw_sl = float(pos_risk["stop_loss_short"])
            capped_sl = min(raw_sl, price + max_dist)
            if self.active_stop_loss <= 0.0:
                self.active_stop_loss = capped_sl
                self.active_take_profit = float(pos_risk["take_profit_short"])
            else:
                # Ratchet DOWN only — never widen stop-loss upward when ATR spikes
                self.active_stop_loss = min(self.active_stop_loss, capped_sl)
                if self.entry_price > 0 and atr > 0:
                    gain = self.entry_price - price
                    if gain >= 1.6 * atr:
                        # Stage 1: Break-Even Lock covering full Maker+Taker fees (-6 bps)
                        self.active_stop_loss = min(self.active_stop_loss, self.entry_price * 0.9994)
                    if gain >= 2.8 * atr:
                        # Stage 2: Chandelier Profit Ratchet
                        self.active_stop_loss = min(self.active_stop_loss, price + 1.5 * atr)

    def execute_order(
        self,
        side: str,
        amount: float,
        current_price: float,
        reason: str,
        reduce_only: bool = False,
        prefer_maker: bool = True,
    ):
        """下單撮合：先經 manual_override 防護，再走 RiskManager 閘道與 OrderManager 冪等下單"""
        if self.manual_controller.is_symbol_locked(self.symbol):
            self.log(f"🛡️ [手動接管防護] {self.symbol} 處於 manual_override 鎖定狀態，Bot 策略下單被抑制！")
            return

        order_side = side.upper()
        # 集中式 RiskManager 檢驗
        import math
        try:
            from crypto_trading.instruments import get_instrument_spec
            _spec = getattr(self, "instrument_spec", None) or get_instrument_spec(self.symbol)
        except ImportError:
            from crypto_trading.instruments import BTC_USDT_BINANCE_TESTNET
            _spec = getattr(self, "instrument_spec", None) or BTC_USDT_BINANCE_TESTNET

        min_qty = float(_spec.min_quantity)
        if not math.isfinite(float(amount)) or float(amount) <= 0.0:
            if reduce_only and abs(float(self.position)) < min_qty:
                self.position = 0.0
                self.position_side = "FLAT"
                self.entry_price = 0.0
            self.log(f"⛔ [數量防護] 攔截無效下單數量 {amount} {self.symbol} (防 -4003)")
            return
        try:
            amount = float(_spec.normalize_quantity(amount))
            current_price = float(_spec.normalize_price(current_price))
        except ValueError as exc:
            if reduce_only and abs(float(self.position)) < min_qty:
                self.position = 0.0
                self.position_side = "FLAT"
                self.entry_price = 0.0
            self.log(f"⛔ [精度防護] {self.symbol} 數量/價格正規化失敗: {exc}")
            return
        if amount <= 0.0 or amount < min_qty:
            if reduce_only and abs(float(self.position)) < min_qty:
                self.position = 0.0
                self.position_side = "FLAT"
                self.entry_price = 0.0
            self.log(f"⛔ [最小數量防護] {amount} < min_quantity {min_qty} ({self.symbol})，拒絕送單")
            return
        _ex_connected = True if (self.exchange_adapter is None or getattr(self.exchange_adapter, "mock_mode", False)) else bool(getattr(self.exchange_adapter, "is_connected", True))
        risk_decision = self.risk_manager.evaluate_order(
            symbol=self.symbol,
            side=order_side,
            order_type="LIMIT" if prefer_maker else "MARKET",
            amount=amount,
            price=current_price,
            is_reduce_only=reduce_only,
            context={
                "data_source": "crypto_trading_bot",
                "data_timestamp": datetime.now(timezone.utc).isoformat(),
                "data_quality_passed": True,
                "reconciliation_fresh": (getattr(self, "reconciliation_service", None).is_ready_for_trading() if getattr(self, "reconciliation_service", None) else not bool(self.exchange_adapter)),
                "has_unknown_orders": False,
                "exchange_connected": _ex_connected,
                "step_size": float(_spec.quantity_step),
                "min_quantity": min_qty,
                "tick_size": float(_spec.price_tick),
                "min_notional": float(_spec.min_notional),
                "auto_quantize_step": True,
            },
        )
        if not risk_decision.allowed:
            self.log(
                f"⛔ [風控閘道攔截] {order_side} {amount:.4f} {self.symbol} 被拒絕: {', '.join(risk_decision.reasons)}"
            )
            return

        # 1. 若有對接真實 Testnet 或 Bitunix Adapter，且處於活動狀態
        if self.exchange_adapter:
            try:
                _exec_order_type = "limit" if prefer_maker else "market"
                res = self.order_manager.execute_with_idempotency(
                    client=self.exchange_adapter,
                    symbol=self.symbol,
                    side=order_side,
                    order_type=_exec_order_type,
                    amount=amount,
                    price=current_price,
                    reduce_only=reduce_only,
                )
                if res.get("success", False):
                    _raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
                    _nested_order = (
                        res.get("order")
                        if isinstance(res.get("order"), dict)
                        else (_raw.get("order") if isinstance(_raw.get("order"), dict) else {})
                    )
                    order_id = (
                        res.get("exchange_order_id")
                        or res.get("order_id")
                        or res.get("orderId")
                        or res.get("id")
                        or _nested_order.get("orderId")
                        or _nested_order.get("id")
                        or f"MOCK_{int(time.time() * 1000)}"
                    )
                    _raw_px = (
                        _nested_order.get("avgPrice")
                        or _nested_order.get("price")
                        or res.get("price")
                        or 0.0
                    )
                    try:
                        _parsed_px = float(_raw_px)
                    except (TypeError, ValueError):
                        _parsed_px = 0.0
                    fill_price = _parsed_px if _parsed_px > 0.0 else float(current_price)
                    self.last_order_info = res
                    self.log(
                        f"⚡ [交易所委託成交] {order_side} {amount:.4f} {self.symbol} @ {fill_price:.2f} | 遠端 OrderID: {order_id} | 原因: {reason}"
                    )

                    _order_fee = self._compute_order_fee(
                        fill_price * amount,
                        order_type=_exec_order_type,
                        post_only=prefer_maker,
                    )
                    self.sync.record_order(
                        symbol=self.symbol,
                        side=order_side,
                        order_type=_exec_order_type.upper(),
                        price=fill_price,
                        amount=amount,
                        status="FILLED",
                        exchange_order_id=order_id,
                        cost=fill_price * amount,
                        fee=_order_fee,
                        reason=reason,
                    )

                    # 雙向持倉會計
                    if (
                        reduce_only
                        or (self.position_side == "LONG" and order_side == "SELL")
                        or (self.position_side == "SHORT" and order_side == "BUY")
                    ):
                        # 平倉
                        gross_pnl = (
                            (fill_price - self.entry_price) * amount
                            if self.position_side == "LONG"
                            else (self.entry_price - fill_price) * amount
                        )
                        pnl = gross_pnl - _order_fee - float(getattr(self, "entry_fee_usdt", 0.0))
                        pnl_pct = (
                            ((fill_price - self.entry_price) / self.entry_price * 100.0)
                            if (self.entry_price > 0 and self.position_side == "LONG")
                            else (
                                ((self.entry_price - fill_price) / self.entry_price * 100.0)
                                if self.entry_price > 0
                                else 0.0
                            )
                        )
                        self.log(
                            f"🔴 [交易所部位平倉] {amount:.4f} {self.symbol} @ {fill_price:.2f} | 損益: {pnl:+.2f} USDT | 原因: {reason}"
                        )
                        self.sync.record_trade_log(
                            symbol=self.symbol,
                            action="CLOSE",
                            side=order_side,
                            price=fill_price,
                            amount=amount,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason=reason,
                        )
                        if pnl < 0:
                            self.sync.record_post_mortem(
                                symbol=self.symbol,
                                entry_price=self.entry_price,
                                exit_price=fill_price,
                                loss_amount=pnl,
                                loss_pct=pnl_pct,
                                cause_analysis=f"真實交易所平倉: {reason}",
                            )
                        self.position = max(0.0, self.position - amount)
                        if self.position == 0.0:
                            self.position_side = "FLAT"
                            self.entry_price = 0.0
                            self.entry_fee_usdt = 0.0
                            self.active_stop_loss = 0.0
                            self.active_take_profit = 0.0
                        self.sync.update_position(
                            symbol=self.symbol,
                            side=self.position_side,
                            contracts=self.position,
                            entry_price=self.entry_price,
                        )
                    else:
                        # 開倉
                        self.position += amount
                        self.position_side = "LONG" if order_side == "BUY" else "SHORT"
                        self.entry_price = fill_price
                        self.entry_fee_usdt = _order_fee
                        self.sync.update_position(
                            symbol=self.symbol,
                            side=self.position_side,
                            contracts=self.position,
                            entry_price=self.entry_price,
                        )
                        self.sync.record_trade_log(
                            symbol=self.symbol,
                            action="OPEN",
                            side=order_side,
                            price=fill_price,
                            amount=amount,
                            reason=reason,
                        )
                    return
                else:
                    err_reason = res.get("error") or res.get("message") or "未知錯誤"
                    self.log(
                        f"⚠️ [交易所下單未成交或被拒] 原因: {err_reason}。保持原狀，嚴禁污染本地沙盒！"
                    )
                    self.sync.record_order(
                        symbol=self.symbol,
                        side=order_side,
                        order_type="MARKET",
                        price=current_price,
                        amount=amount,
                        status="ORDER_REJECTED",
                        exchange_order_id="NONE",
                        cost=0.0,
                        fee=0.0,
                        reason=f"REJECTED: {err_reason}",
                    )
                    return
            except Exception as e:
                self.log(f"⚠️ [交易所下單異常] {e}。保持原狀，嚴禁污染本地沙盒！")
                return

        # 2. 僅在純沙盒模式 (exchange_adapter is None) 呼叫內部沙盒 Paper Order 記帳
        self.execute_paper_order(side, amount, current_price, reason, reduce_only=reduce_only, prefer_maker=prefer_maker)

    def execute_paper_order(
        self,
        side: str,
        amount: float,
        current_price: float,
        reason: str,
        reduce_only: bool = False,
        prefer_maker: bool = True,
    ):
        """執行合約保證金模擬訂單撮合 (支援 USDT Maker 0.02% / Taker 0.05% 手續費與淨損益入帳)"""
        cost = amount * current_price
        _ot = "LIMIT" if prefer_maker else "MARKET"
        fee = self._compute_order_fee(cost, order_type=_ot, post_only=prefer_maker)
        order_side = side.upper()

        is_closing = (
            reduce_only
            or (self.position_side == "LONG" and order_side == "SELL")
            or (self.position_side == "SHORT" and order_side == "BUY")
        )

        if is_closing and self.position > 0:
            close_amount = min(self.position, amount)
            gross_pnl = (
                (current_price - self.entry_price) * close_amount
                if self.position_side == "LONG"
                else (self.entry_price - current_price) * close_amount
            )
            pnl = gross_pnl - fee - float(getattr(self, "entry_fee_usdt", 0.0))
            pnl_pct = (
                ((current_price - self.entry_price) / self.entry_price * 100.0)
                if (self.entry_price > 0 and self.position_side == "LONG")
                else (((self.entry_price - current_price) / self.entry_price * 100.0) if self.entry_price > 0 else 0.0)
            )

            self.balance_usdt += gross_pnl - fee
            self.log(
                f"🔴 [模擬合約平倉] {close_amount:.4f} {self.symbol} ({self.position_side}) @ {current_price:.2f} | 損益: {pnl:+.2f} USDT (手續費: {fee:.2f}) | 原因: {reason}"
            )

            self.sync.record_order(
                symbol=self.symbol,
                side=order_side,
                order_type="MARKET",
                price=current_price,
                amount=close_amount,
                status="FILLED",
                exchange_order_id=f"PAPER_{int(time.time() * 1000)}",
                cost=cost,
                fee=fee,
                reason=reason,
            )
            self.sync.record_trade_log(
                symbol=self.symbol,
                action="CLOSE",
                side=order_side,
                price=current_price,
                amount=close_amount,
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason=reason,
            )

            if pnl < 0:
                self.sync.record_post_mortem(
                    symbol=self.symbol,
                    entry_price=self.entry_price,
                    exit_price=current_price,
                    loss_amount=pnl,
                    loss_pct=pnl_pct,
                    cause_analysis=f"平倉觸發原因: {reason}",
                )

            self.position = max(0.0, self.position - close_amount)
            if self.position == 0.0:
                self.position_side = "FLAT"
                self.entry_price = 0.0
                self.entry_fee_usdt = 0.0
                self.active_stop_loss = 0.0
                self.active_take_profit = 0.0
            self.sync.update_position_record(
                symbol=self.symbol,
                side=self.position_side,
                contracts=self.position,
                entry_price=self.entry_price,
            )
        else:
            # 開倉 (做多或做空)
            self.balance_usdt -= fee
            self.position += amount
            self.position_side = "LONG" if order_side == "BUY" else "SHORT"
            self.entry_price = current_price
            self.entry_fee_usdt = fee
            self.log(
                f"🟢 [模擬合約開倉] {order_side} {amount:.4f} {self.symbol} @ {current_price:.2f} | 名義價值: {cost:.2f} USDT (手續費: {fee:.2f}) | 原因: {reason}"
            )

            self.sync.record_order(
                symbol=self.symbol,
                side=order_side,
                order_type="MARKET",
                price=current_price,
                amount=amount,
                status="FILLED",
                exchange_order_id=f"PAPER_{int(time.time() * 1000)}",
                cost=cost,
                fee=fee,
                reason=reason,
            )
            self.sync.update_position_record(
                symbol=self.symbol,
                side=self.position_side,
                contracts=self.position,
                entry_price=self.entry_price,
            )
            self.sync.record_trade_log(
                symbol=self.symbol,
                action="OPEN",
                side=order_side,
                price=current_price,
                amount=amount,
                reason=reason,
            )

    def get_equity(self, current_price: float) -> float:
        """合約淨值計算：equity = wallet_balance + unrealized_pnl"""
        if self.position_side == "LONG" and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) * self.position
        elif self.position_side == "SHORT" and self.entry_price > 0:
            unrealized_pnl = (self.entry_price - current_price) * self.position
        else:
            unrealized_pnl = 0.0
        return self.balance_usdt + unrealized_pnl

    def step(
        self,
        *,
        max_notional_usdt: float | None = None,
        free_balance_usdt: float | None = None,
    ):
        """單次決策迴圈步進 (整合高階趨勢濾網 Hierarchical Regime Filter 與成本門檻)"""
        # 1. 分離高低時框資料，各取至少 100 根已收盤 K 線
        from crypto_trading.settings import TradingMode, get_settings
        _allow_syn = get_settings().trading_mode == TradingMode.LOCAL_PAPER
        df_exec = self.loader.fetch_ohlcv(limit=250, timeframe=getattr(self, "exec_timeframe", "5m"), allow_synthetic=_allow_syn)
        df_high = self.loader.fetch_ohlcv(limit=100, timeframe="1h", allow_synthetic=_allow_syn)

        df_strat = self.strategy.generate_signals(df_exec)
        curr = df_strat.iloc[-1]
        current_price = float(curr["Close"])
        self.last_price = current_price
        _raw_sig_f = float(curr["signal"])
        raw_signal = 1 if _raw_sig_f >= 0.25 else (-1 if _raw_sig_f <= -0.25 else 0)
        atr = float(curr.get("ATR", current_price * 0.02))

        # 1.1 高階時框 (1h) 市場狀態檢驗與信號濾波 (傳入獨立 df_high)
        regime_info = self.regime_filter.detect_regime(df_high)
        signal = self.regime_filter.filter_micro_signal(raw_signal, regime_info)

        # 1.2 自癒空倉狀態：若 position 歸零或低於最小下單量，立即重置為 FLAT（防 -4003 迴圈）
        try:
            from crypto_trading.instruments import get_instrument_spec
            _spec = getattr(self, "instrument_spec", None) or get_instrument_spec(self.symbol)
            _min_qty = float(_spec.min_quantity)
        except Exception:
            _min_qty = 1e-6
        if self.position_side != "FLAT" and abs(float(self.position)) < _min_qty:
            self.position = 0.0
            self.position_side = "FLAT"
            self.entry_price = 0.0

        # 計算當前總資產淨值 (合約保證金會計)
        total_equity = self.get_equity(current_price)
        self.risk_manager.update_capital(total_equity)
        try:
            self.sync.record_equity(total_equity, self.balance_usdt, total_equity - self.balance_usdt)
        except Exception:
            pass

        # 2. 判斷是否有停損或平倉信號 (棘輪式移動停損：絕不因 ATR 暴增而放寬停損；停損用市價確保成交、停利用 Maker 省 60% 費率)
        if self.position_side == "LONG" and self.position >= _min_qty and self.entry_price > 0:
            self._init_or_ratchet_stops("LONG", current_price if self.active_stop_loss > 0 else self.entry_price, atr)
            if current_price <= self.active_stop_loss:
                self.execute_order(
                    "SELL",
                    self.position,
                    current_price,
                    reason=f"觸發多頭棘輪停損 ({self.active_stop_loss:.2f})",
                    reduce_only=True,
                    prefer_maker=False,
                )
                return
            elif self.active_take_profit > 0 and current_price >= self.active_take_profit:
                self.execute_order(
                    "SELL",
                    self.position,
                    current_price,
                    reason=f"達到多頭 1.5R 目標停利 ({self.active_take_profit:.2f})",
                    reduce_only=True,
                    prefer_maker=True,
                )
                return
        elif self.position_side == "SHORT" and self.position >= _min_qty and self.entry_price > 0:
            self._init_or_ratchet_stops("SHORT", current_price if self.active_stop_loss > 0 else self.entry_price, atr)
            if current_price >= self.active_stop_loss:
                self.execute_order(
                    "BUY",
                    self.position,
                    current_price,
                    reason=f"觸發空頭棘輪停損 ({self.active_stop_loss:.2f})",
                    reduce_only=True,
                    prefer_maker=False,
                )
                return
            elif self.active_take_profit > 0 and current_price <= self.active_take_profit:
                self.execute_order(
                    "BUY",
                    self.position,
                    current_price,
                    reason=f"達到空頭 1.5R 目標停利 ({self.active_take_profit:.2f})",
                    reduce_only=True,
                    prefer_maker=True,
                )
                return

        # 3. 處理策略開平倉信號 (支援 signal == 1 多, signal == -1 空, 反向確認平倉)
        if self.position_side == "LONG" and self.position >= _min_qty and signal == -1:
            self.execute_order(
                "SELL",
                self.position,
                current_price,
                reason="策略反向空頭信號確認平多",
                reduce_only=True,
                prefer_maker=True,
            )
            return
        if self.position_side == "SHORT" and self.position >= _min_qty and signal == 1:
            self.execute_order(
                "BUY",
                self.position,
                current_price,
                reason="策略反向多頭信號確認平空",
                reduce_only=True,
                prefer_maker=True,
            )
            return

        if signal in (1, -1) and self.position_side == "FLAT":
            # 3.1 進場成本門檻檢驗 (當策略提供 expected_edge 且未自帶 cost_gated 時)
            if not getattr(self.strategy, "cost_gated", False) and "expected_edge" in curr.index:
                try:
                    _edge_val = float(curr["expected_edge"])
                    from crypto_trading.execution_cost_filter import evaluate_cost_barrier
                    if not evaluate_cost_barrier(expected_return=_edge_val).trade_allowed:
                        return
                except Exception:
                    pass

            pos_plan = self.risk_manager.calculate_position_size(current_price, atr)
            size = float(pos_plan["size"])
            # 3.2 強制套用 PortfolioRiskAllocator 上限與可用保證金上限 (防 -2019 Margin insufficient)
            notional_caps = []
            if max_notional_usdt is not None and float(max_notional_usdt) > 0:
                notional_caps.append(float(max_notional_usdt))
            if free_balance_usdt is not None and float(free_balance_usdt) >= 0:
                notional_caps.append(float(free_balance_usdt) * 0.90)
            if notional_caps and current_price > 0:
                cap_qty = min(notional_caps) / current_price
                size = min(size, cap_qty)
            if size >= _min_qty:
                side = "BUY" if signal == 1 else "SELL"
                reason = "條件機率+技術指標達標做多" if signal == 1 else "條件機率+技術指標達標做空"
                self.execute_order(side, size, current_price, reason=reason, prefer_maker=True)
                if self.position_side in ("LONG", "SHORT") and self.entry_price > 0:
                    self._init_or_ratchet_stops(self.position_side, self.entry_price, atr)
        elif signal == 0 and self.position_side != "FLAT" and self.position >= _min_qty:
            close_side = "SELL" if self.position_side == "LONG" else "BUY"
            self.execute_order(
                close_side,
                self.position,
                current_price,
                reason="策略離場信號",
                reduce_only=True,
            )

    def run_loop(self, iterations: int | None = None, sleep_sec: int = 2):
        """啟動監聽迴圈 (支援 iterations=None 無限常駐背景操盤)"""
        self.log("🤖 虛擬貨幣量化操盤機器人已啟動！")
        self.log(f"⚙️ 模式: {'【模擬沙盒交易 Paper Trading】' if self.is_paper else '【真實實盤交易 Live Trading】'}")
        self.log(f"💰 當前初始資金: {self.balance_usdt:.2f} USDT | 監控交易對: {self.symbol}")
        self.log("=" * 60)

        step_i = 0
        while True:
            step_i += 1
            if iterations is not None and step_i > iterations:
                break
            cycle_desc = f"Cycle {step_i}" + (f"/{iterations}" if iterations else "")
            self.log(f"--- 執行心跳檢測 {cycle_desc} ---")
            self.step()
            mark_p = float(getattr(self, "last_price", 0.0) or self.entry_price or 0.0)
            total_equity = self.get_equity(mark_p) if mark_p > 0 else self.balance_usdt
            self.log(
                f"📊 當前帳戶淨值: {total_equity:.2f} USDT | 倉位狀態: {self.position_side} ({self.position:.4f} {self.symbol})"
            )
            time.sleep(sleep_sec)

        self.log("✅ 機器人週期自檢完畢，處於隨時可常駐監聽狀態。")


if __name__ == "__main__":
    bot = CryptoTradingBot()
    bot.run_loop(iterations=None, sleep_sec=5)
