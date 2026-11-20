import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

try:
    from crypto_trading.config import RISK_PARAMS
except ImportError:
    RISK_PARAMS = {
        "max_position_pct": 0.20,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "max_drawdown_pct": 0.08,
        "leverage": 1,
        "risk_per_trade_pct": 0.01,
    }
try:
    from crypto_trading.settings import TradingMode, get_settings
except ImportError:
    class TradingMode:
        LIVE = "LIVE"
        PAPER = "PAPER"
        BITUNIX_LIVE = "BITUNIX_LIVE"
        BINANCE_TESTNET = "BINANCE_TESTNET"
        LOCAL_PAPER = "LOCAL_PAPER"
    class _FallbackSettings:
        trading_mode = "PAPER"
    def get_settings():
        return _FallbackSettings()


class DecisionAction(str, Enum):
    APPROVE = "APPROVE"
    REDUCE_ONLY = "REDUCE_ONLY"
    REJECT = "REJECT"


@dataclass
class RiskDecision:
    action: DecisionAction
    allowed: bool
    reasons: list[str]
    max_allowed_amount: float
    details: dict[str, Any]


class RiskManager:
    """
    集中式風控核心閘道 (Centralized Risk Gate)
    所有下單入口 (Bot / Manual / Reconcile) 共用單一 Gate。
    嚴格檢查：
    1. TradingMode 與 unlock 權限
    2. DataSource 標記與品質 (DataQuality, stale data 檢查)
    3. 對帳新鮮度 (reconciliation freshness)
    4. 未確認訂單 (unknown / in-flight orders)
    5. DB 是否可寫 (DB writable)
    6. 交易所連通性 (exchange connectivity)
    7. Notional, exposure, leverage, positions 數量
    8. Daily loss, drawdown 熔斷 (max_drawdown_stop_pct)
    9. 最小名義金額 (min_notional), step, tick 限制
    Fail-closed 原則：任何不確定或異常時，一律只允許 reduce-only 或直接拒絕！
    """

    def __init__(self, initial_capital: float = 10000.0, db_path: str | None = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown_limit = RISK_PARAMS.get("max_drawdown_stop_pct", 0.08)
        self.risk_per_trade = RISK_PARAMS.get("max_risk_per_trade_pct", 0.02)
        self.daily_loss_limit = RISK_PARAMS.get("max_daily_loss_pct", 0.05)
        self.max_open_positions = RISK_PARAMS.get("max_open_positions", 3)
        self.max_leverage = RISK_PARAMS.get("max_leverage", 5)
        self.max_exposure_pct = RISK_PARAMS.get("max_exposure_pct", 1.5)  # 總持倉名義價值上限
        self.min_notional = 5.0
        self.drawdown_reduce_limit = float(RISK_PARAMS.get("drawdown_reduce_pct", 0.05))
        self.cooldown_hours = float(RISK_PARAMS.get("circuit_breaker_cooldown_hours", 24.0))
        self.is_halted = False
        self.halted_at: datetime | None = None
        self.defensive_mode = False
        self._halt_logged = False
        self._capital_anchored = False
        self.daily_pnl = 0.0
        self.db_path = db_path

    def reanchor_capital(self, capital: float, *, defensive: bool = False) -> None:
        """Re-anchor initial, peak, and current capital when syncing balance or recovering after cooldown."""
        if capital is not None and float(capital) > 0:
            c = float(capital)
            self.initial_capital = c
            self.peak_capital = c
            self.current_capital = c
            self.is_halted = False
            self.halted_at = None
            self.defensive_mode = bool(defensive)
            self._halt_logged = False
            self._capital_anchored = True

    def update_capital(self, new_capital: float, *, now: datetime | None = None) -> bool:
        """更新資金並檢查分級回撤與自動冷卻復原（避免永久鎖死）"""
        from datetime import timezone
        curr_time = now or datetime.now(timezone.utc)
        if not self._capital_anchored and self.peak_capital == 10000.0 and 0 < new_capital < 9000.0:
            self.reanchor_capital(new_capital)
            return True
        self._capital_anchored = True

        # Auto-recovery after circuit-breaker cooldown window expires
        if self.is_halted and self.halted_at is not None and self.cooldown_hours > 0:
            elapsed_h = (curr_time - self.halted_at).total_seconds() / 3600.0
            if elapsed_h >= self.cooldown_hours and new_capital > 0:
                self.reanchor_capital(new_capital, defensive=True)
                return True

        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
            self.defensive_mode = False

        drawdown = (self.peak_capital - self.current_capital) / max(self.peak_capital, 1e-9)
        if drawdown >= self.max_drawdown_limit:
            self.is_halted = True
            if self.halted_at is None:
                self.halted_at = curr_time
            if not self._halt_logged:
                self._halt_logged = True
                print(
                    f"[🚨 警報] 帳戶累計回撤達到 {drawdown * 100:.2f}%，超過上限 {self.max_drawdown_limit * 100:.1f}%！啟動交易熔斷保護（{self.cooldown_hours:.0f} 小時冷卻後將自動以當前資金重錨並減半倉位復原）。"
                )
            return False

        self.defensive_mode = drawdown >= self.drawdown_reduce_limit
        self._halt_logged = False
        return True

    def calculate_position_size(self, current_price: float, atr: float) -> dict[str, float]:
        stop_distance = max(atr * RISK_PARAMS.get("atr_stop_multiplier", 2.0), current_price * 0.01)
        levels = {
            "stop_loss_distance": round(stop_distance, 2),
            "stop_loss_long": round(current_price - stop_distance, 2),
            "stop_loss_short": round(current_price + stop_distance, 2),
            "take_profit_long": round(current_price + (stop_distance * 1.5), 2),
            "take_profit_short": round(current_price - (stop_distance * 1.5), 2),
        }
        if self.is_halted:
            return {"size": 0.0, "stop_loss": 0.0, "take_profit": 0.0, **levels}

        risk_mult = 0.5 if self.defensive_mode else 1.0
        risk_amount = self.current_capital * self.risk_per_trade * risk_mult
        position_size = risk_amount / stop_distance

        max_notional = self.current_capital * (0.25 if self.defensive_mode else 0.50)
        if position_size * current_price > max_notional:
            position_size = max_notional / current_price

        return {"size": round(position_size, 4), **levels}

    def calculate_equity_aware_position_size(
        self,
        current_price: float,
        atr: float,
        *,
        signal_strength: float = 1.0,
        target_leverage: float = 2.0,
        maintenance_margin_rate: float = 0.005,
        liquidation_safety_factor: float = 3.0,
        min_notional: float = 5.0,
        step_size: float = 0.001,
        max_capital_utilization: float = 0.90,
        active_symbols_count: int = 1,
    ) -> dict[str, Any]:
        """Dynamic equity-aware sizer with safe isolated-margin leverage for small ($300) to large accounts.

        Guarantees:
          1. Effective leverage is bounded by `max_leverage` AND `liquidation_safety_factor` so the
             isolated liquidation price is always at least 3x farther than the ATR stop-loss.
          2. Maximizes safe capital utilization on live balance while respecting exchange `step_size`
             and `min_notional` (including Bitunix/Binance Tier-1 isolated margin rules).
        """
        from decimal import Decimal, ROUND_DOWN

        base = self.calculate_position_size(current_price, atr)
        if self.is_halted or current_price <= 0 or self.current_capital <= 0:
            return {**base, "size": 0.0, "notional": 0.0, "effective_leverage": 0.0, "margin_used": 0.0, "skip_reason": "HALTED_OR_INVALID_EQUITY"}

        conviction = max(0.0, min(1.0, abs(float(signal_strength))))
        if conviction <= 0.0:
            return {**base, "size": 0.0, "notional": 0.0, "effective_leverage": 0.0, "margin_used": 0.0, "skip_reason": "ZERO_SIGNAL"}

        # Compute liquidation-safe leverage cap from ATR stop distance (Binance/Bitunix isolated margin model)
        stop_dist = float(base["stop_loss_distance"])
        stop_pct = max(stop_dist / current_price, 0.01)
        mmr = max(0.004, float(maintenance_margin_rate))
        # Ensure (1 / lev) - mmr >= stop_pct * liquidation_safety_factor
        max_safe_lev_by_stop = 1.0 / max(stop_pct * float(liquidation_safety_factor) + mmr, 0.05)
        defensive_lev_cap = 1.5 if self.defensive_mode else float(self.max_leverage)
        eff_leverage = max(1.0, min(float(target_leverage), defensive_lev_cap, max_safe_lev_by_stop))

        # Per-symbol isolated margin budget -> leveraged notional
        n_syms = max(1, int(active_symbols_count))
        defensive_scale = 0.5 if self.defensive_mode else 1.0
        margin_budget = (self.current_capital / n_syms) * float(max_capital_utilization) * conviction * defensive_scale
        target_notional = margin_budget * eff_leverage
        raw_qty = target_notional / current_price

        if step_size > 0:
            d_step = Decimal(str(step_size))
            qty = float((Decimal(str(raw_qty)) / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step)
        else:
            qty = round(raw_qty, 6)

        realized_notional = qty * current_price
        if qty <= 0 or realized_notional < float(min_notional):
            return {
                **base,
                "size": 0.0,
                "notional": 0.0,
                "precheck_leverage": round(eff_leverage, 2),
            "effective_leverage": round(eff_leverage, 2),
            "liquidation_source": "PRECHECK_APPROXIMATION_ONLY",
                "margin_used": 0.0,
                "skip_reason": f"BELOW_MIN_NOTIONAL_OR_STEP(notional={realized_notional:.2f}<min={min_notional:.2f})",
            }

        margin_used = realized_notional / eff_leverage
        liq_dist_pct = max(0.01, (1.0 / eff_leverage) - mmr)
        return {
            **base,
            "size": qty,
            "notional": round(realized_notional, 2),
            "margin_used": round(margin_used, 2),
            "precheck_leverage": round(eff_leverage, 2),
            "effective_leverage": round(eff_leverage, 2),
            "liquidation_source": "PRECHECK_APPROXIMATION_ONLY",
            "est_liquidation_price_long": round(current_price * (1.0 - liq_dist_pct), 2),
            "est_liquidation_price_short": round(current_price * (1.0 + liq_dist_pct), 2),
            "capital_utilization_pct": round((margin_used / self.current_capital) * 100.0, 2),
            "defensive_mode": self.defensive_mode,
            "skip_reason": None,
        }

    def evaluate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float,
        is_reduce_only: bool = False,
        context: dict[str, Any] | None = None,
    ) -> RiskDecision:
        """
        全域唯一下單風控入口 (Centralized Risk Gate)
        Fail-closed: 遇到任何風險警報，開倉直接拒絕，平倉放行或轉為 REDUCE_ONLY
        """
        reasons = []
        ctx = context or {}

        # 1. TradingMode 檢查
        settings = get_settings()
        mode = ctx.get("trading_mode", settings.trading_mode)
        if mode == TradingMode.BITUNIX_LIVE and not getattr(settings, "live_trading_unlocked", settings.is_live):
            reasons.append("BITUNIX_LIVE is locked without user confirmation token")

        # 2. 回撤與每日虧損熔斷
        drawdown = (self.peak_capital - self.current_capital) / (self.peak_capital + 1e-9)
        if self.is_halted or drawdown >= self.max_drawdown_limit:
            reasons.append(
                f"Max drawdown circuit breaker active ({drawdown * 100:.2f}% >= {self.max_drawdown_limit * 100}%)"
            )

        if self.daily_pnl < -self.current_capital * self.daily_loss_limit:
            reasons.append(
                f"Daily loss limit reached ({self.daily_pnl:.2f} < {-self.current_capital * self.daily_loss_limit:.2f})"
            )

        # 3. 數據品質與 Stale Data 檢查
        data_source = ctx.get("data_source")
        if not data_source:
            reasons.append("DataSource metadata is missing")

        last_data_time = ctx.get("data_timestamp")
        if last_data_time is not None and last_data_time != "":
            try:
                from datetime import timezone
                if isinstance(last_data_time, (int, float)):
                    dt = datetime.fromtimestamp(float(last_data_time), tz=timezone.utc)
                elif isinstance(last_data_time, datetime):
                    dt = last_data_time if last_data_time.tzinfo is not None else last_data_time.replace(tzinfo=timezone.utc)
                else:
                    s_ts = str(last_data_time).strip()
                    if s_ts.endswith("Z"):
                        s_ts = s_ts[:-1] + "+00:00"
                    dt = datetime.fromisoformat(s_ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                staleness = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()
                max_stale = ctx.get("max_stale_seconds", 300)
                if staleness > max_stale:
                    reasons.append(f"Stale market data: age {staleness:.1f}s > {max_stale}s")
            except Exception:
                reasons.append("Invalid data_timestamp format")

        if ctx.get("data_quality_passed") is False:
            reasons.append("DataQuality check failed (missing/inf/nan candles detected)")

        # 4. 對帳新鮮度與未確認訂單
        if ctx.get("reconciliation_fresh") is False:
            reasons.append("Reconciliation is not fresh or has pending discrepancies")

        if ctx.get("has_unknown_orders", False):
            reasons.append("There are UNKNOWN / in-flight orders waiting for remote resolution")

        # 5. DB 可寫性檢查
        db_path = self.db_path or ctx.get("db_path")
        if db_path:
            try:
                test_conn = sqlite3.connect(db_path, timeout=1.0)
                test_cur = test_conn.cursor()
                test_cur.execute("CREATE TABLE IF NOT EXISTS _db_write_probe (ts TEXT)")
                test_cur.execute(
                    "INSERT INTO _db_write_probe VALUES (?)",
                    (datetime.utcnow().isoformat(),),
                )
                test_conn.commit()
                test_conn.close()
            except Exception as e:
                reasons.append(f"Database write probe failed: {e}")

        # 6. 交易所連通性檢查
        if ctx.get("exchange_connected") is False:
            reasons.append("Exchange connectivity is offline")

        # 7. 名義金額、Tick、Step 與持倉限額
        notional = amount * price
        min_not = ctx.get("min_notional", self.min_notional)
        if notional < min_not:
            reasons.append(f"Notional value {notional:.2f} < minimum required {min_not:.2f}")

        step_size = ctx.get("step_size", 0.001)
        if step_size > 0:
            from decimal import Decimal, ROUND_DOWN
            d_amt = Decimal(str(amount))
            d_step = Decimal(str(step_size))
            if ctx.get("auto_quantize_step", True):
                q_amt = float((d_amt / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step)
                if q_amt <= 0:
                    reasons.append(f"Amount {amount} quantizes to 0 under step_size {step_size}")
                else:
                    amount = q_amt
            else:
                rem = d_amt % d_step
                if rem != Decimal("0"):
                    reasons.append(f"Amount {amount} violates step_size {step_size}")

        tick_size = ctx.get("tick_size", 0.01)
        if tick_size > 0:
            remainder_p = price % tick_size
            if round(remainder_p, 6) != 0 and round(abs(remainder_p - tick_size), 6) != 0:
                reasons.append(f"Price {price} violates tick_size {tick_size}")

        # 部位與槓桿限額
        open_positions = ctx.get("open_positions_count", 0)
        if not is_reduce_only and open_positions >= self.max_open_positions:
            reasons.append(f"Max open positions count reached ({open_positions} >= {self.max_open_positions})")

        total_exposure = ctx.get("total_exposure_notional", 0.0) + (0.0 if is_reduce_only else notional)
        max_exposure = self.current_capital * self.max_exposure_pct
        if not is_reduce_only and total_exposure > max_exposure:
            reasons.append(f"Total exposure {total_exposure:.2f} exceeds cap {max_exposure:.2f}")

        # 8. 決策輸出 (Fail-Closed)
        if not reasons:
            return RiskDecision(
                action=DecisionAction.APPROVE,
                allowed=True,
                reasons=["All centralized risk checks passed"],
                max_allowed_amount=amount,
                details={
                    "notional": notional,
                    "drawdown_pct": round(drawdown * 100, 2),
                },
            )

        # 若有違規理由，檢查是否為減倉 / 平倉
        if is_reduce_only:
            # 只要不是 DB 寫入失敗或連線全斷，減倉原則上放行
            critical_blockers = [r for r in reasons if "Database write probe" in r or "Exchange connectivity" in r]
            if critical_blockers:
                return RiskDecision(
                    action=DecisionAction.REJECT,
                    allowed=False,
                    reasons=critical_blockers,
                    max_allowed_amount=0.0,
                    details={"blocked_on_reduce": True},
                )
            return RiskDecision(
                action=DecisionAction.REDUCE_ONLY,
                allowed=True,
                reasons=reasons,
                max_allowed_amount=amount,
                details={"fallback_to_reduce_only": True},
            )

        # 開倉請求有違規，一律 Fail-Closed 拒絕！
        return RiskDecision(
            action=DecisionAction.REJECT,
            allowed=False,
            reasons=reasons,
            max_allowed_amount=0.0,
            details={"fail_closed": True},
        )

    def check_drawdown(self, current_balance: float) -> bool:
        return self.update_capital(current_balance)

    @property
    def is_trading_halted(self) -> bool:
        return bool(self.is_halted)

    @staticmethod
    def verify_exchange_liquidation_price(
        *,
        entry_price: float,
        stop_loss_price: float,
        side: str,
        exchange_liq_price: float | None,
        local_precheck_liq_price: float | None = None,
        precheck_leverage: float | None = None,
        exchange_confirmed_leverage: float | None = None,
        mmr_tier_known: bool = True,
        max_mismatch_pct: float = 0.05,
        min_liq_to_stop_ratio: float = 2.0,
    ) -> dict[str, Any]:
        """Final fail-closed verification against actual exchange-reported liqPrice (Binance/Bitunix).

        Local `(1/leverage) - MMR` is strictly a PRECHECK; orders are BLOCKED if MMR tier is
        unknown, exchange liqPrice is missing, local vs exchange mismatch exceeds tolerance, or
        liquidation buffer relative to stop-loss is insufficient.
        """
        ex_liq_val = float(exchange_liq_price) if exchange_liq_price is not None else None
        local_liq_val = float(local_precheck_liq_price) if local_precheck_liq_price is not None else None
        liq_diff = (
            round(abs(ex_liq_val - local_liq_val), 6)
            if (ex_liq_val is not None and local_liq_val is not None)
            else None
        )
        audit_fields = {
            "precheck_leverage": precheck_leverage,
            "exchange_confirmed_leverage": exchange_confirmed_leverage,
            "exchange_liquidation_price": ex_liq_val,
            "local_liquidation_estimate": local_liq_val,
            "liquidation_difference": liq_diff,
        }

        if not mmr_tier_known:
            return {"allowed": False, "action": "BLOCK_ORDER", "reason": "UNKNOWN_MMR_TIER", **audit_fields}
        if ex_liq_val is None or ex_liq_val <= 0:
            return {"allowed": False, "action": "BLOCK_ORDER", "reason": "MISSING_EXCHANGE_LIQUIDATION_PRICE", **audit_fields}

        ex_liq = ex_liq_val
        entry = float(entry_price)
        stop = float(stop_loss_price)

        if local_liq_val is not None and local_liq_val > 0:
            mismatch = abs(ex_liq - local_liq_val) / entry
            if mismatch > float(max_mismatch_pct):
                return {
                    "allowed": False,
                    "action": "BLOCK_ORDER",
                    "reason": f"LIQUIDATION_MISMATCH({mismatch*100:.2f}%>{max_mismatch_pct*100:.2f}%)",
                    **audit_fields,
                }

        s_up = side.upper()
        if s_up in ("BUY", "LONG"):
            stop_dist = entry - stop
            liq_dist = entry - ex_liq
        else:
            stop_dist = stop - entry
            liq_dist = ex_liq - entry

        if stop_dist <= 0 or liq_dist <= 0 or liq_dist < stop_dist * float(min_liq_to_stop_ratio):
            return {
                "allowed": False,
                "action": "BLOCK_ORDER",
                "reason": f"INSUFFICIENT_LIQUIDATION_BUFFER(liq_dist={liq_dist:.4f}, stop_dist={stop_dist:.4f})",
                **audit_fields,
            }

        return {
            "allowed": True,
            "action": "APPROVE",
            "reason": "EXCHANGE_LIQUIDATION_CONFIRMED",
            "liq_to_stop_ratio": round(liq_dist / stop_dist, 2),
            **audit_fields,
        }
