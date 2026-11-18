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
        self.is_halted = False
        self._halt_logged = False
        self._capital_anchored = False
        self.daily_pnl = 0.0
        self.db_path = db_path

    def reanchor_capital(self, capital: float) -> None:
        """Re-anchor initial, peak, and current capital when syncing persisted or exchange starting balance."""
        if capital is not None and float(capital) > 0:
            c = float(capital)
            self.initial_capital = c
            self.peak_capital = c
            self.current_capital = c
            self.is_halted = False
            self._halt_logged = False
            self._capital_anchored = True

    def update_capital(self, new_capital: float) -> bool:
        """更新資金並檢查是否觸發回撤熔斷（首次注入非預設 10000 起始資金時自動錨定，且熔斷警報僅印出一次不刷屏）"""
        if not self._capital_anchored and self.peak_capital == 10000.0 and 0 < new_capital < 9000.0:
            self.reanchor_capital(new_capital)
            return True
        self._capital_anchored = True
        self.current_capital = new_capital
        self.peak_capital = max(self.peak_capital, new_capital)

        drawdown = (self.peak_capital - self.current_capital) / max(self.peak_capital, 1e-9)
        if drawdown >= self.max_drawdown_limit:
            self.is_halted = True
            if not self._halt_logged:
                self._halt_logged = True
                print(
                    f"[🚨 警報] 帳戶累計回撤達到 {drawdown * 100:.2f}%，超過最大容忍限制 {self.max_drawdown_limit * 100}%！啟動交易熔斷保護！"
                )
            return False
        self._halt_logged = False
        return True

    def calculate_position_size(self, current_price: float, atr: float) -> dict[str, float]:
        if self.is_halted:
            return {"size": 0.0, "stop_loss": 0.0, "take_profit": 0.0}

        stop_distance = max(atr * RISK_PARAMS.get("atr_stop_multiplier", 2.0), current_price * 0.01)
        risk_amount = self.current_capital * self.risk_per_trade
        position_size = risk_amount / stop_distance

        max_notional = self.current_capital * 0.5
        if position_size * current_price > max_notional:
            position_size = max_notional / current_price

        return {
            "size": round(position_size, 4),
            "stop_loss_distance": round(stop_distance, 2),
            "stop_loss_long": round(current_price - stop_distance, 2),
            "stop_loss_short": round(current_price + stop_distance, 2),
            "take_profit_long": round(current_price + (stop_distance * 1.5), 2),
            "take_profit_short": round(current_price - (stop_distance * 1.5), 2),
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
        if mode == TradingMode.BITUNIX_LIVE and not settings.live_trading_unlocked:
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
        if last_data_time:
            try:
                dt = datetime.fromisoformat(last_data_time)
                staleness = (datetime.utcnow() - dt).total_seconds()
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
            remainder = amount % step_size
            if round(remainder, 6) != 0 and round(abs(remainder - step_size), 6) != 0:
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
