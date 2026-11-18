from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from taiwan_stock.config import DEFAULT_WATCHLIST, PARAMS
from taiwan_stock.universe_registry import TaiwanUniverseRegistry, build_yuanta_style_quote_panel


class StockScreener:
    """
    台股技術面與動量選股引擎
    篩選邏輯：
    1. 均線糾結後向上發散 / 多頭排列 (MA5 > MA20 > MA60)
    2. 成交量顯著放大 (今日量 > 5日均量 * 1.5)
    3. 布林通道突破或壓縮後帶量突破上軌
    4. KD 指標黃金交叉 (K值向上穿越D值且在強勢區間)
    5. 動態提供 ATR 停損點位計算
    """

    def __init__(
        self,
        watchlist: list[str] | None = None,
        top_n: int = 25,
        universe_mode: str | None = None,
        registry: TaiwanUniverseRegistry | None = None,
    ):
        self.registry = registry or TaiwanUniverseRegistry()
        if watchlist is not None:
            self.watchlist = list(watchlist)
            self.universe_mode = "CUSTOM_WATCHLIST"
            self.security_type_filter = "CUSTOM"
            import hashlib, json
            self.universe_sha256 = hashlib.sha256(json.dumps(self.watchlist, separators=(",", ":")).encode("utf-8")).hexdigest()
        else:
            official_equities = self.registry.current_equities(markets={"TWSE", "TIB", "TPEX"}, include_etf=False, limit=2500)
            chosen_mode = universe_mode or ("OFFICIAL_FULL_UNIVERSE" if len(official_equities) >= 1700 else "POPULAR_WATCHLIST")
            if chosen_mode == "POPULAR_WATCHLIST":
                # Only scan top_n common stocks from popular watchlist (never claim full-market)
                eq_items = self.registry.current_equities(markets={"TWSE", "TIB", "TPEX"}, include_etf=False, limit=top_n)
                self.watchlist = [x["symbol"] for x in eq_items] if eq_items else list(DEFAULT_WATCHLIST)
                self.universe_mode = "POPULAR_WATCHLIST"
                self.security_type_filter = "COMMON_STOCK"
                import hashlib, json
                self.universe_sha256 = hashlib.sha256(json.dumps(self.watchlist, separators=(",", ":")).encode("utf-8")).hexdigest()
            else:
                lineage = self.registry.get_universe_with_lineage(mode=chosen_mode, limit=2500)
                self.watchlist = [x["symbol"] for x in lineage["items"]][: max(top_n, 2500) if chosen_mode == "OFFICIAL_FULL_UNIVERSE" else top_n]
                self.universe_mode = lineage["universe_mode"]
                self.security_type_filter = lineage["security_type_filter"]
                self.universe_sha256 = lineage["universe_sha256"]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算關鍵技術指標"""
        df = df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # 移動平均線
        df["MA5"] = close.rolling(5).mean()
        df["MA20"] = close.rolling(20).mean()
        df["MA60"] = close.rolling(60).mean()
        df["VOL_MA5"] = df["Volume"].rolling(5).mean()

        # 布林通道 (Bollinger Bands)
        std20 = close.rolling(20).std()
        df["BB_MID"] = df["MA20"]
        df["BB_UPPER"] = df["BB_MID"] + (std20 * PARAMS["bbands_std"])
        df["BB_LOWER"] = df["BB_MID"] - (std20 * PARAMS["bbands_std"])

        # KD 指標 (Stochastic Oscillator 9,3,3)
        low9 = low.rolling(PARAMS["kd_period"]).min()
        high9 = high.rolling(PARAMS["kd_period"]).max()
        # 嚴格使用 np.maximum 與 epsilon 防禦零分母，避免極端數值暴衝
        kd_denom = np.maximum(high9 - low9, 1e-4)
        rsv = ((close - low9) / kd_denom) * 100

        # 簡易計算 EMA 方式的 K 與 D
        k_list = [50.0]
        d_list = [50.0]
        for val in rsv.fillna(50):
            k = (2 / 3) * k_list[-1] + (1 / 3) * val
            d = (2 / 3) * d_list[-1] + (1 / 3) * k
            k_list.append(k)
            d_list.append(d)
        df["K"] = k_list[1:]
        df["D"] = d_list[1:]

        # ATR 指標 (真實波動區間)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(PARAMS["atr_period"]).mean()

        return df

    def evaluate_stock(self, symbol: str) -> dict[str, Any] | None:
        """針對單一股票評估技術突破信號與分數 (不論是否達到門檻皆返回分析結果)"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="6mo")
            if df.empty or len(df) < 60:
                return None

            df = self.calculate_indicators(df)
            curr = df.iloc[-1]
            prev = df.iloc[-2]

            signals = []
            score = 0

            # 條件 1: 均線多頭排列
            if curr["MA5"] > curr["MA20"] > curr["MA60"]:
                signals.append("均線多頭排列 (MA5>MA20>MA60)")
                score += 25
            elif curr["MA5"] > curr["MA20"]:
                signals.append("短天期均線金叉 (MA5>MA20)")
                score += 10

            # 條件 2: 成交量放大
            vol_denom = max(float(curr["VOL_MA5"]), 1.0)
            if curr["Volume"] > curr["VOL_MA5"] * PARAMS["volume_surge_multiplier"]:
                vol_ratio = round(float(curr["Volume"]) / vol_denom, 2)
                signals.append(f"爆量突破 (成交量為5日均量 {vol_ratio} 倍)")
                score += 25
            elif curr["Volume"] > curr["VOL_MA5"]:
                signals.append("成交量溫和增溫 (>5日均量)")
                score += 10

            # 條件 3: 布林通道突破
            if curr["Close"] >= curr["BB_UPPER"] and prev["Close"] < prev["BB_UPPER"]:
                signals.append("帶量突破布林通道上軌")
                score += 20
            elif curr["Close"] > curr["BB_MID"]:
                score += 10

            # 條件 4: KD 黃金交叉或強勢高檔鈍化
            if curr["K"] > curr["D"] and prev["K"] <= prev["D"]:
                signals.append(f"KD低檔/中檔黃金交叉 (K:{curr['K']:.1f}, D:{curr['D']:.1f})")
                score += 20
            elif curr["K"] > 70 and curr["D"] > 70:
                signals.append(f"KD強勢高檔鈍化 (K:{curr['K']:.1f})")
                score += 15

            # 條件 5: 破線轉弱信號 (空方預警)
            bearish_signals = []
            if curr["Close"] < curr["MA20"] and prev["Close"] >= prev["MA20"]:
                bearish_signals.append("帶量跌破20日月線防守")
            if curr["MA5"] < curr["MA20"] < curr["MA60"]:
                bearish_signals.append("均線完整空頭排列")
            if curr["Close"] < curr["BB_LOWER"]:
                bearish_signals.append("跌破布林下軌 (超跌/弱勢主跌)")

            atr = curr["ATR"] if not np.isnan(curr["ATR"]) else (curr["Close"] * 0.03)
            suggested_stop_loss = round(curr["Close"] - (atr * PARAMS["atr_stop_loss_multiplier"]), 2)
            suggested_target = round(curr["Close"] + (atr * PARAMS["atr_stop_loss_multiplier"] * 2), 2)

            matches = self.registry.search(query=symbol.split(".")[0], limit=1)
            meta = matches[0] if matches else {"name": symbol, "industry": "台股標的"}
            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}
            panel = build_yuanta_style_quote_panel(df, symbol, meta=meta, fundamentals=info)
            suit = panel.get("suitability") or {}
            fund = panel.get("fundamentals") or {}
            blended_score = int(round(0.5 * score + 0.5 * float(suit.get("overall_score", score))))

            return {
                "symbol": symbol,
                "name": meta.get("name", symbol),
                "industry": meta.get("industry", "台股標的"),
                "close": round(curr["Close"], 2),
                "change_pct": round((curr["Close"] - prev["Close"]) / prev["Close"] * 100, 2),
                "volume": int(curr["Volume"]),
                "score": blended_score,
                "tech_breakout_score": score,
                "suitability_verdict": suit.get("verdict", "中性觀望"),
                "suitability_code": suit.get("verdict_code", "HOLD"),
                "pe_ratio": fund.get("trailing_pe"),
                "eps": fund.get("trailing_eps"),
                "dividend_yield_pct": fund.get("dividend_yield_pct"),
                "signals": (signals + (suit.get("reasons") or [])[:2]) if signals else (suit.get("reasons") or ["盤整觀望中"]),
                "bearish_signals": bearish_signals,
                "is_weak": len(bearish_signals) > 0 or (curr["MA5"] < curr["MA20"]),
                "meets_threshold": blended_score >= 55 and suit.get("verdict_code") != "CAUTION",
                "stop_loss": suggested_stop_loss,
                "target_price": suggested_target,
                "risk_reward_ratio": "1 : 2.0",
            }
        except Exception as e:
            print(f"[篩選略過] {symbol} 處理失敗: {e}")
            return None

    def screen_stock(self, symbol: str) -> dict[str, Any] | None:
        """針對單一股票評估技術突破信號 (相容舊版介面)"""
        res = self.evaluate_stock(symbol)
        if res and res["meets_threshold"]:
            return res
        return None

    def run_screener_detailed(self) -> dict[str, Any]:
        """
        執行完整選股掃描，提供選股透明度與空頭避險閉環：
        返回包含:
        - scanned_count: 總掃描檔數
        - qualified_count: 符合門檻檔數
        - picks: 符合門檻 (score >= 50) 之多方強勢推薦標的
        - near_picks: 若 0 檔符合，自動列出前 3 檔最接近門檻標的
        - defense_hedges: 空頭防禦標的 (如破線轉弱空方標的或反向ETF 00632R)
        """
        all_evals = []
        defense_candidates = []
        print(f"[*] 開始掃描台股選股池，共 {len(self.watchlist)} 檔標的...")

        # 額外將反向避險 ETF 00632R.TW (元大台灣50反1) 加入掃描
        scan_pool = list(self.watchlist)
        if "00632R.TW" not in scan_pool:
            scan_pool.append("00632R.TW")

        provider_missing: list[str] = []
        insufficient_history: list[str] = []
        for symbol in scan_pool:
            res = self.evaluate_stock(symbol)
            if res:
                all_evals.append(res)
                if res["symbol"] == "00632R.TW" or res["is_weak"]:
                    defense_candidates.append(res)
            elif symbol != "00632R.TW":
                provider_missing.append(symbol)

        qualified = [e for e in all_evals if e["meets_threshold"] and e["symbol"] != "00632R.TW"]
        qualified.sort(key=lambda x: x["score"], reverse=True)

        all_evals.sort(key=lambda x: x["score"], reverse=True)
        near_picks = all_evals[:3] if len(qualified) == 0 else []

        defense_hedges = []
        # 優先放入反向 ETF
        for d in defense_candidates:
            if d["symbol"] == "00632R.TW":
                defense_hedges.insert(
                    0,
                    {
                        "symbol": "00632R.TW (元大台灣50反1)",
                        "close": d["close"],
                        "change_pct": d["change_pct"],
                        "reason": "大盤空頭防禦反向避險工具 (對沖系統性回檔風險)",
                    },
                )
            elif len(defense_hedges) < 3 and d["bearish_signals"]:
                defense_hedges.append(
                    {
                        "symbol": d["symbol"],
                        "close": d["close"],
                        "change_pct": d["change_pct"],
                        "reason": f"破線轉弱: {', '.join(d['bearish_signals'])}",
                    }
                )

        return {
            "universe_mode": self.universe_mode,
            "universe_count": len(self.watchlist),
            "universe_sha256": self.universe_sha256,
            "security_type_filter": self.security_type_filter,
            "is_full_universe": self.universe_mode == "OFFICIAL_FULL_UNIVERSE" and len(self.watchlist) >= 1700,
            "official_count": len(self.watchlist),
            "provider_available_count": len([e for e in all_evals if e["symbol"] != "00632R.TW"]),
            "screenable_count": len([e for e in all_evals if e["symbol"] != "00632R.TW"]),
            "provider_missing_symbols": provider_missing,
            "insufficient_history_symbols": insufficient_history,
            "scanned_count": len(scan_pool),
            "qualified_count": len(qualified),
            "picks": qualified,
            "near_picks": near_picks,
            "defense_hedges": defense_hedges,
        }

    def run_screener(self) -> list[dict[str, Any]]:
        """批次掃描選股池 (回傳符合門檻或最接近門檻標的)"""
        detailed = self.run_screener_detailed()
        return detailed["picks"]


if __name__ == "__main__":
    screener = StockScreener()
    candidates = screener.run_screener()
    print(f"\n=== 篩選完成，共選出 {len(candidates)} 檔具備進場潛力標的 ===")
    for c in candidates:
        print(
            f"[{c['symbol']}] 分數:{c['score']} 現價:{c['close']} ({c['change_pct']}%) 信號: {', '.join(c['signals'])}"
        )
