from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from taiwan_stock.config import MARKET_INDEX_SYMBOL


class MarketAnalyzer:
    """
    台股大盤環境分析器
    負責評估當前加權指數趨勢、均線架構、波動率與市場多空溫度
    """

    def __init__(self, symbol: str = MARKET_INDEX_SYMBOL):
        self.symbol = symbol

    def fetch_market_data(self, period: str = "6mo") -> pd.DataFrame:
        """獲取大盤歷史 K 線數據"""
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period)
            if df.empty:
                print(f"[警告] 無法取得大盤數據: {self.symbol}")
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"[錯誤] 獲取大盤歷史數據失敗: {e}")
            return pd.DataFrame()

    def analyze(self) -> dict[str, Any]:
        """
        執行大盤多空結構綜合判斷
        重構透明且精準的 0-100 分數公式：
        1. 均線趨勢分 (Trend Score, 滿分 40 分):
           - 多頭排列 (MA5 > MA20 > MA60): 40 分
           - 站上 MA20 且 MA20 > MA60: 30 分
           - 站上 MA20 但 MA20 < MA60: 20 分
           - 站上 MA5 但壓在 MA20 之下 (空頭反彈格局): 12 分
           - 均線全面破線/空頭排列: 0 分
        2. 短期動能分 (Momentum Score, 滿分 25 分):
           - 5日報酬 > +2.0%: 25 分
           - 5日報酬 > 0%: 18 分
           - 5日報酬 -1.5% ~ 0%: 10 分
           - 5日報酬 < -1.5%: 0 分
        3. 20 日年化波動率風險調節 (Volatility Score, 滿分 20 分):
           - 年化波動率 <= 12%: 20 分 (極佳低風險)
           - 年化波動率 <= 18%: 15 分 (平穩)
           - 年化波動率 <= 25%: 8 分 (偏高注意)
           - 年化波動率 > 25%: 0 分 (極度劇烈震盪，風險高)
        4. 市場廣度與量能 (Breadth & Volume, 滿分 15 分):
           - 今日成交量 > 5日均量 且 當日上漲: 15 分 (良性放量走強)
           - 今日成交量 <= 5日均量 但 當日上漲: 10 分 (縮量微漲)
           - 今日成交量 <= 5日均量 且 當日下跌: 6 分 (量縮回檔)
           - 今日成交量 > 5日均量 且 當日下跌: 0 分 (爆量下殺)
        """
        df = self.fetch_market_data()
        if df.empty or len(df) < 60:
            return {
                "status": "DATA_UNAVAILABLE",
                "score": None,
                "regime": "DATA_UNAVAILABLE",
                "summary": "大盤數據取得不足，拒絕以預設 50 分假冒中性市場。",
            }

        close = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
        latest_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = (latest_close - prev_close) / prev_close * 100

        # 計算移動平均線
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])

        # 均線架構與型態說明
        bullish_alignment = ma5 > ma20 > ma60
        bearish_alignment = ma5 < ma20 < ma60
        above_ma5 = latest_close > ma5
        above_ma20 = latest_close > ma20
        above_ma60 = latest_close > ma60

        # 1. 均線趨勢分 (40 分)
        ma_pattern = ""
        if bullish_alignment:
            trend_score = 40
            ma_pattern = "多頭排列格局 (MA5 > MA20 > MA60)"
        elif above_ma20 and (ma20 > ma60):
            trend_score = 30
            ma_pattern = "中多格局 (站穩月季線)"
        elif above_ma20 and not (ma20 > ma60):
            trend_score = 20
            ma_pattern = "月線反彈震盪格局"
        elif above_ma5 and not above_ma20:
            trend_score = 12
            ma_pattern = "空頭反彈格局 (站上MA5但受制MA20下壓)"
        else:
            trend_score = 0
            ma_pattern = "全面空頭破線格局"

        # 2. 短期動能分 (25 分)
        ret_5d = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100) if len(close) >= 6 else change_pct
        if ret_5d >= 2.0:
            momentum_score = 25
        elif ret_5d >= 0.0:
            momentum_score = 18
        elif ret_5d >= -1.5:
            momentum_score = 10
        else:
            momentum_score = 0

        # 3. 20 日歷史年化波動率風險調節 (20 分)
        returns = close.pct_change().dropna()
        volatility_20d = float(returns.tail(20).std() * np.sqrt(250) * 100)
        if np.isnan(volatility_20d):
            volatility_20d = 15.0

        if volatility_20d <= 12.0:
            vol_score = 20
        elif volatility_20d <= 18.0:
            vol_score = 15
        elif volatility_20d <= 25.0:
            vol_score = 8
        else:
            vol_score = 0

        # 4. 市場廣度與量能 (15 分)
        vol_ma5 = float(volume.rolling(5).mean().iloc[-1]) if len(volume) >= 5 else 1.0
        latest_vol = float(volume.iloc[-1]) if len(volume) > 0 else 1.0
        is_up = change_pct >= 0
        volume_surging = latest_vol > vol_ma5

        if volume_surging and is_up:
            breadth_score = 15
            volume_status = "放量上攻"
        elif (not volume_surging) and is_up:
            breadth_score = 10
            volume_status = "量縮緩漲"
        elif (not volume_surging) and (not is_up):
            breadth_score = 6
            volume_status = "量縮回檔"
        else:
            breadth_score = 0
            volume_status = "爆量下跌 (出貨警訊)"

        # 總分計算 (0 ~ 100)
        score = int(trend_score + momentum_score + vol_score + breadth_score)
        score = max(0, min(100, score))

        # 評定市場狀態與建議
        if score >= 75:
            regime = "強烈看多 (積極做多)"
            suggestion = "大盤均線呈多頭排列且站穩月季線，適合提高多單持股水位，鎖定強勢題材股。"
        elif score >= 60:
            regime = "謹慎偏多 (順勢操作)"
            suggestion = "指數偏多但可能面臨震盪，建議精選個股、不盲目追高，設好動態移動停利。"
        elif score <= 35:
            regime = "空頭防禦 (嚴格減倉/避險)"
            suggestion = "指數跌破關鍵均線，下行風險加大，建議降低持股成數或布局避險空單。"
        else:
            regime = "區間震盪 (觀望或高拋低吸)"
            suggestion = "均線糾結且方向不明，控制部位至三成以內，等待明確放量突破信號。"

        return {
            "status": "正常",
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "latest_close": round(latest_close, 2),
            "change_pct": round(change_pct, 2),
            "ma5": round(ma5, 2),
            "ma20": round(ma20, 2),
            "ma60": round(ma60, 2),
            "ma_pattern": ma_pattern,
            "bullish_alignment": bullish_alignment,
            "bearish_alignment": bearish_alignment,
            "above_ma5": above_ma5,
            "above_ma20": above_ma20,
            "above_ma60": above_ma60,
            "volatility_20d": round(volatility_20d, 2),
            "volume_status": volume_status,
            "sub_scores": {
                "trend": trend_score,
                "momentum": momentum_score,
                "volatility": vol_score,
                "breadth": breadth_score,
            },
            "score": score,
            "regime": regime,
            "suggestion": suggestion,
        }


if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    res = analyzer.analyze()
    print("=== 台股大盤分析結果 ===")
    for k, v in res.items():
        print(f"{k}: {v}")
