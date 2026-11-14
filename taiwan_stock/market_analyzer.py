import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
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

    def analyze(self) -> Dict[str, Any]:
        """
        執行大盤多空結構綜合判斷
        返回綜合評級、均線狀態、動量指標與操作建議
        """
        df = self.fetch_market_data()
        if df.empty or len(df) < 60:
            return {
                "status": "數據不足",
                "score": 50,
                "regime": "震盪觀望",
                "summary": "大盤數據取得不足，建議保守觀望。"
            }

        close = df['Close']
        latest_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = (latest_close - prev_close) / prev_close * 100

        # 計算移動平均線
        ma5 = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])

        # 均線排列判斷
        bullish_alignment = (ma5 > ma20) and (ma20 > ma60)
        bearish_alignment = (ma5 < ma20) and (ma20 < ma60)
        above_ma20 = latest_close > ma20
        above_ma60 = latest_close > ma60

        # 20日歷史波動率 (年化)
        returns = close.pct_change().dropna()
        volatility_20d = float(returns.tail(20).std() * np.sqrt(250) * 100)

        # 多空評分系統 (0~100)
        score = 50
        if bullish_alignment:
            score += 25
        elif bearish_alignment:
            score -= 25

        if above_ma20:
            score += 15
        else:
            score -= 15

        if above_ma60:
            score += 10
        else:
            score -= 10

        # 評定市場狀態
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
            "bullish_alignment": bullish_alignment,
            "bearish_alignment": bearish_alignment,
            "above_ma20": above_ma20,
            "above_ma60": above_ma60,
            "volatility_20d": round(volatility_20d, 2),
            "score": score,
            "regime": regime,
            "suggestion": suggestion
        }

if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    res = analyzer.analyze()
    print("=== 台股大盤分析結果 ===")
    for k, v in res.items():
        print(f"{k}: {v}")
