import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from taiwan_stock.config import DEFAULT_WATCHLIST, PARAMS

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
    def __init__(self, watchlist: Optional[List[str]] = None):
        self.watchlist = watchlist or DEFAULT_WATCHLIST

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算關鍵技術指標"""
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']

        # 移動平均線
        df['MA5'] = close.rolling(5).mean()
        df['MA20'] = close.rolling(20).mean()
        df['MA60'] = close.rolling(60).mean()
        df['VOL_MA5'] = df['Volume'].rolling(5).mean()

        # 布林通道 (Bollinger Bands)
        std20 = close.rolling(20).std()
        df['BB_MID'] = df['MA20']
        df['BB_UPPER'] = df['BB_MID'] + (std20 * PARAMS['bbands_std'])
        df['BB_LOWER'] = df['BB_MID'] - (std20 * PARAMS['bbands_std'])

        # KD 指標 (Stochastic Oscillator 9,3,3)
        low9 = low.rolling(PARAMS['kd_period']).min()
        high9 = high.rolling(PARAMS['kd_period']).max()
        rsv = (close - low9) / (high9 - low9 + 1e-9) * 100
        
        # 簡易計算 EMA 方式的 K 與 D
        k_list = [50.0]
        d_list = [50.0]
        for val in rsv.fillna(50):
            k = (2/3) * k_list[-1] + (1/3) * val
            d = (2/3) * d_list[-1] + (1/3) * k
            k_list.append(k)
            d_list.append(d)
        df['K'] = k_list[1:]
        df['D'] = d_list[1:]

        # ATR 指標 (真實波動區間)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(PARAMS['atr_period']).mean()

        return df

    def screen_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """針對單一股票評估技術突破信號"""
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
            if curr['MA5'] > curr['MA20'] > curr['MA60']:
                signals.append("均線多頭排列 (MA5>MA20>MA60)")
                score += 25

            # 條件 2: 成交量放大
            if curr['Volume'] > curr['VOL_MA5'] * PARAMS['volume_surge_multiplier']:
                vol_ratio = round(curr['Volume'] / (curr['VOL_MA5'] + 1e-9), 2)
                signals.append(f"爆量突破 (成交量為5日均量 {vol_ratio} 倍)")
                score += 25

            # 條件 3: 布林通道突破
            if curr['Close'] >= curr['BB_UPPER'] and prev['Close'] < prev['BB_UPPER']:
                signals.append("帶量突破布林通道上軌")
                score += 20
            elif curr['Close'] > curr['BB_MID']:
                score += 10

            # 條件 4: KD 黃金交叉或強勢高檔鈍化
            if curr['K'] > curr['D'] and prev['K'] <= prev['D']:
                signals.append(f"KD低檔/中檔黃金交叉 (K:{curr['K']:.1f}, D:{curr['D']:.1f})")
                score += 20
            elif curr['K'] > 70 and curr['D'] > 70:
                signals.append(f"KD強勢高檔鈍化 (K:{curr['K']:.1f})")
                score += 15

            # 若綜合評分達到門檻 (>= 50 分)，納入推薦選股池
            if score >= 50:
                atr = curr['ATR'] if not np.isnan(curr['ATR']) else (curr['Close'] * 0.03)
                suggested_stop_loss = round(curr['Close'] - (atr * PARAMS['atr_stop_loss_multiplier']), 2)
                suggested_target = round(curr['Close'] + (atr * PARAMS['atr_stop_loss_multiplier'] * 2), 2)

                return {
                    "symbol": symbol,
                    "close": round(curr['Close'], 2),
                    "change_pct": round((curr['Close'] - prev['Close']) / prev['Close'] * 100, 2),
                    "volume": int(curr['Volume']),
                    "score": score,
                    "signals": signals,
                    "stop_loss": suggested_stop_loss,
                    "target_price": suggested_target,
                    "risk_reward_ratio": "1 : 2.0"
                }
            return None
        except Exception as e:
            print(f"[篩選略過] {symbol} 處理失敗: {e}")
            return None

    def run_screener(self) -> List[Dict[str, Any]]:
        """批次掃描選股池"""
        results = []
        print(f"[*] 開始掃描台股選股池，共 {len(self.watchlist)} 檔標的...")
        for symbol in self.watchlist:
            res = self.screen_stock(symbol)
            if res:
                results.append(res)
        # 依照評分高低排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

if __name__ == "__main__":
    screener = StockScreener()
    candidates = screener.run_screener()
    print(f"\n=== 篩選完成，共選出 {len(candidates)} 檔具備進場潛力標的 ===")
    for c in candidates:
        print(f"[{c['symbol']}] 分數:{c['score']} 現價:{c['close']} ({c['change_pct']}%) 信號: {', '.join(c['signals'])}")
