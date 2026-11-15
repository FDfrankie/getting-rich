"""
Cryptocurrency ML Strategy Integration.
Connects trained LightGBM / XGBoost directional predictors into the BacktestEngine and BotRunner.
Signal Output:
  1: Long
 -1: Short
  0: Flat / Neutral
"""

import numpy as np
import pandas as pd
from typing import Optional
from crypto_trading.strategies.momentum_breakout import BaseStrategy
from crypto_trading.ml_trainer import CryptoMLTrainer
from crypto_trading.feature_engineer import CryptoFeatureEngineer

class MLAlphaStrategy(BaseStrategy):
    """
    機器學習預測 Alpha 策略
    - 透過提取多尺度動量、波動度估計值、微觀結構特徵
    - 結合 Walk-forward 訓練的 LightGBM / XGBoost 模型預測未來超額收益
    - 設立動態信心閥值 (Confidence Hurdle) 濾除低勝率噪聲信號
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        trainer: Optional[CryptoMLTrainer] = None,
        confidence_hurdle: float = 0.0015 # 預測收益率需高於 0.15% (扣除手續費) 才開倉
    ):
        self.confidence_hurdle = confidence_hurdle
        if trainer is not None:
            self.trainer = trainer
        elif model_path is not None:
            self.trainer = CryptoMLTrainer.load_weights(model_path)
        else:
            self.trainer = None

    def fit_on_data(self, df_train: pd.DataFrame, model_type: str = "lightgbm", task: str = "regression"):
        """便利接口：直接在傳入的訓練數據上訓練一個輕量級模型"""
        self.trainer = CryptoMLTrainer(model_type=model_type, task=task)
        summary = self.trainer.walk_forward_train(df_train)
        return summary

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.trainer is None:
            # 未載入合格模型與 manifest 時，回傳 0 (Hold) 並標記 MODEL_UNAVAILABLE，禁止現場 fit_on_data 偷看數據
            df['signal'] = 0.0
            df['ml_prediction'] = 0.0
            df['model_status'] = 'MODEL_UNAVAILABLE'
            return df

        predictions = self.trainer.predict(df)
        
        # 轉換為策略持倉信號 (1: 做多, -1: 做空, 0: 觀望)
        signals = np.zeros(len(df))
        valid_indices = range(len(df) - len(predictions), len(df))

        for idx, pred in zip(valid_indices, predictions):
            if pred > self.confidence_hurdle:
                signals[idx] = 1.0
            elif pred < -self.confidence_hurdle:
                signals[idx] = -1.0
            else:
                signals[idx] = 0.0

        df['signal'] = signals
        df['ml_prediction'] = np.pad(predictions, (len(df) - len(predictions), 0), mode='constant', constant_values=0.0)
        return df
