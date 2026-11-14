import unittest
import numpy as np
import pandas as pd
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_trading.feature_engineer import CryptoFeatureEngineer
from crypto_trading.ml_trainer import CryptoMLTrainer
from crypto_trading.rl_env import CryptoTradingEnv
from crypto_trading.rl_agent import PPOAgent
from crypto_trading.strategies.ml_strategy import MLAlphaStrategy
from crypto_trading.strategies.rl_strategy import RLAlphaStrategy
from crypto_trading.backtest_engine import BacktestEngine
from taiwan_stock.ml_ranker import TaiwanStockMLRanker

def generate_mock_crypto_df(bars: int = 250) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=bars, freq="1h")
    ret = np.random.normal(0.0002, 0.01, size=bars)
    price = 60000.0 * np.exp(np.cumsum(ret))
    high = price * (1.0 + abs(np.random.normal(0, 0.003, size=bars)))
    low = price * (1.0 - abs(np.random.normal(0, 0.003, size=bars)))
    open_p = price * (1.0 + np.random.normal(0, 0.002, size=bars))
    vol = np.random.uniform(100, 1000, size=bars)

    df = pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": price,
        "Volume": vol
    }, index=dates)
    return df

class TestMLRLTradingInfrastructure(unittest.TestCase):
    def setUp(self):
        self.df = generate_mock_crypto_df(300)
        self.temp_dir = "temp_test_models"
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_feature_engineer(self):
        fe = CryptoFeatureEngineer()
        processed = fe.compute_features(self.df)
        self.assertGreater(len(fe.feature_columns), 10)
        self.assertIn("log_ret_1", processed.columns)
        self.assertIn("parkinson_vol_14", processed.columns)
        self.assertIn("rsi_14_norm", processed.columns)
        self.assertIn("target_fwd_ret_3", processed.columns)

    def test_crypto_ml_trainer_walk_forward(self):
        trainer = CryptoMLTrainer(model_type="lightgbm", task="regression")
        summary = trainer.walk_forward_train(self.df, n_splits=3)
        self.assertIn("mean_rank_ic", summary)
        self.assertIn("mean_sharpe", summary)
        self.assertGreater(len(summary["top_10_features"]), 0)

        # Test Save & Load
        model_file = os.path.join(self.temp_dir, "test_lgbm.joblib")
        trainer.save_weights(model_file)
        self.assertTrue(os.path.exists(model_file))

        loaded = CryptoMLTrainer.load_weights(model_file)
        preds = loaded.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_crypto_rl_environment(self):
        fe = CryptoFeatureEngineer()
        df_feats = fe.compute_features(self.df).dropna()
        env = CryptoTradingEnv(df_feats, fe.feature_columns, initial_balance=10000.0)
        
        obs = env.reset()
        self.assertEqual(len(obs), env.state_dim)

        # Execute actions
        for action in [0.5, -0.5, 1.0, 0.0]:
            obs, reward, done, info = env.step(action)
            self.assertIsInstance(reward, float)
            self.assertIn("balance", info)

    def test_crypto_ppo_agent(self):
        state_dim = 20
        agent = PPOAgent(state_dim=state_dim, action_dim=1, device="cpu")
        dummy_state = np.zeros(state_dim, dtype=np.float32)
        
        action, log_prob, value = agent.select_action(dummy_state)
        self.assertGreaterEqual(action, -1.0)
        self.assertLessEqual(action, 1.0)

        # Test save & load
        weight_path = os.path.join(self.temp_dir, "test_ppo.pt")
        agent.save(weight_path)
        self.assertTrue(os.path.exists(weight_path))
        agent.load(weight_path)

    def test_backtest_with_ml_and_rl_strategies(self):
        engine = BacktestEngine(initial_capital=10000.0)

        # 1. ML Strategy
        ml_strat = MLAlphaStrategy()
        df_ml = ml_strat.generate_signals(self.df)
        self.assertIn("signal", df_ml.columns)
        ml_metrics = engine.run(df_ml)
        self.assertIn("sharpe_ratio", ml_metrics)

        # 2. RL Strategy
        rl_strat = RLAlphaStrategy()
        df_rl = rl_strat.generate_signals(self.df)
        self.assertIn("signal", df_rl.columns)
        rl_metrics = engine.run(df_rl)
        self.assertIn("sharpe_ratio", rl_metrics)

    def test_taiwan_stock_ml_ranker(self):
        mock_universe = {
            "2330.TW": generate_mock_crypto_df(350),
            "2454.TW": generate_mock_crypto_df(350)
        }
        ranker = TaiwanStockMLRanker(model_type="lightgbm")
        clean_ds, feat_cols = ranker.prepare_dataset(mock_universe)
        res = ranker.train_model(clean_ds)
        self.assertIn("rank_ic", res)

        rank_df = ranker.predict_rankings(mock_universe)
        self.assertEqual(len(rank_df), 2)
        self.assertIn("predicted_fwd_5d_ret_pct", rank_df.columns)

if __name__ == "__main__":
    unittest.main()
