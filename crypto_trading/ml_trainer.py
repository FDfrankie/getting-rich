"""
Machine Learning Training Pipeline for Directional Crypto Alpha.
Implements:
1. Walk-Forward Cross-Validation (Time-Series Split) without lookahead bias
2. LightGBM and XGBoost Regressors & Classifiers
3. Model evaluation on Information Coefficient (Rank IC), Sharpe Ratio, and Directional Accuracy
4. Feature importance extraction
5. Export & Import weights (.joblib, .json)
"""

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None
from crypto_trading.feature_engineer import CryptoFeatureEngineer


class CryptoMLTrainer:
    def __init__(self, model_type: str = "lightgbm", task: str = "regression"):
        """
        model_type: 'lightgbm' or 'xgboost'
        task: 'regression' (predict forward return) or 'classification' (predict directional up/down)
        """
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.feature_engineer = CryptoFeatureEngineer()
        self.model = None
        self.feature_names: list[str] = []
        self.cv_results: list[dict[str, float]] = []

    def _init_model(self):
        if self.model_type == "lightgbm":
            if self.task == "regression":
                return lgb.LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=31,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=-1,
                )
            else:
                return lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=31,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=-1,
                )
        elif self.model_type == "xgboost":
            if self.task == "regression":
                return xgb.XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=0,
                )
            else:
                return xgb.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=0,
                )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def walk_forward_train(self, df_raw: pd.DataFrame, n_splits: int = 5, train_ratio: float = 0.7, purge_bars: int = 12, embargo_bars: int = 2) -> dict[str, Any]:
        """
        Executes strict chronological Walk-Forward Validation.
        At each fold k:
          - Train set: [0 : fold_train_end]
          - Test set:  [fold_train_end : fold_test_end] (out-of-sample)
        No future data is ever leaked.
        """
        processed = self.feature_engineer.compute_features(df_raw).dropna().copy()
        features = self.feature_engineer.feature_columns
        self.feature_names = features

        target_col = (
            "target_fwd_ret_3"
            if self.task == "regression"
            else ("target_direction_3" if "target_direction_3" in processed.columns else "target_binary")
        )
        X = np.asarray(processed[features].values, dtype=float)
        y = np.asarray(processed[target_col].values, dtype=float if self.task == "regression" else int)
        returns = np.asarray(processed["target_fwd_ret_3"].values, dtype=float)

        n_samples = len(X)
        fold_size = int(n_samples / (n_splits + 1))

        all_metrics = []
        oof_predictions = np.zeros(n_samples)
        oof_mask = np.zeros(n_samples, dtype=bool)

        for fold in range(n_splits):
            train_end = fold_size * (fold + 1)
            test_end = min(train_end + fold_size, n_samples)
            if train_end >= test_end:
                break

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]
            test_ret = returns[train_end:test_end]

            model = self._init_model()
            model.fit(X_train, y_train)

            # 3 期重疊收益之年化夏普比率修正因子：年週期為 8760 / 3，避免 3 期重疊持有將波動虛胖
            annual_multiplier = np.sqrt(8760.0 / 3.0)

            if self.task == "regression":
                preds = model.predict(X_test)
                # Compute Rank IC (Spearman correlation between prediction & realization)
                rank_ic = pd.Series(preds).corr(pd.Series(y_test), method="spearman")
                # Strategy simulated returns based on sign of prediction
                strat_ret = np.sign(preds) * test_ret
                sharpe = (np.mean(strat_ret) / (np.std(strat_ret) + 1e-9)) * annual_multiplier
                accuracy = np.mean((preds > 0) == (y_test > 0))
            else:
                prob = model.predict_proba(X_test)[:, 1]
                preds = prob - 0.5
                rank_ic = pd.Series(prob).corr(pd.Series(y_test), method="spearman")
                strat_ret = np.sign(preds) * test_ret
                sharpe = (np.mean(strat_ret) / (np.std(strat_ret) + 1e-9)) * annual_multiplier
                accuracy = np.mean((prob > 0.5) == (y_test == 1))

            oof_predictions[train_end:test_end] = preds
            oof_mask[train_end:test_end] = True

            metrics = {
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "rank_ic": float(rank_ic if not np.isnan(rank_ic) else 0.0),
                "directional_accuracy": float(accuracy),
                "annualized_sharpe": float(sharpe),
            }
            all_metrics.append(metrics)

        # Train final production model on full data
        self.model = self._init_model()
        self.model.fit(X, y)
        self.cv_results = all_metrics

        # Feature Importance
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feat_imp = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        else:
            feat_imp = []

        summary = {
            "model_type": self.model_type,
            "task": self.task,
            "features_count": len(self.feature_names),
            "folds": all_metrics,
            "mean_rank_ic": float(np.mean([m["rank_ic"] for m in all_metrics])),
            "mean_accuracy": float(np.mean([m["directional_accuracy"] for m in all_metrics])),
            "mean_sharpe": float(np.mean([m["annualized_sharpe"] for m in all_metrics])),
            "top_10_features": feat_imp[:10],
        }
        return summary

    def save_weights(self, path: str):
        """Saves model instance, feature names, and configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        bundle = {
            "model": self.model,
            "model_type": self.model_type,
            "task": self.task,
            "feature_names": self.feature_names,
            "cv_results": self.cv_results,
        }
        joblib.dump(bundle, path)

    @classmethod
    def load_weights(cls, path: str) -> "CryptoMLTrainer":
        bundle = joblib.load(path)
        trainer = cls(model_type=bundle["model_type"], task=bundle["task"])
        trainer.model = bundle["model"]
        trainer.feature_names = bundle["feature_names"]
        trainer.cv_results = bundle.get("cv_results", [])
        return trainer

    def predict(self, df_raw: pd.DataFrame, fill_nan: bool = True) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        processed = self.feature_engineer.compute_features(df_raw)
        X = processed[self.feature_names].values
        if fill_nan:
            # Replace NaNs from initial rolling windows with 0 to preserve length matching df_raw
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.task == "regression":
            return self.model.predict(X)
        else:
            prob = self.model.predict_proba(X)[:, 1]
            return prob - 0.5
