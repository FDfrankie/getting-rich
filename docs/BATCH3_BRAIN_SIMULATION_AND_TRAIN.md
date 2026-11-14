# BATCH 3: WorldQuant BRAIN 遺傳挖掘與 ML/RL 模型訓練微調 (Brain Simulation & Model Training)

## 一、核心目標與任務清單
1. **ContinuousAlphaMiner 24/7 背景自主演算法挖掘**：
   - 基於遺傳演算法 (Genetic Algorithm) 與運算元語法樹，自動生成並變異跨市場 Alpha 公式。
   - 嚴格結合歷史 K 線執行事後樣本外 (OOS) 模擬檢驗，唯有通過 Sharpe >= 1.25, Fitness >= 1.0, Turnover <= 70% 門檻方得納入合格庫存。
2. **WorldQuant BRAIN 認證標準對齊與相關性過濾**：
   - 實作向量化相關係數矩陣，新挖掘之因子與已存在因子庫進行 Pearson / Spearman 相關性審查，確保 Self-Correlation < 0.70。
3. **加密貨幣 LightGBM 與 PyTorch PPO 模型真實訓練管線**：
   - 特徵工程集成多週期技術指標、波動率偏態、截面相關性（如 ETH_corr_BTC_24, SOL_to_ETH_ratio）。
   - 採用 Walk-Forward Rolling 訓練與可微夏普損失函數 (SoftSharpeLoss, SoftRankICLoss)，產出實體權重 crypto_lgbm_weights.joblib 與 crypto_ppo_agent.pt。
4. **模型熱加載與策略切換 (Strategy Dynamic Hot-Reload)**：
   - 操盤引擎支援在不中斷後台守護進程的情況下，動態切換策略並自動加載最新訓練之 ML / RL 權重。
