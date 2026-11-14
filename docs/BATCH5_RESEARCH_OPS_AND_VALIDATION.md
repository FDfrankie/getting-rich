# BATCH 5: 頂尖量化研究架構、防未來函數嚴格審計與驗證 (Research Ops & Validation)

## 一、核心目標與任務清單
1. **嚴格防未來函數檢驗 (Strict No Look-Ahead Bias Audit)**：
   - 執行 	ests/test_no_lookahead_audit.py，全方位驗證：
     1. 特徵計算僅依賴歷史與當期 Bar，追加未來 K 線時已產生之特徵數值 100% 保持恆定。
     2. 台股機器學習 Ranker 導入嚴格 5 交易日 Embargo (隔離期)，徹底防止未來資訊滲漏。
     3. 波動率與夏普比率嚴格套用動態年化常數。
2. **可微損失函數與深度學習架構落地**：
   - 實現並驗證 crypto_trading/loss_functions.py 之 SoftSharpeLoss 與 SoftRankICLoss，將夏普比率直接作為神經網絡梯度優化目標。
3. **學術文獻與量化工程知識庫整合**：
   - 整合 arXiv:2607.09426 (RL 做市與資金費率對沖)、arXiv:2510.07943 (梯度提升樹直接優化夏普比率)、arXiv:2604.15973 (訂單流不平衡 L2 微觀特徵) 等理論至系統架構。
4. **自動化單元測試全數驗收 (100% Pytest Pass)**：
   - 執行包含架構升級、多空撮合、風控金鑰保險箱、機器學習訓練等 28 項 pytest 測試，確保系統無任何 Regression。
