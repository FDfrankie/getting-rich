# 👤 使用者實務操作指南 (User Action Guide & Workflow Checklist)

本專案已實現高階量化工程與多市場串接。為了協助您最高效率運用本系統，以下列出明確的**使用者操作清單**，清楚劃分「全自動執行項目」與「需要您手動介入操作的項目」。

---

## 📅 一、例行週期操作清單 (Routine Checklist)

### 1. 每日例行操作 (Daily - 建議花費 2-3 分鐘)
- [ ] **查看台股 08:00 晨報**：
  - 開啟 Web 儀表板，點擊「台股大盤決策」分頁。
  - 檢視今日大盤加權指數多空趨勢、外資期貨淨未平倉口數，以及多因子選股模型（動量、均線乖離率、成交量周轉率）推薦的前 5 檔優質標的。
- [ ] **檢查加密貨幣機器人狀態**：
  - 在「加密自動操盤」分頁確認機器人即時淨值曲線與未實現盈虧。
  - 若正在運行策略（動量突破 / 機器學習預測 / PPO 強化學習），確認策略發出最新信號是否與預期一致。

### 2. 每週 / 每月定期維護 (Weekly / Monthly - 建議花費 10 分鐘)
- [ ] **Kaggle 雲端免費 GPU 一鍵更新模型權重**：
  - 本專案已將完整的模型訓練管線打包至 `kaggle_notebooks/kaggle_crypto_ml_rl_training.ipynb`。
  - **步驟**：
    1. 前往 Kaggle 開啟該 Notebook。
    2. 開啟免費 T4 GPU 加速，點擊 **"Run All"**。
    3. 筆記本將自動下載歷史高頻 K 線、計算多因子特徵、呼叫 `SoftSharpeLoss` / `SoftRankICLoss` 進行訓練，並輸出最新權重檔：
       - `crypto_ml_model.joblib`
       - `crypto_ppo_agent.pt`
       - `taiwan_stock_ranker.joblib`
    4. 下載新產生的權重檔，放置於專案對應目錄中，重啟服務即可完成線上無縫升級！

---

## 🔑 二、由模擬盤切換為實盤真金白銀操作 (Real-Money Execution)

目前系統預設啟動**雙軌沙盒模擬盤 (PAPER_TRADING = True)**，不會扣除您的真實資產。當您評估回測與模擬成效滿意，欲切換為實盤時，請遵循以下步驟：

### 1. 幣安 (Binance) 永續合約實盤金鑰配置
1. 登入 [Binance 官方網站](https://www.binance.com)，進入「API 管理」。
2. 建立一組全新 API Key，並設定以下安全權限：
   - 勾選 **「讀取」** 與 **「啟用合約交易」**。
   - **切勿勾選** 「啟用提現 (Withdrawal)」！確保資金 100% 安全。
   - 建議綁定您主機或雲端伺服器的固定 IP。
3. 在專案根目錄建立或修改 `.env` 檔案（或於 Render 環境變數中設定）：
   ```env
   BINANCE_API_KEY=你的幣安實盤金鑰
   BINANCE_API_SECRET=你的幣安實盤私鑰
   PAPER_TRADING=False
   ```
4. 儲存後重啟 Web UI 或機器人，系統將以真實資金開始依訊號下單，並自動套用 ATR 風險管理與動態止損！

---

## 🌐 三、WorldQuant BRAIN 顧問資格申請與量化因子提交 (WorldQuant BRAIN)

本專案的 `worldquant_brain/` 模組內建了全自動表達式解析器與高達 8 檔通過官方 2026 最新指標標準（Sharpe > 1.25, Turnover < 70%, Margin > 0.0015）的驗證因子庫。

### 使用者操作步驟：
1. **申請帳號與顧問資格**：
   - 前往 [WorldQuant BRAIN 官方平台](https://platform.worldquantbrain.com) 註冊研究者帳號。
   - 累積通過的 Alpha 因子積分，即可申請簽約成為官方給薪研究顧問 (Research Consultant)。
2. **獲取 API 憑證**：
   - 在個人個人檔案設定中取得認證 Token 或使用登入帳號密碼。
   - 配置至 `.env`：
     ```env
     BRAIN_USERNAME=your_email@example.com
     BRAIN_PASSWORD=your_brain_password
     ```
3. **在 Web 儀表板一鍵批次模擬與提交**：
   - 切換至「WorldQuant BRAIN」分頁。
   - 點擊「載入驗證因子庫」，可直接在瀏覽器上查看各因子的回測指標雷達圖。
   - 點擊「一鍵提交驗證通過之 Alpha」，系統將自動調用官方 API 進行排隊模擬與正式提交！

---

## 🎯 四、自動化與手動責任對照表 (Automation Matrix)

| 功能模組 | 系統全自動完成的部分 | 需要您手動操作的部分 |
| :--- | :--- | :--- |
| **台股大盤與晨報** | 每日定時爬取 TWSE、計算技術與外資期貨多空、生成視覺化 HTML 晨報 | 每日開盤前花 2 分鐘閱讀晨報重點 |
| **加密貨幣交易** | 24/7 收集 K 線、計算特徵、推斷部位、訂單成交模擬、移動止損風控 | 切換實盤時填入 API Key；定期查看損益 |
| **AI 模型訓練** | 提供完整的 Walk-Forward 分割、可微損失函數、RL PPO 框架與 Kaggle 腳本 | 在 Kaggle 點擊 "Run All" 並下載最新權重 (每週/月一次) |
| **WorldQuant BRAIN** | 因子語法構建、去相關性中性化檢查、驗證因子庫存儲 | 申請 BRAIN 帳號與填入登入憑證 |
| **雲端部署維護** | Render / Docker 自動持續運作、健康檢查與崩潰重啟 | 若修改代碼時執行 `git push origin main` |
