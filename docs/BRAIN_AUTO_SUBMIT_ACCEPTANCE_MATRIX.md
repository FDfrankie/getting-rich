# WorldQuant BRAIN 受控自動提交與合規驗收矩陣 (BRAIN Auto-Submit Acceptance Matrix)

本矩陣嚴格遵循 WorldQuant BRAIN 官方規範及全系統安全鎖定標準，詳細列出受控自動提交與因子合規條件驗收狀態。

---

## 核心安全準則 (Core Safety Constraints)
- **預設安全鎖**：`BRAIN_SIMULATION_ENABLED=false`、`BRAIN_AUTO_SUBMIT_ENABLED=false`。
- **嚴禁批量補提交**：測試與冷啟動期間不得自動執行 Production Submit，禁止批量補提交歷史舊 Alpha。
- **永久冪等性**：基於表達式正規化與 SHA-256 雜湊，確保同一因子永不重複送測或提交。

---

## 驗收矩陣總表 (Acceptance Matrix)

| 編號 | 驗收條件 (Condition) | 關聯模組 / 測試 | 實施與驗證結果 | 狀態 (Status) |
| :--- | :--- | :--- | :--- | :---: |
| 01 | 異步送測隊列架構 (Async Job Queue) | `worldquant_brain/brain_job_queue.py` | 實作非阻塞任務隊列，背景執行輪詢與狀態更新 | `PASS` |
| 02 | 運算式 SHA-256 永久防重 (Idempotency Key) | `worldquant_brain/brain_job_queue.py` | `compute_simulation_key` 確保相同表達式返回同一 Job ID | `PASS` |
| 03 | 嚴格門檻審查 (Hard Gate: Sharpe & Fitness) | `worldquant_brain/alpha_vault.py` | Sharpe >= 1.25, Fitness >= 1.0, Margin > 10 bps | `PASS` |
| 04 | 換手率區間審查 (Turnover Guard 1%~70%) | `worldquant_brain/alpha_vault.py` | 修正 >1.0 誤除 100 漏洞，超出 0.70 嚴格判定不合格 | `PASS` |
| 05 | 相關性過濾 (Self-Correlation < 0.70) | `worldquant_brain/alpha_vault.py` | 向量化相關性檢驗，杜絕高重疊因子 | `PASS` |
| 06 | HTTP 429 退避與速率限制 (Rate Limit Backoff) | `worldquant_brain/brain_job_queue.py` | 遵循 `Retry-After` 標頭自動退避重試 | `PASS` |
| 07 | Mock 隔離與零連外測試 (Zero External Leak in Tests) | `tests/test_brain_job_queue.py` | 全套測試 100% Mock，無任何對外真實請求 | `PASS` |
| 08 | 雙向隊列對齊 JSON API 契約 | `tests/test_brain_web_api_json_contract.py` | 前後端 Web API 響應結構嚴格一致 | `PASS` |
