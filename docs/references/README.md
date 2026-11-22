# 專案量化研究與工程參考知識庫索引

本目錄記錄 WorldQuant BRAIN 官方規則與跨市場量化工程準則（最新完整雲端版同步存放於 Google Drive `炒幣/`）：

- **WorldQuant BRAIN 官方指南與 Alpha 研究知識庫（Google Doc）**: `1audaVbf9TWFb2imkb3NIo8c9fTjcvc0FlH4UJTwOJac`
- **機器可驗證官方知識庫 JSON (v6)**: `worldquant_brain_official_kb_v6.json`（涵蓋 137 篇 Help Center 文章、Simulate/Alphas 介面實測、66 個 Bronze 已解鎖運算子白名單、Learn 課程與進階文件、Community 論壇 22 篇高價值研究貼文）
- **Bronze 帳號實測限制（`FJ10089`）**:
  - Neutralization 僅開放 `NONE`、`MARKET`、`SECTOR`、`INDUSTRY`、`SUBINDUSTRY`（Consultant-only 中性化預設由 `BRAIN_CONSULTANT_MODE=false` 阻擋）
  - 運算子白名單為 66 個基礎運算子；避開已停用欄位 `opt6_nexterntod`、`opt6_lasterntod`
  - 防過擬合紅線：Test Period 80-20 下降 > 50%、Rank/Binary 測試下降 > 70%
