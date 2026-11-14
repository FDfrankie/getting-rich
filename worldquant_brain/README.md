# 🧠 WorldQuant BRAIN 量化研發與獲利全指南 (Consultant Model)

WorldQuant BRAIN 是全球頂級量化對沖基金 **WorldQuant (世坤投資)** 推出的全球量化研究者平台。透過在該平台上構建、回測並提交預測股票回報的量化因子（稱為 **Alpha**），你可以直接參與基金獲利分成並獲得高額美金報酬。

---

## 💰 一、WorldQuant BRAIN 究竟如何賺錢？（獲利機制拆解）

WorldQuant BRAIN 的賺錢途徑主要分為四大管道：

### 1. 簽約成為量化研究顧問 (Research Consultant) — **核心穩定獲利來源**
- **晉升階梯**：
  1. **Webinar / Challenger**: 初學者階段，透過回測累積積分。
  2. **Expert**: 通過基本門檻，解鎖更多高階數據集（Fundamental, Model, Sentiment）。
  3. **Master**: 達到指定積分（如 10,000+ 分）且具備多檔優質 Alpha。
  4. **Research Consultant**: 收到官方簽約邀請，正式成為特聘量化顧問。
- **獲利方式**：
  - **季度分潤獎金 (Quarterly Payouts)**：你提交並通過審核的 Alpha 會被納入 WorldQuant 基金的實盤投資組合（Portfolio）。基金依據你的 Alpha 在實盤中的 Sharpe 比率、收益貢獻度與相關性，每季向你發放美金分潤。
  - **簽約/績效津貼 (Stipend / Milestone Bonus)**：每月提交指定數量達標 Alpha，即可領取固定達標獎金。

### 2. 全球量化挑戰賽 (Global Alphathon / Competitions)
- WorldQuant 定期舉辦地區與全球性量化大賽（如 International Quant Championship）。
- 前三名與優勝隊伍直接瓜分數萬至數十萬美金的現金獎金池，並直接獲得全職 Quant Researcher / Trader 的面試或 Offer。

### 3. 一次性 Alpha 採購與特殊獎勵活動
- 官方經常推出「指定數據集專案」（例如指定開發 ESG、供應鏈或分析師預期數據）。符合要求的優質 Alpha 每檔可額外獲得特定點數或單檔現金獎金。

---

## 🎯 二、通過審核並能賺錢的 Alpha 標準 (Simulation Criteria)

若要讓 Alpha 具備變現價值並通過 WorldQuant 的機器人與人工審查，必須滿足下列嚴格標準：

| 指標 | 要求標準 | 說明與調優技巧 |
| :--- | :--- | :--- |
| **Sharpe Ratio** | **> 1.25** (建議 > 1.58) | 衡量風險調整後收益，曲線需平滑穩定，不可依賴單一牛市年份 |
| **Turnover (換手率)** | **1% ~ 70%** (最佳在 20%~40%) | 低於 1% 缺乏信號動態；高於 70% 扣除交易手續費後將無法在實盤獲利 |
| **Fitness** | **> 1.0** | 綜合報酬、夏普與換手率的指標：$\text{Fitness} = \text{Sharpe} \times \sqrt{\frac{\text{Returns}}{\text{Turnover}}}$ |
| **Margin** | **> 10 bps** (0.10%) | 每筆交易的平均利潤空間，確保能覆蓋實盤市場衝擊成本 |
| **Sub-universe 檢驗** | 所有子池皆表現良好 | 在 Top 3000、Top 1000、Top 500 流動性池中不得出現斷崖式下跌（防過擬合）|
| **Correlation (相關性)**| **< 0.70** | 與平台上已存在的 Alpha 相關度必須低於 0.7，提倡原創經濟學邏輯 |

---

## 🛠️ 三、如何開始您的 WorldQuant BRAIN 之旅

1. **前往官網註冊**：[WorldQuant BRAIN 官方平台](https://platform.worldquantbrain.com/)。
2. **研讀本專案提供的工具庫**：
   - 參閱 [operators_cheatsheet.md](operators_cheatsheet.md)：掌握 BRAIN 語法。
   - 參閱 [data_fields_guide.md](data_fields_guide.md)：了解量價、財務報表與另類數據。
   - 參閱 [alpha_templates/](alpha_templates/)：直接複製經過驗證的 20+ 組高分 Alpha 表達式。
3. **回測與提交**：
   - 在 BRAIN 平台 Simulation 介面貼上表達式，設定 Settings（Region: US, Universe: TOP3000, Delay: 1, Decay: 4, Neutralization: Subindustry）。
   - 檢視回測報告，確認各項數值達標後點擊 **Submit** 提交。
