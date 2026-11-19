# Neutralization 消融實驗評分表（100 分）

| 維度 | 權重 | 評分原則 |
|---|---:|---|
| Sharpe 保留／改善 | 24 | 與同一 Baseline 比較，不只看絕對最高值 |
| Return 保留 | 18 | 至少保留 70%；低於 25% 標記 Risk-factor Dependency |
| Return-to-Drawdown | 18 | 獎勵回撤下降且收益仍保留 |
| Fitness | 12 | 同時反映 Sharpe、Return 與 Turnover |
| OS 穩定性 | 10 | Test／Semi-OS／年度穩定綜合值 |
| 低 Self-correlation | 8 | 僅使用官方 PnL-based correlation |
| Sub-universe | 5 | 官方通過才得分 |
| Exposure Reduction | 5 | Market beta 等曝險下降 |

## 扣分
- Risk-factor Dependency：-20
- Over-neutralized：-12
- 未驗證的 Double Neutralization：-5
- Missing final neutralization：Hard Block
- Long/Short imbalance >20%：Hard Block
- Unofficial metrics：Hard Block

## 成功門檻
- Sharpe 不得比 Baseline 下降超過 0.10
- Return 保留至少 70%
- Drawdown 不高於 Baseline
- |Market beta| <= 0.30
- Sub-universe PASS
- OS stability >= 0.60

## 推薦實驗矩陣
同一 Expression、Region、Universe、Delay、Decay、Truncation 下比較：
`MARKET / SECTOR / INDUSTRY / SUBINDUSTRY`。若帳號可用，再加入 `SLOW_FACTORS / FAST_FACTORS / SLOW_FAST_FACTORS`。

不得跨不同 Baseline 混合評分，也不得只選 Sharpe 最高者。
