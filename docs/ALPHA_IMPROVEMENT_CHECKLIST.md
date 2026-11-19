# Alpha 改进检查清单

## 1. 官方身份与硬门槛
- [ ] Metrics 来源为 `WORLDQUANT_BRAIN`
- [ ] `official_simulation_id` 与 `brain_alpha_id` 存在且分开
- [ ] D0 Sharpe 严格 `>2.0`；D1 严格 `>1.25`
- [ ] Fitness 严格 `>1.0`
- [ ] `0.01 < turnover < 0.70`
- [ ] Self-correlation 严格 `<0.70`
- [ ] Weight、Bias、Sub-universe 官方检查通过
- [ ] 非公开公式、非重复、非参数微调冒充新假说
- [ ] Test Period 未被用于反复调参

## 2. PnL 跳变诊断
- [ ] 检查 NaN/非 NaN 切换；证据来源已标记
- [ ] 检查 Alpha 值快速变化；不得在无 Vector 时伪造精确指标
- [ ] 检查最大单股权重是否 `<10%`
- [ ] 检查 Truncation，优先测试 `0.05–0.10`
- [ ] 检查 Market、Sector、Industry、Subindustry 曝险
- [ ] 检查单一年份绝对 PnL 是否 `>40%`
- [ ] 检查事件集中、Regime 依赖和早期年份最大回撤

## 3. Backfill
- [ ] Backfill 为因果且不使用未来资料
- [ ] 记录原始 Coverage、Backfilled Coverage、Window、Staleness
- [ ] News、Analyst、Event Field 没有被长窗口 Backfill 变成陈旧信号
- [ ] 完成有／无 Backfill 消融
- [ ] Point-in-time 状态已验证

## 4. Turnover 与交易成本
- [ ] `<12.5%` 时不再因继续降低 Turnover 自动加分
- [ ] `12.5–30%` 为成本偏好区
- [ ] `30–40%` 可接受但需关注 Margin
- [ ] `40–70%` 必须做 1x／1.5x／2x Cost Stress
- [ ] 高换手先判断 Productive 或 Wasteful
- [ ] Productive Turnover 不机械压低
- [ ] Wasteful Turnover 才测试 hump、trade_when、提高 Decay

## 5. 信号纯度与过拟合
- [ ] 一个主要经济假说
- [ ] 最多一个确认信号
- [ ] 最多一个风险控制信号
- [ ] 默认 Conditions 不超过 3 个
- [ ] 多个独立 Signal 拆为原子 Alpha，在 Portfolio／SuperAlpha 层组合
- [ ] 没有无意义常数、静态行业翻号或只改 Window 的 IS 微调

## 6. 风险与平滑动作
- [ ] NaN 跳变才优先测试 causal backfill
- [ ] Alpha 变化过快才测试 Decay／平均
- [ ] 权重集中才测试 Rank／Winsorize／Truncation
- [ ] 市场曝险才测试 Neutralization／Volatility Scaling
- [ ] Neutralization 后 Return 消失时标记 Risk-factor Dependency
- [ ] 不把平滑后的漂亮 PnL 误当成预测力提升

## 7. 稳健性与 OS
- [ ] Clean Test Period
- [ ] Positive-year ratio `>=60%`
- [ ] 单一年份绝对 PnL `<=40%`
- [ ] 参数、Universe、Regime、Cost Stress 完成
- [ ] DSR、PBO、CPCV 或等价多重测试检查完成
- [ ] OS 稳健性优先级高于 IS 单点 Sharpe

## 8. IQC 组合价值
- [ ] 官方 Before／After Submission 或内部 Merged Analysis 完成
- [ ] Incremental Merged Sharpe 为正
- [ ] Incremental Return-to-Drawdown 为正
- [ ] Merged Turnover 没有无价值恶化
- [ ] Idea、Dataset、Delay、Universe 增加实际多样性
- [ ] 单条 GOOD 但 Merged Score 下降时默认不提交

## 9. Research Memory
- [ ] 保存 Parent／Child、Diagnosis、Action、Before／After Metrics
- [ ] Evidence 消费冪等
- [ ] 保存成功与失败修复
- [ ] 新候选生成前调用 `recommend_fixes()`
- [ ] 不同 Family／Domain／Region／Universe／Delay 不错误共享经验

## 10. 提交
- [ ] 统一 PreSubmissionGate 通过
- [ ] IQC Candidate Score 与 Timing Score 达标
- [ ] 今日最多 1–3 条；同 Idea Family 默认最多一条
- [ ] Server-side Auto-submit Policy、Quota、Kill Switch 通过
- [ ] 可提交不等于应该提交
