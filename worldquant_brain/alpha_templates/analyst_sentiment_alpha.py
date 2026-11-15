"""
WorldQuant BRAIN 分析師預期與情緒 Alpha 模板庫
適用設定 (Settings)：
- Universe: TOP3000
- Delay: 1
- Neutralization: Subindustry
- Decay: 5 ~ 10
"""

SENTIMENT_ALPHAS = [
    {
        "id": "SENT_ALPHA_01",
        "name": "分析師目標價隱含空間 (Analyst Target Price Upside)",
        "expression": "group_neutralize(rank((target_price - close) / close), subindustry)",
        "intuition": "分析師目標價與現價之折溢價差，經細分行業中性化後，代表專業研究員給予最高上行預期的股票組合。",
        "expected_sharpe": "> 1.45",
        "typical_turnover": "15% ~ 28%",
    },
    {
        "id": "SENT_ALPHA_02",
        "name": "EPS 預期上修動能 (EPS Consensus Revision Momentum)",
        "expression": "ts_decay_linear(rank(eps_consensus_mean - ts_delay(eps_consensus_mean, 20)), 5)",
        "intuition": "當多家投行分析師在近一個月內連續上修 EPS 預期，往往暗示公司訂單超預期或有重大基本面拐點。",
        "expected_sharpe": "> 1.6",
        "typical_turnover": "12% ~ 22%",
    },
]


def print_catalog():
    print("=" * 80)
    print("🧠 WorldQuant BRAIN 分析師與市場情緒 Alpha 庫 (Sentiment & Analyst Library)")
    print("=" * 80)
    for a in SENTIMENT_ALPHAS:
        print(f"\n【{a['id']}】{a['name']}")
        print(f"表達式 (Expression): {a['expression']}")
        print(f"經濟學直覺: {a['intuition']}")
        print(f"預期 Sharpe: {a['expected_sharpe']} | 預期換手率: {a['typical_turnover']}")


if __name__ == "__main__":
    print_catalog()
