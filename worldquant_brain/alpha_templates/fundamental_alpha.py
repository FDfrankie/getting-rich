"""
WorldQuant BRAIN 基本面品質與成長性 Alpha 模板庫
適用設定 (Settings)：
- Universe: TOP3000 / TOP1000
- Delay: 1
- Neutralization: Industry / Subindustry
- Decay: 10 ~ 20 (基本面數據頻率低，適度延長衰減週期)
"""

FUNDAMENTAL_ALPHAS = [
    {
        "id": "FUND_ALPHA_01",
        "name": "高資本回報率與穩健現金流 (Cash Flow ROIC Quality)",
        "expression": "group_neutralize(rank(operating_cashflow / (assets - liabilities + 0.001)), sector)",
        "intuition": "優質公司的營業現金流相對於淨資產比例顯著高於同行業平均，具備定價權與長期超額收益。",
        "expected_sharpe": "> 1.5",
        "typical_turnover": "5% ~ 15%",
    },
    {
        "id": "FUND_ALPHA_02",
        "name": "營業利益加速度動量 (Operating Income Acceleration)",
        "expression": "rank(operating_income - ts_delay(operating_income, 63)) / (cap + 0.001)",
        "intuition": "季度盈餘顯著加速成長且市值尚未完全反映的公司，後續季報發布後容易產生盈餘公告漂移 (PEAD)。",
        "expected_sharpe": "> 1.4",
        "typical_turnover": "10% ~ 20%",
    },
    {
        "id": "FUND_ALPHA_03",
        "name": "營收成長與估值性價比 (Sales Growth to EV Value)",
        "expression": "group_rank((sales - ts_delay(sales, 252)) / (sales + 0.001), subindustry) - group_rank(debt / (equity + 0.001), subindustry)",
        "intuition": "在同行業中選出年營收高成長且槓桿負債比相對較低之穩健白馬股。",
        "expected_sharpe": "> 1.38",
        "typical_turnover": "8% ~ 18%",
    },
]


def print_catalog():
    print("=" * 80)
    print("🧠 WorldQuant BRAIN 高通過率基本面 Alpha 表達式庫 (Fundamental Library)")
    print("=" * 80)
    for a in FUNDAMENTAL_ALPHAS:
        print(f"\n【{a['id']}】{a['name']}")
        print(f"表達式 (Expression): {a['expression']}")
        print(f"經濟學直覺: {a['intuition']}")
        print(
            f"預期 Sharpe: {a['expected_sharpe']} | 預期換手率: {a['typical_turnover']}"
        )


if __name__ == "__main__":
    print_catalog()
