import os
from datetime import datetime
from taiwan_stock.market_analyzer import MarketAnalyzer
from taiwan_stock.stock_screener import StockScreener
from taiwan_stock.ml_ranker import TaiwanStockMLRanker
from taiwan_stock.config import REPORTS_DIR, DEFAULT_WATCHLIST
import yfinance as yf

class ReportGenerator:
    """
    台股每日晨報自動生成器
    整合大盤多空結構、技術突破選股與 ML 多因子預測排序，輸出格式化 Markdown 報告與文字摘要
    """
    def __init__(self):
        self.market_analyzer = MarketAnalyzer()
        self.stock_screener = StockScreener()
        self.ml_ranker = TaiwanStockMLRanker(model_type="lightgbm")

    def generate(self) -> str:
        today_str = datetime.now().strftime("%Y-%m-%d")
        now_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("[*] 正在進行大盤多空診斷...")
        market_info = self.market_analyzer.analyze()

        print("[*] 正在掃描潛力強勢標的...")
        picks = self.stock_screener.run_screener()

        report_lines = [
            f"# 📊 台股早盤量化決策晨報 ({today_str})",
            f"*生成時間: {now_time_str} | 系統: Antigravity Automated Quant Agent*",
            "",
            "---",
            "## 🧭 1. 大盤環境與多空溫度診斷",
            f"- **加權指數最新收盤**: `{market_info.get('latest_close', 'N/A')}` ({market_info.get('change_pct', 0):+0.2f}%)",
            f"- **關鍵均線狀況**: MA5=`{market_info.get('ma5')}` | MA20(月線)=`{market_info.get('ma20')}` | MA60(季線)=`{market_info.get('ma60')}`",
            f"- **多空評分**: **{market_info.get('score', 50)} / 100**",
            f"- **市場狀態**: **【{market_info.get('regime', '觀望')}】**",
            f"- **20日年化波動率**: `{market_info.get('volatility_20d', 0)}%`",
            f"- **操作方針**: > {market_info.get('suggestion', '無')}",
            "",
            "---",
            "## 🎯 2. 今日主動精選潛力投資標的",
        ]

        if not picks:
            report_lines.append("> ⚠️ 今日篩選池中無標的同時滿足嚴格突破條件，建議持盈保泰、等待更好風險報酬比之時機。")
        else:
            report_lines.append("| 股票代碼 | 收盤價 | 漲跌幅 | 量化評分 | 觸發技術信號 | 建議防守停損 | 目標價位 | 風險報酬比 |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for p in picks:
                sigs = "<br>".join(p['signals'])
                report_lines.append(
                    f"| **{p['symbol']}** | {p['close']} | {p['change_pct']:+0.2f}% | **{p['score']}** | {sigs} | {p['stop_loss']} | {p['target_price']} | {p['risk_reward_ratio']} |"
                )

        # 區塊 3: 機器學習多因子 Alpha 排名
        print("[*] 正在運行 ML 多因子預測排序模型...")
        try:
            stock_data_map = {}
            for sym in DEFAULT_WATCHLIST[:8]:
                t = yf.Ticker(sym)
                h = t.history(period="1y")
                if len(h) >= 80:
                    stock_data_map[sym] = h

            if stock_data_map:
                clean_ds, _ = self.ml_ranker.prepare_dataset(stock_data_map)
                self.ml_ranker.train_model(clean_ds)
                rank_df = self.ml_ranker.predict_rankings(stock_data_map)

                report_lines.extend([
                    "",
                    "---",
                    "## 🤖 3. AI / ML 多因子未來 5 日超額報酬排序榜",
                    "| 排名 | 標的 | 最新收盤 | 預測 5 日報酬 (%) | 月線乖離率 (%) | RSI(14) |",
                    "| :---: | :--- | :--- | :---: | :---: | :---: |"
                ])
                for _, row in rank_df.head(5).iterrows():
                    report_lines.append(
                        f"| #{int(row['rank'])} | **{row['symbol']}** | {row['current_price']} | **{row['predicted_fwd_5d_ret_pct']:+0.2f}%** | {row['bias_ma20']:+0.2f}% | {row['rsi_14']} |"
                    )
        except Exception as e:
            report_lines.append(f"\n> ℹ️ ML 多因子預測模型運行略過: {str(e)}")

        report_lines.extend([
            "",
            "---",
            "## 🛡️ 4. 紀律與風控指引",
            "1. **大盤濾網原則**：若大盤評分低於 40 分，嚴格控制總多單部位於 30% 以下，切勿重倉追高。",
            "2. **單筆風險限制**：任何單一標的承擔之損失金額不可超過總投資部位之 1.5%~2.0%。",
            "3. **嚴格執行停損**：跌破上述「建議防守停損」無條件離場，保本永遠是獲利的第一前提。",
            "",
            "---",
            "*免責聲明：本報告由量化統計模型自動生成，所提供之數據與資訊僅供學術研究與投資策略參考，不構成任何具體投資邀約或保證獲利承諾。*"
        ])

        report_content = "\n".join(report_lines)

        # 寫入歷史報告存檔
        report_filename = f"morning_report_{today_str}.md"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"[✓] 晨報已成功生成並存檔至: {report_path}")
        return report_content

if __name__ == "__main__":
    generator = ReportGenerator()
    content = generator.generate()
    print("\n" + "="*50)
    print(content)
