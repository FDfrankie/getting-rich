import os
from datetime import datetime
from taiwan_stock.market_analyzer import MarketAnalyzer
from taiwan_stock.stock_screener import StockScreener
from taiwan_stock.ml_ranker import TaiwanStockMLRanker
from taiwan_stock.config import REPORTS_DIR, DEFAULT_WATCHLIST
from data_persistence.gdrive_sync import GDrivePersistenceSync
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
        self.sync = GDrivePersistenceSync()

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

        # 區塊 4: 每日覆盤與虧損檢討 (Post-Mortem Analysis)
        post_mortems = self.sync.get_recent_post_mortems(limit=5)
        report_lines.extend([
            "",
            "---",
            "## 🔍 4. 每日自動化覆盤與虧損歸因檢討 (Daily Post-Mortem Loss Analysis)"
        ])
        if not post_mortems:
            report_lines.append("> ✅ 最近期無觸發停損或重大虧損事件，系統執行紀律率 100%。")
        else:
            report_lines.append("| 日期 | 標的 | 進場價 | 出場價 | 虧損金額 | 虧損幅度 | 歸因分析與優化對策 |")
            report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for pm in post_mortems:
                report_lines.append(
                    f"| {pm['date']} | **{pm['symbol']}** | {pm['entry_price']} | {pm['exit_price']} | -${abs(pm['loss_amount']):.2f} | {pm['loss_pct']:+.2f}% | {pm['cause_analysis']} |"
                )

        report_lines.extend([
            "",
            "---",
            "## 🛡️ 5. 紀律與風控指引",
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

    def generate_html_report(self) -> str:
        """
        生成具備精緻視覺排版、色彩標籤、多空燈號與響應式表格的晨報 HTML 視圖
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        now_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        market_info = self.market_analyzer.analyze()
        picks = self.stock_screener.run_screener()
        score = market_info.get('score', 50)
        regime = market_info.get('regime', '震盪整理')
        
        # 燈號顏色邏輯
        if score >= 70:
            badge_class = "bg-rose-500/20 text-rose-400 border border-rose-500/30"
            light_icon = "🟢 多頭強勢格局"
        elif score <= 40:
            badge_class = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
            light_icon = "🔴 空頭防守格局"
        else:
            badge_class = "bg-amber-500/20 text-amber-400 border border-amber-500/30"
            light_icon = "🟡 中性震盪盤整"

        html = f"""
        <div class="space-y-6 text-slate-100 font-sans">
            <!-- 頂部 Header 卡片 -->
            <div class="bg-gradient-to-r from-slate-900 to-indigo-950/80 border border-indigo-500/30 rounded-2xl p-5 shadow-2xl">
                <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
                    <div>
                        <div class="flex items-center space-x-2">
                            <span class="px-2.5 py-0.5 rounded-full text-xs font-bold {badge_class}">{light_icon}</span>
                            <span class="text-xs text-slate-400 font-mono">📅 {today_str} 早盤 08:00 發布</span>
                        </div>
                        <h2 class="text-2xl font-black mt-2 text-white tracking-wide">台股早盤量化決策晨報</h2>
                        <p class="text-xs text-slate-400 mt-1">產出時間: {now_time_str} | Antigravity AI Quantitative Core</p>
                    </div>
                    <div class="text-right bg-slate-950/60 p-3 rounded-xl border border-slate-800">
                        <span class="text-[11px] text-slate-400">大盤多空能量指數</span>
                        <div class="text-2xl font-black text-rose-400">{score} <span class="text-xs text-slate-400">/ 100</span></div>
                    </div>
                </div>
            </div>

            <!-- 大盤關鍵均線與操作方針 -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-xl">
                    <span class="text-xs text-slate-400">加權指數最新點位</span>
                    <div class="text-xl font-bold text-white mt-1">{market_info.get('latest_close', 'N/A')} <span class="text-xs {'text-rose-400' if market_info.get('change_pct',0)>=0 else 'text-emerald-400'}">({market_info.get('change_pct',0):+0.2f}%)</span></div>
                    <div class="text-[11px] text-slate-500 mt-1">MA5: {market_info.get('ma5')} | MA20: {market_info.get('ma20')}</div>
                </div>
                <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-xl">
                    <span class="text-xs text-slate-400">市場狀態判定</span>
                    <div class="text-xl font-bold text-amber-400 mt-1">【{regime}】</div>
                    <div class="text-[11px] text-slate-500 mt-1">20日年化波動率: {market_info.get('volatility_20d', 0)}%</div>
                </div>
                <div class="bg-slate-900/90 border border-slate-800 p-4 rounded-xl">
                    <span class="text-xs text-slate-400">今日操盤方針</span>
                    <div class="text-xs text-slate-200 mt-1.5 leading-relaxed bg-slate-950 p-2 rounded border border-slate-800/80">{market_info.get('suggestion', '順勢而為，控制總部位風險。')}</div>
                </div>
            </div>

            <!-- 主動精選強勢突破標的表格 -->
            <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="font-bold text-sm text-slate-200 flex items-center space-x-2">
                        <span class="w-2 h-2 rounded-full bg-rose-500 animate-ping"></span>
                        <span>🎯 今日主動量化強勢突破選股池</span>
                    </h3>
                    <span class="text-xs text-slate-400 font-mono">共掃描 {len(picks)} 檔強勢股</span>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-slate-400 border-b border-slate-800 bg-slate-950/50">
                            <tr>
                                <th class="py-2.5 px-3">股票代碼</th>
                                <th class="py-2.5 px-3">最新價</th>
                                <th class="py-2.5 px-3">漲跌幅</th>
                                <th class="py-2.5 px-3">量化評分</th>
                                <th class="py-2.5 px-3">觸發信號</th>
                                <th class="py-2.5 px-3">防守停損</th>
                                <th class="py-2.5 px-3">目標價位</th>
                                <th class="py-2.5 px-3">風報比</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/60 font-mono">
        """

        if not picks:
            html += """
                            <tr>
                                <td colspan="8" class="text-center py-6 text-slate-500 font-sans">今日無標的完全符合突破門檻，建議持盈保泰、等待更佳風險報酬比。</td>
                            </tr>
            """
        else:
            for p in picks:
                change_color = "text-rose-400 font-bold" if p['change_pct'] >= 0 else "text-emerald-400 font-bold"
                sigs = " ".join([f"<span class='px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] mr-1'>{s}</span>" for s in p['signals']])
                html += f"""
                            <tr class="hover:bg-slate-800/40 transition">
                                <td class="py-2.5 px-3 font-bold text-white font-sans">{p['symbol']}</td>
                                <td class="py-2.5 px-3 text-slate-200">${p['close']}</td>
                                <td class="py-2.5 px-3 {change_color}">{p['change_pct']:+0.2f}%</td>
                                <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 font-bold">{p['score']}分</span></td>
                                <td class="py-2.5 px-3 font-sans">{sigs}</td>
                                <td class="py-2.5 px-3 text-emerald-400">${p['stop_loss']}</td>
                                <td class="py-2.5 px-3 text-amber-300">${p['target_price']}</td>
                                <td class="py-2.5 px-3 text-indigo-300 font-bold">{p['risk_reward_ratio']}</td>
                            </tr>
                """

        post_mortems = self.sync.get_recent_post_mortems(limit=5)
        html += """
            <!-- 覆盤與虧損檢討區塊 -->
            <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="font-bold text-sm text-slate-200 flex items-center space-x-2">
                        <span class="w-2 h-2 rounded-full bg-amber-500"></span>
                        <span>🔍 每日自動化覆盤與虧損歸因檢討 (Post-Mortem Loss Analysis)</span>
                    </h3>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-slate-400 border-b border-slate-800 bg-slate-950/50">
                            <tr>
                                <th class="py-2.5 px-3">日期</th>
                                <th class="py-2.5 px-3">標的</th>
                                <th class="py-2.5 px-3">進場價</th>
                                <th class="py-2.5 px-3">出場價</th>
                                <th class="py-2.5 px-3">虧損金額</th>
                                <th class="py-2.5 px-3">虧損幅度</th>
                                <th class="py-2.5 px-3">歸因分析與優化對策</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-800/60 font-mono">
        """
        if not post_mortems:
            html += """
                            <tr>
                                <td colspan="7" class="text-center py-4 text-slate-500 font-sans">近期無停損或虧損事件，系統執行紀律良好。</td>
                            </tr>
            """
        else:
            for pm in post_mortems:
                html += f"""
                            <tr class="hover:bg-slate-800/40 transition">
                                <td class="py-2.5 px-3 text-slate-300 font-sans">{pm['date']}</td>
                                <td class="py-2.5 px-3 font-bold text-white font-sans">{pm['symbol']}</td>
                                <td class="py-2.5 px-3 text-slate-200">${pm['entry_price']}</td>
                                <td class="py-2.5 px-3 text-slate-200">${pm['exit_price']}</td>
                                <td class="py-2.5 px-3 text-emerald-400 font-bold">-${abs(pm['loss_amount']):.2f}</td>
                                <td class="py-2.5 px-3 text-emerald-400 font-bold">{pm['loss_pct']:+.2f}%</td>
                                <td class="py-2.5 px-3 text-slate-300 font-sans">{pm['cause_analysis']}</td>
                            </tr>
                """
        html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """
        return html

if __name__ == "__main__":
    generator = ReportGenerator()
    content = generator.generate()
    print("\n" + "="*50)
    print(content)
