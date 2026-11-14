import time
import schedule
from datetime import datetime
from taiwan_stock.report_generator import ReportGenerator
from taiwan_stock.config import WEBHOOK_SETTINGS

# 雙時段排程 (Asia/Taipei 時區)
MORNING_SCHEDULE_TIME = "08:00"   # 早上盤前 08:00
AFTERNOON_SCHEDULE_TIME = "16:30" # 下午盤後 16:30

def scheduled_job(session_type: str = "盤前晨報"):
    """定時執行的台股量化決策報告產出任務 (支援盤前與盤後雙時段)"""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{now_str}] ⏰ 觸發台股【{session_type}】自動產出任務 (Asia/Taipei)...")
    try:
        generator = ReportGenerator()
        report_text = generator.generate(session_type=session_type)
        html_report = generator.generate_html_report(session_type=session_type)
        print(f"[✓] 【{session_type}】產出成功！(含 Markdown 與視覺化 HTML/PDF)")

        # 檢查是否啟用外部推播 (例如 LINE Notify 或 Discord Webhook)
        if WEBHOOK_SETTINGS.get("enable_push"):
            print(f"[i] 推播功能已配置，若需通知可在此發送 Webhook【{session_type}】。")
    except Exception as e:
        print(f"[!] 執行【{session_type}】任務時發生異常: {e}")

def main():
    print("=" * 60)
    print(f"🕒 台股自動決策報告守護程式 (Dual-Session Scheduler) 已啟動")
    print(f"📅 排程設定 (Asia/Taipei)：")
    print(f"   1. 下午盤後分析：每日 【{AFTERNOON_SCHEDULE_TIME}】")
    print(f"   2. 早上盤前晨報：每日 【{MORNING_SCHEDULE_TIME}】")
    print("=" * 60)

    # 註冊雙時段定時排程
    schedule.every().day.at(AFTERNOON_SCHEDULE_TIME).do(scheduled_job, session_type="下午盤後報告")
    schedule.every().day.at(MORNING_SCHEDULE_TIME).do(scheduled_job, session_type="早上盤前晨報")

    print("[*] 系統進入常駐監聽迴圈 (按 Ctrl+C 可停止)...")
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    main()
