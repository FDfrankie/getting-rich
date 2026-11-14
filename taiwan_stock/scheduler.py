import time
import schedule
from datetime import datetime
from taiwan_stock.report_generator import ReportGenerator
from taiwan_stock.config import REPORT_SCHEDULE_TIME, WEBHOOK_SETTINGS

def scheduled_job():
    """每日定時執行的早報產出任務"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏰ 觸發每日早晨 {REPORT_SCHEDULE_TIME} 自動晨報任務...")
    try:
        generator = ReportGenerator()
        report_text = generator.generate()
        print("[✓] 每日晨報產出成功！")

        # 檢查是否啟用外部推播 (例如 LINE Notify 或 Discord Webhook)
        if WEBHOOK_SETTINGS.get("enable_push"):
            # 可在此串接 requests.post 推播到通訊軟體
            print("[i] 推播功能已配置，若需通知可在此發送 Webhook。")
    except Exception as e:
        print(f"[!] 執行晨報任務時發生異常: {e}")

def main():
    print("=" * 60)
    print(f"🕒 台股自動早報守護程式 (Scheduler) 已啟動")
    print(f"📅 排程設定：每日台灣時間 【{REPORT_SCHEDULE_TIME}】 自動生成市場報告")
    print("=" * 60)

    # 註冊每日定時排程
    schedule.every().day.at(REPORT_SCHEDULE_TIME).do(scheduled_job)

    # 初次啟動時先詢問或立即執行一次測試
    print("[*] 系統進入常駐監聽迴圈 (按 Ctrl+C 可停止)...")
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    main()
