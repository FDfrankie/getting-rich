from datetime import datetime
from taiwan_stock.twse_calendar import TAIPEI_TZ as _TAIPEI_TZ, is_twse_trading_day
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
except ImportError:
    BlockingScheduler = None
    CronTrigger = None


from taiwan_stock.config import WEBHOOK_SETTINGS
from taiwan_stock.report_generator import ReportGenerator

# 雙時段排程 (Asia/Taipei 時區)
MORNING_SCHEDULE_TIME = "08:00"  # 早上盤前 08:00
AFTERNOON_SCHEDULE_TIME = "16:30"  # 下午盤後 16:30


def scheduled_job(session_type: str = "盤前晨報"):
    """定時執行的台股量化決策報告產出任務 (支援盤前與盤後雙時段)"""
    now_str = datetime.now(_TAIPEI_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now_str}] ⏰ 觸發台股【{session_type}】自動產出任務 (Asia/Taipei)...")
    try:
        generator = ReportGenerator()
        md_content = generator.generate(session_type=session_type)
        generator.generate_html_report(session_type=session_type)
        print(f"[✓] 【{session_type}】產出成功！(含 Markdown 與視覺化 HTML/PDF)")

        webhook_url = WEBHOOK_SETTINGS.get("webhook_url") or WEBHOOK_SETTINGS.get("discord_webhook_url")
        if WEBHOOK_SETTINGS.get("enable_push") and webhook_url:
            from notifications.alert_manager import send_webhook_notification

            ok = send_webhook_notification(webhook_url, f"台股【{session_type}】", str(md_content or "")[:1500])
            print(f"[{'✓' if ok else '!'}] Webhook 推播{'成功' if ok else '失敗'}")
    except Exception as e:
        print(f"[!] 執行【{session_type}】任務時發生異常: {e}")


def main():
    print("=" * 60)
    print("🕒 台股自動決策報告守護程式 (Dual-Session Scheduler) 已啟動")
    print("📅 排程設定 (Asia/Taipei)：")
    print(f"   1. 下午盤後分析：每日 【{AFTERNOON_SCHEDULE_TIME}】")
    print(f"   2. 早上盤前晨報：每日 【{MORNING_SCHEDULE_TIME}】")
    print("=" * 60)

    # 註冊雙時段定時排程
    if BlockingScheduler is None or CronTrigger is None:
        raise RuntimeError("APScheduler is required for timezone-safe scheduling (Asia/Taipei)")
    sched = BlockingScheduler(timezone=_TAIPEI_TZ)
    sched.add_job(lambda: is_twse_trading_day() and scheduled_job("下午盤後報告"), CronTrigger(hour=16, minute=30, timezone=_TAIPEI_TZ))
    sched.add_job(lambda: is_twse_trading_day() and scheduled_job("早上盤前晨報"), CronTrigger(hour=8, minute=0, timezone=_TAIPEI_TZ))
    sched.start()


if __name__ == "__main__":
    main()
