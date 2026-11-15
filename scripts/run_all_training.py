"""
統一訓練轉發腳本：
假資料合成已清理，統一導向 scripts/train_models.py 執行真實歷史訓練與因果驗證。
"""

import subprocess
import sys

if __name__ == "__main__":
    print("[*] 正在透過 scripts/train_models.py 執行因果無洩漏之真實歷史訓練管線...")
    cmd = [sys.executable, "scripts/train_models.py", "--target", "all"]
    ret = subprocess.call(cmd)
    sys.exit(ret)
