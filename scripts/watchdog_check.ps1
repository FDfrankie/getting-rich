$ErrorActionPreference = "SilentlyContinue"
$dir = "G:\我的雲端硬碟\炒幣"
if (Test-Path "$dir\data_persistence\watchdog.paused") { exit 0 }

function Resolve-QuantPython {
    $candidates = @(
        "C:\quant_envs\antigravity\Scripts\python.exe",
        "C:\quant_envs\venv\Scripts\python.exe",
        "G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return "python"
}

$pyExe = Resolve-QuantPython

# 1. Keep Quant_BrainMiner running as a single instance
$minerProcs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "headless_brain_miner\.py" } | Sort-Object CreationDate -Descending)
if ($minerProcs.Count -eq 0) {
    Start-ScheduledTask -TaskName "Quant_BrainMiner"
} elseif ($minerProcs.Count -gt 1) {
    $minerProcs | Select-Object -Skip 1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

# 2. Keep headless_crypto_bot.py running as a single instance with visible logs (P59 fix)
try {
    $cryptoProcs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "headless_crypto_bot\.py" } | Sort-Object CreationDate -Descending)
    if ($cryptoProcs.Count -eq 0) {
        $botOut = "$dir\data_persistence\headless_crypto_bot.log"
        $botErr = "$dir\data_persistence\headless_crypto_bot_err.log"
        Start-Process -FilePath $pyExe -ArgumentList "`"$dir\scripts\headless_crypto_bot.py`"" -WorkingDirectory $dir -WindowStyle Hidden -RedirectStandardOutput $botOut -RedirectStandardError $botErr
    } elseif ($cryptoProcs.Count -gt 1) {
        $cryptoProcs | Select-Object -Skip 1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
} catch {
    # Non-fatal: never block BrainMiner watchdog
}

# 3. Single Throttled 5-minute Cloud Crypto PnL Dashboard Sync (zero-lock read-only exporter)
try {
    $pnlFile = "$dir\data_persistence\crypto_pnl_status.json"
    $needsSync = $true
    if (Test-Path $pnlFile) {
        $ageSec = ((Get-Date) - (Get-Item $pnlFile).LastWriteTime).TotalSeconds
        if ($ageSec -lt 280) {
            $needsSync = $false
        }
    }
    if ($needsSync) {
        Start-Process -FilePath $pyExe -ArgumentList "`"$dir\scripts\export_crypto_pnl_to_drive.py`" --once" -WorkingDirectory $dir -WindowStyle Hidden -RedirectStandardError "$dir\data_persistence\pnl_sync_err.log"
    }
} catch {
    # Non-fatal
}
