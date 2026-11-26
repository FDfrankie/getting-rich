$ErrorActionPreference = "SilentlyContinue"
$dir = "G:\我的雲端硬碟\炒幣"
if (Test-Path "$dir\data_persistence\watchdog.paused") { exit 0 }

# 1. Keep Quant_BrainMiner running as a single instance
$minerProcs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "headless_brain_miner\.py" } | Sort-Object CreationDate -Descending)
if ($minerProcs.Count -eq 0) {
    Start-ScheduledTask -TaskName "Quant_BrainMiner"
} elseif ($minerProcs.Count -gt 1) {
    $minerProcs | Select-Object -Skip 1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

# 2. Throttled 5-minute Cloud Crypto PnL Dashboard Sync (zero-lock read-only exporter)
$pnlFile = "$dir\data_persistence\crypto_pnl_status.json"
$needsSync = $true
if (Test-Path $pnlFile) {
    $ageSec = ((Get-Date) - (Get-Item $pnlFile).LastWriteTime).TotalSeconds
    if ($ageSec -lt 280) {
        $needsSync = $false
    }
}
if ($needsSync) {
    $pyExe = "G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe"
    if (-not (Test-Path $pyExe)) { $pyExe = "python" }
    Start-Process -FilePath $pyExe -ArgumentList "`"$dir\scripts\export_crypto_pnl_to_drive.py`" --once" -WorkingDirectory $dir -WindowStyle Hidden -RedirectStandardError "$dir\data_persistence\pnl_sync_err.log"
}

# P55 Self-Healing Crypto Bot + Cloud PnL Sync
try {
    $cryptoProc = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*headless_crypto_bot*" }
    if (-not $cryptoProc) {
        $pyExe = "C:\quant_envs\venv\Scripts\python.exe"
        if (-not (Test-Path $pyExe)) { $pyExe = "python" }
        Start-Process -FilePath $pyExe -ArgumentList "scripts\headless_crypto_bot.py" -WorkingDirectory "G:\我的雲端硬碟\炒幣" -WindowStyle Hidden -ErrorAction SilentlyContinue
    }
    $pyExeSync = "C:\quant_envs\venv\Scripts\python.exe"
    if (-not (Test-Path $pyExeSync)) { $pyExeSync = "python" }
    & $pyExeSync "G:\我的雲端硬碟\炒幣\scripts\export_crypto_pnl_to_drive.py" --once 2>&1 | Out-Null
} catch {
    # Non-fatal: never block BrainMiner watchdog
}
