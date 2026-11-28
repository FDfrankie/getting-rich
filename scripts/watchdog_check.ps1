$ErrorActionPreference = "Continue"
$dir = "G:\我的雲端硬碟\炒幣"
if (Test-Path "$dir\data_persistence\watchdog.paused") { exit 0 }

$auditLog = "$dir\data_persistence\watchdog_audit.log"
function Write-Audit([string]$msg) {
    try {
        $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        Add-Content -Path $auditLog -Value "[$ts] $msg" -Encoding UTF8 -ErrorAction SilentlyContinue
        if ((Test-Path $auditLog) -and ((Get-Item $auditLog).Length -gt 65536)) {
            $tail = Get-Content -Path $auditLog -Tail 150 -ErrorAction SilentlyContinue
            Set-Content -Path $auditLog -Value $tail -Encoding UTF8 -ErrorAction SilentlyContinue
        }
    } catch {}
}

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
try {
    $minerProcs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "headless_brain_miner\.py" } | Sort-Object CreationDate -Descending)
    if ($minerProcs.Count -eq 0) {
        Start-ScheduledTask -TaskName "Quant_BrainMiner" -ErrorAction SilentlyContinue
        Write-Audit "STARTED Quant_BrainMiner (0 instances found)"
    } elseif ($minerProcs.Count -gt 1) {
        $minerProcs | Select-Object -Skip 1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Write-Audit "DEDUP Quant_BrainMiner (kept newest PID=$($minerProcs[0].ProcessId))"
    }
} catch {
    Write-Audit "ERR BrainMiner watchdog: $($_.Exception.Message)"
}

# 2. Keep headless_crypto_bot.py running as a single instance on walk-forward 3m Maker profile (P60)
try {
    $cryptoProcs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "headless_crypto_bot\.py" } | Sort-Object CreationDate -Descending)
    if ($cryptoProcs.Count -eq 0) {
        $env:CRYPTO_STRATEGY_PROFILE = "condprob_3m"
        $env:CRYPTO_EXEC_TIMEFRAME = "3m"
        $botOut = "$dir\data_persistence\headless_crypto_bot.log"
        $botErr = "$dir\data_persistence\headless_crypto_bot_err.log"
        $p = Start-Process -FilePath $pyExe -ArgumentList "`"$dir\scripts\headless_crypto_bot.py`"" -WorkingDirectory $dir -WindowStyle Hidden -RedirectStandardOutput $botOut -RedirectStandardError $botErr -PassThru
        Write-Audit "STARTED headless_crypto_bot.py PID=$($p.Id) profile=condprob_3m tf=3m"
    } elseif ($cryptoProcs.Count -gt 1) {
        $cryptoProcs | Select-Object -Skip 1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Write-Audit "DEDUP headless_crypto_bot.py (kept newest PID=$($cryptoProcs[0].ProcessId))"
    }
} catch {
    Write-Audit "ERR CryptoBot watchdog: $($_.Exception.Message)"
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
        Write-Audit "SYNCED crypto_pnl_status.json via export_crypto_pnl_to_drive.py --once"
    }
} catch {
    Write-Audit "ERR PnL sync: $($_.Exception.Message)"
}
