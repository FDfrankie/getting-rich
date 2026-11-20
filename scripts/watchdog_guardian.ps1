[CmdletBinding()]
param(
    [switch]$CheckOnce,
    [switch]$Loop,
    [int]$IntervalSeconds = 10,
    [int]$Port = 8999
)

$ErrorActionPreference = "SilentlyContinue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $RepoRoot "data_persistence"
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
}

$PauseFlag = Join-Path $DataDir "watchdog.paused"
$LogPath = Join-Path $DataDir "watchdog.log"
$StdoutLog = Join-Path $DataDir "web_ui_stdout.log"
$StderrLog = Join-Path $DataDir "web_ui_stderr.log"

function Get-SilentPythonPath {
    $venvDir = Join-Path (Split-Path -Parent $RepoRoot) ".quant_envs\antigravity\Scripts"
    $pyw = Join-Path $venvDir "pythonw.exe"
    if (Test-Path $pyw) { return $pyw }
    $py = Join-Path $venvDir "python.exe"
    if (Test-Path $py) { return $py }
    return "pythonw.exe"
}

function Write-WatchdogLog {
    param([string]$Message)
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$ts] $Message"
    try {
        if ((Test-Path $LogPath) -and ((Get-Item $LogPath).Length -gt 5242880)) {
            $rotPath = "$LogPath.1"
            if (Test-Path $rotPath) { Remove-Item -Force $rotPath -ErrorAction SilentlyContinue }
            Move-Item -Force $LogPath $rotPath -ErrorAction SilentlyContinue
        }
        Add-Content -Path $LogPath -Value $line -Encoding UTF8
    } catch {
        # Best-effort logging
    }
}

function Test-PlatformRunning {
    param([int]$TargetPort)
    $listening = Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue
    if ($listening) {
        return $true
    }
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$TargetPort/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            return $true
        }
    } catch {
        # Not listening yet
    }
    try {
        $procs = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" -ErrorAction SilentlyContinue
        foreach ($p in $procs) {
            $cmdLine = [string]$p.CommandLine
            if (-not $cmdLine) { continue }
            if ($cmdLine -match 'web_ui[\\/\.]app\.py' -or $cmdLine -match 'web_ui\.server') {
                return $true
            }
        }
    } catch {
        # Ignore CIM permission errors
    }
    return $false
}

function Invoke-WatchdogCheck {
    if (Test-Path $PauseFlag) {
        return
    }
    if (-not (Test-PlatformRunning -TargetPort $Port)) {
        $pyExe = Get-SilentPythonPath
        Write-WatchdogLog "[Watchdog] Detected platform stopped; starting silently via $pyExe (WindowStyle Hidden)"
        Start-Process -FilePath $pyExe `
            -ArgumentList @("-B", "-u", "web_ui\app.py") `
            -WorkingDirectory $RepoRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $StdoutLog `
            -RedirectStandardError $StderrLog | Out-Null
    }
}

if ($CheckOnce) {
    Invoke-WatchdogCheck
    exit 0
}

# Continuous daemon mode: enforce single daemon instance via Named Mutex
$mutexName = "Local\AntigravityQuantWatchdogDaemon_$Port"
$createdNew = $false
$mutex = New-Object System.Threading.Mutex($true, $mutexName, [ref]$createdNew)
if (-not $createdNew) {
    exit 0
}

try {
    Write-WatchdogLog "[Watchdog] Guardian daemon started (port=$Port, interval=${IntervalSeconds}s)"
    while ($true) {
        try {
            Invoke-WatchdogCheck
        } catch {
            Write-WatchdogLog "[Watchdog] Check iteration error: $($_.Exception.Message)"
        }
        Start-Sleep -Seconds ([Math]::Max(3, $IntervalSeconds))
    }
} finally {
    try { $mutex.ReleaseMutex() } catch {}
    $mutex.Dispose()
}
