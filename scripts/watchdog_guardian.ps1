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
$BatName = (-join ([char[]]@(0x555F, 0x52D5, 0x91CF, 0x5316, 0x4EA4, 0x6613, 0x5E73, 0x53F0))) + ".bat"
$BatPath = Join-Path $RepoRoot $BatName

function Write-WatchdogLog {
    param([string]$Message)
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$ts] $Message"
    try {
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
    $procs = Get-CimInstance Win32_Process -Filter "Name LIKE 'python%' OR Name = 'cmd.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        $cmdLine = [string]$p.CommandLine
        if (-not $cmdLine) { continue }
        if ($cmdLine -match 'web_ui[\\/\.]app\.py' -or $cmdLine -match 'web_ui\.server' -or $cmdLine -match ([regex]::Escape($BatName))) {
            return $true
        }
    }
    return $false
}

function Invoke-WatchdogCheck {
    if (Test-Path $PauseFlag) {
        return
    }
    if (-not (Test-Path $BatPath)) {
        Write-WatchdogLog "[Watchdog] Launcher batch file not found: $BatPath"
        return
    }
    if (-not (Test-PlatformRunning -TargetPort $Port)) {
        Write-WatchdogLog ("[Watchdog] Detected platform stopped; auto-executing " + $BatName + " --watchdog")
        Start-Process -FilePath $BatPath -ArgumentList "--watchdog" -WorkingDirectory $RepoRoot -WindowStyle Minimized | Out-Null
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
