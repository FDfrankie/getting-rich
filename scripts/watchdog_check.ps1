$ErrorActionPreference = "SilentlyContinue"
$dir = "G:\我的雲端硬碟\炒幣"
if (Test-Path "$dir\data_persistence\watchdog.paused") { exit 0 }

$minerProcs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "headless_brain_miner\.py" } | Sort-Object CreationDate -Descending)
if ($minerProcs.Count -eq 0) {
    Start-ScheduledTask -TaskName "Quant_BrainMiner"
} elseif ($minerProcs.Count -gt 1) {
    # Keep newest instance and terminate older orphan duplicates so Quant_BrainMiner stays single-instance
    $minerProcs | Select-Object -Skip 1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
