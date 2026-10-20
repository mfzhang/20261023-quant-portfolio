# Daily B0 paper-trading run for Windows Task Scheduler.
# Logs go to output/paper_logs/ (gitignored: the KRX login can print the account ID).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "output\paper_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("b0_{0:yyyyMMdd_HHmm}.log" -f (Get-Date))

Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$p = Start-Process -FilePath "C:\Users\82102\AppData\Local\Programs\Python\Python314\python.exe" `
    -ArgumentList "-X", "utf8", "paper\b0_paper.py" -WorkingDirectory $root -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err"
exit $p.ExitCode
