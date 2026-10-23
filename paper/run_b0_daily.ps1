# Daily paper-trading run (B0, then H1) for Windows Task Scheduler.
# Logs go to output/paper_logs/ (gitignored: the KRX login can print the account ID).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "output\paper_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = "{0:yyyyMMdd_HHmm}" -f (Get-Date)

Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$code = 0
foreach ($book in "b0", "h1") {
    $log = Join-Path $logDir ("{0}_{1}.log" -f $book, $stamp)
    $p = Start-Process -FilePath "C:\Users\82102\AppData\Local\Programs\Python\Python314\python.exe" `
        -ArgumentList "-X", "utf8", "paper\${book}_paper.py" -WorkingDirectory $root -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    if ($p.ExitCode -ne 0) { $code = $p.ExitCode }
}
exit $code
