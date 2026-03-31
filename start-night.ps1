$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = "src"
$logDir = Join-Path $projectRoot "data"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$collectorLog = Join-Path $logDir "collector-night.log"
$dashboardLog = Join-Path $logDir "dashboard-night.log"

$collectorCommand = @"
Set-Location '$projectRoot'
\$env:PYTHONPATH='$pythonPath'
py -m kalshi_btc_bot.cli --log-level INFO collect-forever *>> '$collectorLog'
"@

$dashboardCommand = @"
Set-Location '$projectRoot'
\$env:PYTHONPATH='$pythonPath'
py -m streamlit run app.py --server.headless true --server.port 8502 *>> '$dashboardLog'
"@

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $collectorCommand
) -WorkingDirectory $projectRoot

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    $dashboardCommand
) -WorkingDirectory $projectRoot

Write-Host ""
Write-Host "Overnight processes started."
Write-Host "Collector log:  $collectorLog"
Write-Host "Dashboard log:  $dashboardLog"
Write-Host "Dashboard URL:  http://localhost:8502"
