$ErrorActionPreference = "Stop"

$days = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

$start = Get-Date
Write-Host "============================================================"
Write-Host "Label Matcher Batch - 5 days"
Write-Host "Started at $start"
Write-Host "============================================================"

foreach ($day in $days) {
    $dayStart = Get-Date
    $flowsFile  = "data\processed\multiday_packets\${day}_flow_features.csv"
    $labelsFile = "data\processed\multiday_labels\${day}_combined.csv"
    $outFile    = "data\processed\multiday_packets\${day}_labeled.csv"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[$day] Starting at $dayStart"
    Write-Host "  Flows : $flowsFile"
    Write-Host "  Labels: $labelsFile"
    Write-Host "  Out   : $outFile"
    Write-Host "============================================================"

    if (-not (Test-Path $flowsFile)) {
        Write-Host "[$day] [SKIP] Flow file not found: $flowsFile"
        continue
    }
    if (-not (Test-Path $labelsFile)) {
        Write-Host "[$day] [SKIP] Label file not found: $labelsFile"
        continue
    }

    python src\label_matcher.py --flows $flowsFile --labels $labelsFile --out $outFile

    $dayEnd = Get-Date
    $elapsed = ($dayEnd - $dayStart).TotalMinutes
    Write-Host "[$day] Finished in $([math]::Round($elapsed, 1)) min"
}

$end = Get-Date
$totalMin = ($end - $start).TotalMinutes
Write-Host ""
Write-Host "============================================================"
Write-Host "All done at $end"
Write-Host "Total: $([math]::Round($totalMin, 1)) min"
Write-Host "============================================================"