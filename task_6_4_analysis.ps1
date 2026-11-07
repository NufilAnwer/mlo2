
$ErrorActionPreference = "Continue"

Write-Host "`n===================================="
Write-Host "VISIONGUARD - TASK 6.4 IMAGE ANALYSIS"
Write-Host "===================================="

$Report = "task_6_4_analysis_report.txt"

Start-Transcript -Path $Report -Force

Write-Host "`nSTEP 1: Largest installed Python packages"
docker compose exec web sh -c "du -sh /usr/local/lib/python3.11/site-packages/* 2>/dev/null | sort -h | tail -20"

Write-Host "`nSTEP 2: TensorFlow package information"
docker compose exec web python -m pip show tensorflow-cpu

Write-Host "`nSTEP 3: Installed Python packages"
docker compose exec web python -m pip list

Write-Host "`nSTEP 4: Current Docker image size"
$Before = docker image inspect visionguard:task-6.3 --format '{{.Size}}'

if ($LASTEXITCODE -eq 0 -and $Before) {
    $BeforeMB = [math]::Round(([double]$Before / 1MB), 2)
    Write-Host "Before optimization: $BeforeMB MB"
}

Write-Host "`nSTEP 5: Docker disk usage"
docker system df

Write-Host "`nSTEP 6: Available C drive space"
Get-PSDrive C | Select-Object Name, @{
    Name = "FreeGB"
    Expression = { [math]::Round($_.Free / 1GB, 2) }
} | Format-Table -AutoSize

Write-Host "`nSTEP 7: Save original Docker configuration"

if (!(Test-Path "Dockerfile.before-optimization")) {
    Copy-Item Dockerfile Dockerfile.before-optimization
}

if (!(Test-Path "requirements-docker.before-optimization.txt")) {
    Copy-Item requirements-docker.txt requirements-docker.before-optimization.txt
}

Write-Host "Original Docker configuration backed up."

Stop-Transcript

Write-Host "`nAnalysis completed."
Write-Host "Report saved to: $Report"
Write-Host "No Docker images, containers, or volumes were deleted."
Write-Host "No model training or rebuilding was performed."
