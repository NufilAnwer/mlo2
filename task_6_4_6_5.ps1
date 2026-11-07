
$ErrorActionPreference = "Continue"

$Report = "task_6_4_6_5_report.txt"

Start-Transcript -Path $Report -Force

function Show-Section {
    param([string]$Title)
    Write-Host "`n========================================"
    Write-Host $Title
    Write-Host "========================================"
}

# ==========================================
# TASK 6.4 - DOCKER IMAGE ANALYSIS
# ==========================================

Show-Section "TASK 6.4 - IMAGE SIZE AND OPTIMIZATION"

Write-Host "`n1. Current Docker image size"

$ImageSize = docker image inspect visionguard:task-6.3 --format '{{.Size}}'

if ($LASTEXITCODE -eq 0 -and $ImageSize) {
    $ImageMB = [math]::Round(([double]$ImageSize / 1MB), 2)
    Write-Host "Current image size: $ImageMB MB"
}

Write-Host "`n2. Docker image history"
docker history visionguard:task-6.3

Write-Host "`n3. Docker disk usage"
docker system df

Write-Host "`n4. Verify Docker build cache cleanup"
docker builder du

Write-Host "`n5. Verify Dockerfile optimization backup"

if (Test-Path "Dockerfile.before-optimization") {
    Write-Host "Original Dockerfile backup exists."
}
else {
    Write-Host "Original Dockerfile backup missing."
}

if (Test-Path "requirements-docker.before-optimization.txt") {
    Write-Host "Original requirements backup exists."
}
else {
    Write-Host "Original requirements backup missing."
}

Write-Host "`n6. Current C drive space"

Get-PSDrive C | Select-Object Name, @{
    Name = "FreeGB"
    Expression = { [math]::Round($_.Free / 1GB, 2) }
} | Format-Table -AutoSize

Write-Host "`n7. Verify application HTTP response"

try {
    $Response = Invoke-WebRequest http://localhost:8000/ -UseBasicParsing
    Write-Host "HTTP status:" $Response.StatusCode
}
catch {
    Write-Host "HTTP check failed:" $_.Exception.Message
}

Write-Host "`nTask 6.4 analysis recorded."
Write-Host "Note: Image rebuild/size reduction has NOT been performed."
Write-Host "Cache cleanup reclaimed 2.112 GB."

# ==========================================
# TASK 6.5 - DOCKER OPERATIONS
# ==========================================

Show-Section "TASK 6.5 - DOCKER OPERATIONS"

Write-Host "`n1. Check running containers"
docker compose ps

Write-Host "`n2. Check application container"
docker compose exec web whoami

Write-Host "`n3. Check application working directory"
docker compose exec web pwd

Write-Host "`n4. Check application files"
docker compose exec web ls -la /app

Write-Host "`n5. Check PostgreSQL readiness"
docker compose exec database pg_isready -U visionguard -d visionguard

Write-Host "`n6. Check database connectivity from Flask container"

docker compose exec web python -c "import os, psycopg2; c=psycopg2.connect(os.environ['DATABASE_URL']); q=c.cursor(); q.execute('SELECT 1'); print('Database connection successful:', q.fetchone()[0]); c.close()"

Write-Host "`n7. Check prediction table"
docker compose exec database psql -U visionguard -d visionguard -c "SELECT COUNT(*) AS total_predictions FROM predictions;"

Write-Host "`n8. Display recent application logs"
docker compose logs --tail=30 web

Write-Host "`n9. Display recent database logs"
docker compose logs --tail=20 database

# ==========================================
# TASK 6.5 - STOP AND START TEST
# ==========================================

Show-Section "TASK 6.5 - STOP AND START OPERATIONS"

Write-Host "`n10. Stop only the web service"
docker compose stop web

Write-Host "`n11. Verify web service stopped"
docker compose ps

Write-Host "`n12. Start the web service again"
docker compose start web

Write-Host "`n13. Wait for web container health"

$Healthy = $false

for ($i = 1; $i -le 24; $i++) {
    Start-Sleep -Seconds 5

    $Health = docker inspect visionguard-web --format '{{.State.Health.Status}}' 2>$null

    Write-Host "Health check $i : $Health"

    if ($Health -eq "healthy") {
        $Healthy = $true
        break
    }
}

if ($Healthy) {
    Write-Host "Web service is healthy."
}
else {
    Write-Host "WARNING: Web service did not become healthy within 120 seconds."
    docker compose logs --tail=50 web
}

# ==========================================
# TASK 6.5 - RESTART AND TROUBLESHOOTING
# ==========================================

Show-Section "TASK 6.5 - RESTART AND TROUBLESHOOTING"

Write-Host "`n14. Restart web service"
docker compose restart web

Write-Host "`n15. Wait for application recovery"

$Healthy = $false

for ($i = 1; $i -le 24; $i++) {
    Start-Sleep -Seconds 5

    $Health = docker inspect visionguard-web --format '{{.State.Health.Status}}' 2>$null

    Write-Host "Recovery check $i : $Health"

    if ($Health -eq "healthy") {
        $Healthy = $true
        break
    }
}

if ($Healthy) {
    Write-Host "Application recovered successfully."
}
else {
    Write-Host "WARNING: Application recovery needs investigation."
    docker compose logs --tail=50 web
}

Write-Host "`n16. Test application HTTP response"

try {
    $Response = Invoke-WebRequest http://localhost:8000/ -UseBasicParsing
    Write-Host "HTTP status:" $Response.StatusCode
}
catch {
    Write-Host "HTTP check failed:" $_.Exception.Message
}

Write-Host "`n17. Inspect final service status"
docker compose ps

Write-Host "`n18. Inspect final application logs"
docker compose logs --tail=30 web

Write-Host "`n19. Inspect final database readiness"
docker compose exec database pg_isready -U visionguard -d visionguard

# ==========================================
# FINAL REPORT
# ==========================================

Show-Section "TASK 6.4 AND TASK 6.5 SUMMARY"

Write-Host "Task 6.4: Image analysis and cache cleanup recorded."
Write-Host "Task 6.4: Image rebuild not performed."
Write-Host "Task 6.5: Container operations and connectivity tested."
Write-Host "Task 6.5: Stop/start and restart recovery tested."
Write-Host "PostgreSQL data volume was preserved."
Write-Host "Champion model was not retrained."
Write-Host "`nDetailed output: $Report"

Stop-Transcript

Write-Host "`nScript completed."
