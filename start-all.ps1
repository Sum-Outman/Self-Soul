<#
.SYNOPSIS
# Starts all core components of the Self Soul AGI System.
.DESCRIPTION
This script starts the main API server, frontend application, realtime stream manager and performance monitoring service simultaneously.
It creates separate terminal windows for each component for better management.
.EXAMPLE
PS> .\start-all.ps1
#>

# Stop any existing processes on required ports to avoid conflicts
Write-Host "Stopping any processes using required ports..." -ForegroundColor Yellow
try {
    Get-NetTCPConnection -LocalPort 5175, 8000, 8766, 8080 -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped process $($_.OwningProcess) on port $($_.LocalPort)"
    }
} catch {
    Write-Host "No existing processes to stop."
}

# Using Anaconda Python which already has all dependencies installed
$anacondaPython = "C:\ProgramData\Anaconda3\python.exe"

# Start the main API server in a new PowerShell window
Write-Host "Starting Main API Server on port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"& '$anacondaPython' '$PSScriptRoot\core\main.py'`""

# Start the realtime stream manager in a new PowerShell window
Write-Host "Starting Realtime Stream Manager on port 8766..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"& '$anacondaPython' '$PSScriptRoot\core\realtime_stream_manager.py'`""

# Wait a moment to ensure the stream manager starts properly
Start-Sleep -Seconds 3

# Start the performance monitoring service in a new PowerShell window
Write-Host "Starting Performance Monitoring Service on port 8080..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"& '$anacondaPython' '$PSScriptRoot\core\system_monitor.py'`""

# Wait a moment to ensure all services start properly
Start-Sleep -Seconds 3

# Start the frontend application in a new PowerShell window
Write-Host "Starting AGI Soul System frontend on port 5175..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"Set-Location -Path '$PSScriptRoot\app'; npm install; npm run dev`""

# Display instructions for the user
Write-Host "\nAGI Soul System Components Started:" -ForegroundColor Cyan
Write-Host "- Main API Server: http://localhost:8000"
Write-Host "- Frontend Application: http://localhost:5175"
Write-Host "\nTo stop all components, simply close the terminal windows." -ForegroundColor Yellow
