<#
.SYNOPSIS
Starts all components of the AGI Brain System.
.DESCRIPTION
This script starts both the frontend application and the mock API server simultaneously.
It creates separate terminal windows for each component for better management.
.EXAMPLE
PS> .\start-all.ps1
#>

# Stop any existing processes on required ports to avoid conflicts
Write-Host "Stopping any processes using required ports..." -ForegroundColor Yellow
try {
    Get-NetTCPConnection -LocalPort 5175, 8000 -ErrorAction SilentlyContinue | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped process $($_.OwningProcess) on port $($_.LocalPort)"
    }
} catch {
    Write-Host "No existing processes to stop."
}

# Start the mock API server in a new PowerShell window
Write-Host "Starting Mock API Server on port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"python '$PSScriptRoot\simple_mock_server.py'`""

# Wait a moment to ensure the server starts properly
Start-Sleep -Seconds 2

# Start the frontend application in a new PowerShell window
Write-Host "Starting AGI Brain System frontend on port 5175..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit -Command `"Set-Location -Path '$PSScriptRoot\app'; npm run dev`""

# Display instructions for the user
Write-Host "\nAGI Brain System Components Started:" -ForegroundColor Cyan
Write-Host "- Mock API Server: http://localhost:8000"
Write-Host "- Frontend Application: http://localhost:5175"
Write-Host "\nTo stop all components, simply close the terminal windows." -ForegroundColor Yellow