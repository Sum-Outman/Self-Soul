<#
.SYNOPSIS
Starts the AGI Brain System frontend application.
.DESCRIPTION
This script changes to the app directory and starts the Vue.js development server.
It's designed to work with PowerShell, avoiding issues with && operator compatibility.
.EXAMPLE
PS> .\start-app.ps1
#>

# Change to the app directory
Set-Location -Path "$PSScriptRoot\app"

# Check if npm is installed
if (-not (Get-Command "npm" -ErrorAction SilentlyContinue)) {
    Write-Host "npm is not installed. Please install Node.js and npm first." -ForegroundColor Red
    exit 1
}

# Start the development server
Write-Host "Starting AGI Brain System frontend on port 5175..." -ForegroundColor Green
npm run dev