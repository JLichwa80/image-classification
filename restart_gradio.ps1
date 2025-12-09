# Restart the Gradio app for the pneumonia image-classification project
# Usage (from PowerShell in this folder):
#   .\restart_gradio.ps1

param(
    [string]$PythonExe = "python"
)

Write-Host "Stopping existing Python/Gradio processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

$AppPath = "deployment\gradio-app\app.py"
Write-Host "Starting Gradio app using" $PythonExe $AppPath "..." -ForegroundColor Green

# Disable Gradio's auto-reload by setting environment variable
$env:GRADIO_WATCH_DIRS = "false"

& $PythonExe $AppPath
