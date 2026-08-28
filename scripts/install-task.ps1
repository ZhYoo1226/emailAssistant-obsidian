# Run the email assistant as a scheduled task (auto-restart on crash, at logon).
# Usage (admin PowerShell):
#   .\scripts\install-task.ps1            # install for current user
#   .\scripts\install-task.ps1 -Remove    # uninstall
param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$taskName = "EmailAssistant"
$projectDir = Split-Path $PSScriptRoot -Parent

if ($Remove) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Task '$taskName' removed."
    exit 0
}

# Proxy is required for Gmail/Qdrant Cloud access on this machine.
# Adjust or remove -Proxy below if your network differs.
$python = Join-Path $projectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtualenv not found at $python. Run 'uv sync' first."
}

$action = New-ScheduledTaskAction -Execute $python -Argument "main.py" -WorkingDirectory $projectDir
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 365) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Task '$taskName' installed. It starts at logon; start it now with:"
Write-Host "  Start-ScheduledTask -TaskName '$taskName'"
Write-Host "Logs go to the Windows event log; for file logs consider redirecting main.py output."
