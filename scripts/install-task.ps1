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
$proxy = "http://127.0.0.1:7897"
$python = Join-Path $projectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtualenv not found at $python. Run 'uv sync' first."
}

# Scheduled tasks start with a clean environment, so the proxy must be set
# inside the task's command line — env vars from the installing shell do not
# carry over.
$cmd = "`$env:HTTPS_PROXY='$proxy'; `$env:HTTP_PROXY='$proxy'; & '$python' main.py *>> '$projectDir\assistant.log'"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -Command `"$cmd`"" `
    -WorkingDirectory $projectDir
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 365) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Task '$taskName' installed (proxy $proxy, log: $projectDir\assistant.log)."
Write-Host "It starts at logon; start it now with:"
Write-Host "  Start-ScheduledTask -TaskName '$taskName'"
