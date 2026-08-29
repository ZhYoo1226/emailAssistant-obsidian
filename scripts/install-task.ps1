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

# config.py loads HTTPS_PROXY from .env, so the task needs no proxy setup
# here — only output redirection for diagnosis.
$python = Join-Path $projectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtualenv not found at $python. Run 'uv sync' first."
}

$cmd = "& '$python' main.py *>> '$projectDir\assistant.log'"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -Command `"$cmd`"" `
    -WorkingDirectory $projectDir
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 365) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Task '$taskName' installed (proxy comes from .env; log: $projectDir\assistant.log)."
Write-Host "It starts at logon; start it now with:"
Write-Host "  Start-ScheduledTask -TaskName '$taskName'"
