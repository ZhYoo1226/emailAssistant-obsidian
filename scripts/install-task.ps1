# 把邮件助手注册为计划任务（崩溃自动重启，登录时自启）。
# 用法（管理员 PowerShell）：
#   .\scripts\install-task.ps1            # 为当前用户安装
#   .\scripts\install-task.ps1 -Remove    # 卸载
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

# config.py 会从 .env 读取 HTTPS_PROXY，所以这里不需要给任务配置代理
# ——只需要输出重定向以便诊断。
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
