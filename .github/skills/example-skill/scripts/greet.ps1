# 示例脚本：输出欢迎信息
param(
    [string]$Name = "World"
)

Write-Host "Hello, $Name!" -ForegroundColor Green
Write-Host "This is an example skill script." -ForegroundColor Cyan
Write-Host "Skill directory: $PSScriptRoot" -ForegroundColor Yellow
