# Skill 配套脚本模板
# 将此脚本放在 .github/skills/<skill-name>/scripts/ 目录下

param(
    [string]$Param1 = "default"
)

Write-Host "=== Skill 脚本执行开始 ===" -ForegroundColor Cyan
Write-Host "参数: $Param1" -ForegroundColor Yellow

# --- 在这里添加你的脚本逻辑 ---

Write-Host "=== Skill 脚本执行完成 ===" -ForegroundColor Green
