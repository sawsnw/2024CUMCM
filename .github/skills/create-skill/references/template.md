# SKILL.md 模板

## YAML 头部模板

```yaml
---
name: your-skill-name          # 与目录名一致
description: '描述该 Skill 的用途和触发场景。包含关键触发词，如"适用于xxx"、"当需要xxx时使用"。'
argument-hint: '斜杠命令的提示文字'
user-invocable: true           # 是否显示为斜杠命令
---
```

## 正文模板
 
```markdown
# Skill 名称

## 用途
简要说明这个 Skill 的作用。

## 何时使用
- 触发场景 1
- 触发场景 2
- 触发场景 3

## 步骤
### 第 1 步：xxx
具体操作说明

### 第 2 步：xxx
具体操作说明，可引用[脚本](./scripts/xxx.ps1)或[参考文档](./references/xxx.md)

### 第 3 步：xxx
具体操作说明
```

## 提示

1. **description** 是 Agent 发现的关键，务必包含触发词
2. 引用资源使用相对路径 `./scripts/xxx` 或 `./references/xxx`
3. SKILL.md 正文建议控制在 500 行以内
