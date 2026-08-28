---
name: example-skill
description: '一个示例 Skill，演示如何在 VS Code 中创建和使用自定义 Skill。适用于学习 Skill 的结构和工作流程。'
argument-hint: '提供要处理的任务描述'
user-invocable: true
---

# 示例 Skill

## 用途
这是一个演示用的示例 Skill，帮助你理解 Skill 的结构和使用方式。

## 何时使用
- 学习如何创建 Skill
- 测试 Skill 的自动发现机制
- 作为新 Skill 的模板参考

## 结构说明

```
.github/skills/example-skill/
├── SKILL.md           # Skill 的核心指令文件
├── scripts/           # 存放可执行脚本
│   └── greet.ps1      # 示例脚本
└── references/        # 存放参考文档
```

## 使用方式

### 方式一：自动发现
当你的任务描述中包含 Skill 描述中的关键词（如"示例"、"学习 Skill"等）时，Agent 会自动加载此 Skill。

### 方式二：斜杠命令
在聊天框中输入 `/` 可以看到此 Skill 并手动选择调用。

### 方式三：手动引用
在对话中提及此 Skill 的名称即可触发加载。

## 步骤
1. Agent 发现匹配的 Skill
2. 加载 SKILL.md 正文
3. 按需引用 `scripts/` 或 `references/` 中的资源
4. 执行 Skill 中定义的流程
