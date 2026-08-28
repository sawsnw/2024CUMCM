---
name: create-skill
description: '创建新的 VS Code Copilot Skill。适用于将重复性工作流封装为可复用的 Skill，包含 SKILL.md、脚本、参考文档的完整脚手架搭建。'
argument-hint: '描述你想创建的 Skill 的用途和功能'
user-invocable: true
---

# 创建 Skill 的 Skill

## 用途
根据用户的需求，自动生成一个完整的 Skill 目录结构，包括 `SKILL.md`、配套脚本和参考文档。

## 何时使用
- 想将一个重复性工作流封装为 Skill
- 需要一个新 Skill 的脚手架模板
- 快速创建标准化的 Skill 目录

## 步骤

### 第 1 步：分析需求
询问用户以下信息：
1. **Skill 名称** — 目录名，小写字母、数字和连字符（如 `code-review`）
2. **用途描述** — Skill 的简短描述（将写入 `description` 字段）
3. **触发关键词** — 哪些词可以让 Agent 自动发现此 Skill
4. **工作流步骤** — Skill 要执行的具体步骤
5. **是否需要脚本** — 是否需要配套的可执行脚本
6. **是否需要参考文档** — 是否需要附加的参考资料

### 第 2 步：创建目录结构

```
.github/skills/<skill-name>/
├── SKILL.md           # 必须 — 核心指令文件
├── scripts/           # 可选 — 可执行脚本
│   └── *.ps1 / *.py / *.sh
└── references/        # 可选 — 参考文档
    └── *.md
```

### 第 3 步：生成 SKILL.md

使用[模板](./references/template.md) 生成 SKILL.md，填充 YAML 头部和正文。

**YAML 头部规则：**
- `name` — 与目录名一致，小写字母数字+连字符，1-64 字符
- `description` — 包含触发关键词，最长 1024 字符
- `argument-hint` — 斜杠命令的提示文字
- `user-invocable` — 默认 `true`（显示为斜杠命令）

### 第 4 步：生成脚本（如需要）

根据 Skill 的需求创建脚本文件，使用[脚本模板](./scripts/template.ps1)。

### 第 5 步：生成参考文档（如需要）

创建 `references/` 下的说明文档。

### 第 6 步：验证

检查生成的文件：
- `name` 是否与文件夹名一致
- YAML 头部语法是否正确（冒号、引号）
- `description` 是否包含触发关键词
- 文件路径引用是否正确（使用 `./` 相对路径）
- 正文是否清晰、步骤是否完整
