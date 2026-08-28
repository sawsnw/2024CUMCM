# 项目指南

## 代码风格
- 使用中文注释
- 遵循 PEP 8（Python）/ 项目现有风格
- 所有函数必须写类型注解和文档字符串

## 架构
```
Test/
├── src/                 # 核心代码模块
│   ├── config.py        # 全局配置
│   ├── evaluation/      # 评估模块
│   ├── models/          # 模型模块
│   ├── utils/           # 工具函数
│   └── visualization/   # 可视化模块
├── data/                # 数据文件
├── docs/                # 文档
│   ├── model_design/    # 模型设计
│   ├── problem_analysis/# 问题分析
│   └── references/      # 参考资料
├── results/             # 输出结果
│   ├── figures/         # 图表
│   ├── logs/            # 日志
│   └── tables/          # 表格
└── scripts/             # 入口脚本
```

## 约定
- 所有参数和路径等统一在 `config.py` 中定义，并在代码中通过 `from config import *` 导入。
- 可运行的入口脚本放在 `scripts/` 目录下，通过 `from src.xxx import *` 导入 `src/` 中的模块。
- `src/` 下的模块不可直接运行，只提供可导入的函数和类。
- 所有输出结果除了日志文件外，均覆盖旧文件
- 日志文件命名规则为 `{脚本名}_{YYYY-MM-DD}_{HHMMSS}.log`，例如 `train_model_2026-07-14_143052.log`
- 日志文件不覆盖旧文件，每次运行追加新的日志文件

