# 2024 高教社杯全国大学生数学建模竞赛 A 题 —— "板凳龙"闹元宵

## 简介

"板凳龙"（又称"盘龙"）是浙闽地区的传统民俗活动，人们将数十至上百条板凳首尾相连，形成蜿蜒曲折的板凳龙。盘龙时龙头在前领头，龙身与龙尾相随盘旋，整体呈圆盘状。在能够自如盘入和盘出的前提下，盘龙所需面积越小、行进速度越快，观赏性越好。

本仓库针对 2024 年全国大学生数学建模竞赛 A 题，对一支由 **223 节板凳**（1 节龙头 + 221 节龙身 + 1 节龙尾）首尾相连组成的舞龙队，沿等距螺线盘入、盘出及调头过程中的**运动学建模、碰撞检测与速度优化**问题进行了完整建模与求解，并提供了可复现的代码、文档与结果。

## 问题概览与核心结果

| 问题 | 内容 | 核心结果 |
|---|---|---|
| 一 | 沿等距螺线盘入的位置与速度模型 | 224 个把手中心 $t=0\sim300\,\mathrm{s}$ 的位置与速度 |
| 二 | 盘入终止时刻的碰撞检测模型 | 最晚盘入时刻 $t^{*}=412.474\,\mathrm{s}$，接触对 $(1,9)$ |
| 三 | 满足调头空间约束的最小螺距模型 | 最小螺距 $p^{*}=0.449610\,\mathrm{m}$ |
| 四 | S 形调头曲线优化及板凳龙运动模型 | 最短无碰撞调头曲线 $L^{*}=1.287452\,\mathrm{m}$，较基准缩短 $90.48\%$ |
| 五 | 基于速度传递系数的龙头最大速度优化 | 龙头最大恒定速度 $v_{\mathrm{h}}^{\max}=0.569398\,\mathrm{m/s}$ |

## 项目结构

```
2024CUMCM/
├── pyproject.toml            # 项目配置与依赖（numpy / openpyxl / matplotlib）
├── README.md
├── A题_text.txt              # 赛题原文
├── data/                     # 数据文件（result1/2/4.xlsx 模板，只读）
├── docs/
│   ├── model_design/         # 模型设计文档
│   ├── problem_analysis/     # 问题分析与模型求解文档（问题一~五）
│   └── references/           # 参考资料
├── results/
│   ├── figures/              # 图表输出（problem2/3/4/5_*.png 等）
│   ├── logs/                 # 日志（不覆盖旧文件）
│   └── tables/               # 表格结果（result1/2/4.xlsx、problem3/5_*.xlsx 等）
├── scripts/                  # 入口脚本（solve_problem1~5.py 等）
└── src/                      # 核心代码模块（不可直接运行）
    ├── config.py             # 全局配置（参数与路径统一在此定义）
    ├── evaluation/           # 评估模块
    ├── models/               # 模型模块
    │   ├── spiral_dragon.py  # 问题一/二：等距螺线盘入模型
    │   ├── collision.py      # 问题二：碰撞检测
    │   ├── problem3.py       # 问题三：最小螺距模型
    │   ├── problem4.py       # 问题四：S 形调头曲线模型
    │   └── problem5.py       # 问题五：速度传递系数模型
    ├── utils/                # 工具函数（logger.py 等）
    └── visualization/        # 可视化模块（problem3/4/5_plots.py 等）
```

## 快速开始

### 环境要求

- Python 3.9+
- 依赖：`numpy`、`openpyxl`、`matplotlib`（已声明于 `pyproject.toml`）

### 安装

```bash
pip install -e .
```

### 运行入口脚本

各问题均有独立入口脚本，运行方式（以问题一为例）：

```bash
python scripts/solve_problem1.py
```

| 脚本 | 对应问题 | 主要输出 |
|---|---|---|
| `scripts/solve_problem1.py` | 问题一 | `results/tables/result1.xlsx` |
| `scripts/solve_problem2.py` | 问题二 | `results/tables/result2.xlsx`、`problem2_global_margin.xlsx`、`problem2_global_margin.png` |
| `scripts/solve_problem3.py` | 问题三 | `results/tables/problem3_*.xlsx`、`results/figures/problem3_*.png` |
| `scripts/solve_problem4.py` | 问题四 | `results/tables/result4.xlsx`、`problem4_turning_comparison.png` |
| `scripts/solve_problem5.py` | 问题五 | `results/tables/problem5_*.xlsx/json`、`results/figures/problem5_*.png` |
| `scripts/plot_problem4_route.py` | 问题四 | 调头路线示意图 `problem4_route.png` |

## 问题与模型概述

### 问题一：板凳龙沿等距螺线盘入的位置与速度模型

舞龙队沿螺距 $p=0.55\,\mathrm{m}$ 的等距螺线顺时针向内盘入，龙头前把手位于第 16 圈，以 $1\,\mathrm{m/s}$ 匀速行进。基于等距螺线 $r=b\theta$（$b=p/(2\pi)$）与相邻把手间的**固定弦长**约束，建立位置递推与速度递推模型，求解 $t=0,1,\ldots,300\,\mathrm{s}$ 时全部 224 个把手中心的位置与速度。

### 问题二：板凳龙盘入终止时刻的碰撞检测模型

在问题一模型基础上，重建每节板凳的矩形区域，对所有非相邻板凳进行两两碰撞检测，定义全局裕量 $G(t)$，沿时间轴搜索 $G(t)$ 首次由正变零的时刻。结果为 $t^{*}=412.473838\,\mathrm{s}$，首次接触发生在龙头与第 8 节龙身之间（板凳对 $(1,9)$）。

### 问题三：满足调头空间约束的最小螺距模型

舞龙队须在半径 $R=4.5\,\mathrm{m}$ 的圆形调头空间内由顺时针盘入切换为逆时针盘出。第三问要求确定盘入螺线的**最小螺距** $p^{*}$，使龙头能够沿等距螺线安全盘入至调头空间边界而不发生碰撞。求解得 $p^{*}=0.449610\,\mathrm{m}$，临界状态发生在龙头极径 $r_0=4.6174\,\mathrm{m}$ 处，最危险板凳对为 $(1,18)$。

### 问题四：S 形调头曲线优化及板凳龙运动模型

盘入螺线螺距 $p=1.7\,\mathrm{m}$，盘出螺线为中心对称像，调头路径由两段相切圆弧（第一段半径为第二段的 2 倍）组成。通过二维网格搜索 + 局部细化 + 双朝向扰动验证，求得满足碰撞约束的最短 S 形曲线 $L^{*}=1.287452\,\mathrm{m}$（切点参数 $\theta_a=5.602429$，$\theta_c=2.656490$），相对基准曲线 $13.525\,\mathrm{m}$ 缩短约 $90.48\%$，并计算 $t=-100,-99,\ldots,100\,\mathrm{s}$ 全部把手的位置与速度。

### 问题五：基于速度传递系数的龙头最大速度优化模型

利用相邻把手固定距离约束导出的**速度传递系数** $\lambda_i(s)$（速度关于龙头速度齐次线性），在问题四最优路径上扫描全部构型，得到最大放大系数 $\Lambda_{\max}=3.512480$（出现在 $s_0=0.964\,\mathrm{m}$，临界把手为第 1 节龙身），从而龙头最大恒定速度为

$$
v_{\mathrm{h}}^{\max}=\frac{2}{\Lambda_{\max}}=0.569398\,\mathrm{m/s}.
$$

## 目录约定

- 所有参数和路径统一在 `src/config.py` 中定义，代码通过 `from src.config import *` 导入。
- 可运行入口脚本位于 `scripts/` 目录，通过 `from src.xxx import *` 导入 `src/` 中的模块。
- `src/` 下的模块不可直接运行，仅提供可导入的函数和类。
- 除日志文件外，所有输出结果均覆盖旧文件。
- 日志文件命名规则为 `{脚本名}_{YYYY-MM-DD}_{HHMMSS}.log`（如 `solve_problem1_2026-07-14_143052.log`），不覆盖旧文件，每次运行追加新的日志文件。

## 文档

- 各问题的详细模型建立与求解过程见 `docs/problem_analysis/问题一~五_模型建立与求解.md`。
- 模型设计文档见 `docs/model_design/`，参考资料见 `docs/references/`。