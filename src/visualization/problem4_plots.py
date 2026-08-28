"""问题四可视化模块：最优调头曲线对比图绘制。

绘制调头空间内的基准 S 形曲线与优化后 S 形曲线，直观展示调头曲线变短。

绘图约定见 src/visualization/约定.md：
    - 绘图文字使用中文；
    - 绘图不在图中加图名，只通过文件名体现。
"""

import os

import matplotlib

matplotlib.use("Agg")  # 无界面后端，供入口脚本调用

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, TURN_SPACE_RADIUS
from src.models.problem4 import s_curve_points

# 中文字体（Windows 常见字体，按顺序回退）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def plot_turning_comparison(
    g_baseline: dict,
    g_optimal: dict,
    output_path: str | None = None,
) -> str:
    """绘制基准与最优 S 形调头曲线对比图。

    Args:
        g_baseline: 基准配置几何 dict（solve_s_curve 返回）。
        g_optimal: 优化后配置几何 dict。
        output_path: 输出图片路径，默认 problem4_turning_comparison.png。

    Returns:
        输出图片的绝对路径。
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "problem4_turning_comparison.png")

    circ = np.linspace(0.0, 2.0 * np.pi, 200)
    cx = TURN_SPACE_RADIUS * np.cos(circ)
    cy = TURN_SPACE_RADIUS * np.sin(circ)

    fig, ax = plt.subplots(figsize=(7.2, 7.0))

    # 调头空间边界圆
    ax.plot(cx, cy, "--", color="k", lw=1.2, alpha=0.7, label="调头空间边界")

    # 基准曲线
    a1b, a2b = s_curve_points(g_baseline, n1=150, n2=150)
    sb = np.vstack([a1b, a2b[1:]])
    ax.plot(sb[:, 0], sb[:, 1], "-", color="#1f77b4", lw=2.6,
            label=f"基准曲线 L0={g_baseline['L_S']:.2f} m")
    ax.scatter([g_baseline["A"][0]], [g_baseline["A"][1]], color="#1f77b4", s=30)
    ax.scatter([g_baseline["C"][0]], [g_baseline["C"][1]], color="#1f77b4", s=30)

    # 最优曲线
    a1o, a2o = s_curve_points(g_optimal, n1=150, n2=150)
    so = np.vstack([a1o, a2o[1:]])
    ax.plot(so[:, 0], so[:, 1], "-", color="#d62728", lw=2.6,
            label=f"最优曲线 L*={g_optimal['L_S']:.2f} m")
    ax.scatter([g_optimal["A"][0]], [g_optimal["A"][1]], color="#d62728", s=30)
    ax.scatter([g_optimal["C"][0]], [g_optimal["C"][1]], color="#d62728", s=30)

    # 圆心
    for c, color, name in [(g_baseline["c1"], "#1f77b4", "O1"), (g_baseline["c2"], "#1f77b4", "O2"),
                           (g_optimal["c1"], "#d62728", "O1'"), (g_optimal["c2"], "#d62728", "O2'")]:
        ax.plot([c[0]], [c[1]], "o", color=color, ms=5)

    ax.scatter([0.0], [0.0], color="k", s=25)
    ax.annotate("螺线中心 O", xy=(0.0, 0.0), xytext=(0.8, -1.1),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)

    # 缩短量标注
    dL = g_baseline["L_S"] - g_optimal["L_S"]
    eta = 100.0 * dL / g_baseline["L_S"]
    ax.text(
        0.0, -TURN_SPACE_RADIUS * 1.12,
        f"缩短量 ΔL = {dL:.2f} m，缩短比例 η = {eta:.2f}%",
        fontsize=10, ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7e6", ec="#d62728", lw=0.8),
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(-TURN_SPACE_RADIUS * 1.3, TURN_SPACE_RADIUS * 1.3)
    ax.set_ylim(-TURN_SPACE_RADIUS * 1.3, TURN_SPACE_RADIUS * 1.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
