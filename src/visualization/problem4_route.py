"""问题四可视化模块：S 形调头路线图绘制。

绘制双面板路线图：
    左：整体路线——盘入螺线 → 调头空间 → S 形调头曲线 → 盘出螺线；
    右：调头空间放大——两段相切圆弧、圆心、切点与半径标注。

绘图约定见 src/visualization/约定.md：
    - 绘图文字使用中文；
    - 绘图不在图中加图名，只通过文件名体现。
"""

import os

import matplotlib

matplotlib.use("Agg")  # 无界面后端，供入口脚本调用

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, P4_B, TURN_SPACE_RADIUS
from src.models.problem4 import (
    inward_spiral_point,
    outward_spiral_point,
    s_curve_points,
    turning_length,
)

# 中文字体（Windows 常见字体，按顺序回退）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# 龙头初始参数（第 16 圈，与问题一一致）
HEAD_INITIAL_THETA = 32.0 * np.pi


def _spiral_curve_points(
    theta_end: float,
    theta_start: float,
    n: int = 800,
    outward: bool = False,
) -> np.ndarray:
    """采样一段螺线的坐标点（沿行进方向 θ 递减/递增）。

    Args:
        theta_end: 终点参数（弧度）。
        theta_start: 起点参数（弧度）。
        n: 采样点数。
        outward: 是否采样盘出螺线（中心对称像）。

    Returns:
        (n+1, 2) 的坐标点数组。
    """
    thetas = np.linspace(theta_start, theta_end, n + 1)
    if outward:
        return np.array([outward_spiral_point(t) for t in thetas])
    return np.array([inward_spiral_point(t) for t in thetas])


def _draw_direction_arrows(ax, pts: np.ndarray, n_arrows: int = 4, color: str = "k"):
    """在路径上均匀放置行进方向箭头。

    Args:
        ax: matplotlib 坐标轴。
        pts: 路径采样点 (N, 2)。
        n_arrows: 箭头数量。
        color: 箭头颜色。
    """
    idx = np.linspace(0, len(pts) - 2, n_arrows, dtype=int)
    for i in idx:
        p0 = pts[i]
        v = pts[i + 1] - pts[i]
        ax.annotate(
            "",
            xy=p0 + 0.4 * v,
            xytext=p0,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.2, mutation_scale=12
            ),
        )


def plot_problem4_route(g: dict, output_path: str | None = None) -> str:
    """绘制问题四 S 形调头路线图。

    Args:
        g: solve_s_curve 返回的几何 dict（给定配置）。
        output_path: 输出图片路径，默认 results/figures/problem4_route.png。

    Returns:
        输出图片的绝对路径。
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "problem4_route.png")

    arc1, arc2 = s_curve_points(g, n1=150, n2=150)
    A, C, B = g["A"], g["C"], g["B"]
    c1, c2, R = g["c1"], g["c2"], g["R"]
    theta_a = g["theta_a"]
    theta_c = g["theta_c"]

    # 螺线参数范围：整体路线从龙头初始圈（θ=32π）开始
    theta_start = HEAD_INITIAL_THETA
    spiral_in = _spiral_curve_points(theta_a, theta_start, n=1500)
    spiral_out = _spiral_curve_points(theta_c, theta_start, n=1500, outward=True)

    # 调头空间边界圆
    circ = np.linspace(0.0, 2.0 * np.pi, 200)
    cx = TURN_SPACE_RADIUS * np.cos(circ)
    cy = TURN_SPACE_RADIUS * np.sin(circ)

    # 龙头起点（θ=32π 处，r=16p=27.2 m，位于 +x 轴）
    head_start = inward_spiral_point(HEAD_INITIAL_THETA)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.8))

    # ================= 左：整体路线 =================
    ax = axes[0]
    ax.plot(spiral_in[:, 0], spiral_in[:, 1], "-", color="#1f77b4", lw=1.2,
            label="盘入螺线（p=1.7 m）")
    ax.plot(spiral_out[:, 0], spiral_out[:, 1], "-", color="#2ca02c", lw=1.2,
            label="盘出螺线（中心对称）")
    ax.plot(cx, cy, "--", color="k", lw=1.0, alpha=0.7,
            label="调头空间（直径 9 m）")
    s_curve = np.vstack([arc1, arc2[1:]])
    ax.plot(s_curve[:, 0], s_curve[:, 1], "-", color="#ff7f0e", lw=2.6,
            label="S 形调头曲线")
    ax.scatter([head_start[0]], [head_start[1]], color="k", s=25, zorder=5)
    ax.annotate("龙头起点", xy=(head_start[0], head_start[1]),
                xytext=(head_start[0] - 3.2, head_start[1] - 1.6),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    ax.scatter([0.0], [0.0], color="k", s=25, zorder=5)
    ax.annotate("螺线中心 O", xy=(0.0, 0.0), xytext=(1.2, -1.6),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    # 调头空间边界与 S 形曲线在整体图中的位置提示
    ax.annotate("调头空间", xy=(0.0, TURN_SPACE_RADIUS),
                xytext=(-10.0, 4.5), fontsize=9, color="k", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="k", alpha=0.7))
    _draw_direction_arrows(ax, spiral_in, n_arrows=3, color="#1f77b4")
    _draw_direction_arrows(ax, spiral_out, n_arrows=3, color="#2ca02c")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

    # ================= 右：调头空间放大 =================
    ax2 = axes[1]
    ax2.plot(cx, cy, "--", color="k", lw=1.2, alpha=0.7)
    ax2.plot(arc1[:, 0], arc1[:, 1], "-", color="#ff7f0e", lw=3.0,
             label="前段圆弧（半径 2R）")
    ax2.plot(arc2[:, 0], arc2[:, 1], "-", color="#d62728", lw=3.0,
             label="后段圆弧（半径 R）")
    # 盘入/盘出螺线局部（贴近切点部分）
    theta_plot = np.linspace(theta_a - 0.6, theta_a + 0.6, 300)
    local_in = np.array([inward_spiral_point(t) for t in theta_plot])
    local_out = np.array([outward_spiral_point(t) for t in theta_plot])
    ax2.plot(local_in[:, 0], local_in[:, 1], "-", color="#1f77b4", lw=1.4)
    ax2.plot(local_out[:, 0], local_out[:, 1], "-", color="#2ca02c", lw=1.4)

    # 圆心与半径示意
    ax2.plot([c1[0]], [c1[1]], "o", color="#ff7f0e", ms=6)
    ax2.annotate("圆心 O1", xy=(c1[0], c1[1]), xytext=(c1[0] - 0.1, c1[1] - 0.85),
                 fontsize=9, color="#ff7f0e")
    ax2.plot([c2[0]], [c2[1]], "o", color="#d62728", ms=6)
    ax2.annotate("圆心 O2", xy=(c2[0], c2[1]), xytext=(c2[0] - 0.1, c2[1] + 0.85),
                 fontsize=9, color="#d62728")
    # 半径标注线（圆心到弧上一点）
    a_mid = arc1[len(arc1) // 2]
    b_mid = arc2[len(arc2) // 2]
    ax2.plot([c1[0], a_mid[0]], [c1[1], a_mid[1]], "-", color="#ff7f0e", lw=0.9,
             alpha=0.6)
    ax2.plot([c2[0], b_mid[0]], [c2[1], b_mid[1]], "-", color="#d62728", lw=0.9,
             alpha=0.6)
    ax2.text(a_mid[0] + 0.15, a_mid[1] + 0.1, f"2R≈{2 * R:.2f} m",
             color="#ff7f0e", fontsize=9)
    ax2.text(b_mid[0] - 0.05, b_mid[1] + 0.15, f"R≈{R:.2f} m",
             color="#d62728", fontsize=9)

    # 切点标注
    for pt, name, color, off in [
        (A, "A", "r", (0.5, -0.9)),
        (B, "B", "k", (0.5, 0.9)),
        (C, "C", "m", (-1.3, 0.6)),
    ]:
        ax2.scatter([pt[0]], [pt[1]], color=color, s=35, zorder=6)
        ax2.annotate(name, xy=(pt[0], pt[1]),
                     xytext=(pt[0] + off[0], pt[1] + off[1]),
                     arrowprops=dict(arrowstyle="->", lw=0.7, color=color),
                     color=color, fontsize=11, fontweight="bold")

    ax2.scatter([0.0], [0.0], color="k", s=25, zorder=6)
    ax2.annotate("O", xy=(0.0, 0.0), xytext=(-0.7, -0.9), fontsize=11,
                 fontweight="bold")
    ax2.text(
        -TURN_SPACE_RADIUS * 0.98, TURN_SPACE_RADIUS * 1.04,
        "调头空间", fontsize=10, color="k", ha="left",
    )
    ax2.text(0.0, TURN_SPACE_RADIUS - 0.55, f"ρ = {TURN_SPACE_RADIUS} m",
             fontsize=9, color="k", ha="center", va="top")

    # 行进方向箭头
    _draw_direction_arrows(ax2, arc1, n_arrows=2, color="#ff7f0e")
    _draw_direction_arrows(ax2, arc2, n_arrows=2, color="#d62728")

    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_aspect("equal")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.set_xlim(-TURN_SPACE_RADIUS * 1.25, TURN_SPACE_RADIUS * 1.25)
    ax2.set_ylim(-TURN_SPACE_RADIUS * 1.25, TURN_SPACE_RADIUS * 1.25)

    # 调头曲线总长说明（问题四关注量）
    ax2.text(
        0.0, -TURN_SPACE_RADIUS * 1.12,
        f"调头曲线总长 L = 2R·α1 + R·α2 ≈ {turning_length(g):.2f} m",
        fontsize=10, ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fff7e6", ec="#ff7f0e", lw=0.8),
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
