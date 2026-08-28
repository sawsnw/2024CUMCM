"""问题五可视化模块：龙头最大速度优化相关图表绘制。

绘图约定见 ``src/visualization/约定.md``：
    - 绘图文字使用中文；
    - 绘图不在图中加图名，只通过文件名体现。

共三张图：
    1. 速度放大系数 ``Λ(s0)`` 随龙头弧长位置的变化曲线（标注临界点）；
    2. 临界构型下各把手速度分布（含 2 m/s 上限线与临界把手）；
    3. 临界构型板凳龙位置图（突出临界把手）。
"""

import os

import matplotlib

matplotlib.use("Agg")  # 无界面后端，供入口脚本调用

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, TURN_SPACE_RADIUS
from src.models.problem4 import full_path_points

# 中文字体（Windows 常见字体，按顺序回退）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def plot_lambda_curve(
    s0_grid: np.ndarray,
    lambda_curve: np.ndarray,
    s0_star: float,
    lambda_max: float,
    output_path: str | None = None,
    xlim: tuple[float, float] | None = None,
    fine_s0: np.ndarray | None = None,
    fine_lambda: np.ndarray | None = None,
) -> str:
    """绘制速度放大系数 ``Λ(s0)`` 随龙头弧长位置的变化曲线。

    Args:
        s0_grid: 粗扫龙头弧长网格（m）。
        lambda_curve: 各网格点上的 ``Λ(s0)``。
        s0_star: 临界龙头弧长位置（m）。
        lambda_max: 全程最大放大系数。
        output_path: 输出图片路径，默认 ``problem5_lambda_curve.png``。
        xlim: 可选的横轴显示范围 ``(xmin, xmax)``（m）。
        fine_s0: 可选的最大值附近精细采样网格（m），用于局部放大子图。
        fine_lambda: 可选的精细采样对应的 ``Λ(s0)``。

    Returns:
        输出图片的绝对路径。
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "problem5_lambda_curve.png")

    has_fine = (
        fine_s0 is not None and fine_lambda is not None and len(fine_s0) > 1
    )

    if has_fine:
        # 分成两个独立子图：左 = 全程曲线，右 = 最大值附近局部放大
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))

        # 左图：全程 Λ(s0)
        ax1.plot(s0_grid, lambda_curve, "-", color="#1f77b4", lw=1.4,
                 label="速度放大系数 Λ(s0)")
        ax1.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.7,
                    label="Λ = 1（无放大）")
        ax1.axvline(s0_star, color="#d62728", ls=":", lw=1.4)
        ax1.annotate(
            f"Λ_max = {lambda_max:.4f}\ns0* = {s0_star:.3f} m",
            xy=(s0_star, lambda_max),
            xytext=(s0_star + 0.6, lambda_max * 0.80),
            fontsize=10, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
        )
        if xlim is not None:
            ax1.set_xlim(xlim)
        ax1.set_xlabel("龙头弧长位置 s0 (m)")
        ax1.set_ylabel("最大速度放大系数 Λ(s0)")
        ax1.set_title("(a) 全程速度放大系数 Λ(s0)", fontsize=11)
        ax1.grid(True, ls=":", alpha=0.4)
        ax1.legend(fontsize=9, loc="upper right")

        # 右图：最大值附近局部放大（精细采样曲线）
        ax2.plot(fine_s0, fine_lambda, "-", color="#1f77b4", lw=1.6,
                 label="速度放大系数 Λ(s0)")
        ax2.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.7,
                    label="Λ = 1（无放大）")
        ax2.axvline(s0_star, color="#d62728", ls=":", lw=1.4)
        ax2.scatter([s0_star], [lambda_max], color="#d62728", s=40, zorder=5)
        ax2.annotate(
            f"Λ_max = {lambda_max:.4f}\ns0* = {s0_star:.3f} m",
            xy=(s0_star, lambda_max),
            xytext=(s0_star + 0.15, lambda_max * 0.70),
            fontsize=10, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
        )
        ax2.set_xlabel("s0 (m)")
        ax2.set_ylabel("Λ(s0)")
        ax2.set_title("(b) 最大值附近局部放大", fontsize=11)
        ax2.grid(True, ls=":", alpha=0.4)
        ax2.legend(fontsize=9, loc="lower left")
    else:
        # 无精细数据：单图（全程曲线）
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(s0_grid, lambda_curve, "-", color="#1f77b4", lw=1.4,
                label="速度放大系数 Λ(s0)")
        ax.axhline(1.0, color="gray", ls="--", lw=1.0, alpha=0.7,
                   label="Λ = 1（无放大）")
        ax.axvline(s0_star, color="#d62728", ls=":", lw=1.4)
        ax.annotate(
            f"Λ_max = {lambda_max:.4f}\ns0* = {s0_star:.3f} m",
            xy=(s0_star, lambda_max),
            xytext=(s0_star + 0.5, lambda_max * 0.82),
            fontsize=10, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
        )
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_xlabel("龙头弧长位置 s0 (m)")
        ax.set_ylabel("最大速度放大系数 Λ(s0)")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_critical_speeds(
    lambda_star: np.ndarray,
    i_star: int,
    v_h_max: float,
    s0_star: float,
    speed_limit: float = 2.0,
    output_path: str | None = None,
) -> str:
    """绘制临界构型下各把手的速度分布。

    实际速度为放大系数与龙头最大速度之积 ``v_i = λ_i·v_h_max``，
    并绘制 2 m/s 速度上限线、标注临界把手。

    Args:
        lambda_star: 临界构型下全部把手放大系数（shape (224,)）。
        i_star: 临界把手编号。
        v_h_max: 龙头最大恒定速度（m/s）。
        s0_star: 临界龙头弧长位置（m）。
        speed_limit: 各把手速度上限（m/s）。
        output_path: 输出图片路径，默认 ``problem5_critical_speeds.png``。

    Returns:
        输出图片的绝对路径。
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "problem5_critical_speeds.png")

    speeds = lambda_star * v_h_max
    idx = np.arange(speeds.size)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(idx, speeds, "-", color="#1f77b4", lw=1.3, label="把手速度 v_i")
    ax.axhline(speed_limit, color="#d62728", ls="--", lw=1.4,
               label=f"速度上限 {speed_limit:.0f} m/s")
    ax.scatter([i_star], [speeds[i_star]], color="#d62728", s=55, zorder=5)
    ax.annotate(
        f"临界把手 i*={i_star}\nv = {speeds[i_star]:.3f} m/s",
        xy=(i_star, speeds[i_star]),
        xytext=(i_star + 4, speeds[i_star] - 0.15),
        fontsize=10, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
    )
    ax.set_xlabel("把手编号 i")
    ax.set_ylabel("速度 v (m/s)")
    ax.set_title("临界构型下各把手速度（s0* = %.3f m，v_h = %.4f m/s）"
                 % (s0_star, v_h_max), fontsize=10)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_critical_configuration(
    g: dict,
    positions: np.ndarray,
    i_star: int,
    s0_star: float,
    output_path: str | None = None,
) -> str:
    """绘制临界构型的板凳龙位置图，突出临界把手。

    Args:
        g: ``solve_s_curve`` 返回的几何 dict。
        positions: 临界构型下全部把手中心坐标（shape (224, 2)）。
        i_star: 临界把手编号。
        s0_star: 临界龙头弧长位置（m）。
        output_path: 输出图片路径，默认 ``problem5_critical_config.png``。

    Returns:
        输出图片的绝对路径。
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "problem5_critical_config.png")

    circ = np.linspace(0.0, 2.0 * np.pi, 200)
    cx = TURN_SPACE_RADIUS * np.cos(circ)
    cy = TURN_SPACE_RADIUS * np.sin(circ)

    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.plot(cx, cy, "--", color="k", lw=1.1, alpha=0.6, label="调头空间边界")

    # 完整路径
    fp = full_path_points(g, n=400)
    ax.plot(fp[:, 0], fp[:, 1], "-", color="gray", lw=1.0, alpha=0.5,
            label="最优调头路径")

    # 板凳（把手连线）
    ax.plot(positions[:, 0], positions[:, 1], "-", color="#1f77b4", lw=1.4,
            label="板凳龙")
    ax.scatter(positions[:, 0], positions[:, 1], color="#1f77b4", s=12)
    # 龙头与临界把手
    ax.scatter(positions[0, 0], positions[0, 1], color="#2ca02c", s=45,
               label="龙头（第 0 把手）")
    ax.scatter(positions[i_star, 0], positions[i_star, 1], color="#d62728", s=60,
               zorder=5, label=f"临界把手 i*={i_star}")
    ax.annotate(
        f"临界把手 i*={i_star}",
        xy=(positions[i_star, 0], positions[i_star, 1]),
        xytext=(0.4, 0.4), textcoords="offset points",
        fontsize=9, color="#d62728",
    )
    head = positions[0]
    ax.text(
        0.0, -TURN_SPACE_RADIUS * 1.14,
        f"临界构型：s0* = {s0_star:.3f} m，龙头 @ ({head[0]:.3f}, {head[1]:.3f}) m",
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
