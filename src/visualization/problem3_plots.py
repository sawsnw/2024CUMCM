"""问题三可视化模块：螺距—裕量曲线与临界构型图绘制。

绘图约定见 src/visualization/约定.md：
    - 绘图文字使用中文；
    - 绘图不在图中加图名，只通过文件名体现。
"""

import os

import matplotlib

matplotlib.use("Agg")  # 无界面后端，供入口脚本调用

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, TURN_RADIUS

# 中文字体（Windows 常见字体，按顺序回退）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def plot_phi_vs_pitch(
    p_grid: list[float],
    phi_grid: list[float],
    p_star: float,
    phi_star: float,
    output_path: str | None = None,
) -> str:
    """绘制全过程最小裕量 Φ(p) 随螺距 p 的变化曲线。

    用于显示安全—碰撞临界点：Φ(p) ≥ 0 可行，Φ(p) < 0 不可行，
    并用红色虚线标注最小可行螺距 p*。

    Args:
        p_grid: 螺距采样值列表（m）。
        phi_grid: 对应的全过程最小裕量 Φ(p) 列表（m）。
        p_star: 求解得到的最小可行螺距（m）。
        phi_star: p* 处的全过程最小裕量（m）。
        output_path: 输出图片路径，默认 problem3_phi_vs_pitch.png。

    Returns:
        输出图片的绝对路径。
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "problem3_phi_vs_pitch.png")

    pp = np.array(p_grid, dtype=float)
    ph = np.array(phi_grid, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(pp, ph, "-o", ms=3.5, lw=1.4, color="#1f77b4",
            label="全过程最小裕量 Φ(p)")
    ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(p_star, color="r", lw=1.3, ls="--", alpha=0.85,
               label=f"最小螺距 p* = {p_star:.6f} m")
    ax.scatter([p_star], [phi_star], color="r", s=45, zorder=5)
    ax.annotate(
        f"临界点 ({p_star:.4f}, {phi_star:.2e})",
        xy=(p_star, phi_star),
        xytext=(p_star + (pp.max() - pp.min()) * 0.08, ph.max() * 0.5),
        arrowprops=dict(arrowstyle="->", color="r", lw=1.0),
        color="r", fontsize=9,
    )
    ax.fill_between(pp, ph, 0.0, where=(ph >= 0), color="g", alpha=0.12,
                    label="可行区 Φ(p) ≥ 0")
    ax.fill_between(pp, ph, 0.0, where=(ph < 0), color="r", alpha=0.12,
                    label="碰撞区 Φ(p) < 0")
    ax.set_xlabel("螺距 p (m)")
    ax.set_ylabel("全过程最小裕量 Φ(p) (m)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlim(float(pp.min()), float(pp.max()))
    pad = (float(ph.max()) - float(ph.min())) * 0.08 + 1e-9
    ax.set_ylim(float(ph.min()) - pad, float(ph.max()) + pad)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_margin_vs_radius(
    p_star: float,
    r_grid: list[float],
    margin_grid: list[float],
    critical_r: float,
    critical_pair: tuple[int, int],
    output_path: str | None = None,
) -> str:
    """绘制最优螺距下全局裕量 G(p*, r0) 随龙头极径 r0 的变化曲线。

    用于验证最危险构型是否位于调头边界（文档 10.3、13 节）。

    Args:
        p_star: 最优螺距（m）。
        r_grid: 龙头极径采样值列表（m）。
        margin_grid: 对应的全局裕量 G(p*, r0) 列表（m）。
        critical_r: 最危险构型的龙头极径（m）。
        critical_pair: 最危险板凳对 (i*, j*)。
        output_path: 输出图片路径，默认 problem3_margin_vs_radius.png。

    Returns:
        输出图片的绝对路径。
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "problem3_margin_vs_radius.png")

    rr = np.array(r_grid, dtype=float)
    gg = np.array(margin_grid, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(rr, gg, "-o", ms=3.0, lw=1.3, color="#2ca02c",
            label="全局碰撞裕量 G(p*, r0)")
    ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(critical_r, color="r", lw=1.3, ls="--", alpha=0.85,
               label=f"最危险极径 r0 = {critical_r:.3f} m")
    ax.scatter([critical_r], [float(gg.min())], color="r", s=45, zorder=5)
    ax.annotate(
        f"临界板凳对 ({critical_pair[0]}, {critical_pair[1]})",
        xy=(critical_r, float(gg.min())),
        xytext=(critical_r - (rr.max() - rr.min()) * 0.30, float(gg.max()) * 0.55),
        arrowprops=dict(arrowstyle="->", color="r", lw=1.0),
        color="r", fontsize=9,
    )
    # 标注调头边界位置
    ax.axvline(TURN_RADIUS, color="b", lw=1.0, ls=":", alpha=0.7,
               label=f"调头边界 r0 = {TURN_RADIUS} m")
    ax.set_xlabel("龙头前把手极径 r0 (m)")
    ax.set_ylabel("全局碰撞裕量 G(p*, r0) (m)")
    ax.set_title(f"最优螺距 p* = {p_star:.6f} m", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlim(float(rr.min()), float(rr.max()))
    pad = (float(gg.max()) - float(gg.min())) * 0.12 + 1e-9
    ax.set_ylim(float(gg.min()) - pad, float(gg.max()) + pad)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_critical_configuration(
    positions: list[tuple[float, float]],
    critical_pair: tuple[int, int],
    p_star: float,
    output_path: str | None = None,
) -> str:
    """绘制临界构型整体图与临界板凳对局部放大图。

    Args:
        positions: 224 个把手中心坐标。
        critical_pair: 临界接触板凳对 (i*, j*)（1-based 编号）。
        p_star: 最优螺距（m）。
        output_path: 输出图片路径，默认 problem3_critical_config.png。

    Returns:
        输出图片的绝对路径。
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "problem3_critical_config.png")

    from src.models.problem3 import critical_rectangles

    rects = critical_rectangles(p_star, TURN_RADIUS)
    P = np.asarray(positions, dtype=float)

    # 调头空间边界圆
    theta_circle = np.linspace(0.0, 2.0 * np.pi, 200)
    cx = TURN_RADIUS * np.cos(theta_circle)
    cy = TURN_RADIUS * np.sin(theta_circle)

    def draw_rect(ax, rect, color="C0", alpha=0.8, lw=1.0):
        C, e, n, a, c = rect[:2], rect[2:4], rect[4:6], rect[6], rect[7]
        corners = np.array([
            C + a * e + c * n,
            C + a * e - c * n,
            C - a * e - c * n,
            C - a * e + c * n,
            C + a * e + c * n,
        ])
        ax.plot(corners[:, 0], corners[:, 1], color=color, lw=lw, alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))

    # ---- 左：整体构型 ----
    ax = axes[0]
    for k in range(rects.shape[0]):
        if k + 1 in critical_pair:
            draw_rect(ax, rects[k], color="r", lw=1.6)
        else:
            draw_rect(ax, rects[k], color="#9ecae1", lw=0.5, alpha=0.7)
    ax.plot(P[:, 0], P[:, 1], "-", color="k", lw=0.6, alpha=0.5)
    ax.plot(cx, cy, "--", color="b", lw=1.2)
    ax.scatter([P[0, 0]], [P[0, 1]], color="r", s=30, zorder=5)
    ax.annotate("龙头前把手", xy=(P[0, 0], P[0, 1]),
                xytext=(P[0, 0] + 0.6, P[0, 1] + 0.6),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)

    # ---- 右：临界板凳对局部放大 ----
    ax2 = axes[1]
    i, j = critical_pair
    ri = rects[i - 1]
    rj = rects[j - 1]
    draw_rect(ax2, ri, color="r", lw=1.8)
    draw_rect(ax2, rj, color="#2ca02c", lw=1.8)
    ax2.annotate(f"板凳 {i}", xy=ri[:2], xytext=(ri[0], ri[1] + 0.15),
                 ha="center", fontsize=9, color="r")
    ax2.annotate(f"板凳 {j}", xy=rj[:2], xytext=(rj[0], rj[1] - 0.18),
                 ha="center", fontsize=9, color="#2ca02c")
    # 包围盒局部范围
    pts = np.vstack([
        ri[:2] + ri[6] * ri[2:4] + ri[7] * ri[4:6],
        ri[:2] + ri[6] * ri[2:4] - ri[7] * ri[4:6],
        ri[:2] - ri[6] * ri[2:4] - ri[7] * ri[4:6],
        ri[:2] - ri[6] * ri[2:4] + ri[7] * ri[4:6],
        rj[:2] + rj[6] * rj[2:4] + rj[7] * rj[4:6],
        rj[:2] + rj[6] * rj[2:4] - rj[7] * rj[4:6],
        rj[:2] - rj[6] * rj[2:4] - rj[7] * rj[4:6],
        rj[:2] - rj[6] * rj[2:4] + rj[7] * rj[4:6],
    ])
    xmin, xmax = pts[:, 0].min() - 0.1, pts[:, 0].max() + 0.1
    ymin, ymax = pts[:, 1].min() - 0.1, pts[:, 1].max() + 0.1
    span = max(xmax - xmin, ymax - ymin)
    xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    ax2.set_xlim(xmid - span / 2, xmid + span / 2)
    ax2.set_ylim(ymid - span / 2, ymid + span / 2)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_aspect("equal")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.set_title(f"临界板凳对 ({i}, {j}) 局部放大", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
