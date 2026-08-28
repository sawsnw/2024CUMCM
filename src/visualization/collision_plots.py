"""问题二可视化模块：全局碰撞裕量 G(t) 曲线绘制。

绘图约定见 src/visualization/约定.md：
    - 绘图文字使用中文；
    - 绘图不在图中加图名，只通过文件名体现。
"""

import os

import matplotlib

matplotlib.use("Agg")  # 无界面后端，供入口脚本调用

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR

# 中文字体（Windows 常见字体，按顺序回退）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def plot_global_margin(
    trace: list[tuple[float, float, int, int]],
    t_star: float,
    pair_star: tuple[int, int],
    fine_points: list[tuple[float, float]] | None = None,
    output_path: str | None = None,
) -> str:
    """绘制全局碰撞裕量 G(t) 随时间变化曲线。

    主图使用一秒步长粗搜索轨迹；若提供加密扫描点，则在右侧绘制终止时刻
    附近的局部放大图，标注首次接触时刻与接触板凳对。

    Args:
        trace: 粗搜索轨迹，元素为 (t, G, i_star, j_star)。
        t_star: 盘入终止时刻（s）。
        pair_star: 终止时刻最危险板凳对 (i*, j*)。
        fine_points: 加密扫描点列表 [(t, G), ...]，用于局部放大图（可选）。
        output_path: 输出图片路径，默认 results/figures/problem2_global_margin.png。

    Returns:
        输出图片的绝对路径。
    """
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "problem2_global_margin.png")

    ts = np.array([p[0] for p in trace], dtype=float)
    gs = np.array([p[1] for p in trace], dtype=float)

    n_panels = 2 if fine_points else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(12.5 if n_panels == 2 else 7, 4.6)
    )
    ax_main = axes[0] if n_panels == 2 else axes

    # ---- 主图：全程 G(t)（一秒步长采样） ----
    ax_main.plot(ts, gs, "-o", ms=3, lw=1.2, color="#1f77b4",
                 label="一秒步长采样")
    ax_main.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.7)
    ax_main.axvline(t_star, color="r", lw=1.2, ls="--", alpha=0.8,
                    label=f"t* = {t_star:.3f} s")
    ax_main.scatter([t_star], [0.0], color="r", s=40, zorder=5)
    ax_main.annotate(
        f"首次接触 ({pair_star[0]}, {pair_star[1]})",
        xy=(t_star, 0.0),
        xytext=(t_star * 0.38, float(np.max(gs)) * 0.62),
        arrowprops=dict(arrowstyle="->", color="r", lw=1.0),
        color="r", fontsize=9,
    )
    ax_main.set_xlabel("时间 t (s)")
    ax_main.set_ylabel("全局碰撞裕量 G(t) (m)")
    ax_main.legend(fontsize=9, loc="upper right")
    ax_main.grid(True, ls=":", alpha=0.5)
    ax_main.set_xlim(float(ts.min()), float(ts.max()))
    y_min = min(float(gs.min()), -0.01)
    ax_main.set_ylim(y_min, float(gs.max()) * 1.05)

    # ---- 局部放大图：终止时刻附近（加密扫描） ----
    if fine_points and n_panels == 2:
        fts = np.array([p[0] for p in fine_points], dtype=float)
        fgs = np.array([p[1] for p in fine_points], dtype=float)
        ax_fine = axes[1]
        ax_fine.plot(fts, fgs, "-o", ms=2.5, lw=1.0, color="#d62728")
        ax_fine.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.7)
        ax_fine.axvline(t_star, color="r", lw=1.2, ls="--", alpha=0.8)
        ax_fine.annotate(
            f"t* = {t_star:.3f} s",
            xy=(t_star, 0.0),
            xytext=(t_star - (fts.max() - fts.min()) * 0.45,
                    float(fgs.max()) * 0.7),
            arrowprops=dict(arrowstyle="->", color="r", lw=1.0),
            color="r", fontsize=9,
        )
        ax_fine.set_xlabel("时间 t (s)")
        ax_fine.set_ylabel("G(t) (m)")
        ax_fine.set_title("终止时刻附近（加密扫描）", fontsize=10)
        ax_fine.grid(True, ls=":", alpha=0.5)
        ax_fine.set_xlim(float(fts.min()), float(fts.max()))
        pad = max(abs(float(fgs.max())), abs(float(fgs.min()))) * 0.25 + 1e-9
        ax_fine.set_ylim(float(fgs.min()) - pad, float(fgs.max()) + pad)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
