"""问题四路线图绘制脚本：S 形调头路线。

运行方式:
    python scripts/plot_problem4_route.py

功能:
    1. 求解问题四“给定配置”的 S 形调头曲线几何（两段圆弧与调头空间圆内切、
       与盘入/盘出螺线相切）；
    2. 绘制双面板路线图（整体路线 + 调头空间放大），输出到
       results/figures/problem4_route.png（覆盖旧文件）；
    3. 输出几何参数表格到 results/tables/problem4_route.xlsx（覆盖旧文件）；
    4. 日志写入 results/logs/plot_problem4_route_YYYY-MM-DD_HHMMSS.log（不覆盖旧文件）。

几何模型见 src/models/problem4.py，文档见 docs/problem_analysis/问题四_模型建立与求解.md。
"""

import os
import time

import numpy as np
import openpyxl

from src.config import P4_ROUTE_FIG
from src.models.problem4 import (
    ORIENTATION_MINUS,
    solve_given_configuration,
    s_curve_points,
    turning_length,
)
from src.utils.logger import setup_logger
from src.visualization.problem4_route import plot_problem4_route

# 几何参数表格输出路径
ROUTE_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "tables",
    "problem4_route.xlsx",
)


def save_route_table(g: dict, output_path: str) -> None:
    """将 S 形调头曲线几何参数保存到 xlsx 表格（覆盖旧文件）。

    Args:
        g: solve_s_curve 返回的几何 dict。
        output_path: 输出表格路径。
    """
    R = g["R"]
    L = turning_length(g)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "几何参数"
    ws.append(["项目", "符号", "数值", "单位"])
    ws.append(["盘入螺线螺距", "p", 1.7, "m"])
    ws.append(["调头空间半径", "ρ", 4.5, "m"])
    ws.append(["前段圆弧半径", "2R", round(2 * R, 6), "m"])
    ws.append(["后段圆弧半径", "R", round(R, 6), "m"])
    ws.append(["切点 A 参数", "θa", round(g["theta_a"], 9), "rad"])
    ws.append(["切点 C 参数", "θc", round(g["theta_c"], 9), "rad"])
    for name, key in [("切点 A", "A"), ("切点 B", "B"), ("切点 C", "C")]:
        pt = g[key]
        ws.append([name + " 坐标", f"({pt[0]:.6f}, {pt[1]:.6f})", "", "m"])
    for name, key in [("前段圆心 O1", "c1"), ("后段圆心 O2", "c2")]:
        pt = g[key]
        ws.append([name, f"({pt[0]:.6f}, {pt[1]:.6f})", "", "m"])
    ws.append(["调头曲线总长", "L", round(L, 6), "m"])
    wb.save(output_path)


def main() -> None:
    t0 = time.time()
    logger = setup_logger(__name__, script_name="plot_problem4_route")
    logger.info("问题四：S 形调头路线图绘制开始")

    # 1. 求解给定配置
    logger.info("步骤 1：求解给定配置（两圆弧与调头空间圆内切）")
    state, f, g = solve_given_configuration(
        theta_a0=16.5715, theta_c0=16.6169, orientation=ORIENTATION_MINUS
    )
    logger.info("    牛顿迭代收敛：θa=%.9f rad，θc=%.9f rad，残差=(%.2e, %.2e)",
                state[0], state[1], f[0], f[1])
    logger.info("    R=%.6f m（前段 2R=%.6f m，后段 R=%.6f m）", g["R"], 2 * g["R"], g["R"])
    logger.info("    |A|=%.6f m，|C|=%.6f m，|B|=%.6f m",
                np.linalg.norm(g["A"]), np.linalg.norm(g["C"]), np.linalg.norm(g["B"]))
    logger.info("    调头曲线总长 L=%.6f m", turning_length(g))

    # 2. 绘制路线图
    logger.info("步骤 2：绘制 S 形调头路线图")
    out = plot_problem4_route(g, output_path=P4_ROUTE_FIG)
    logger.info("    路线图已保存：%s", out)

    # 3. 保存几何参数表格
    logger.info("步骤 3：保存几何参数表格")
    save_route_table(g, ROUTE_TABLE_PATH)
    logger.info("    表格已保存：%s", ROUTE_TABLE_PATH)

    logger.info("完成，总耗时 %.2f s", time.time() - t0)


if __name__ == "__main__":
    main()
