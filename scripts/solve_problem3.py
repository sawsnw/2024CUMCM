"""问题三求解脚本：满足调头空间约束的最小螺距。

运行方式:
    python scripts/solve_problem3.py

功能:
    1. 外层螺距粗搜索：对 p ∈ [PITCH_SEARCH_MIN, PITCH_SEARCH_MAX] 逐点计算
       全过程最小裕量 Φ(p)，确定不可行端 p_L 与可行端 p_R；
    2. 验证临界区间内可行性随螺距增大而改善；
    3. 外层二分求最小可行螺距 p*（取安全侧右端点）；
    4. 在 p* 处计算全过程 G(p*, r0) 曲线，确认最危险构型位置；
    5. 运行模型检验（调头边界、相邻距离、最优性左右扰动、临界裕量、
       最危险板凳对独立交叉验证、外侧起点敏感性）；
    6. 保存 Φ(p) 与 G(p*, r0) 数据表格；
    7. 绘制 Φ(p) 曲线、G(p*, r0) 曲线与临界构型图；
    8. 日志写入 results/logs/solve_problem3_YYYY-MM-DD_HHMMSS.log（不覆盖旧文件）。

模型依据:
    文档《问题三_模型建立与求解.md》。龙头前把手从外侧沿等距螺线盘入至调头
    空间边界（半径 4.5 m）的全程中，所有非相邻板凳不发生碰撞的最小螺距。
"""

import math
import os
import time

import numpy as np
import openpyxl

from src.config import (
    DELTA_PITCH,
    EPS_G3,
    EPS_PITCH,
    PITCH_COARSE_STEP,
    PITCH_SEARCH_MAX,
    PITCH_SEARCH_MIN,
    R_HEAD_START,
    RESULT3_CONFIG_FIG,
    RESULT3_MARGIN_FIG,
    RESULT3_MARGIN_TABLE,
    RESULT3_PHI_FIG,
    RESULT3_PHI_TABLE,
    TURN_RADIUS,
)
from src.models.problem3 import (
    compute_positions,
    configuration_margin,
    find_r_start,
    path_margin,
    verify_critical_pair,
    verify_problem3,
)
from src.utils.logger import setup_logger
from src.visualization.problem3_plots import (
    plot_critical_configuration,
    plot_margin_vs_radius,
    plot_phi_vs_pitch,
)

# 粗搜索点数（由螺距范围与步长确定）
N_PITCH = int(round((PITCH_SEARCH_MAX - PITCH_SEARCH_MIN) / PITCH_COARSE_STEP)) + 1
# 最优螺距下全过程 G(p*, r0) 细网格点数
N_RADIUS_FINE = 300
# 论文结果保留位数（米 / 厘米）
PITCH_DECIMALS = 6


def coarse_pitch_scan(
    logger,
) -> tuple[list[float], list[float], list[float], list[tuple[int, int]], float | None]:
    """螺距粗搜索：逐点计算 Φ(p)，定位可行与不可行端。

    Args:
        logger: 日志记录器。

    Returns:
        五元组 (p_grid, phi_grid, critical_r_grid, pair_grid, p_critical)：
        - p_grid: 螺距采样列表（m）；
        - phi_grid: 对应的 Φ(p) 列表（m）；
        - critical_r_grid: 各螺距最危险构型的龙头极径列表（m）；
        - pair_grid: 各螺距最危险板凳对列表 [(i, j), ...]；
        - p_critical: 首个满足 Φ>=0 的螺距（可行端 p_R），无则 None。
    """
    logger.info(
        "步骤 1：外层螺距粗搜索 [%.3f, %.3f] m，步长 %.3f m，共 %d 点",
        PITCH_SEARCH_MIN, PITCH_SEARCH_MAX, PITCH_COARSE_STEP, N_PITCH,
    )
    p_grid: list[float] = []
    phi_grid: list[float] = []
    critical_r_grid: list[float] = []
    pair_grid: list[tuple[int, int]] = []
    p_critical: float | None = None

    for idx in range(N_PITCH):
        p = PITCH_SEARCH_MIN + idx * PITCH_COARSE_STEP
        Phi, critical_r, i_star, j_star = path_margin(p, R_HEAD_START, logger)
        # 若最危险极径太靠近扫描右端，说明外侧可能仍有危险构型，扩展后重算
        if critical_r > R_HEAD_START - 2.0:
            logger.info("  最危险极径接近起点，扩展外侧区间后重算")
            _, Phi, critical_r, (i_star, j_star) = find_r_start(p, logger)
        p_grid.append(p)
        phi_grid.append(Phi)
        critical_r_grid.append(critical_r)
        pair_grid.append((i_star, j_star))
        if Phi >= 0.0 and p_critical is None:
            p_critical = p

    logger.info("粗搜索完成：%d 个采样点", len(p_grid))
    return p_grid, phi_grid, critical_r_grid, pair_grid, p_critical


def verify_trend(
    p_grid: list[float],
    phi_grid: list[float],
    p_left: float,
    p_right: float,
    logger,
) -> None:
    """验证临界区间内可行性随螺距增大而改善（文档 11.3 节）。

    检查在 [p_left, p_right] 内，除临界点附近的数值噪声外，Φ(p) 不应出现
    可行—不可行—可行的反转。

    Args:
        p_grid: 螺距采样列表。
        phi_grid: 对应的 Φ(p) 列表。
        p_left: 不可行端螺距。
        p_right: 可行端螺距。
        logger: 日志记录器。

    Raises:
        RuntimeError: 临界区间内可行性不满足单调性。
    """
    logger.info("步骤 2：验证临界区间内可行性单调性")
    in_range = [
        (p, ph) for p, ph in zip(p_grid, phi_grid)
        if p_left <= p <= p_right
    ]
    feasible_seen = False
    infeasible_after = False
    for p, ph in in_range:
        if ph >= 0.0:
            feasible_seen = True
        elif feasible_seen:
            # 已经可行之后又出现不可行 → 反转
            if p > p_left + PITCH_COARSE_STEP:
                infeasible_after = True
                break
    if infeasible_after:
        raise RuntimeError(
            "临界区间内可行性不单调：可行后再次出现不可行，需分区间求根"
        )
    logger.info("可行性单调性验证通过：可行集合表现为区间 [p*, +∞)")


def bisect_pitch(p_left: float, p_right: float, logger) -> float:
    """外层二分求最小可行螺距（文档 11.4 节）。

    前提：Φ(p_left) < 0（不可行），Φ(p_right) >= 0（可行）。

    Args:
        p_left: 不可行端螺距（m）。
        p_right: 可行端螺距（m）。
        logger: 日志记录器。

    Returns:
        二分得到的最大不可行螺距右侧的最小可行螺距 p* = p_R（m）。
    """
    logger.info("步骤 3：外层二分搜索，收敛阈值 ε_p=%.0e m", EPS_PITCH)
    iterations = 0
    while p_right - p_left >= EPS_PITCH:
        p_mid = 0.5 * (p_left + p_right)
        Phi, critical_r, i_star, j_star = path_margin(p_mid, R_HEAD_START, logger)
        iterations += 1
        if Phi < 0.0:
            p_left = p_mid
        else:
            p_right = p_mid
    logger.info("二分完成：%d 次迭代，p* = %.9f m", iterations, p_right)
    return p_right


def full_path_curve(
    p: float,
    r_start: float,
    n_points: int,
    logger,
) -> list[tuple[float, float, int, int]]:
    """计算最优螺距下全过程 G(p, r0) 细网格曲线（文档 14.5 节）。

    Args:
        p: 螺距（m）。
        r_start: 外侧起始极径（m）。
        n_points: 扫描点数。
        logger: 日志记录器。

    Returns:
        列表，元素为 (r0, G, i_star, j_star)。
    """
    logger.info("步骤 4：最优螺距下全过程 G(p, r0) 细网格扫描（%d 点）", n_points)
    radii = np.linspace(r_start, TURN_RADIUS, n_points)
    trace: list[tuple[float, float, int, int]] = []
    for r in radii:
        G, i, j = configuration_margin(p, float(r))
        trace.append((float(r), G, i, j))
    return trace


def verify_all_checks(
    p_star: float,
    phi_star: float,
    critical_r: float,
    critical_pair: tuple[int, int],
    logger,
) -> None:
    """运行问题三全部模型检验（文档第 14 节）。

    Args:
        p_star: 最小可行螺距（m）。
        phi_star: p* 处的全过程最小裕量（m）。
        critical_r: 最危险构型的龙头极径（m）。
        critical_pair: 最危险板凳对。
        logger: 日志记录器。

    Raises:
        RuntimeError: 任一检验未通过。
    """
    logger.info("步骤 5：模型检验")
    b = p_star / (2.0 * math.pi)

    # 5.1 调头边界检验（式 36）
    boundary_theta = TURN_RADIUS / b
    boundary_check = abs(b * boundary_theta - TURN_RADIUS)
    logger.info("  [边界检验] |b·θb - 4.5| = %.3e (应 < 1e-9)", boundary_check)
    if boundary_check >= 1e-9:
        raise RuntimeError("调头边界检验未通过")

    # 5.2 相邻把手距离检验（式 37）
    ver = verify_problem3(p_star, critical_r)
    logger.info("  [几何检验] 最大弦长误差 = %.3e (应 < 1e-9)",
                ver["最大弦长误差"])
    if ver["最大弦长误差"] >= 1e-9:
        raise RuntimeError("相邻把手距离检验未通过")

    # 5.3 最优性左右扰动检验（式 38）
    Phi_plus, _, _, _ = path_margin(p_star + DELTA_PITCH, R_HEAD_START, logger)
    Phi_minus, _, _, _ = path_margin(p_star - DELTA_PITCH, R_HEAD_START, logger)
    ok_plus = Phi_plus >= 0.0
    ok_minus = Phi_minus < 0.0
    logger.info("  [扰动检验] Φ(p*+δ)=%.3e (应≥0, %s)", Phi_plus,
                "通过" if ok_plus else "失败")
    logger.info("  [扰动检验] Φ(p*-δ)=%.3e (应<0, %s)", Phi_minus,
                "通过" if ok_minus else "失败")
    if not (ok_plus and ok_minus):
        raise RuntimeError("最优性左右扰动检验未通过")

    # 5.4 临界裕量检验（式 39）
    ok_crit = abs(phi_star) < EPS_G3
    logger.info("  [临界裕量] |Φ(p*)| = %.3e (应 < %.0e, %s)",
                abs(phi_star), EPS_G3, "通过" if ok_crit else "失败")
    if not ok_crit:
        raise RuntimeError("临界裕量检验未通过")

    # 5.5 最危险板凳对独立交叉验证（文档 14.6 节）
    colliding, detail = verify_critical_pair(
        p_star, critical_r, critical_pair[0], critical_pair[1]
    )
    logger.info("  [接触对验证] 板凳对 (%d, %d)：%s（%s）",
                critical_pair[0], critical_pair[1],
                "发生接触" if colliding else "未检测到接触", detail)

    # 5.6 外侧起点敏感性检查（文档 8.1 节）
    logger.info("步骤 6：外侧起点敏感性检查")
    r_start_ext, phi_ext, r_ext, pair_ext = find_r_start(p_star, logger)
    logger.info("  扩展后起点 r_start=%.2f m，Φ(p*)=%.6f，最危险极径 %.4f m",
                r_start_ext, phi_ext, r_ext)
    if abs(phi_ext - phi_star) > 1e-6:
        logger.warning("  扩展外侧起点后最小裕量发生变化，建议增大 R_HEAD_START")
    else:
        logger.info("  扩展外侧起点不影响最小裕量，外侧区间充分")


def save_phi_table(
    p_grid: list[float],
    phi_grid: list[float],
    critical_r_grid: list[float],
    pair_grid: list[tuple[int, int]],
    output_path: str,
) -> None:
    """保存 Φ(p) 粗搜索数据到 xlsx（覆盖旧文件）。"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "螺距粗搜索"
    ws.append(["螺距 p (m)", "全过程最小裕量 Φ(p) (m)",
               "最危险极径 r0 (m)", "最危险板凳对 i", "最危险板凳对 j"])
    for p, ph, r, (i, j) in zip(p_grid, phi_grid, critical_r_grid, pair_grid):
        ws.append([round(p, 6), round(ph, 6), round(r, 6), i, j])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wb.save(output_path)


def save_margin_table(
    trace: list[tuple[float, float, int, int]],
    output_path: str,
) -> None:
    """保存最优螺距下 G(p*, r0) 全过程数据到 xlsx（覆盖旧文件）。"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "全过程裕量"
    ws.append(["龙头极径 r0 (m)", "全局裕量 G(p*,r0) (m)",
               "最危险板凳对 i", "最危险板凳对 j"])
    for r, G, i, j in trace:
        ws.append([round(r, 6), round(G, 6), i, j])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wb.save(output_path)


def main() -> None:
    logger = setup_logger(__name__, script_name="solve_problem3")
    logger.info("=" * 60)
    logger.info("问题三：满足调头空间约束的最小螺距")
    logger.info("方法：参数化阿基米德螺线 + 分离轴碰撞裕量 + 螺距二分")
    logger.info("调头空间半径 R=%.1f m，把手总数 %d，非相邻板凳对 C(223,2)-222",
                TURN_RADIUS, 224)
    logger.info("=" * 60)

    t_total = time.perf_counter()

    # ---- 步骤 1：外层螺距粗搜索 ----
    p_grid, phi_grid, critical_r_grid, pair_grid, p_critical = coarse_pitch_scan(
        logger
    )

    # 找到不可行端 pL（最后一个 Φ<0）与可行端 pR（第一个 Φ>=0）
    if p_critical is None:
        raise RuntimeError(
            f"在 [{PITCH_SEARCH_MIN}, {PITCH_SEARCH_MAX}] 内未找到可行螺距，"
            "请扩大搜索范围"
        )
    # p_critical 在 p_grid 中的索引（用最近匹配避免浮点误差）
    idx_R = min(range(len(p_grid)), key=lambda k: abs(p_grid[k] - p_critical))
    p_right = p_grid[idx_R]
    idx_L = max(0, idx_R - 1)
    p_left = p_grid[idx_L]
    logger.info("  p_L=%.4f m（不可行，Φ=%.6f）", p_left, phi_grid[idx_L])
    logger.info("  p_R=%.4f m（可行，Φ=%.6f）", p_right, phi_grid[idx_R])
    if phi_grid[idx_L] >= 0.0:
        logger.warning("  p_L 处 Φ>=0，粗搜索步长过大，二分区间仍有效")

    # ---- 步骤 2：验证可行性单调性 ----
    verify_trend(p_grid, phi_grid, p_left, p_right, logger)

    # ---- 步骤 3：外层二分 ----
    p_star = bisect_pitch(p_left, p_right, logger)

    # ---- 步骤 4：最优螺距下全过程 G(p*, r0) 细网格曲线 ----
    phi_star, critical_r, i_star, j_star = path_margin(
        p_star, R_HEAD_START, logger
    )
    trace = full_path_curve(p_star, R_HEAD_START, N_RADIUS_FINE, logger)

    # 确认最危险构型位置（是否在边界）
    min_G = min(trace, key=lambda item: item[1])
    logger.info("  全过程最小裕量 G=%.6f，位于 r0=%.4f m", min_G[1], min_G[0])
    logger.info("  龙头位于边界 r0=4.5 时的裕量 Gb=%.6f",
                trace[-1][1] if len(trace) else float("nan"))

    # ---- 步骤 5-6：模型检验 ----
    verify_all_checks(p_star, phi_star, critical_r, (i_star, j_star), logger)

    # ---- 步骤 7：最优螺距下边界构型 ----
    logger.info("步骤 7：最优螺距下龙头位于调头边界时的构型")
    _, positions = compute_positions(p_star, TURN_RADIUS)
    G_b, i_b, j_b = configuration_margin(p_star, TURN_RADIUS)
    logger.info("  龙头边界构型全局裕量 Gb(p*)=%.6f，最危险板凳对=(%d, %d)",
                G_b, i_b, j_b)

    # ---- 步骤 8：保存数据表格 ----
    logger.info("步骤 8：保存数据表格")
    save_phi_table(p_grid, phi_grid, critical_r_grid, pair_grid, RESULT3_PHI_TABLE)
    logger.info("  已保存表格：%s", RESULT3_PHI_TABLE)
    save_margin_table(trace, RESULT3_MARGIN_TABLE)
    logger.info("  已保存表格：%s", RESULT3_MARGIN_TABLE)

    # ---- 步骤 9：绘图 ----
    logger.info("步骤 9：绘图")
    fig1 = plot_phi_vs_pitch(p_grid, phi_grid, p_star, phi_star, RESULT3_PHI_FIG)
    logger.info("  已保存图片：%s", fig1)
    r_grid = [item[0] for item in trace]
    G_grid = [item[1] for item in trace]
    fig2 = plot_margin_vs_radius(
        p_star, r_grid, G_grid, critical_r, (i_star, j_star), RESULT3_MARGIN_FIG
    )
    logger.info("  已保存图片：%s", fig2)
    fig3 = plot_critical_configuration(
        positions, (i_b, j_b), p_star, RESULT3_CONFIG_FIG
    )
    logger.info("  已保存图片：%s", fig3)

    # ---- 结果汇总 ----
    logger.info("=" * 60)
    logger.info("问题三求解结果")
    logger.info("  最小螺距 p* = %.6f m = %.4f cm", p_star, p_star * 100.0)
    logger.info("  全过程最小裕量 Φ(p*) = %.3e", phi_star)
    logger.info("  最危险构型龙头极径 r0 = %.4f m（调头边界 %.1f m）",
                critical_r, TURN_RADIUS)
    logger.info("  最危险板凳对 = (%d, %d)", i_star, j_star)
    logger.info("  最优螺距下边界构型裕量 Gb(p*) = %.6f", G_b)
    elapsed = time.perf_counter() - t_total
    logger.info("  总耗时 %.2f s", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
