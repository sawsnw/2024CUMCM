"""问题三：满足调头空间约束的最小螺距模型。

实现文档《问题三_模型建立与求解.md》第 5~11 节的候选螺距下全部把手位置递推、
全局碰撞裕量与全过程安全裕量计算。

模型要点：
    1. 螺距 p 为决策变量，螺线参数 b = p/(2π)，螺线方程 r = b·θ（式 3）；
    2. 以龙头前把手极径 r0 描述盘入过程，θ0 = 2π·r0/p（式 4）；
    3. 相邻把手保持固定弦长，由式 (8) 逐节递推取第一个外侧根确定 θi；
    4. 复用问题二路径无关的有向矩形分离轴碰撞模型（src/models/collision.py）
       计算全局碰撞裕量 G(p, r0)（式 20）；
    5. 内层对 r0 ∈ [4.5, r_start] 扫描取最小值得到全过程裕量 Φ(p)（式 23）。

本模块与 src/models/spiral_dragon.py 的区别：spiral_dragon.py 使用固定的
SPIRAL_B（问题一/二专用），本模块把螺线参数 b 显式作为函数参数，供问题三
在螺距方向上进行优化扫描。
"""

import math

import numpy as np

from src.config import (
    BODY_HOLE_DISTANCE,
    HEAD_HOLE_DISTANCE,
    R_COARSE_POINTS,
    R_FINE_POINTS,
    R_FINE_STEP,
    TOTAL_HANDLES,
    TURN_RADIUS,
)
from src.models.collision import (
    build_rectangles,
    global_margin,
    pair_margins,
    verify_pair_independent,
)
from src.models.spiral_dragon import handle_distances


def spiral_point(theta: float, b: float) -> tuple[float, float]:
    """计算螺线参数为 b 的等距螺线上参数 θ 对应点的坐标（式 3）。

    Args:
        theta: 螺线参数（极角，弧度）。
        b: 螺线参数 b = p/(2π)（m/rad）。

    Returns:
        (x, y) 坐标，单位 m。
    """
    return (b * theta * math.cos(theta), b * theta * math.sin(theta))


def _chord_error(theta: float, theta_prev: float, b: float, distance: float) -> float:
    """相邻把手距离方程左端：b²[θ² + θ₀² - 2θθ₀cos(θ - θ₀)] - d²（式 8）。

    Args:
        theta: 待求把手的螺线参数（弧度）。
        theta_prev: 前一把手的螺线参数（弧度）。
        b: 螺线参数（m/rad）。
        distance: 固定弦长（m）。

    Returns:
        方程残差；返回 0 时两点弦长恰好等于 distance。
    """
    return (
        b**2
        * (
            theta**2
            + theta_prev**2
            - 2.0 * theta * theta_prev * math.cos(theta - theta_prev)
        )
        - distance**2
    )


def solve_next_theta(theta_prev: float, distance: float, b: float) -> float:
    """求与前一把手相距 distance 的外侧第一个把手的螺线参数（式 8、9）。

    以弧长近似式给出初值，再沿参数增大方向搜索第一个符号变化区间，
    最后在区间内二分求根，取最靠近 θ_{i-1} 的外侧交点。

    Args:
        theta_prev: 前一把手的螺线参数（弧度）。
        distance: 相邻把手之间的固定弦长（m）。
        b: 螺线参数（m/rad）。

    Returns:
        后一把手的螺线参数（弧度），满足 θ > theta_prev。

    Raises:
        RuntimeError: 未能在合理参数范围内找到外侧根。
    """

    def error(th: float) -> float:
        return _chord_error(th, theta_prev, b, distance)

    # 弧长近似初值：因弦长小于弧长，真实根略大于该初值
    theta_est = theta_prev + distance / (b * math.sqrt(1.0 + theta_prev**2))

    lo, hi = theta_prev, theta_est
    if error(hi) < 0.0:
        # 初值处仍未越过根，沿参数增大方向搜索符号变化区间
        step = 1e-3
        count = 0
        while error(hi) < 0.0:
            lo = hi
            hi += step
            count += 1
            if count > 10_000_000:
                raise RuntimeError("未能在合理范围内找到外侧根")

    # 在 [lo, hi] 内二分求根（error(lo) <= 0，error(hi) >= 0）
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if error(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def compute_positions(
    p: float, r_head: float
) -> tuple[list[float], list[tuple[float, float]]]:
    """给定螺距 p 与龙头极径 r0，递推全部 224 个把手的位置（式 10）。

    Args:
        p: 螺距（m）。
        r_head: 龙头前把手极径 r0（m）。

    Returns:
        二元组 (thetas, positions)：
        - thetas: 224 个把手的螺线参数列表（弧度）；
        - positions: 224 个把手中心坐标列表。
    """
    b = p / (2.0 * math.pi)
    distances = handle_distances()

    thetas = [r_head / b]
    for i in range(1, TOTAL_HANDLES):
        thetas.append(solve_next_theta(thetas[i - 1], distances[i], b))
    positions = [spiral_point(th, b) for th in thetas]
    return thetas, positions


def configuration_margin(p: float, r_head: float) -> tuple[float, int, int]:
    """计算给定螺距与龙头极径下的全局碰撞裕量（式 20~22）。

    Args:
        p: 螺距（m）。
        r_head: 龙头前把手极径 r0（m）。

    Returns:
        三元组 (G, i_star, j_star)：
        - G: 全局碰撞裕量（m），正为分离、零为接触、负为重叠；
        - i_star, j_star: 最危险板凳对（1-based 板凳编号）。
    """
    _, positions = compute_positions(p, r_head)
    return global_margin(positions)


def _scan_radius_range(
    p: float,
    r_hi: float,
    r_lo: float,
    n_points: int,
) -> list[tuple[float, float, int, int]]:
    """对龙头极径 r0 在 [r_lo, r_hi] 内等距扫描，返回裕量轨迹。

    Args:
        p: 螺距（m）。
        r_hi: 扫描区间上界（外侧，m）。
        r_lo: 扫描区间下界（内侧，m）。
        n_points: 扫描点数（含两端）。

    Returns:
        列表，元素为 (r0, G, i_star, j_star)。
    """
    radii = np.linspace(r_hi, r_lo, n_points)
    trace: list[tuple[float, float, int, int]] = []
    for r in radii:
        G, i, j = configuration_margin(p, float(r))
        trace.append((float(r), G, i, j))
    return trace


def path_margin(
    p: float,
    r_start: float,
    logger=None,
) -> tuple[float, float, int, int]:
    """计算螺距 p 下盘入全过程的最小安全裕量 Φ(p)（式 23）。

    流程：先对 r0 ∈ [4.5, r_start] 做粗扫描定位最危险构型，再在最危险极径
    附近两侧用细步长局部加密，避免遗漏粗网格之间的局部最小值（文档 11.2 节）。

    Args:
        p: 螺距（m）。
        r_start: 外侧起始极径（m）。
        logger: 日志记录器（可选）。

    Returns:
        四元组 (Phi, critical_r, i_star, j_star)：
        - Phi: 全过程最小裕量 Φ(p)（m）；
        - critical_r: 取得最小裕量的龙头极径（m）；
        - i_star, j_star: 最危险构型对应的板凳对（1-based 编号）。
    """
    # 第一阶段：粗扫描定位最危险极径
    coarse = _scan_radius_range(p, r_start, TURN_RADIUS, R_COARSE_POINTS)
    k_min = min(range(len(coarse)), key=lambda k: coarse[k][1])
    r_coarse = coarse[k_min][0]
    G_coarse = coarse[k_min][1]

    # 第二阶段：在粗扫描最危险极径附近两侧局部细化
    delta = max((r_start - TURN_RADIUS) / (R_COARSE_POINTS - 1), R_FINE_STEP)
    r_lo = max(TURN_RADIUS, r_coarse - delta)
    r_hi = min(r_start, r_coarse + delta)
    fine = _scan_radius_range(p, r_hi, r_lo, R_FINE_POINTS)

    # 合并两阶段轨迹后取全局最小（细扫描优先）
    best = min(fine, key=lambda item: item[1])
    if coarse[k_min][1] < best[1]:
        best = (r_coarse, G_coarse, coarse[k_min][2], coarse[k_min][3])

    Phi, critical_r, i_star, j_star = best[1], best[0], best[2], best[3]
    if logger is not None:
        logger.info(
            "  p=%7.4f m  Φ=%.6f  最危险极径 r0=%.4f m  板凳对=(%d, %d)",
            p, Phi, critical_r, i_star, j_star,
        )
    return Phi, critical_r, i_star, j_star


def find_r_start(
    p: float, logger=None
) -> tuple[float, float, float, tuple[int, int]]:
    """确定螺距 p 下足够大的外侧起始极径（文档 8.1 节）。

    从初值 R_HEAD_START 出发，若最危险极径接近扫描区间右端（即外侧可能
    存在更危险构型），则不断扩展起点重新计算，直到继续扩展不再改变最小裕量
    或达到 R_HEAD_MAX 上限。

    Args:
        p: 螺距（m）。
        logger: 日志记录器（可选）。

    Returns:
        四元组 (r_start, Phi, critical_r, critical_pair)：
        - r_start: 收敛后的外侧起始极径（m）；
        - Phi: 该起点下全过程最小裕量（m）；
        - critical_r: 最危险构型的龙头极径（m）；
        - critical_pair: 最危险板凳对 (i, j)（1-based 编号）。
    """
    from src.config import R_HEAD_MAX, R_HEAD_START

    r_start = R_HEAD_START
    prev_phi: float | None = None
    while r_start <= R_HEAD_MAX:
        Phi, critical_r, i_star, j_star = path_margin(p, r_start, logger)
        pair = (i_star, j_star)
        if prev_phi is not None and abs(Phi - prev_phi) < 1e-10:
            return r_start, Phi, critical_r, pair
        # 最危险极径太靠近扫描右端时，外侧可能仍有危险构型，需扩展
        if critical_r > r_start - 2.0 and r_start < R_HEAD_MAX:
            prev_phi = Phi
            r_start = min(r_start * 2.0, R_HEAD_MAX)
            continue
        return r_start, Phi, critical_r, pair
    return R_HEAD_MAX, prev_phi or 0.0, 0.0, (0, 0)


def verify_problem3(p: float, r_head: float) -> dict[str, float]:
    """运行问题三的几何尺寸检验（文档 14.1、14.2 节）。

    Args:
        p: 螺距（m）。
        r_head: 龙头前把手极径（m）。

    Returns:
        各检验指标误差字典：
        - 调头边界误差：|bθ0 - r_head|，即龙头是否精确落在极径 r_head 的圆上；
        - 最大弦长误差：式(37)，|‖P_i - P_{i-1}‖ - d_i| 的最大值。
    """
    thetas, positions = compute_positions(p, r_head)
    b = p / (2.0 * math.pi)

    boundary_error = abs(b * thetas[0] - r_head)

    distances = handle_distances()
    max_chord_error = 0.0
    for i in range(1, TOTAL_HANDLES):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        max_chord_error = max(max_chord_error, abs(math.hypot(dx, dy) - distances[i]))

    return {
        "调头边界误差": boundary_error,
        "最大弦长误差": max_chord_error,
    }


def verify_critical_pair(
    p: float, r_head: float, i_star: int, j_star: int
) -> tuple[bool, str]:
    """用线段相交与点在矩形内判断独立交叉验证最危险板凳对（文档 14.6 节）。

    Args:
        p: 螺距（m）。
        r_head: 龙头前把手极径（m）。
        i_star, j_star: 报告的临界接触板凳对（1-based 编号）。

    Returns:
        (colliding, detail) 二元组，语义与 collision.verify_pair_independent 相同。
    """
    _, positions = compute_positions(p, r_head)
    return verify_pair_independent(positions, i_star, j_star)


def critical_rectangles(p: float, r_head: float) -> np.ndarray:
    """返回指定构型下全部板凳矩形参数（供绘图使用）。

    Args:
        p: 螺距（m）。
        r_head: 龙头前把手极径（m）。

    Returns:
        shape (223, 8) 的数组，见 collision.build_rectangles。
    """
    _, positions = compute_positions(p, r_head)
    return build_rectangles(positions)
