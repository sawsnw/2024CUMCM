"""问题五：龙头最大行进速度模型。

问题四的最优路径确定后，龙头位于弧长位置 ``s0`` 时，其余把手由固定弦长方程
（问题四式 37）沿路径向后递推确定；速度由刚性约束求导递推（问题四式 40），
且递推系数只由几何构型决定。因此各把手速度与龙头速度严格成一次齐次比例：

.. math::

    v_i(s_0) = |\\lambda_i(s_0)|\\, v_{\\mathrm h},

其中 ``\\lambda_i`` 为纯几何放大系数。于是龙头最大恒定速度为

.. math::

    v_{\\mathrm h}^{\\max} = \\dfrac{2}{\\Lambda_{\\max}},
    \\qquad
    \\Lambda_{\\max} = \\max_{s_0\\in\\mathcal S}\\max_i |\\lambda_i(s_0)|.

数值流程（文档第 10 节）：
    1. 加载问题四最优配置（``solve_s_curve``）；
    2. 在宽范围 ``[P5_SCAN_MIN, P5_SCAN_MAX]`` 内粗扫 ``\\Lambda(s_0)``，
       定位局部峰值候选，并同步监测最小几何分母 ``D_i``（奇异构型）；
    3. 对候选峰值与三处切点（A、B、C）附近做局部加密 + 精确细化；
    4. 输出 ``\\Lambda_{\\max}``、临界构型 ``s_0^*``、临界把手 ``i^*``、``v_h^max``；
    5. 提供刚性距离、速度约束、线性缩放、最优边界、奇异分母等模型检验函数。

模块约定：``src/models`` 下的模块不可直接运行，只提供可导入的函数和类。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.config import (
    P5_COARSE_STEP,
    P5_FINE_STEP,
    P5_OPTIMAL_ORIENTATION,
    P5_OPTIMAL_THETA_A,
    P5_OPTIMAL_THETA_C,
    P5_REFINE_RADIUS,
    P5_SCAN_MAX,
    P5_SCAN_MIN,
    P5_SPEED_LIMIT,
    P5_ULTRA_FINE_STEP,
    TOTAL_HANDLES,
)
from src.models.problem4 import (
    path_point,
    path_tangent,
    solve_handle_arc_parameter,
    solve_s_curve,
)
from src.models.spiral_dragon import handle_distances

# 粗扫时参与局部加密的候选峰数量上限
MAX_REFINE_CANDIDATES = 5
# 超过该放大倍数的粗扫局部峰值才作为候选（低于阈值视为无放大区）
PEAK_THRESHOLD = 1.05

# 日志回调类型（与 logger.info 签名兼容）
_Logger = Callable[..., None]


def load_optimal_geometry() -> dict:
    """加载问题四最优配置的 S 形曲线几何。

    Returns:
        ``solve_s_curve`` 返回的几何 dict。

    Raises:
        RuntimeError: 最优切点参数无法构造 S 形曲线。
    """
    g = solve_s_curve(
        P5_OPTIMAL_THETA_A, P5_OPTIMAL_THETA_C, P5_OPTIMAL_ORIENTATION
    )
    if g is None:
        raise RuntimeError("问题四最优配置（θa=%.6f, θc=%.6f）无法求解"
                           % (P5_OPTIMAL_THETA_A, P5_OPTIMAL_THETA_C))
    return g


def _speeds_with_head_speed(
    g: dict,
    s0: float,
    v0: float,
    bisect_iter: int = 45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """龙头弧长 ``s0``、龙头速度 ``v0`` 下全部把手的速度与位置。

    显式复现问题四的位置递推（式 37）与速度递推（式 40），龙头速度作为
    递推锚点（``speeds[0] = v0``），不预先假设速度与龙头速度的线性关系，
    供线性缩放、最优边界等独立性检验使用。

    Args:
        g: ``solve_s_curve`` 返回的几何 dict。
        s0: 龙头前把手沿路径的弧长位置（m）。
        v0: 龙头前把手行进速度（m/s）。
        bisect_iter: 弦长方程二分迭代次数（越大越精确、越慢）。

    Returns:
        ``(speeds, s_params, positions)``：
            speeds: shape ``(224,)`` 的把手速度大小（m/s）；
            s_params: shape ``(224,)`` 的把手弧长参数；
            positions: shape ``(224, 2)`` 的把手中心坐标。
    """
    distances = handle_distances()
    s_params = np.zeros(TOTAL_HANDLES)
    positions = np.zeros((TOTAL_HANDLES, 2))
    s_params[0] = s0
    positions[0] = path_point(s0, g)
    for i in range(1, TOTAL_HANDLES):
        s_params[i] = solve_handle_arc_parameter(
            s_params[i - 1], distances[i], g, max_iter=bisect_iter
        )
        positions[i] = path_point(s_params[i], g)
    tau_all = np.array([path_tangent(float(s), g) for s in s_params])  # (224, 2)
    dP = np.zeros((TOTAL_HANDLES, 2))
    dP[1:] = positions[1:] - positions[:-1]  # 板凳方向（有长度）
    dot_prev = np.einsum("ij,ij->i", dP[1:], tau_all[:-1])
    dot_i = np.einsum("ij,ij->i", dP[1:], tau_all[1:])
    speeds = np.empty(TOTAL_HANDLES)
    speeds[0] = v0
    speeds[1:] = v0 * np.cumprod(np.abs(dot_prev / dot_i))
    return speeds, s_params, positions


def speed_amplification(g: dict, s0: float, bisect_iter: int = 45) -> np.ndarray:
    """龙头位于弧长 ``s0`` 时全部把手的速度放大系数 ``|λ_i|``。

    龙头速度为 1 m/s 时，各把手速度数值即放大系数（问题四式 37、40）。

    Args:
        g: 几何 dict。
        s0: 龙头弧长位置（m）。
        bisect_iter: 弦长方程二分迭代次数。

    Returns:
        shape ``(224,)`` 的速度放大系数。
    """
    speeds, _, _ = _speeds_with_head_speed(g, s0, 1.0, bisect_iter)
    return speeds


def coarse_scan(
    g: dict,
    scan_min: float = P5_SCAN_MIN,
    scan_max: float = P5_SCAN_MAX,
    coarse_step: float = P5_COARSE_STEP,
    bisect_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """宽范围粗扫：计算 ``Λ(s0)`` 曲线与最小几何分母。

    Args:
        g: 几何 dict。
        scan_min: 搜索区间下界（m）。
        scan_max: 搜索区间上界（m）。
        coarse_step: 粗扫步长（m）。
        bisect_iter: 弦长方程二分迭代次数（粗扫用较低值加速）。

    Returns:
        ``(s0_grid, lambda_curve, d_min, s_dmin)``：
            s0_grid: 粗扫龙头弧长网格；
            lambda_curve: 各网格点上的 ``Λ(s0) = max_i|λ_i|``；
            d_min: 全网格最小几何分母 ``min|D_i|``（奇异监测，式 52）；
            s_dmin: 最小分母对应的龙头弧长。
    """
    n = int(round((scan_max - scan_min) / coarse_step)) + 1
    s0_grid = scan_min + coarse_step * np.arange(n)
    lambda_curve = np.zeros(n)
    d_min = 1e9
    s_dmin = float(scan_min)
    for k, s0 in enumerate(s0_grid):
        speeds, s_params, positions = _speeds_with_head_speed(
            g, float(s0), 1.0, bisect_iter
        )
        lambda_curve[k] = float(speeds.max())
        tau_all = np.array([path_tangent(float(s), g) for s in s_params])
        dP = positions[1:] - positions[:-1]
        di = np.abs(np.einsum("ij,ij->i", dP, tau_all[1:]))
        m = float(di.min())
        if m < d_min:
            d_min = m
            s_dmin = float(s0)
    return s0_grid, lambda_curve, d_min, s_dmin


def local_peaks(
    lambda_curve: np.ndarray,
    s0_grid: np.ndarray,
    threshold: float = PEAK_THRESHOLD,
) -> list[tuple[float, float]]:
    """从粗扫曲线中提取局部峰值候选（按放大系数降序）。

    Args:
        lambda_curve: 粗扫 ``Λ(s0)`` 曲线。
        s0_grid: 对应的龙头弧长网格。
        threshold: 放大系数阈值，低于该值的局部极大不视为候选。

    Returns:
        按 ``Λ`` 降序排列的 ``[(Λ, s0), ...]`` 列表。
    """
    peaks: list[tuple[float, float]] = []
    n = len(lambda_curve)
    for k in range(1, n - 1):
        if (
            lambda_curve[k] > lambda_curve[k - 1]
            and lambda_curve[k] >= lambda_curve[k + 1]
            and lambda_curve[k] > threshold
        ):
            peaks.append((float(lambda_curve[k]), float(s0_grid[k])))
    peaks.sort(reverse=True)
    return peaks


def fine_lambda_curve(
    g: dict,
    center: float,
    radius: float,
    step: float,
    bisect_iter: int = 45,
) -> tuple[np.ndarray, np.ndarray]:
    """在 ``center ± radius`` 内以细步长生成精细 ``Λ(s0)`` 曲线。

    用于绘制最大值附近的局部放大图：粗扫步长较疏，而最优配置的放大
    峰值宽度仅约 0.2 m，必须以细步长重新采样才能完整呈现峰值形状。

    Args:
        g: 几何 dict。
        center: 局部区间中心（龙头弧长，m）。
        radius: 区间半径（m）。
        step: 采样步长（m）。
        bisect_iter: 弦长方程二分迭代次数。

    Returns:
        ``(fine_s0, fine_lambda)``：精细采样网格与对应 ``Λ(s0)``。
    """
    n = int(round(2.0 * radius / step)) + 1
    fine_s0 = center - radius + step * np.arange(n)
    fine_lambda = np.zeros(n)
    for k, s0 in enumerate(fine_s0):
        speeds, _, _ = _speeds_with_head_speed(g, float(s0), 1.0, bisect_iter)
        fine_lambda[k] = float(speeds.max())
    return fine_s0, fine_lambda


def refine_near(
    g: dict,
    s0_center: float,
    radius: float,
    fine_step: float,
    bisect_iter: int = 45,
) -> tuple[float, float, np.ndarray]:
    """在 ``s0_center ± radius`` 内以 ``fine_step`` 加密扫描，求精确峰值。

    Args:
        g: 几何 dict。
        s0_center: 加密区间中心（m）。
        radius: 加密半径（m）。
        fine_step: 加密步长（m）。
        bisect_iter: 弦长方程二分迭代次数。

    Returns:
        ``(Λ, s0, speeds)``：区间内最大放大系数、对应龙头弧长、该构型速度分布。
    """
    best: tuple[float, float, np.ndarray | None] = (0.0, s0_center, None)
    s0 = s0_center - radius
    while s0 <= s0_center + radius + 1e-12:
        speeds, _, _ = _speeds_with_head_speed(g, float(s0), 1.0, bisect_iter)
        m = float(speeds.max())
        if m > best[0]:
            best = (m, float(s0), speeds)
        s0 += fine_step
    assert best[2] is not None
    return best  # type: ignore[return-value]


def solve_max_amplification(
    g: dict,
    logger: _Logger | None = None,
) -> dict:
    """求解全程最大速度放大系数与临界构型/临界把手。

    流程（文档第 10 节）：
        1. 宽范围粗扫 ``Λ(s0)``，定位局部峰值候选并监测最小分母；
        2. 对候选峰值与三处切点（A=0、B=L1、C=L_S）附近局部加密；
        3. 对最优位置做超细精确细化。

    Args:
        g: 几何 dict。
        logger: 日志记录器（可选），用于打印进度。

    Returns:
        结果 dict，包含：
            ``g``: 几何 dict；
            ``s0_grid``: 粗扫网格；
            ``lambda_curve``: 粗扫 ``Λ(s0)`` 曲线；
            ``Lambda_max``: 全程最大速度放大系数；
            ``s0_star``: 临界龙头弧长位置（m）；
            ``i_star``: 临界把手编号（0=龙头，223=龙尾后把手）；
            ``lambda_star``: 临界构型下全部把手放大系数；
            ``v_h_max``: 龙头最大恒定速度（m/s）；
            ``d_min`` / ``s_dmin``: 最小几何分母及其位置；
            ``grid_lambda_max``: 仅由粗扫网格得到的最大值（用于对比）；
            ``peaks``: 粗扫局部峰值候选。
    """
    if logger is not None:
        logger.info("步骤 1：宽范围粗扫 [%.1f, %.1f] m，步长 %.2f m",
                    P5_SCAN_MIN, P5_SCAN_MAX, P5_COARSE_STEP)
    s0_grid, lambda_curve, d_min, s_dmin = coarse_scan(
        g, bisect_iter=30
    )
    grid_lambda_max = float(lambda_curve.max())
    peaks = local_peaks(lambda_curve, s0_grid)
    if logger is not None:
        logger.info("    粗扫完成：网格最大 Λ=%.6f，局部峰值数=%d，D_min=%.4e @ s0=%.2f",
                    grid_lambda_max, len(peaks), d_min, s_dmin)
        for lam, s0 in peaks[:MAX_REFINE_CANDIDATES]:
            logger.info("    候选峰值：s0=%.2f m，Λ=%.6f", s0, lam)

    # 候选 = 粗扫局部峰值（前若干个）+ 三处分段连接点
    candidates = [s0 for _, s0 in peaks[:MAX_REFINE_CANDIDATES]]
    candidates.extend([0.0, g["L1"], g["L_S"]])
    candidates = list(dict.fromkeys(candidates))  # 去重保序

    if logger is not None:
        logger.info("步骤 2：候选局部加密（步长 %.3f m，半径 %.2f m）",
                    P5_FINE_STEP, P5_REFINE_RADIUS)
    best: tuple[float, float, np.ndarray | None] = (0.0, candidates[0], None)
    for sc in candidates:
        lam, s0r, sp = refine_near(g, sc, P5_REFINE_RADIUS, P5_FINE_STEP, 45)
        if logger is not None:
            logger.info("    候选 s0=%.2f → 峰值 Λ=%.6f @ s0=%.3f", sc, lam, s0r)
        if lam > best[0]:
            best = (lam, s0r, sp)

    if logger is not None:
        logger.info("步骤 3：最优位置超细细化（步长 %.4f m）", P5_ULTRA_FINE_STEP)
    lam_f, s0_f, sp_f = refine_near(g, best[1], 0.02, P5_ULTRA_FINE_STEP, 45)
    if lam_f > best[0]:
        best = (lam_f, s0_f, sp_f)

    Lambda_max, s0_star, lambda_star = best
    assert lambda_star is not None
    i_star = int(np.argmax(lambda_star))
    v_h_max = P5_SPEED_LIMIT / Lambda_max

    if logger is not None:
        logger.info("    全程最大放大系数 Λ_max=%.9f @ s0*=%.4f m，临界把手 i*=%d",
                    Lambda_max, s0_star, i_star)
        logger.info("    龙头最大速度 v_h^max = 2/Λ_max = %.9f m/s", v_h_max)

    return {
        "g": g,
        "s0_grid": s0_grid,
        "lambda_curve": lambda_curve,
        "grid_lambda_max": grid_lambda_max,
        "peaks": peaks,
        "Lambda_max": Lambda_max,
        "s0_star": s0_star,
        "i_star": i_star,
        "lambda_star": lambda_star,
        "v_h_max": v_h_max,
        "d_min": d_min,
        "s_dmin": s_dmin,
    }


# ============================================================
# 模型检验（文档第 15 节）
# ============================================================

def verify_rigid_distances(g: dict, s0: float, bisect_iter: int = 45) -> tuple[float, int]:
    """刚性距离检验：``max |‖P_i-P_{i-1}‖ - d_i|``。

    Args:
        g: 几何 dict。
        s0: 龙头弧长位置（m）。
        bisect_iter: 弦长方程二分迭代次数。

    Returns:
        ``(max_error, handle_idx)``：最大误差及其把手编号（1-based）。
    """
    _, _, positions = _speeds_with_head_speed(g, s0, 1.0, bisect_iter)
    distances = handle_distances()
    errs = np.abs(
        np.linalg.norm(positions[1:] - positions[:-1], axis=1) - distances[1:]
    )
    k = int(np.argmax(errs))
    return float(errs[k]), k + 1


def verify_speed_constraints(g: dict, s0: float, bisect_iter: int = 45) -> tuple[float, int]:
    """速度刚性约束残差：``max |(P_i-P_{i-1})·(u_i τ_i - u_{i-1} τ_{i-1})|``。

    Args:
        g: 几何 dict。
        s0: 龙头弧长位置（m）。
        bisect_iter: 弦长方程二分迭代次数。

    Returns:
        ``(max_residual, handle_idx)``：最大残差及其板凳编号（1-based）。
    """
    speeds, s_params, positions = _speeds_with_head_speed(g, s0, 1.0, bisect_iter)
    del speeds
    tau_all = np.array([path_tangent(float(s), g) for s in s_params])
    dP = positions[1:] - positions[:-1]
    # 带符号弧长变化率 ṡ（式 40）
    sdot = np.empty(TOTAL_HANDLES)
    sdot[0] = 1.0
    for i in range(1, TOTAL_HANDLES):
        sdot[i] = sdot[i - 1] * (
            np.dot(dP[i - 1], tau_all[i - 1]) / np.dot(dP[i - 1], tau_all[i])
        )
    res = np.empty(TOTAL_HANDLES - 1)
    for i in range(1, TOTAL_HANDLES):
        res[i - 1] = abs(
            float(np.dot(dP[i - 1], sdot[i] * tau_all[i] - sdot[i - 1] * tau_all[i - 1]))
        )
    k = int(np.argmax(res))
    return float(res[k]), k + 1


def verify_linear_scaling(
    g: dict,
    s0: float,
    head_speeds: tuple[float, ...] = (0.5, 1.5, 2.0, 3.0),
    bisect_iter: int = 45,
) -> float:
    """线性缩放检验：``v_i(v0)/v0`` 应与单位速度放大系数一致（与 v0 无关）。

    对多个龙头速度 ``v0``，用完整递推计算把手速度，再除以 ``v0``，
    与单位速度放大系数比较，返回最大绝对偏差。

    Args:
        g: 几何 dict。
        s0: 龙头弧长位置（m）。
        head_speeds: 用于检验的龙头速度取值。
        bisect_iter: 弦长方程二分迭代次数。

    Returns:
        最大偏差（量级为浮点舍入误差即视为通过）。
    """
    ref, _, _ = _speeds_with_head_speed(g, s0, 1.0, bisect_iter)
    max_err = 0.0
    for c in head_speeds:
        sp, _, _ = _speeds_with_head_speed(g, s0, c, bisect_iter)
        max_err = max(max_err, float(np.max(np.abs(sp / c - ref))))
    return max_err


def verify_optimal_boundary(
    g: dict,
    s0_star: float,
    i_star: int,
    v_h_max: float,
    bisect_iter: int = 45,
    eps: float = 1e-6,
) -> dict:
    """最优边界检验（文档 15.4 节）。

    在 ``v_h_max`` 下，临界把手速度应等于 2 m/s；略增大时应有把手超过上限，
    略减小时应全部满足约束。

    Args:
        g: 几何 dict。
        s0_star: 临界龙头弧长位置（m）。
        i_star: 临界把手编号。
        v_h_max: 计算得到的龙头最大速度（m/s）。
        bisect_iter: 弦长方程二分迭代次数。
        eps: 最优解左右扰动的相对幅度。

    Returns:
        dict：含最优解处最大速度、临界把手速度、扰动 ±eps 后的最大速度。
    """
    sp_opt, _, _ = _speeds_with_head_speed(g, s0_star, v_h_max, bisect_iter)
    sp_hi, _, _ = _speeds_with_head_speed(g, s0_star, v_h_max * (1.0 + eps), bisect_iter)
    sp_lo, _, _ = _speeds_with_head_speed(g, s0_star, v_h_max * (1.0 - eps), bisect_iter)
    return {
        "limit": P5_SPEED_LIMIT,
        "v_max_at_opt": float(sp_opt.max()),
        "v_i_star_at_opt": float(sp_opt[i_star]),
        "v_max_plus": float(sp_hi.max()),
        "v_max_minus": float(sp_lo.max()),
        "eps": eps,
    }


def extend_scan_check(
    g: dict,
    s0_star: float,
    extend: float = 100.0,
    step: float = 5.0,
    bisect_iter: int = 30,
) -> tuple[float, float]:
    """搜索区间扩展检验（文档 15.6 节）。

    向当前最优位置两侧延伸扫描，确认未出现更大的速度放大系数。

    Args:
        g: 几何 dict。
        s0_star: 临界龙头弧长位置（m）。
        extend: 向两侧额外延伸的弧长长度（m）。
        step: 延伸扫描步长（m）。
        bisect_iter: 弦长方程二分迭代次数。

    Returns:
        ``(Lambda_ext, d_ext)``：延伸区间内的最大放大系数及其位置。
    """
    best: tuple[float, float] = (0.0, s0_star)
    for lo, hi in [
        (s0_star - extend, s0_star),
        (s0_star, s0_star + extend),
    ]:
        s0 = lo
        while s0 <= hi + 1e-12:
            speeds, _, _ = _speeds_with_head_speed(g, float(s0), 1.0, bisect_iter)
            m = float(speeds.max())
            if m > best[0]:
                best = (m, float(s0))
            s0 += step
    return best


def verify_all(
    result: dict,
    logger: _Logger | None = None,
) -> dict:
    """运行问题五全部模型检验（文档第 15 节）。

    Args:
        result: ``solve_max_amplification`` 返回的结果 dict。
        logger: 日志记录器（可选）。

    Returns:
        检验结果 dict。
    """
    g = result["g"]
    s0_star = result["s0_star"]
    i_star = result["i_star"]
    v_h_max = result["v_h_max"]

    if logger is not None:
        logger.info("步骤 5：模型检验（文档第 15 节）")

    # 15.1 刚性距离检验
    max_chord, idx_chord = verify_rigid_distances(g, s0_star)
    if logger is not None:
        logger.info("  刚性距离最大误差 = %.3e（应≈0，@把手 %d）", max_chord, idx_chord)

    # 15.2 速度约束残差检验
    max_rig, idx_rig = verify_speed_constraints(g, s0_star)
    if logger is not None:
        logger.info("  速度刚性约束最大残差 = %.3e（应≈0，@板凳 %d）", max_rig, idx_rig)

    # 15.3 线性缩放检验
    scaling_err = verify_linear_scaling(g, s0_star)
    if logger is not None:
        logger.info("  线性缩放最大偏差 = %.3e（应≈0）", scaling_err)

    # 15.4 最优边界检验
    boundary = verify_optimal_boundary(g, s0_star, i_star, v_h_max)
    if logger is not None:
        logger.info("  最优边界：v_max@opt=%.9f（应=2），v_i*(@opt)=%.9f，"
                    "v_max(+eps)=%.9f，v_max(-eps)=%.9f",
                    boundary["v_max_at_opt"], boundary["v_i_star_at_opt"],
                    boundary["v_max_plus"], boundary["v_max_minus"])

    # 15.5 网格收敛检验（粗扫 vs 加密）
    grid_lam = result["grid_lambda_max"]
    lam = result["Lambda_max"]
    if logger is not None:
        logger.info("  网格收敛：粗扫 Λ=%.6f vs 加密 Λ=%.9f（相对差 %.2f%%）",
                    grid_lam, lam, 100.0 * (lam - grid_lam) / lam)

    # 15.6 搜索区间扩展检验
    lam_ext, s_ext = extend_scan_check(g, s0_star)
    if logger is not None:
        logger.info("  搜索区间扩展：延伸区内最大 Λ=%.6f @ s0=%.2f（应≤%.6f）",
                    lam_ext, s_ext, lam)

    # 奇异分母监测（式 52、53）
    d_min = result["d_min"]
    if logger is not None:
        if d_min < 1e-3:
            logger.warning("  警告：最小几何分母 D_min=%.4e 接近零，可能存在奇异构型", d_min)
        else:
            logger.info("  最小几何分母 D_min = %.4e @ s0=%.2f（远离零，无奇异）",
                        d_min, result["s_dmin"])

    return {
        "max_chord_error": (max_chord, idx_chord),
        "max_speed_residual": (max_rig, idx_rig),
        "linear_scaling_error": scaling_err,
        "boundary": boundary,
        "grid_vs_fine_rel_diff": 100.0 * (lam - grid_lam) / lam,
        "extend_scan": (lam_ext, s_ext),
    }


def build_summary(result: dict, checks: dict) -> dict:
    """汇总问题五全部关键数值，供脚本写 JSON 与文档引用。

    Args:
        result: ``solve_max_amplification`` 的结果 dict。
        checks: ``verify_all`` 返回的检验结果 dict。

    Returns:
        纯 JSON 可序列化的汇总 dict。
    """
    return {
        "Lambda_max": round(result["Lambda_max"], 12),
        "s0_star": round(result["s0_star"], 6),
        "i_star": result["i_star"],
        "v_h_max": round(result["v_h_max"], 12),
        "critical_time": round(result["s0_star"] / result["v_h_max"], 6),
        "grid_lambda_max": round(result["grid_lambda_max"], 9),
        "grid_v_h_max": round(P5_SPEED_LIMIT / result["grid_lambda_max"], 9),
        "D_min": round(result["d_min"], 12),
        "s_dmin": round(result["s_dmin"], 4),
        "speed_limit": P5_SPEED_LIMIT,
        "geometry": {
            "theta_a": round(result["g"]["theta_a"], 9),
            "theta_c": round(result["g"]["theta_c"], 9),
            "R": round(result["g"]["R"], 9),
            "L1": round(result["g"]["L1"], 9),
            "L2": round(result["g"]["L2"], 9),
            "L_S": round(result["g"]["L_S"], 9),
        },
        "checks": {
            "max_chord_error": round(checks["max_chord_error"][0], 12),
            "linear_scaling_error": round(checks["linear_scaling_error"], 12),
            "grid_vs_fine_rel_diff_percent": round(checks["grid_vs_fine_rel_diff"], 4),
        },
    }
