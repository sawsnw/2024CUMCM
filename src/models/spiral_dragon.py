"""板凳龙沿等距螺线盘入模型（问题一）。

实现阿基米德螺线上 224 个把手中心的位置与速度递推求解。
公式推导详见 docs/problem_analysis/问题一_模型建立与求解.md。

模型要点：
    1. 螺线方程 r = b·θ，b = p/(2π)，盘入方向为 θ 减小；
    2. 龙头前把手以 1 m/s 沿螺线弧长运动，由弧长方程二分求解 θ₀(t)；
    3. 相邻把手间保持固定弦长，逐节求解第一个外侧根确定 θ_i(t)；
    4. 对刚性约束求导，逐节递推参数变化率与速度。
"""

import math

from src.config import (
    BODY_HOLE_DISTANCE,
    HEAD_HOLE_DISTANCE,
    HEAD_INITIAL_THETA,
    HEAD_SPEED,
    SPIRAL_B,
    TOTAL_HANDLES,
)


def spiral_point(theta: float) -> tuple[float, float]:
    """计算螺线上参数为 theta 的点的直角坐标 P(θ) = (bθcosθ, bθsinθ)。

    Args:
        theta: 螺线参数（极角，弧度）。

    Returns:
        (x, y) 坐标，单位 m。
    """
    return (SPIRAL_B * theta * math.cos(theta), SPIRAL_B * theta * math.sin(theta))


def spiral_derivative(theta: float) -> tuple[float, float]:
    """计算螺线对参数的导数向量 P'(θ)。

    Args:
        theta: 螺线参数（弧度）。

    Returns:
        (dx/dθ, dy/dθ)。
    """
    return (
        SPIRAL_B * (math.cos(theta) - theta * math.sin(theta)),
        SPIRAL_B * (math.sin(theta) + theta * math.cos(theta)),
    )


def _arc_antiderivative(theta: float) -> float:
    """弧长被积函数的原函数 G(θ) = θ√(1+θ²) + arsinh(θ)。

    满足 ∫√(1+θ²)dθ = G(θ)/2，用于龙头弧长方程的解析积分。
    """
    return theta * math.sqrt(1.0 + theta**2) + math.asinh(theta)


def solve_head_theta(t: float) -> float:
    """求解时刻 t 龙头前把手的螺线参数 θ₀(t)。

    由弧长方程 (b/2)·[G(32π) - G(θ₀)] = t 使用二分法求解。
    因 G 在 θ > 0 上严格单调递增，方程对每个 t 都有唯一解。

    Args:
        t: 时间（s）。

    Returns:
        龙头前把手的螺线参数（弧度）。
    """
    # 目标值：G(θ₀) = G(32π) - 2t/b
    target = _arc_antiderivative(HEAD_INITIAL_THETA) - 2.0 * t / SPIRAL_B
    lo, hi = 0.0, HEAD_INITIAL_THETA
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _arc_antiderivative(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _chord_error(theta: float, theta_prev: float, distance: float) -> float:
    """相邻把手距离方程左端：b²[θ² + θ₀² - 2θθ₀cos(θ - θ₀)] - d²。

    返回 0 时表示两点之间的弦长恰好等于 distance。

    Args:
        theta: 待求把手的螺线参数（弧度）。
        theta_prev: 前一把手的螺线参数（弧度）。
        distance: 固定弦长（m）。

    Returns:
        方程残差。
    """
    return (
        SPIRAL_B**2
        * (
            theta**2
            + theta_prev**2
            - 2.0 * theta * theta_prev * math.cos(theta - theta_prev)
        )
        - distance**2
    )


def solve_next_theta(theta_prev: float, distance: float) -> float:
    """求与前一把手相距 distance 的外侧第一个把手的螺线参数。

    采用文档第 7.3 节策略：以弧长近似式 (25) 给出初值，再沿参数增大方向
    搜索第一个符号变化区间，最后在区间内二分求根。

    Args:
        theta_prev: 前一把手的螺线参数（弧度）。
        distance: 相邻把手之间的固定弦长（m）。

    Returns:
        后一把手的螺线参数（弧度），满足 θ > theta_prev。

    Raises:
        RuntimeError: 未能在合理参数范围内找到外侧根。
    """

    def error(th: float) -> float:
        return _chord_error(th, theta_prev, distance)

    # 弧长近似初值（式 25）：因弦长小于弧长，真实根略大于该初值
    theta_est = theta_prev + distance / (
        SPIRAL_B * math.sqrt(1.0 + theta_prev**2)
    )

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

    # 在 [lo, hi] 内二分求根（error(lo) <= 0，error(hi) >= 0），
    # 区间宽度小于双精度分辨率的 1e-14 倍时提前终止，避免固定 100 次迭代浪费
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if error(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def handle_distances() -> list[float]:
    """返回相邻把手之间的固定距离列表 d[0..223]。

    索引与把手编号对应：d[i] 为把手 P_{i-1} 与 P_i 之间的距离。
    d[1] 为龙头两孔中心距 2.86 m（连接把手 0 与把手 1），
    d[2..223] 均为龙身/龙尾两孔中心距 1.65 m；
    d[0] 不使用（龙头前把手无前驱），置 0 占位。

    Returns:
        长度为 224 的距离列表。
    """
    return [0.0, HEAD_HOLE_DISTANCE] + [BODY_HOLE_DISTANCE] * (TOTAL_HANDLES - 2)


def compute_handle_states(
    thetas: list[float],
) -> tuple[list[float], list[tuple[float, float]], list[float]]:
    """由把手螺线参数递推参数变化率、速度向量与速度大小。

    Args:
        thetas: 224 个把手的螺线参数列表（弧度）。

    Returns:
        三元组 (theta_dot, velocities, speeds)：
        - theta_dot: 参数变化率列表（rad/s）；
        - velocities: 速度向量列表（m/s）；
        - speeds: 速度大小列表（m/s）。
    """
    n = len(thetas)
    theta_dot = [0.0] * n
    velocities = [(0.0, 0.0)] * n
    speeds = [0.0] * n

    # 步骤 2：龙头参数变化率与速度（速度大小恒为 1 m/s）
    theta_dot[0] = -HEAD_SPEED / (SPIRAL_B * math.sqrt(1.0 + thetas[0] ** 2))
    d0 = spiral_derivative(thetas[0])
    velocities[0] = (d0[0] * theta_dot[0], d0[1] * theta_dot[0])
    speeds[0] = HEAD_SPEED

    # 步骤 4：逐节递推参数变化率与速度
    for i in range(1, n):
        p_i = spiral_point(thetas[i])
        p_prev = spiral_point(thetas[i - 1])
        # Q_i = P_i - P_{i-1}（沿板凳轴线方向）
        q = (p_i[0] - p_prev[0], p_i[1] - p_prev[1])
        d_prev = spiral_derivative(thetas[i - 1])
        d_i = spiral_derivative(thetas[i])
        # 式(31)：θ̇_i = [Q·P'(θ_{i-1}) / Q·P'(θ_i)] · θ̇_{i-1}
        q_dot_dprev = q[0] * d_prev[0] + q[1] * d_prev[1]
        q_dot_di = q[0] * d_i[0] + q[1] * d_i[1]
        theta_dot[i] = (q_dot_dprev / q_dot_di) * theta_dot[i - 1]
        # 式(32)(33)：速度向量与速度大小
        velocities[i] = (d_i[0] * theta_dot[i], d_i[1] * theta_dot[i])
        speeds[i] = math.hypot(velocities[i][0], velocities[i][1])

    return theta_dot, velocities, speeds


def compute_dragon_at(
    t: float,
    need_speed: bool = True,
) -> tuple[
    list[tuple[float, float]],
    list[float],
    list[float],
    list[float],
    list[tuple[float, float]],
]:
    """计算时刻 t 全部 224 个把手的位置、速度大小及中间量。

    Args:
        t: 时间（s）。
        need_speed: 是否同时计算参数变化率与速度。粗搜索阶段只需位置，
            传入 False 可跳过速度递推以节省计算量（详见问题二文档第 5.3 节）。

    Returns:
        五元组 (positions, speeds, thetas, theta_dot, velocities)：
        - positions: 224 个把手中心坐标列表；
        - speeds: 224 个把手的速度大小列表（m/s）；
        - thetas: 224 个把手的螺线参数列表（弧度）；
        - theta_dot: 224 个把手的参数变化率列表（rad/s）；
        - velocities: 224 个把手的速度向量列表（m/s）。
    """
    distances = handle_distances()

    # 步骤 1、3：龙头参数 + 逐节递推位置
    thetas = [solve_head_theta(t)]
    for i in range(1, TOTAL_HANDLES):
        thetas.append(solve_next_theta(thetas[i - 1], distances[i]))
    positions = [spiral_point(th) for th in thetas]

    # 步骤 2、4：参数变化率与速度（need_speed=False 时跳过）
    if not need_speed:
        theta_dot = [0.0] * TOTAL_HANDLES
        velocities = [(0.0, 0.0)] * TOTAL_HANDLES
        speeds = [0.0] * TOTAL_HANDLES
        return positions, speeds, thetas, theta_dot, velocities

    theta_dot, velocities, speeds = compute_handle_states(thetas)

    return positions, speeds, thetas, theta_dot, velocities


def verify_configuration(
    t: float,
    thetas: list[float],
    positions: list[tuple[float, float]],
    theta_dot: list[float],
    velocities: list[tuple[float, float]],
    speeds: list[float],
) -> dict[str, float]:
    """运行文档第 12 节模型检验，返回各指标误差。

    Args:
        t: 当前时刻（s）。
        thetas: 224 个把手的螺线参数。
        positions: 224 个把手的位置。
        theta_dot: 224 个把手的参数变化率。
        velocities: 224 个把手的速度向量。
        speeds: 224 个把手的速度大小。

    Returns:
        各检验指标的误差字典：
        - 龙头弧长误差：式(34)；
        - 最大弦长误差：式(35)；
        - 参数递增：0.0 表示通过，1.0 表示存在次序错误（式(36)）；
        - 龙头速度误差：式(37)；
        - 最大速度约束误差：式(38)；
        - 龙头速度：当前时刻龙头速度大小。
    """
    distances = handle_distances()

    # 12.1 龙头弧长检验
    arc_error = abs(
        SPIRAL_B
        / 2.0
        * (
            _arc_antiderivative(HEAD_INITIAL_THETA)
            - _arc_antiderivative(thetas[0])
        )
        - t
    )

    # 12.2 相邻把手距离检验
    max_chord_error = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        max_chord_error = max(max_chord_error, abs(math.hypot(dx, dy) - distances[i]))

    # 12.3 参数次序检验（返回 0.0 表示通过）
    ordered = all(thetas[i] > thetas[i - 1] for i in range(1, len(thetas)))

    # 12.4 龙头速度检验
    head_speed_error = abs(
        SPIRAL_B * math.sqrt(1.0 + thetas[0] ** 2) * abs(theta_dot[0]) - HEAD_SPEED
    )

    # 12.5 刚性速度约束检验：|Q_i·(v_i - v_{i-1})|
    max_proj_error = 0.0
    for i in range(1, len(positions)):
        q = (
            positions[i][0] - positions[i - 1][0],
            positions[i][1] - positions[i - 1][1],
        )
        dv = (
            velocities[i][0] - velocities[i - 1][0],
            velocities[i][1] - velocities[i - 1][1],
        )
        max_proj_error = max(max_proj_error, abs(q[0] * dv[0] + q[1] * dv[1]))

    return {
        "龙头弧长误差": arc_error,
        "最大弦长误差": max_chord_error,
        "参数递增": 0.0 if ordered else 1.0,
        "龙头速度误差": head_speed_error,
        "最大速度约束误差": max_proj_error,
        "龙头速度": speeds[0],
    }
