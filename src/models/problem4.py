"""问题四：S 形调头曲线模型。

实现盘入/盘出螺线与 S 形调头曲线（两段相切圆弧）的几何建模，以及
最优圆弧求解和把手位置/速度递推。

模型要点：
    1. 盘入螺线 P(θ) = b·θ·(cosθ, sinθ)，b = p/(2π)，p = 1.7 m，龙头沿 θ 减小方向盘入；
    2. 盘出螺线为盘入螺线关于螺线中心 O 的中心对称像：Q(θ) = -P(θ)，龙头沿 θ 增大方向盘出；
    3. 调头路径为两段圆弧相切连接而成的 S 形曲线：前段（盘入一侧）圆弧半径 R1 = 2R，
       后段圆弧半径 R2 = R，两圆外切，且分别与盘入、盘出螺线相切；
    4. “基准（给定）”配置：两圆弧均与调头空间圆（半径 ρ = 4.5 m）内切，
       即 |c1| + 2R = ρ 且 |c2| + R = ρ（用二维牛顿迭代求精确解）；
    5. 圆弧优化：以两个切点参数 (θ1, θ2) 为决策变量，在 0 < θ1,θ2 ≤ θb（θb=4.5/b）
       内枚举两种 S 形朝向，只保留圆弧均在调头区域内且无绕行的分支，
       目标为最小化调头曲线总长 L_S = 2R·α1 + R·α2（二维粗搜索 + 局部细化）；
    6. 完整路径 Γ(s)：盘入螺线—圆弧1—圆弧2—盘出螺线按弧长 s 参数化（式 27），
       龙头 s0(t) = t，其余把手由固定弦长方程向后递推（式 37），
       速度由刚性约束求导递推（式 40）。

切点记号：
    - A：S 形曲线与盘入螺线的切点，A = P(θa)；
    - C：S 形曲线与盘出螺线的切点，C = -P(θc)；
    - B：两段圆弧的外切点，B = (c1 + 2c2)/3；
    - c1、c2：两段圆弧的圆心（半径分别为 2R、R）。
"""

import math

import numpy as np

from src.config import P4_B, TOTAL_HANDLES, TURN_RADIUS
from src.models.spiral_dragon import handle_distances

# S 形朝向常数
ORIENTATION_PLUS = 1    # 弧 1 逆时针、弧 2 顺时针
ORIENTATION_MINUS = -1  # 弧 1 顺时针、弧 2 逆时针（问题四给定配置采用此朝向）


def inward_spiral_point(theta: float) -> np.ndarray:
    """计算盘入螺线 P(θ) = bθ(cosθ, sinθ) 上参数 θ 对应点的坐标。

    Args:
        theta: 螺线参数（极角，弧度）。

    Returns:
        (x, y) 坐标数组，单位 m。
    """
    return np.array([P4_B * theta * math.cos(theta), P4_B * theta * math.sin(theta)])


def inward_spiral_derivative(theta: float) -> np.ndarray:
    """计算盘入螺线对参数的导数向量 P'(θ)。

    Args:
        theta: 螺线参数（弧度）。

    Returns:
        (dx/dθ, dy/dθ)。
    """
    return np.array(
        [
            P4_B * (math.cos(theta) - theta * math.sin(theta)),
            P4_B * (math.sin(theta) + theta * math.cos(theta)),
        ]
    )


def outward_spiral_point(theta: float) -> np.ndarray:
    """计算盘出螺线 Q(θ) = -P(θ) 上参数 θ 对应点的坐标。

    盘出螺线为盘入螺线关于螺线中心 O 的中心对称像。

    Args:
        theta: 螺线参数（弧度）。

    Returns:
        (x, y) 坐标数组，单位 m。
    """
    return -inward_spiral_point(theta)


def _rot90(v: np.ndarray) -> np.ndarray:
    """将二维向量逆时针旋转 90°（复平面中乘以 i）。"""
    return np.array([-v[1], v[0]])


def _norm(v: np.ndarray) -> float:
    """二维向量模长。"""
    return float(np.linalg.norm(v))


def solve_s_curve(
    theta_a: float,
    theta_c: float,
    orientation: int = ORIENTATION_MINUS,
) -> dict | None:
    """给定切点参数与 S 形朝向，求解调头曲线几何。

    由两圆外切方程 |c1 - c2| = R1 + R2 = 3R 求解半径 R：
        |d + R·w| = 3R，其中 d = A - C，w = ±i·(2·tA + tC)；
    展开为关于 R 的一元二次方程，取正根。

    Args:
        theta_a: 盘入螺线切点 A 的参数（弧度）。
        theta_c: 盘出螺线切点 C 的参数（弧度）。
        orientation: S 形朝向，取 ORIENTATION_PLUS 或 ORIENTATION_MINUS。

    Returns:
        dict 包含 R、A、C、c1、c2、B、tA、tC、sgn1、sgn2；
        无正根时返回 None。
    """
    A = inward_spiral_point(theta_a)
    C = outward_spiral_point(theta_c)
    tA = -inward_spiral_derivative(theta_a) / _norm(inward_spiral_derivative(theta_a))
    tC = -inward_spiral_derivative(theta_c) / _norm(inward_spiral_derivative(theta_c))
    d = A - C
    if orientation == ORIENTATION_PLUS:
        w = _rot90(2.0 * tA + tC)
    else:
        w = _rot90(-(2.0 * tA + tC))

    a_coef = _norm(w) ** 2 - 9.0
    b_coef = 2.0 * np.dot(d, w)
    c_coef = _norm(d) ** 2
    disc = b_coef**2 - 4.0 * a_coef * c_coef
    if disc < 0:
        return None

    roots: list[float] = []
    if abs(a_coef) < 1e-14:
        if abs(b_coef) > 1e-14:
            roots.append(-c_coef / b_coef)
    else:
        roots.append((-b_coef + math.sqrt(disc)) / (2.0 * a_coef))
        roots.append((-b_coef - math.sqrt(disc)) / (2.0 * a_coef))

    for R in roots:
        if R <= 0 or not math.isfinite(R):
            continue
        if orientation == ORIENTATION_PLUS:
            c1 = A + 2.0 * R * _rot90(tA)
            c2 = C - R * _rot90(tC)
        else:
            c1 = A - 2.0 * R * _rot90(tA)
            c2 = C + R * _rot90(tC)
        B = (c1 + 2.0 * c2) / 3.0  # 两圆外切点
        # 两段圆弧的起点角（相对各自圆心）与分段弧长
        phi1 = math.atan2(A[1] - c1[1], A[0] - c1[0])
        phi2 = math.atan2(B[1] - c2[1], B[0] - c2[0])
        ang1 = angle_span(c1, A, B, 1.0 if orientation == ORIENTATION_PLUS else -1.0)
        ang2 = angle_span(c2, B, C, -1.0 if orientation == ORIENTATION_PLUS else 1.0)
        L1 = 2.0 * R * ang1
        L2 = R * ang2
        return {
            "R": R,
            "theta_a": theta_a,
            "theta_c": theta_c,
            "A": A,
            "C": C,
            "c1": c1,
            "c2": c2,
            "B": B,
            "tA": tA,
            "tC": tC,
            "sgn1": 1.0 if orientation == ORIENTATION_PLUS else -1.0,
            "sgn2": -1.0 if orientation == ORIENTATION_PLUS else 1.0,
            "ang1": ang1,
            "ang2": ang2,
            "phi1": phi1,
            "phi2": phi2,
            "L1": L1,
            "L2": L2,
            "L_S": L1 + L2,
            # 预计算切点处的弧长原函数值，加速螺线段弧长反解
            "H_ref_in": _H(theta_a),
            "H_ref_out": _H(theta_c),
        }
    return None


def _residuals(theta_a: float, theta_c: float, orientation: int) -> tuple[np.ndarray, dict | None]:
    """给定配置的残差向量（用于牛顿迭代）。

    给定配置要求两圆弧均与调头空间圆内切：
        f1 = |c1| + 2R - ρ，f2 = |c2| + R - ρ。

    Args:
        theta_a: 切点 A 参数（弧度）。
        theta_c: 切点 C 参数（弧度）。
        orientation: S 形朝向。

    Returns:
        (f, g)：残差向量 (f1, f2) 与几何 dict（无解时为 None）。
    """
    g = solve_s_curve(theta_a, theta_c, orientation)
    if g is None:
        return np.array([1e6, 1e6]), None
    f1 = _norm(g["c1"]) + 2.0 * g["R"] - TURN_RADIUS
    f2 = _norm(g["c2"]) + g["R"] - TURN_RADIUS
    return np.array([f1, f2]), g


def solve_given_configuration(
    theta_a0: float = 16.5715,
    theta_c0: float = 16.6169,
    orientation: int = ORIENTATION_MINUS,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """用二维牛顿迭代求解给定配置（两圆弧与调头空间圆内切）。

    Args:
        theta_a0: 切点 A 参数初值（弧度）。
        theta_c0: 切点 C 参数初值（弧度）。
        orientation: S 形朝向。
        tol: 残差范数收敛阈值。

    Returns:
        (state, f, g)：state=(θa, θc)，f=最终残差，g=几何 dict。

    Raises:
        RuntimeError: 牛顿迭代不收敛或雅可比奇异。
    """
    state = np.array([theta_a0, theta_c0], dtype=float)
    for _ in range(100):
        f, g = _residuals(state[0], state[1], orientation)
        if g is None:
            raise RuntimeError("牛顿迭代中 S 形曲线无解")
        if _norm(f) < tol:
            return state, f, g
        eps = 1e-7
        J = np.zeros((2, 2))
        for k in range(2):
            s = state.copy()
            s[k] += eps
            fk, _ = _residuals(s[0], s[1], orientation)
            J[:, k] = (fk - f) / eps
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            raise RuntimeError("牛顿迭代雅可比奇异")
        if _norm(step) < 1e-14:
            return state, f, g
        state = state + step
    raise RuntimeError("牛顿迭代不收敛")


def angle_span(c: np.ndarray, p0: np.ndarray, p1: np.ndarray, sgn: float) -> float:
    """圆弧从 p0 到 p1（绕圆心 c、半径由 p 确定）的有向圆心角。

    Args:
        c: 圆心坐标。
        p0: 起点坐标。
        p1: 终点坐标。
        sgn: 旋转方向（+1 逆时针，-1 顺时针）。

    Returns:
        圆心角（弧度），取 (0, 2π)。
    """
    v0, v1 = p0 - c, p1 - c
    a0 = math.atan2(v0[1], v0[0])
    a1 = math.atan2(v1[1], v1[0])
    diff = (a1 - a0) % (2 * math.pi)
    if sgn < 0:
        diff = (2 * math.pi - diff) % (2 * math.pi)
    return diff


def s_curve_points(g: dict, n1: int = 200, n2: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """采样 S 形调头曲线的两段圆弧。

    Args:
        g: solve_s_curve 返回的几何 dict。
        n1: 前段圆弧（半径 2R）采样点数。
        n2: 后段圆弧（半径 R）采样点数。

    Returns:
        (arc1_pts, arc2_pts)：两段圆弧的采样点数组（不含端点 B 重复）。
    """
    R = g["R"]

    def _arc(c, r, p0, p1, sgn, n):
        v0, v1 = p0 - c, p1 - c
        a0 = math.atan2(v0[1], v0[0])
        a1 = math.atan2(v1[1], v1[0])
        diff = (a1 - a0) % (2 * math.pi)
        if sgn < 0:
            diff = (2 * math.pi - diff) % (2 * math.pi)
        pts = []
        for k in range(n + 1):
            a = a0 + sgn * diff * k / n
            pts.append(c + r * np.array([math.cos(a), math.sin(a)]))
        return np.array(pts)

    arc1 = _arc(g["c1"], 2 * R, g["A"], g["B"], g["sgn1"], n1)
    arc2 = _arc(g["c2"], R, g["B"], g["C"], g["sgn2"], n2)
    return arc1, arc2


def turning_length(g: dict) -> float:
    """计算 S 形调头曲线总长 L = 2R·α1 + R·α2。

    Args:
        g: solve_s_curve 返回的几何 dict。

    Returns:
        调头曲线总长（m）。
    """
    R = g["R"]
    ang1 = angle_span(g["c1"], g["A"], g["B"], g["sgn1"])
    ang2 = angle_span(g["c2"], g["B"], g["C"], g["sgn2"])
    return 2.0 * R * ang1 + R * ang2


# ============================================================
# 完整路径 Γ(s) 的弧长参数化（文档第 9 节）
# ============================================================

def _H(theta: float) -> float:
    """螺线弧长原函数 H(θ) = θ√(1+θ²) + arsinhθ（文档式 28）。"""
    return theta * math.sqrt(1.0 + theta**2) + math.asinh(theta)


def _solve_spiral_theta(arc_length: float, h_ref: float) -> float:
    """求 θ ≥ θ_ref 使螺线从 θ_ref 到 θ 的弧长等于 arc_length（式 28、32）。

    由弧长关系 (b/2)[H(θ) - H(θ_ref)] = arc_length，解 H(θ) = target。
    H 为凸函数且初值 sqrt(target) ≥ 真根，牛顿法单调收敛。

    Args:
        arc_length: 从 θ_ref 起的螺线弧长（m，非负）。
        h_ref: 切点处的弧长原函数值 H(θ_ref)（预计算）。

    Returns:
        满足弧长关系的螺线参数 θ（rad）。
    """
    target = h_ref + 2.0 * arc_length / P4_B
    theta = math.sqrt(target)
    for _ in range(8):
        h = theta * math.sqrt(1.0 + theta * theta) + math.asinh(theta)
        val = h - target
        if abs(val) < 1e-14:
            break
        theta -= val / (2.0 * math.sqrt(1.0 + theta * theta))
    return theta


def _arc_point(c: np.ndarray, r: float, kappa: float, phi: float, s: float) -> np.ndarray:
    """圆弧上弧长 s 处的点（文档式 30、31）。

    Args:
        c: 圆心。
        r: 半径。
        kappa: 曲率符号（+1 逆时针、-1 顺时针）。
        phi: 起点角（相对圆心）。
        s: 从起点起的弧长。

    Returns:
        (x, y) 坐标。
    """
    a = phi + kappa * s / r
    return c + r * np.array([math.cos(a), math.sin(a)])


def _arc_tangent(kappa: float, phi: float, r: float, s: float) -> np.ndarray:
    """圆弧上弧长 s 处的单位切向量（沿行进方向）。"""
    a = phi + kappa * s / r
    return kappa * np.array([-math.sin(a), math.cos(a)])


def path_point(s: float, g: dict) -> np.ndarray:
    """完整分段路径 Γ(s) 上的点（文档式 27）。

    s 为以圆弧1起点（盘入切点 A）为原点的行进弧长：
        - s < 0：盘入螺线段；
        - 0 ≤ s < L1：圆弧1；
        - L1 ≤ s < L_S：圆弧2；
        - s ≥ L_S：盘出螺线段。

    Args:
        s: 行进弧长参数（m）。
        g: solve_s_curve 返回的几何 dict。

    Returns:
        (x, y) 坐标。
    """
    if s < 0.0:
        theta = _solve_spiral_theta(-s, g["H_ref_in"])
        return inward_spiral_point(theta)
    if s < g["L1"]:
        return _arc_point(g["c1"], 2.0 * g["R"], g["sgn1"], g["phi1"], s)
    if s < g["L_S"]:
        return _arc_point(g["c2"], g["R"], g["sgn2"], g["phi2"], s - g["L1"])
    theta = _solve_spiral_theta(s - g["L_S"], g["H_ref_out"])
    return outward_spiral_point(theta)


def path_tangent(s: float, g: dict) -> np.ndarray:
    """完整分段路径 Γ(s) 在弧长 s 处的单位切向量 τ(s)。

    Args:
        s: 行进弧长参数（m）。
        g: solve_s_curve 返回的几何 dict。

    Returns:
        单位切向量 (τx, τy)。
    """
    if s < 0.0:
        theta = _solve_spiral_theta(-s, g["H_ref_in"])
        return -inward_spiral_derivative(theta) / _norm(inward_spiral_derivative(theta))
    if s < g["L1"]:
        return _arc_tangent(g["sgn1"], g["phi1"], 2.0 * g["R"], s)
    if s < g["L_S"]:
        return _arc_tangent(g["sgn2"], g["phi2"], g["R"], s - g["L1"])
    theta = _solve_spiral_theta(s - g["L_S"], g["H_ref_out"])
    return -inward_spiral_derivative(theta) / _norm(inward_spiral_derivative(theta))


# ============================================================
# 各把手位置与速度递推（文档第 10、11 节）
# ============================================================

def solve_handle_arc_parameter(s_prev: float, distance: float, g: dict) -> float:
    """求前一把手后侧、与前一位置相距 distance 的把手弧长参数（式 37）。

    沿路径向后搜索使 ‖Γ(s) - Γ(s_prev)‖ = distance 的最近后向解：
    从 s_prev 向后大步（步长倍增）找到第一个弦长≥distance 的区间，
    再在该区间内二分。返回的是最靠近 s_prev 的解。

    Args:
        s_prev: 前一把手的弧长参数（m）。
        distance: 固定弦长（m），d_i = 2.86（龙头）或 1.65（龙身/龙尾）。
        g: solve_s_curve 返回的几何 dict。

    Returns:
        后一把手的弧长参数 s（m，满足 s < s_prev）。
    """
    if distance <= 0.0:
        return s_prev
    P_prev = path_point(s_prev, g)
    pprev_x, pprev_y = float(P_prev[0]), float(P_prev[1])

    def f(s: float) -> float:
        p = path_point(s, g)
        return math.hypot(p[0] - pprev_x, p[1] - pprev_y) - distance

    lo = s_prev
    hi = s_prev - distance
    if f(hi) < 0.0:
        h = distance
        for _ in range(60):
            lo = hi
            h *= 2.0
            hi = s_prev - h
            if f(hi) >= 0.0:
                break
        else:
            raise RuntimeError("向后搜索 60 次倍增仍未找到弦长解")
    # 二分求根：f(lo) < 0 ≤ f(hi)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compute_dragon_at(t: float, g: dict, need_speed: bool = True) -> tuple:
    """计算时刻 t 全部 224 个把手的位置与速度。

    龙头前把手 s0 = t（速度恒 1 m/s，式 35、36）；其余把手由固定弦长
    方程沿路径向后递推（式 37、38），速度由刚性约束求导递推（式 40）。

    Args:
        t: 时间（s），以调头开始（龙头到达圆弧1起点）为零时刻。
        g: solve_s_curve 返回的几何 dict。
        need_speed: 是否计算速度（默认 True）。

    Returns:
        need_speed=True 时返回 (positions, speeds)：
            positions: shape (224, 2) 的把手中心坐标；
            speeds: shape (224,) 的速度大小（m/s）。
        need_speed=False 时只返回 positions。
    """
    distances = handle_distances()
    s_params = np.zeros(TOTAL_HANDLES)
    positions = np.zeros((TOTAL_HANDLES, 2))
    s_params[0] = t
    positions[0] = path_point(t, g)
    for i in range(1, TOTAL_HANDLES):
        s_i = solve_handle_arc_parameter(s_params[i - 1], distances[i], g)
        s_params[i] = s_i
        positions[i] = path_point(s_i, g)
    if not need_speed:
        return positions
    speeds = np.zeros(TOTAL_HANDLES)
    speeds[0] = 1.0
    for i in range(1, TOTAL_HANDLES):
        tau_prev = path_tangent(s_params[i - 1], g)
        tau_i = path_tangent(s_params[i], g)
        dP = positions[i] - positions[i - 1]
        dot_prev = np.dot(dP, tau_prev)
        dot_i = np.dot(dP, tau_i)
        speeds[i] = abs((dot_prev / dot_i) * speeds[i - 1])
    return positions, speeds


# ============================================================
# 圆弧优化（文档第 7、8 节）
# ============================================================

def turning_candidate(
    theta_a: float,
    theta_c: float,
    orientation: int,
    arc_sample: int = 80,
) -> tuple[float, dict] | None:
    """检查一组切点参数的圆弧分支是否可行，并返回弧长与几何。

    可行条件（文档 7.1 节条件 1~5）：式(15) 有正根、圆弧均在调头区域内、
    无全圆绕行（α1, α2 < 2π）。

    Args:
        theta_a: 盘入螺线切点参数（rad）。
        theta_c: 盘出螺线切点参数（rad）。
        orientation: S 形朝向（ORIENTATION_PLUS / ORIENTATION_MINUS）。
        arc_sample: 圆弧越界检查的采样点数。

    Returns:
        (L_S, g)：可行时返回调头曲线总长与几何 dict；否则返回 None。
    """
    g = solve_s_curve(theta_a, theta_c, orientation)
    if g is None:
        return None
    if g["ang1"] >= 2.0 * math.pi - 1e-12 or g["ang2"] >= 2.0 * math.pi - 1e-12:
        return None
    arc1, arc2 = s_curve_points(g, n1=arc_sample, n2=arc_sample)
    if np.linalg.norm(arc1, axis=1).max() > TURN_RADIUS + 1e-9:
        return None
    if np.linalg.norm(arc2, axis=1).max() > TURN_RADIUS + 1e-9:
        return None
    return g["L_S"], g


def optimize_turning_curve(
    n_grid: int = 220,
    top_k: int = 5,
    zoom_rounds: int = 5,
    zoom_grid: int = 21,
    arc_sample: int = 80,
) -> tuple[tuple, list]:
    """二维全局粗搜索 + 多起点局部细化求解最短 S 形调头曲线（式 24）。

    Args:
        n_grid: 粗搜索每维网格点数。
        top_k: 局部细化选取的最优候选个数。
        zoom_rounds: 局部细化轮数。
        zoom_grid: 每轮局部网格每维点数。
        arc_sample: 圆弧越界检查采样点数。

    Returns:
        (best, candidates)：
            best: 最优几何候选元组 (L, θa, θc, orientation, g)；
            candidates: 全部可行候选按弧长升序排列的列表。
    """
    theta_b = TURN_RADIUS / P4_B
    thetas = np.linspace(1e-6, theta_b, n_grid)
    candidates: list[tuple[float, float, float, int, dict]] = []
    for theta_a in thetas:
        for theta_c in thetas:
            for orientation in (ORIENTATION_PLUS, ORIENTATION_MINUS):
                r = turning_candidate(theta_a, theta_c, orientation, arc_sample)
                if r is not None:
                    candidates.append((r[0], theta_a, theta_c, orientation, r[1]))
    candidates.sort(key=lambda x: x[0])
    if not candidates:
        raise RuntimeError("粗搜索未找到任何可行圆弧")

    best = candidates[0]
    step = (theta_b - 1e-6) / (n_grid - 1)
    for (L0, ta0, tc0, ori0, _g0) in candidates[:top_k]:
        ta, tc = ta0, tc0
        span = step
        local_best: tuple | None = None
        for _ in range(zoom_rounds):
            ts = np.linspace(max(1e-6, ta - span), ta + span, zoom_grid)
            cs = np.linspace(max(1e-6, tc - span), tc + span, zoom_grid)
            for t1 in ts:
                for t2 in cs:
                    r = turning_candidate(t1, t2, ori0, arc_sample)
                    if r is None:
                        continue
                    if local_best is None or r[0] < local_best[0]:
                        local_best = (r[0], t1, t2, ori0, r[1])
            if local_best is None:
                break
            ta, tc = local_best[1], local_best[2]
            span *= 0.4
        if local_best is not None and local_best[0] < best[0]:
            best = local_best

    return best, candidates
