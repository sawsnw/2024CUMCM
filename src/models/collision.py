"""问题二：板凳龙碰撞检测模型（路径无关）。

实现文档《问题二_模型建立与求解.md》第 6~8 节的板凳矩形重建、分离轴碰撞检测
与全局裕量计算，并附第 15 节的模型检验工具。

设计要点：
    1. 只依赖任意时刻全部把手中心坐标 positions，与具体运动路径无关，
       后续问题更换运动路径时可直接复用；
    2. 使用 numpy 对全部 24531 对非相邻板凳批量向量化分离轴检测，
       单次裕量计算由纯 Python 的约 65 ms 降至毫秒级；
    3. 对外统一使用文档符号：板凳编号 1..223（1-based），内部使用 0-based 数组。

公式依据：
    板凳矩形 R_k = {C_k + α·e_k + β·n_k : |α| ≤ a_k, |β| ≤ c}（式 15）；
    候选轴集合 A_ij = {e_i, n_i, e_j, n_j}（式 20）；
    板凳对裕量 g_ij = max_u s_ij(u)（式 21）；
    全局裕量 G = min_{(i,j)∈Ω} g_ij（式 25）。
"""

import numpy as np

from src.config import (
    BOARD_WIDTH,
    BODY_HOLE_DISTANCE,
    BODY_LENGTH,
    HEAD_HOLE_DISTANCE,
    HEAD_LENGTH,
    TOTAL_HANDLES,
)

# 板凳数 = 把手数 - 1 = 223
BOARD_NUM = TOTAL_HANDLES - 1


def build_rectangles(positions: object) -> np.ndarray:
    """由全部把手中心重建 223 块板凳的有向矩形参数。

    Args:
        positions: 224 个把手中心坐标的可迭代对象（如 compute_dragon_at 返回的
            (x, y) 列表）。positions[k-1] 与 positions[k] 为板凳 k 的前、后把手。

    Returns:
        shape (223, 8) 的 float 数组，每行依次为
        (Cx, Cy, ex, ey, nx, ny, a, c)：其中 (Cx, Cy) 为矩形中心（式 12），
        (ex, ey) 为纵向单位向量（式 10），(nx, ny) 为横向单位向量（式 11），
        a 为半长度（式 13）、c 为半宽度（式 14）。
    """
    P = np.asarray(positions, dtype=float)
    if P.shape != (TOTAL_HANDLES, 2):
        raise ValueError(
            f"positions 应为 {TOTAL_HANDLES}×2 数组，实际形状为 {P.shape}"
        )
    A = P[:-1]  # 板凳 k 的前把手 P_{k-1}
    B = P[1:]   # 板凳 k 的后把手 P_k
    d = B - A
    norm = np.linalg.norm(d, axis=1)
    e = d / norm[:, None]                     # 纵向单位向量（式 10）
    n = np.column_stack((-e[:, 1], e[:, 0]))  # 横向单位向量（式 11）
    C = 0.5 * (A + B)                         # 矩形中心（式 12）
    a = np.full(BOARD_NUM, 0.5 * BODY_LENGTH)
    a[0] = 0.5 * HEAD_LENGTH                  # 龙头板长（式 13）
    c = np.full(BOARD_NUM, 0.5 * BOARD_WIDTH)  # 半宽度（式 14）
    return np.column_stack(
        (C[:, 0], C[:, 1], e[:, 0], e[:, 1], n[:, 0], n[:, 1], a, c)
    )


def pair_margins(rects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量计算全部非相邻板凳对的分离轴碰撞裕量（式 21）。

    Args:
        rects: build_rectangles 返回的 (223, 8) 数组。

    Returns:
        三元组 (margins, i, j)：
        - margins: shape (N,) 的裕量数组，N = C(223,2) - 222 = 24531；
        - i, j: 对应的 1-based 板凳编号数组，满足 j - i >= 2。
    """
    n = rects.shape[0]
    # 非相邻板凳对集合 Ω（式 23）：1 ≤ i < j ≤ 223，j - i ≥ 2
    iu = np.triu_indices(n, k=2)
    ri = rects[iu[0]]
    rj = rects[iu[1]]

    # 候选轴集合 A_ij（式 20）：{e_i, n_i, e_j, n_j}，shape (N, 4, 2)
    axes = np.stack(
        (ri[:, 2:4], ri[:, 4:6], rj[:, 2:4], rj[:, 4:6]), axis=1
    )

    # 投影中心差 (C_j - C_i)·u（式 19 前半部分）
    dc = rj[:, :2] - ri[:, :2]
    proj_center = np.einsum("nd,nkd->nk", dc, axes)  # (N, 4)

    # 各板凳在每条轴上的投影半长 ρ（式 17）
    dot_ei = np.einsum("nd,nkd->nk", ri[:, 2:4], axes)
    dot_ni = np.einsum("nd,nkd->nk", ri[:, 4:6], axes)
    rho_i = ri[:, 6:7] * np.abs(dot_ei) + ri[:, 7:8] * np.abs(dot_ni)
    dot_ej = np.einsum("nd,nkd->nk", rj[:, 2:4], axes)
    dot_nj = np.einsum("nd,nkd->nk", rj[:, 4:6], axes)
    rho_j = rj[:, 6:7] * np.abs(dot_ej) + rj[:, 7:8] * np.abs(dot_nj)

    # 有符号分离量 s_ij(u)（式 19），对候选轴取最大值得到 g_ij（式 21）
    sep = np.abs(proj_center) - rho_i - rho_j
    margins = sep.max(axis=1)

    return margins, iu[0] + 1, iu[1] + 1


def global_margin(positions: object) -> tuple[float, int, int]:
    """计算整条板凳龙的全局碰撞裕量及最危险板凳对（式 25、27）。

    Args:
        positions: 224 个把手中心坐标。

    Returns:
        三元组 (G, i_star, j_star)：
        - G: 全局碰撞裕量（m），正为分离、零为接触、负为重叠；
        - i_star, j_star: 最危险板凳对（1-based 板凳编号）。
    """
    rects = build_rectangles(positions)
    margins, i, j = pair_margins(rects)
    k = int(np.argmin(margins))
    return float(margins[k]), int(i[k]), int(j[k])


def verify_geometry(positions: object) -> dict[str, float]:
    """运行板凳几何尺寸检验（文档 15.1 节）。

    Args:
        positions: 224 个把手中心坐标。

    Returns:
        各检验指标误差的字典：
        - 最大弦长误差：式(35)，|B_k - A_k| 与期望两孔中心距之差；
        - 最大单位向量误差：纵向单位向量模长与 1 之差；
        - 最大正交误差：横向与纵向单位向量点积（应接近 0）；
        - 最大板长误差：由矩形参数得到的板长与理论板长之差；
        - 最大板宽误差：半宽度与理论半宽度之差。
    """
    rects = build_rectangles(positions)
    P = np.asarray(positions, dtype=float)
    A, B = P[:-1], P[1:]
    dists = np.linalg.norm(B - A, axis=1)
    expected_d = np.full(BOARD_NUM, BODY_HOLE_DISTANCE)
    expected_d[0] = HEAD_HOLE_DISTANCE
    max_chord = float(np.max(np.abs(dists - expected_d)))

    e = rects[:, 2:4]
    n = rects[:, 4:6]
    max_e_norm = float(np.max(np.abs(np.linalg.norm(e, axis=1) - 1.0)))
    max_orth = float(np.max(np.abs(np.einsum("nd,nd->n", e, n))))

    expected_a = np.full(BOARD_NUM, 0.5 * BODY_LENGTH)
    expected_a[0] = 0.5 * HEAD_LENGTH
    max_len = float(np.max(np.abs(rects[:, 6] - expected_a)))
    max_width = float(np.max(np.abs(rects[:, 7] - 0.5 * BOARD_WIDTH)))

    return {
        "最大弦长误差": max_chord,
        "最大单位向量误差": max_e_norm,
        "最大正交误差": max_orth,
        "最大板长误差": max_len,
        "最大板宽误差": max_width,
    }


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """三点有向面积（叉积 z 分量），用于线段相交判定。"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float) -> bool:
    """判断点 p 是否在线段 ab 的包围盒内（含边界）。"""
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(p1: np.ndarray, p2: np.ndarray,
                        p3: np.ndarray, p4: np.ndarray,
                        eps: float = 1e-12) -> bool:
    """判断线段 p1p2 与 p3p4 是否相交（含端点和共线重叠）。"""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    ):
        return True
    if abs(d1) <= eps and _on_segment(p3, p4, p1, eps):
        return True
    if abs(d2) <= eps and _on_segment(p3, p4, p2, eps):
        return True
    if abs(d3) <= eps and _on_segment(p1, p2, p3, eps):
        return True
    if abs(d4) <= eps and _on_segment(p1, p2, p4, eps):
        return True
    return False


def _point_in_rect(p: np.ndarray, C: np.ndarray, e: np.ndarray,
                   n: np.ndarray, a: float, c: float, eps: float = 1e-12) -> bool:
    """判断点 p 是否在矩形 R(C, e, n, a, c) 内（含边界）。"""
    dx = p[0] - C[0]
    dy = p[1] - C[1]
    alpha = dx * e[0] + dy * e[1]
    beta = dx * n[0] + dy * n[1]
    return -a - eps <= alpha <= a + eps and -c - eps <= beta <= c + eps


def verify_pair_independent(
    positions: object, i: int, j: int
) -> tuple[bool, str]:
    """用线段相交与点在矩形内判断独立交叉验证板凳对（文档 15.4 节）。

    Args:
        positions: 224 个把手中心坐标。
        i, j: 1-based 板凳编号（i < j）。

    Returns:
        (colliding, detail) 二元组：
        - colliding: 是否发生矩形边界相交或顶点包含；
        - detail: 具体判定说明字符串。
    """
    rects = build_rectangles(positions)

    def vertices(rect: np.ndarray) -> list[np.ndarray]:
        C = rect[:2]
        e = rect[2:4]
        n = rect[4:6]
        a = rect[6]
        c = rect[7]
        return [
            C + a * e + c * n,
            C + a * e - c * n,
            C - a * e - c * n,
            C - a * e + c * n,
        ]

    vi = vertices(rects[i - 1])
    vj = vertices(rects[j - 1])

    # 16 组边界线段相交检查
    for k in range(4):
        for l in range(4):
            if _segments_intersect(vi[k], vi[(k + 1) % 4], vj[l], vj[(l + 1) % 4]):
                return True, f"矩形边相交：i 边{k + 1}与 j 边{l + 1}"

    ri = rects[i - 1]
    rj = rects[j - 1]
    Ci, ei, ni, ai, ci = ri[:2], ri[2:4], ri[4:6], ri[6], ri[7]
    Cj, ej, nj, aj, cj = rj[:2], rj[2:4], rj[4:6], rj[6], rj[7]
    for k, v in enumerate(vj):
        if _point_in_rect(v, Ci, ei, ni, ai, ci):
            return True, f"矩形 j 的顶点 {k + 1} 位于矩形 i 内部"
    for k, v in enumerate(vi):
        if _point_in_rect(v, Cj, ej, nj, aj, cj):
            return True, f"矩形 i 的顶点 {k + 1} 位于矩形 j 内部"

    return False, "边界不相交、顶点互不包含"
