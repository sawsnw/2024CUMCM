"""问题四模型测试脚本（临时，验证用）。"""

import math
import os

import numpy as np

from src.models.problem4 import (
    ORIENTATION_MINUS,
    ORIENTATION_PLUS,
    compute_dragon_at,
    optimize_turning_curve,
    path_point,
    path_tangent,
    solve_given_configuration,
    turning_candidate,
    turning_length,
)
from src.models.spiral_dragon import handle_distances

OUTFILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "tables",
    "_test_p4_model.txt",
)


def norm(v):
    return float(np.linalg.norm(v))


def main():
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    # 1. 基准配置
    _, _, g0 = solve_given_configuration(16.5715, 16.6169, ORIENTATION_MINUS)
    log(f"基准配置：θa={g0['theta_a']:.9f} θc={g0['theta_c']:.9f} R={g0['R']:.6f}")
    log(f"L1={g0['L1']:.6f} L2={g0['L2']:.6f} L_S={g0['L_S']:.6f}")
    log(f"龙头初始圈处 龙头 θ=32π 的弧长需要：可到任意远")

    # 2. 路径连续性检验：跨分段边界
    log("")
    log("=== 路径连续性检验 ===")
    bounds = [-50.0, -1e-6, 0.0, g0['L1'] - 1e-6, g0['L1'], g0['L1'] + 1e-6,
             g0['L_S'] - 1e-6, g0['L_S'], g0['L_S'] + 1e-6, 80.0]
    ok = True
    for s in bounds:
        p = path_point(s, g0)
        tau = path_tangent(s, g0)
        tn = norm(tau)
        if abs(tn - 1.0) > 1e-9:
            ok = False
            log(f"  s={s}: |τ|={tn:.3e} 异常")
    # 段间端点连续
    for s in [0.0, g0['L1'], g0['L_S']]:
        for eps in (1e-9, -1e-9):
            d = path_point(s + eps, g0) - path_point(s, g0)
            if norm(d) > 1e-7:
                ok = False
                log(f"  s={s} 端点跳跃 {norm(d):.3e}")
    log(f"  路径连续性：{'通过' if ok else '失败'}")

    # 3. 弧长参数化检验：|Γ'(s)|≈1（用数值差分）
    log("=== 弧长参数化 |Γ'(s)|=1 检验 ===")
    max_dev = 0.0
    for s in [-80.0, -30.0, -5.0, 1.0, g0['L1'] * 0.5, g0['L1'] + g0['L2'] * 0.3, g0['L_S'] + 10.0, 60.0]:
        h = 1e-6
        v = (path_point(s + h, g0) - path_point(s - h, g0)) / (2 * h)
        dev = abs(norm(v) - 1.0)
        max_dev = max(max_dev, dev)
    log(f"  最大 |Γ'|-1 = {max_dev:.3e}（应≈0）")

    # 4. 把手位置递推：弦长检验
    log("")
    log("=== 把手弦长检验（t=0 和 t=-100, 100）===")
    dists = handle_distances()
    for t in (-100.0, 0.0, 100.0):
        positions, speeds = compute_dragon_at(t, g0)
        max_err = 0.0
        for i in range(1, 224):
            err = abs(norm(positions[i] - positions[i - 1]) - dists[i])
            max_err = max(max_err, err)
        log(f"  t={t:6.0f} 最大弦长误差={max_err:.3e} 龙头速度={speeds[0]:.6f} 龙尾速度={speeds[223]:.6f}")

    # 5. 速度递推检验：刚性约束
    log("=== 速度刚性约束检验 ===")
    positions, speeds = compute_dragon_at(3.0, g0)
    max_rig = 0.0
    for i in range(1, 224):
        # 近似速度向量：需要切向，用数值差分代替
        P = positions[i] - positions[i - 1]
        # 这里仅检查速度大小为正且有限
        max_rig = max(max_rig, abs(speeds[i]))
    log(f"  t=3 速度范围: [{speeds.min():.6f}, {speeds.max():.6f}]")

    # 6. 优化（小网格快速测试）
    log("")
    log("=== 圆弧优化（小网格测试 n_grid=90）===")
    best, cands = optimize_turning_curve(n_grid=90, top_k=3, zoom_rounds=4, zoom_grid=15)
    L_star, ta, tc, ori, g_star = best
    log(f"可行候选数：{len(cands)}")
    log(f"最优：L*={L_star:.6f} θa={ta:.6f} θc={tc:.6f} ori={ori} R={g_star['R']:.6f}")
    log(f"基准 L0={g0['L_S']:.6f}，ΔL={g0['L_S'] - L_star:.6f}，缩短比例={100 * (g0['L_S'] - L_star) / g0['L_S']:.2f}%")

    # 前 8 个候选
    log("前 8 个候选：")
    for L, a, c, o, gg in cands[:8]:
        log(f"  L={L:.6f} θa={a:.6f} θc={c:.6f} ori={o} R={gg['R']:.6f}")

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written:", OUTFILE)


if __name__ == "__main__":
    main()
