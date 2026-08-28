"""碰撞约束下最短 S 形调头曲线筛选（临时）。

几何候选按弧长升序排列，从最短开始计算全时段碰撞裕量，
找到第一个无碰撞（min G > 0）的候选作为碰撞约束最优解。
用 2s 粗网格快速拒绝，对通过的候选做 1s 全网格复核。
"""

import os
import time

import numpy as np

from src.models.collision import global_margin
from src.models.problem4 import (
    ORIENTATION_MINUS,
    ORIENTATION_PLUS,
    compute_dragon_at,
    optimize_turning_curve,
    turning_candidate,
)

OUTFILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "tables",
    "_collision_opt.txt",
)

COARSE_STEP = 2  # 粗筛时间步长（s）
FULL_STEP = 1    # 复核时间步长（s）


def min_margin(g: dict, step: float) -> tuple[float, float, int, int]:
    """计算几何候选在 t=-100..100（步长 step）内的最小碰撞裕量。"""
    min_g = 1e9
    worst = None
    t = -100.0
    while t <= 100.0 + 1e-12:
        positions = compute_dragon_at(t, g, need_speed=False)
        G, i, j = global_margin(positions)
        if G < min_g:
            min_g = G
            worst = (t, i, j)
        t += step
    return min_g, worst[0], worst[1], worst[2]


def main():
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    t0 = time.time()
    log("步骤 1：几何粗搜索 + 局部细化")
    best, candidates = optimize_turning_curve(n_grid=200, top_k=5, zoom_rounds=5, zoom_grid=21)
    L_geom, ta, tc, ori, g_geom = best
    log(f"几何最优：L={L_geom:.6f} θa={ta:.6f} θc={tc:.6f} ori={ori}")
    log(f"可行几何候选数：{len(candidates)}，耗时 {time.time()-t0:.1f}s")

    # 记录候选几何（可能很多，保留前 200 个）
    cand_list = candidates[:200]
    log("")
    log("步骤 2：按弧长从小到大检查碰撞")
    first_safe = None
    for idx, (L, ta, tc, ori, g) in enumerate(cand_list):
        t1 = time.time()
        try:
            m_coarse, tw, iw, jw = min_margin(g, COARSE_STEP)
        except Exception as e:
            log(f"  候选 {idx}: L={L:.4f} θa={ta:.4f} θc={tc:.4f} R={g['R']:.4f} "
                f"计算异常({e})，视为碰撞")
            continue
        if m_coarse <= 0:
            log(f"  候选 {idx}: L={L:.4f} θa={ta:.4f} θc={tc:.4f} R={g['R']:.4f} "
                f"粗筛 minG={m_coarse:.4f} @ t={tw:g}({iw},{jw})  [碰撞] {time.time()-t1:.0f}s")
            continue
        # 粗筛通过 → 1s 全网格复核
        m_full, tw, iw, jw = min_margin(g, FULL_STEP)
        log(f"  候选 {idx}: L={L:.4f} θa={ta:.4f} θc={tc:.4f} R={g['R']:.4f} "
            f"粗筛 minG={m_coarse:.4f}，1s 复核 minG={m_full:.4f} @ t={tw:g}({iw},{jw}) "
            f"{time.time()-t1:.0f}s")
        if m_full > 0:
            first_safe = (L, ta, tc, ori, g, m_full, tw, iw, jw)
            log(f"  ★ 首个无碰撞候选：L*={L:.6f} θa={ta:.6f} θc={tc:.6f} "
                f"R={g['R']:.6f} minG={m_full:.6f}")
            break
        if idx > 60:
            log("  已检查 60+ 个候选仍未找到无碰撞方案，停止")
            break

    log("")
    log(f"总耗时 {time.time()-t0:.1f}s")
    if first_safe:
        L_star, ta, tc, ori, g_star, m, tw, iw, jw = first_safe
        log(f"=== 结果 ===")
        log(f"几何最优 L_geom = {L_geom:.6f}")
        log(f"碰撞约束最优 L* = {L_star:.6f} (θa={ta:.6f}, θc={tc:.6f}, R={g_star['R']:.6f})")
        log(f"最小裕量 minG = {m:.6f} @ t={tw:g}, 板凳对({iw},{jw})")
    else:
        log("未找到无碰撞候选")

    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written:", OUTFILE)


if __name__ == "__main__":
    main()
