"""基线配置全时段碰撞裕量分析（临时）。"""

import os
import time

import numpy as np

from src.config import TOTAL_HANDLES
from src.models.collision import global_margin
from src.models.problem4 import (
    ORIENTATION_MINUS,
    compute_dragon_at,
    solve_given_configuration,
)

OUTFILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "tables",
    "_margin_baseline.txt",
)


def main():
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    _, _, g0 = solve_given_configuration(16.5715, 16.6169, ORIENTATION_MINUS)
    log(f"基线 L0={g0['L_S']:.6f}")

    t_start = time.time()
    min_g = 1e9
    worst = None
    for t in range(-100, 101):
        positions = compute_dragon_at(float(t), g0, need_speed=False)
        G, i, j = global_margin(positions)
        if G < min_g:
            min_g = G
            worst = (t, i, j, G)
        if t % 25 == 0:
            log(f"  t={t:4d}  G={G:.6f}  (worst pair {i},{j})")
    log(f"最小裕量 min G = {min_g:.6f} m，出现在 t={worst[0]}, 板凳对 ({worst[1]},{worst[2]})")
    log(f"总耗时 {time.time() - t_start:.1f} s")

    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written:", OUTFILE)


if __name__ == "__main__":
    main()
