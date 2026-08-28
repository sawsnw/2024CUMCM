"""测量 compute_dragon_at 与候选碰撞检查的成本（临时）。"""

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
    "_timing_p4.txt",
)


def main():
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    _, _, g0 = solve_given_configuration(16.5715, 16.6169, ORIENTATION_MINUS)

    # 单时刻位置+速度
    t0 = time.time()
    positions, speeds = compute_dragon_at(0.0, g0)
    t1 = time.time()
    log(f"单时刻 compute_dragon_at(需速度): {t1 - t0:.3f} s")

    t0 = time.time()
    positions = compute_dragon_at(0.0, g0, need_speed=False)
    t1 = time.time()
    log(f"单时刻 compute_dragon_at(仅位置): {t1 - t0:.3f} s")

    # 201 个时刻（仅位置）
    t0 = time.time()
    for t in range(-100, 101):
        compute_dragon_at(float(t), g0, need_speed=False)
    t1 = time.time()
    log(f"201 时刻（仅位置）: {t1 - t0:.3f} s")

    # 201 个时刻（位置+速度）
    t0 = time.time()
    for t in range(-100, 101):
        compute_dragon_at(float(t), g0)
    t1 = time.time()
    log(f"201 时刻（位置+速度）: {t1 - t0:.3f} s")

    # 201 次全局裕量
    t0 = time.time()
    for t in range(-100, 101):
        positions, _ = compute_dragon_at(float(t), g0, need_speed=False)
        global_margin(positions)
    t1 = time.time()
    log(f"201 时刻（位置+碰撞裕量）: {t1 - t0:.3f} s")

    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("written:", OUTFILE)


if __name__ == "__main__":
    main()
