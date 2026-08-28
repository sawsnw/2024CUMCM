"""剖析 compute_dragon_at 的热点（临时）。"""

import cProfile
import pstats
import io

from src.models.problem4 import ORIENTATION_MINUS, compute_dragon_at, solve_given_configuration


def main():
    _, _, g0 = solve_given_configuration(16.5715, 16.6169, ORIENTATION_MINUS)
    pr = cProfile.Profile()
    pr.enable()
    for t in range(-100, 101, 10):  # 21 个时刻
        compute_dragon_at(float(t), g0, need_speed=False)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(18)
    print(s.getvalue())


if __name__ == "__main__":
    main()
