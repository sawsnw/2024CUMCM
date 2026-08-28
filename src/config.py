"""全局配置模块。

所有参数和路径等统一在此定义，通过 `from config import *` 导入。
"""

import math
import os
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# ============================================================
# 日志配置
# ============================================================
# 日志文件命名格式: {脚本名}_{YYYY-MM-DD}_{HHMMSS}.log
# 例如: train_model_2026-07-14_143052.log
# 日志文件不覆盖旧文件，每次运行生成新的日志文件
LOG_FILENAME_FORMAT = "{script_name}_{date}_{time}.log"
LOG_DATE_FORMAT = "%Y-%m-%d"
LOG_TIME_FORMAT = "%H%M%S"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_log_filename(script_name: str) -> str:
    """根据脚本名生成日志文件名。

    Args:
        script_name: 脚本名称（不含扩展名）。

    Returns:
        日志文件名，例如 "train_model_2026-07-14_143052.log"。
    """
    now = datetime.now()
    return LOG_FILENAME_FORMAT.format(
        script_name=script_name,
        date=now.strftime(LOG_DATE_FORMAT),
        time=now.strftime(LOG_TIME_FORMAT),
    )


def get_log_filepath(script_name: str) -> str:
    """获取日志文件的完整路径。

    Args:
        script_name: 脚本名称（不含扩展名）。

    Returns:
        日志文件的绝对路径。
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    return os.path.join(LOGS_DIR, get_log_filename(script_name))


# ============================================================
# 问题一：板凳龙等距螺线模型参数
# ============================================================
# 板凳尺寸（单位：m）
HEAD_LENGTH = 3.41          # 龙头板长
BODY_LENGTH = 2.20          # 龙身与龙尾板长
BOARD_WIDTH = 0.30          # 板凳宽度
HOLE_OFFSET = 0.275         # 孔中心距最近板头距离

# 相邻把手中心之间的固定距离（两孔中心距）
HEAD_HOLE_DISTANCE = HEAD_LENGTH - 2.0 * HOLE_OFFSET   # 龙头：2.86 m
BODY_HOLE_DISTANCE = BODY_LENGTH - 2.0 * HOLE_OFFSET   # 龙身/龙尾：1.65 m

# 螺线参数
SPIRAL_PITCH = 0.55                     # 螺距 p（m）
SPIRAL_B = SPIRAL_PITCH / (2.0 * math.pi)   # 阿基米德螺线参数 b = p/(2π)

# 龙头初始状态
HEAD_INITIAL_CIRCLE = 16                    # 初始位于第 16 圈
HEAD_INITIAL_THETA = HEAD_INITIAL_CIRCLE * 2.0 * math.pi   # 32π（rad）
HEAD_SPEED = 1.0                            # 龙头前把手沿螺线行进速度（m/s）

# 板凳节数与把手数
HEAD_BOARD_NUM = 1      # 龙头节数
BODY_BOARD_NUM = 221    # 龙身节数
TAIL_BOARD_NUM = 1      # 龙尾节数
BOARD_NUM = HEAD_BOARD_NUM + BODY_BOARD_NUM + TAIL_BOARD_NUM   # 223 节
TOTAL_HANDLES = BOARD_NUM + 1               # 224 个把手中心

# 求解时间范围（s）
T_START = 0
T_END = 300

# 问题一结果文件
# 模板位于 data/ 目录（不修改），结果输出到 results/tables/ 下
RESULT1_TEMPLATE_PATH = os.path.join(DATA_DIR, "result1.xlsx")
RESULT1_PATH = os.path.join(TABLES_DIR, "result1.xlsx")

# 问题二结果文件
RESULT2_TEMPLATE_PATH = os.path.join(DATA_DIR, "result2.xlsx")
RESULT2_PATH = os.path.join(TABLES_DIR, "result2.xlsx")

# 问题二 G(t) 数据表格（粗搜索轨迹 + 加密扫描点）
G_TABLE_PATH = os.path.join(TABLES_DIR, "problem2_global_margin.xlsx")

# ============================================================
# 问题三：满足调头空间约束的最小螺距模型参数
# ============================================================
TURN_RADIUS = 4.5               # 调头空间半径 R（m），对应直径 9 m 的圆形区域
TURN_DIAMETER = 9.0             # 调头空间直径（m）

# 内层全过程扫描：龙头极径 r0 ∈ [4.5, R_HEAD_START]
R_HEAD_START = 20.0             # 外侧起始极径初值（m），自动扩展直至收敛
R_HEAD_MAX = 60.0               # 外侧起始极径上限（m），防止无限扩展
R_COARSE_POINTS = 45            # 内层极径粗扫描点数
R_FINE_STEP = 0.01              # 内层极径局部细化步长（m）
R_FINE_POINTS = 25              # 内层极径局部细化点数

# 外层螺距搜索
PITCH_SEARCH_MIN = 0.2          # 螺距粗搜索下界（m）
PITCH_SEARCH_MAX = 1.0          # 螺距粗搜索上界（m）
PITCH_COARSE_STEP = 0.02        # 螺距粗搜索步长（m）
EPS_PITCH = 1e-8                # 外层螺距二分收敛阈值（m）
EPS_G3 = 1e-6                   # 临界裕量容许误差（m）
DELTA_PITCH = 1e-3              # 最优性左右扰动步长（m）

# 问题三结果文件
RESULT3_PHI_TABLE = os.path.join(TABLES_DIR, "problem3_phi_vs_pitch.xlsx")
RESULT3_MARGIN_TABLE = os.path.join(TABLES_DIR, "problem3_margin_vs_radius.xlsx")
RESULT3_PHI_FIG = os.path.join(FIGURES_DIR, "problem3_phi_vs_pitch.png")
RESULT3_MARGIN_FIG = os.path.join(FIGURES_DIR, "problem3_margin_vs_radius.png")
RESULT3_CONFIG_FIG = os.path.join(FIGURES_DIR, "problem3_critical_config.png")

# ============================================================
# 问题四：S 形调头曲线模型参数
# ============================================================
P4_PITCH = 1.7                      # 盘入螺线螺距 p（m）
P4_B = P4_PITCH / (2.0 * math.pi)   # 螺线参数 b = p/(2π)（m/rad）
TURN_SPACE_RADIUS = TURN_RADIUS     # 调头空间半径 ρ = 4.5 m（复用问题三）

# 问题四结果文件
P4_RESULT_TEMPLATE_PATH = os.path.join(DATA_DIR, "result4.xlsx")
P4_RESULT_PATH = os.path.join(TABLES_DIR, "result4.xlsx")
P4_ROUTE_FIG = os.path.join(FIGURES_DIR, "problem4_route.png")
