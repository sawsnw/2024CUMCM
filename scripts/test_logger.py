"""验证日志生成的示例脚本。

运行方式:
    python scripts/test_logger.py

预期输出:
    1. 控制台打印 INFO/DEBUG/WARNING 级别的日志
    2. results/logs/ 下生成 test_logger_2026-07-14_*.log 文件
"""

from src.utils.logger import setup_logger

# 配置日志记录器，传入脚本名（不含扩展名）
logger = setup_logger(__name__, script_name="test_logger")

# 测试不同日志级别
logger.info("这是一条 INFO 级别的日志")
logger.debug("这是一条 DEBUG 级别的日志（默认不会输出到控制台）")
logger.warning("这是一条 WARNING 级别的日志")

# 模拟分阶段任务
logger.info("=" * 50)
logger.info("阶段一：数据加载")
logger.info("数据加载完成，共 1000 条记录")

logger.info("=" * 50)
logger.info("阶段二：模型训练")
logger.info("Epoch 1/10, loss=0.5234")
logger.info("Epoch 2/10, loss=0.4121")
logger.info("Epoch 3/10, loss=0.3892")
logger.info("模型训练完成")

logger.info("=" * 50)
logger.info("阶段三：结果保存")
logger.info("结果已保存至 results/figures/")
logger.info("全部任务执行完毕")
