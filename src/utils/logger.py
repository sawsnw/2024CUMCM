"""日志工具模块。

提供统一的日志配置函数，供各入口脚本使用。
"""

import logging
import sys

from src.config import (
    LOGS_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATETIME_FORMAT,
    get_log_filepath,
)


def setup_logger(
    name: str = __name__,
    script_name: str | None = None,
    level: str | int | None = None,
    add_console: bool = True,
) -> logging.Logger:
    """配置并返回日志记录器。

    Args:
        name: 日志记录器名称，默认为调用模块的 __name__。
        script_name: 脚本名称（不含扩展名），用于日志文件名。
                     如果为 None，则只添加控制台日志。
        level: 日志级别，默认使用 config.py 中的 LOG_LEVEL。
        add_console: 是否添加控制台日志处理器，默认为 True。

    Returns:
        配置好的日志记录器。
    """
    logger = logging.getLogger(name)

    # 设置日志级别
    log_level = level or LOG_LEVEL
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT)

    # 文件日志处理器（日志文件不覆盖，追加写入）
    if script_name:
        log_path = get_log_filepath(script_name)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 控制台日志处理器（终端可能为 GBK 编码，用 errors='replace' 避免特殊字符报错）
    if add_console:
        console_handler = logging.StreamHandler(sys.stdout)
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
