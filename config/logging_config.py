"""
日志配置。控制台 INFO，文件 DEBUG，格式统一。
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | Path = "logs",
    log_file: str = "app.log",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    """初始化日志：控制台 + 文件。"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # 降低第三方库日志级别
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)
