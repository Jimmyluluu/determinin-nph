#!/usr/bin/env python3
"""
日誌管理模組
提供統一的 logging 介面
"""
import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str = "MindScope",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    設定並返回 logger

    Parameters:
        name (str): Logger 名稱
        level (int): 日誌級別
        log_file (Optional[str]): 日誌檔案路徑，None 表示不寫入檔案
        console_output (bool): 是否輸出到終端

    Returns:
        logging.Logger: 設定好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除現有的 handlers
    logger.handlers.clear()

    # 格式設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 簡化的終端格式（不顯示時間戳）
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # 添加終端輸出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 添加檔案輸出
    if log_file:
        # 確保日誌目錄存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 全域 logger 實例
_global_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """
    取得全域 logger 實例

    Returns:
        logging.Logger: Logger 實例
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger()
    return _global_logger


def set_log_level(level: int):
    """
    設定全域 logger 的日誌級別

    Parameters:
        level (int): 日誌級別 (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = get_logger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


class AnalysisLogger:
    """
    分析專用的 logger 包裝器
    提供更便利的日誌記錄方法
    """

    def __init__(self, case_name: str = ""):
        self.logger = get_logger()
        self.case_name = case_name

    def _format_message(self, message: str) -> str:
        """格式化訊息（加入案例名稱）"""
        if self.case_name:
            return f"[{self.case_name}] {message}"
        return message

    def info(self, message: str):
        """記錄資訊"""
        self.logger.info(self._format_message(message))

    def warning(self, message: str):
        """記錄警告"""
        self.logger.warning(self._format_message(message))

    def error(self, message: str):
        """記錄錯誤"""
        self.logger.error(self._format_message(message))

    def debug(self, message: str):
        """記錄除錯資訊"""
        self.logger.debug(self._format_message(message))

    def critical(self, message: str):
        """記錄嚴重錯誤"""
        self.logger.critical(self._format_message(message))

    def success(self, message: str):
        """記錄成功訊息（使用 INFO 級別）"""
        self.logger.info(self._format_message(f"✅ {message}"))

    def failure(self, message: str):
        """記錄失敗訊息（使用 ERROR 級別）"""
        self.logger.error(self._format_message(f"❌ {message}"))

    def progress(self, current: int, total: int, message: str = ""):
        """記錄進度"""
        progress_msg = f"進度: {current}/{total}"
        if message:
            progress_msg += f" - {message}"
        self.logger.info(self._format_message(progress_msg))


# 便利函數
def log_section(title: str, width: int = 80):
    """
    記錄區段標題

    Parameters:
        title (str): 標題文字
        width (int): 總寬度
    """
    logger = get_logger()
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)


def log_subsection(title: str, width: int = 80):
    """
    記錄子區段標題

    Parameters:
        title (str): 標題文字
        width (int): 總寬度
    """
    logger = get_logger()
    logger.info("-" * width)
    logger.info(title)
    logger.info("-" * width)
