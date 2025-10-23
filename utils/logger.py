"""로깅 유틸리티"""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """컬러 로깅 포맷터"""

    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        # 레벨에 따라 색상 적용
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.BOLD}{self.COLORS[levelname]}{levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    로거 설정

    Args:
        name: 로거 이름
        level: 로깅 레벨

    Returns:
        설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 기존 핸들러가 있으면 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 포맷 설정
    formatter = ColoredFormatter(
        fmt='%(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.propagate = False  # 상위 로거로 전파 방지

    return logger
