import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional


def get_config_dir(name: str = ".bbstrader") -> Path:
    """
    Get the path to the configuration directory.

    Args:
        name: The name of the configuration directory.

    Returns:
        The path to the configuration directory.
    """
    home_dir = Path.home() / name
    if not home_dir.exists():
        home_dir.mkdir()
    return home_dir


BBSTRADER_DIR = get_config_dir()


class LogLevelFilter(logging.Filter):
    def __init__(self, levels: List[int]) -> None:
        """
        Initializes the filter with specific logging levels.

        Args:
            levels: A list of logging level values (integers) to include.
        """
        super().__init__()
        self.levels = levels

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filters log records based on their level.

        Args:
            record: The log record to check.

        Returns:
            True if the record's level is in the allowed levels, False otherwise.
        """
        return record.levelno in self.levels


class CustomFormatter(logging.Formatter):
    def formatTime(
        self, record: logging.LogRecord, datefmt: Optional[str] = None
    ) -> str:
        if hasattr(record, "custom_time"):
            # Use the custom time if provided
            record.created = record.custom_time.timestamp() # type: ignore
        return super().formatTime(record, datefmt)


class CustomLogger(logging.Logger):
    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)

    def log(
        self,
        level: int,
        msg: object,
        *args: object,
        custom_time: Optional[datetime] = None,
        **kwargs: Any,
    ) -> None:
        if custom_time:
            if "extra" not in kwargs or kwargs["extra"] is None:
                kwargs["extra"] = {}
            kwargs["extra"]["custom_time"] = custom_time
        super().log(level, msg, *args, **kwargs)


def config_logger(log_file: str, console_log: bool = True) -> logging.Logger:
    logging.setLoggerClass(CustomLogger)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = CustomFormatter(
        "%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
