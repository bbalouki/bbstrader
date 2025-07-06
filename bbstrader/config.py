import logging
from pathlib import Path
from typing import List


TERMINAL = "/terminal64.exe"
BASE_FOLDER = "C:/Program Files/"

AMG_PATH = BASE_FOLDER + "Admirals Group MT5 Terminal" + TERMINAL
PGL_PATH = BASE_FOLDER + "Pepperstone MetaTrader 5" + TERMINAL
FTMO_PATH = BASE_FOLDER + "FTMO MetaTrader 5" + TERMINAL
JGM_PATH = BASE_FOLDER + "JustMarkets MetaTrader 5" + TERMINAL

BROKERS_PATHS = {
    "AMG": AMG_PATH,
    "FTMO": FTMO_PATH,
    "PGL": PGL_PATH,
    "JGM": JGM_PATH,
}


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
    def __init__(self, levels: List[int]):
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
    def formatTime(self, record, datefmt=None):
        if hasattr(record, "custom_time"):
            # Use the custom time if provided
            record.created = record.custom_time.timestamp()
        return super().formatTime(record, datefmt)


class CustomLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
        custom_time=None,
    ):
        if extra is None:
            extra = {}
        # Add custom_time to the extra dictionary if provided
        if custom_time:
            extra["custom_time"] = custom_time
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

    def info(self, msg, *args, custom_time=None, **kwargs):
        self._log(logging.INFO, msg, args, custom_time=custom_time, **kwargs)

    def debug(self, msg, *args, custom_time=None, **kwargs):
        self._log(logging.DEBUG, msg, args, custom_time=custom_time, **kwargs)

    def warning(self, msg, *args, custom_time=None, **kwargs):
        self._log(logging.WARNING, msg, args, custom_time=custom_time, **kwargs)

    def error(self, msg, *args, custom_time=None, **kwargs):
        self._log(logging.ERROR, msg, args, custom_time=custom_time, **kwargs)

    def critical(self, msg, *args, custom_time=None, **kwargs):
        self._log(logging.CRITICAL, msg, args, custom_time=custom_time, **kwargs)


def config_logger(log_file: str, console_log=True):
    # Use the CustomLogger
    logging.setLoggerClass(CustomLogger)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Custom formatter
    formatter = CustomFormatter(
        "%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    if console_log:
        # Handler for the console with a different level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
