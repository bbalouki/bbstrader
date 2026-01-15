"""
Simplified Investment & Trading Toolkit with Python & C++

"""

__author__ = "Bertin Balouki SIMYELI"
__copyright__ = "2023-2026 Bertin Balouki SIMYELI"
__email__ = "bertin@bbs-trading.com"
__license__ = "MIT"

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bbstrader")
except PackageNotFoundError:
    __version__ = "unknown"


from bbstrader import compat  # noqa: F401
from bbstrader import core  # noqa: F401
from bbstrader import btengine  # noqa: F401
from bbstrader import metatrader  # noqa: F401
from bbstrader import models  # noqa: F401
from bbstrader import trading  # noqa: F401
from bbstrader.config import config_logger  # noqa: F401

version = __version__
