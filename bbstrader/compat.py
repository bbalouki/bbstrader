import platform
import sys


def setup_mock_modules():
    """Mock some modules not available on some OS to prevent import errors."""
    from unittest.mock import MagicMock

    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            return MagicMock()
        
    MOCK_MODULES = []

    # Mock Metatrader5 on Linux and MacOS
    if platform.system() != "Windows":
        MOCK_MODULES.append("MetaTrader5")

    # Mock posix On windows
    if platform.system() == "Windows":
        MOCK_MODULES.append("posix")

    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


setup_mock_modules()
