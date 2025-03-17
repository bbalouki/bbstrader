import platform
import sys


def setup_mock_metatrader():
    """Mock MetaTrader5 on Linux and MacOS to prevent import errors."""
    if platform.system() != "Windows":
        from unittest.mock import MagicMock

        class Mock(MagicMock):
            @classmethod
            def __getattr__(cls, name):
                return MagicMock()

        MOCK_MODULES = ["MetaTrader5"]
        sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


setup_mock_metatrader()
