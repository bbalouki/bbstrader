import platform
import sys


def setup_mock_metatrader():
    """Mock MetaTrader5 on Linux to prevent import errors."""
    if platform.system() == "Linux":
        from unittest.mock import MagicMock

        class Mock(MagicMock):
            @classmethod
            def __getattr__(cls, name):
                return MagicMock()

        MOCK_MODULES = ["MetaTrader5"]
        sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

        print(
            "Warning: MetaTrader5 is not available on Linux. A mock version is being used."
        )


setup_mock_metatrader()
