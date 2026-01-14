from operator import attrgetter

import pandas as pd

from bbstrader.api.handlers import Mt5Handlers
from bbstrader.api.metatrader_client import *  # type: ignore # noqa: F403

# ruff: noqa: F405
classes_to_patch = [
    AccountInfo,
    BookInfo,
    OrderCheckResult,
    OrderSentResult,
    RateInfo,
    SymbolInfo,
    TerminalInfo,
    TickInfo,
    TradeDeal,
    TradeOrder,
    TradePosition,
    TradeRequest,
]


def dynamic_str(self):
    fields = set()
    for name in dir(self):
        if name.startswith("_"):
            continue
        try:
            value = getattr(self, name)
            if not callable(value):
                fields.add(f"{name}={value!r}")
        except Exception:
            pass
    return f"{type(self).__name__}({', '.join(fields)})"


for cls in classes_to_patch:
    cls.__str__ = dynamic_str
    cls.__repr__ = dynamic_str


def trade_object_to_df(obj_list):
    """
    Fast conversion of a list of C++ bound objects to a pandas DataFrame.
    """
    if not obj_list:
        return pd.DataFrame()

    first_obj = obj_list[0]
    columns = [
        name
        for name in dir(first_obj)
        if not name.startswith("_") and not callable(getattr(first_obj, name))
    ]
    fetcher = attrgetter(*columns)
    data = [fetcher(obj) for obj in obj_list]
    df = pd.DataFrame(data, columns=columns)
    return df


Mt5client = MetaTraderClient(Mt5Handlers)
