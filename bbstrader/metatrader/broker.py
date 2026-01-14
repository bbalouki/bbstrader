import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from bbstrader.api import Mt5client as client
from bbstrader.metatrader.utils import INIT_MSG, SymbolType, raise_mt5_error

try:
    import MetaTrader5 as mt5
except ImportError:
    import bbstrader.compat  # noqa: F401

COUNTRIES_STOCKS = {
    "USA": r"\b(US|USA)\b",
    "AUS": r"\b(Australia)\b",
    "BEL": r"\b(Belgium)\b",
    "DNK": r"\b(Denmark)\b",
    "FIN": r"\b(Finland)\b",
    "FRA": r"\b(France)\b",
    "DEU": r"\b(Germany)\b",
    "NLD": r"\b(Netherlands)\b",
    "NOR": r"\b(Norway)\b",
    "PRT": r"\b(Portugal)\b",
    "ESP": r"\b(Spain)\b",
    "SWE": r"\b(Sweden)\b",
    "GBR": r"\b(UK)\b",
    "CHE": r"\b(Switzerland)\b",
    "HKG": r"\b(Hong Kong)\b",
    "IRL": r"\b(Ireland)\b",
    "AUT": r"\b(Austria)\b",
}

EXCHANGES = {
    "XASX": r"Australia.*\(ASX\)",
    "XBRU": r"Belgium.*\(Euronext\)",
    "XCSE": r"Denmark.*\(CSE\)",
    "XHEL": r"Finland.*\(NASDAQ\)",
    "XPAR": r"France.*\(Euronext\)",
    "XETR": r"Germany.*\(Xetra\)",
    "XAMS": r"Netherlands.*\(Euronext\)",
    "XOSL": r"Norway.*\(NASDAQ\)",
    "XLIS": r"Portugal.*\(Euronext\)",
    "XMAD": r"Spain.*\(BME\)",
    "XSTO": r"Sweden.*\(NASDAQ\)",
    "XLON": r"UK.*\(LSE\)",
    "XNYS": r"US.*\((NYSE|ARCA|AMEX)\)",
    "NYSE": r"US.*\(NYSE\)",
    "ARCA": r"US.*\(ARCA\)",
    "AMEX": r"US.*\(AMEX\)",
    "NASDAQ": r"US.*\(NASDAQ\)",
    "BATS": r"US.*\(BATS\)",
    "XSWX": r"Switzerland.*\(SWX\)",
}

SYMBOLS_TYPE = {
    SymbolType.ETFs: r"\b(ETFs?|Exchange\s?Traded\s?Funds?|Trackers?)\b",
    SymbolType.BONDS: r"\b(Treasuries|Bonds|Bunds|Gilts|T-Notes|Fixed\s?Income)\b",
    SymbolType.FOREX: r"\b(Forex|FX|Currencies|Exotics?|Majors?|Minors?)\b",
    SymbolType.FUTURES: r"\b(Futures?|Forwards|Expiring|Front\s?Month)\b",
    SymbolType.STOCKS: r"\b(Stocks?|Equities?|Shares?|Blue\s?Chips?|Large\s?Cap)\b",
    SymbolType.INDICES: r"\b(Indices?|Index|Cash|Spot\s?Indices|Benchmarks)\b(?![^$]*(UKOIL|USOIL|WTI|BRENT))",
    SymbolType.COMMODITIES: r"\b(Commodit(ies|y)|Metals?|Precious|Bullion|Agricultures?|Energies?|Oil|Crude|WTI|BRENT|UKOIL|USOIL|Gas|NATGAS)\b",
    SymbolType.CRYPTO: r"\b(Cryptos?|Cryptocurrencies?|Digital\s?Assets?|DeFi|Altcoins)\b",
}


def check_mt5_connection(
    *,
    path=None,
    login=None,
    password=None,
    server=None,
    timeout=60_000,
    portable=False,
    **kwargs,
) -> bool:
    """
    Initialize the connection to the MetaTrader 5 terminal.

    Parameters
    ----------
    path : str, optional
        Path to the MetaTrader 5 terminal executable file.
        Defaults to ``None`` (e.g., ``"C:/Program Files/MetaTrader 5/terminal64.exe"``).
    login : int, optional
        The login ID of the trading account. Defaults to ``None``.
    password : str, optional
        The password of the trading account. Defaults to ``None``.
    server : str, optional
        The name of the trade server to which the client terminal is connected.
        Defaults to ``None``.
    timeout : int, optional
        Connection timeout in milliseconds. Defaults to ``60_000``.
    portable : bool, optional
        If ``True``, the portable mode of the terminal is used.
        Defaults to ``False``.
        See: https://www.metatrader5.com/en/terminal/help/start_advanced/start#portable

    Returns
    -------
    bool
        ``True`` if the connection is successfully established, otherwise ``False``.

    Notes
    -----
    If you want to launch multiple terminal instances:

    * First, launch each terminal in **portable mode**.
    * See instructions: https://www.metatrader5.com/en/terminal/help/start_advanced/start#configuration_file
    """

    if login is not None and server is not None:
        account_info = mt5.account_info()
        if account_info is not None:
            if account_info.login == login and account_info.server == server:
                return True

    init = False
    if path is None and (login or password or server):
        raise ValueError(
            "You must provide a path to the terminal executable file"
            "when providing login, password or server"
        )
    try:
        if path is not None:
            if login is not None and password is not None and server is not None:
                init = mt5.initialize(
                    path=path,
                    login=login,
                    password=password,
                    server=server,
                    timeout=timeout,
                    portable=portable,
                )
            else:
                init = mt5.initialize(path=path)
        else:
            init = mt5.initialize()
        if not init:
            raise_mt5_error(str(mt5.last_error()) + INIT_MSG)
    except Exception:
        raise_mt5_error(str(mt5.last_error()) + INIT_MSG)
    return init


class Broker(object):
    def __init__(
        self,
        name: str,
        timezone: Optional[str] = None,
        custom_patterns: Optional[Dict[SymbolType, str]] = None,
        custom_countries_stocks: Optional[Dict[str, str]] = None,
        custom_exchanges: Optional[Dict[str, str]] = None,
    ):
        self._name = name
        self._timezone = timezone
        self._patterns = {**SYMBOLS_TYPE, **(custom_patterns or {})}
        self._countries_stocks = {**COUNTRIES_STOCKS, **(custom_countries_stocks or {})}
        self._exchanges = {**EXCHANGES, **(custom_exchanges or {})}

    @property
    def name(self):
        return self._name

    @property
    def timezone(self):
        return self._timezone

    @property
    def countries_stocks(self):
        return self._countries_stocks

    @property
    def exchanges(self):
        return self._exchanges

    def __str__(self):
        return self.name

    def __eq__(self, other) -> bool:
        return self.name == other.name

    def __ne__(self, other) -> bool:
        return self.name != other.name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __hash__(self):
        return hash(self.name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timezone": self.timezone,
            "patterns": self._patterns,
            "countries_stocks": self._countries_stocks,
            "exchanges": self._exchanges,
        }

    def initialize_connection(self, **kwargs) -> bool:
        """Broker-specific connection initialization."""
        return check_mt5_connection(**kwargs)

    def get_terminal_timezone(self) -> str:
        """Fetch or override terminal timezone."""
        if self._timezone is not None:
            return self._timezone

        symbol = self.get_symbols()[0]
        tick = client.symbol_info_tick(symbol)

        if tick is None:
            return "Unknown (Market might be closed)"

        server_time = tick.time
        utc_now = datetime.now(timezone.utc).timestamp()

        # Check if the tick is stale (e.g., older than 10 hours).
        # This prevents calculating offsets based on weekend gaps.
        if abs(server_time - utc_now) > 3600 * 10:
            # Most Forex/CFD brokers use PLT/EEST (UTC+2 or UTC+3)
            # which maps to Europe/Nicosia or Europe/Athens.
            return "Europe/Nicosia"

        offset_hours = round((server_time - utc_now) / 3600)

        if offset_hours == 0:
            return "UTC"
        elif offset_hours in [2, 3]:
            return "Europe/Nicosia"
        elif offset_hours == 7:
            return "Asia/Bangkok"
        else:
            if -12 <= offset_hours <= 14:
                # Note: Etc/GMT signs are inverted.
                # If offset is +2 (server is ahead), we need Etc/GMT-2
                return f"Etc/GMT{-offset_hours:+d}"
            else:
                return "UTC"

    def get_broker_time(self, time: str, format: str):
        broker_time = datetime.strptime(time, format)
        broker_tz = self.get_terminal_timezone()
        broker_tz = ZoneInfo(broker_tz)
        broker_time = broker_time.replace(tzinfo=ZoneInfo("UTC"))
        return broker_time.astimezone(broker_tz)

    def get_symbol_type(self, symbol: str) -> SymbolType:
        info = client.symbol_info(symbol)
        if info is None:
            return SymbolType.unknown
        for sym_type, pattern in self._patterns.items():
            if re.search(re.compile(pattern, re.IGNORECASE), info.path):
                return sym_type
        return SymbolType.unknown

    def get_symbols(
        self,
        symbol_type: SymbolType | str = "ALL",
        check_etf: bool = False,
        save: bool = False,
        file_name: str = "symbols",
        include_desc: bool = False,
        display_total: bool = False,
    ) -> List[str]:
        symbols = client.symbols_get()
        if not symbols:
            raise_mt5_error("Failed to get symbols")

        symbol_list = []
        if symbol_type != "ALL":
            if (
                not isinstance(symbol_type, SymbolType)
                or symbol_type not in self._patterns
            ):
                raise ValueError(f"Unsupported symbol type: {symbol_type}")

        def check_etfs(info):
            if (
                symbol_type == SymbolType.ETFs
                and check_etf
                and "ETF" not in info.description
            ):
                raise ValueError(
                    f"{info.name} doesn't have 'ETF' in its description. "
                    "If this is intended, set check_etf=False."
                )

        if save:
            max_length = max(len(s.name) for s in symbols)
            file_path = f"{file_name}.txt"
            with open(file_path, mode="w", encoding="utf-8") as file:
                for s in symbols:
                    info = client.symbol_info(s.name)
                    if symbol_type == "ALL":
                        self._write_symbol(file, info, include_desc, max_length)
                        symbol_list.append(s.name)
                    else:
                        pattern = re.compile(self._patterns[symbol_type], re.IGNORECASE)
                        if re.search(pattern, info.path):
                            check_etfs(info)
                            self._write_symbol(file, info, include_desc, max_length)
                            symbol_list.append(s.name)
        else:
            for s in symbols:
                info = client.symbol_info(s.name)
                if symbol_type == "ALL":
                    symbol_list.append(s.name)
                else:
                    pattern = re.compile(self._patterns[symbol_type], re.IGNORECASE)
                    if re.search(pattern, info.path):
                        check_etfs(info)
                        symbol_list.append(s.name)

        if display_total:
            name = symbol_type if isinstance(symbol_type, str) else symbol_type.name
            print(
                f"Total number of {name} symbols: {len(symbol_list)}"
                if symbol_type != "ALL"
                else f"Total symbols: {len(symbol_list)}"
            )

        return symbol_list

    def _write_symbol(self, file, info, include_desc, max_length):
        if include_desc:
            space = " " * (max_length - len(info.name))
            file.write(info.name + space + "|" + info.description + "\n")
        else:
            file.write(info.name + "\n")

    def get_symbols_by_category(
        self, symbol_type: SymbolType | str, category: str, category_map: Dict[str, str]
    ) -> List[str]:
        if category not in category_map:
            raise ValueError(
                f"Unsupported category: {category}. Choose from: {', '.join(category_map)}"
            )

        symbols = self.get_symbols(symbol_type=symbol_type)
        pattern = re.compile(category_map[category], re.IGNORECASE)
        symbol_list = []
        for s in symbols:
            info = client.symbol_info(s)
            if re.search(pattern, info.path):
                symbol_list.append(s)
        return symbol_list

    def get_leverage_for_symbol(
        self, symbol: str, account_leverage: bool = True
    ) -> int:
        if account_leverage:
            return client.account_info().leverage
        s_info = client.symbol_info(symbol)
        if not s_info:
            raise ValueError(f"Symbol {symbol} not found")
        volume_min = s_info.volume_min
        contract_size = s_info.trade_contract_size
        av_price = (s_info.bid + s_info.ask) / 2
        action = random.choice([mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL])
        margin = client.order_calc_margin(action, symbol, volume_min, av_price)
        if margin is None or margin == 0:
            return client.account_info().leverage  # Fallback
        return round((volume_min * contract_size * av_price) / margin)

    def adjust_tick_values(
        self,
        symbol: str,
        tick_value_loss: float,
        tick_value_profit: float,
        contract_size: float,
    ) -> Tuple[float, float]:
        symbol_type = self.get_symbol_type(symbol)
        if (
            symbol_type == SymbolType.COMMODITIES
            or symbol_type == SymbolType.FUTURES
            or symbol_type == SymbolType.CRYPTO
            and contract_size > 1
        ):
            tick_value_loss = tick_value_loss / contract_size
            tick_value_profit = tick_value_profit / contract_size
        return tick_value_loss, tick_value_profit

    def get_min_stop_level(self, symbol: str) -> int:
        s_info = client.symbol_info(symbol)
        return s_info.trade_stops_level if s_info else 0

    def validate_lot_size(self, symbol: str, lot: float) -> float:
        s_info = client.symbol_info(symbol)
        if not s_info:
            raise ValueError(f"Symbol {symbol} not found")
        if lot > s_info.volume_max:
            return s_info.volume_max / 2
        if lot < s_info.volume_min:
            return s_info.volume_min
        steps = self._volume_step(s_info.volume_step)
        return round(lot, steps) if steps > 0 else round(lot)

    def _volume_step(self, value: float) -> int:
        value_str = str(value)
        if "." in value_str and value_str != "1.0":
            decimal_index = value_str.index(".")
            return len(value_str) - decimal_index - 1
        return 0

    def get_currency_conversion_factor(
        self, symbol: str, base_currency: str, account_currency: str
    ) -> float:
        if base_currency == account_currency:
            return 1.0
        conversion_symbol = f"{base_currency}{account_currency}"
        info = client.symbol_info_tick(conversion_symbol)
        if info:
            return (info.ask + info.bid) / 2
        return 1.0
