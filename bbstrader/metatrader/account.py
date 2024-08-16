import re
import os
import pandas as pd
import urllib.request
from datetime import datetime
import MetaTrader5 as mt5
from currency_converter import SINGLE_DAY_ECB_URL, CurrencyConverter
from bbstrader.metatrader.utils import (
    _BROKERS_, _ADMIRAL_MARKETS_URL_, _JUST_MARKETS_URL_, INIT_MSG,
    raise_mt5_error, AccountInfo, TerminalInfo, _SYMBOLS_TYPE_,
    SymbolInfo, TickInfo, TradeRequest, OrderCheckResult,
    OrderSentResult, TradePosition, TradeOrder, TradeDeal,
)
from typing import Tuple, Union, List, Dict, Optional, Literal


class Account(object):
    """
    The `Account` class is utilized to retrieve information about 
    the current trading account or a specific account. 
    It enables interaction with the MT5 terminal to manage account details, 
    including account informations, terminal status, financial instrument details, 
    active orders, open positions, and trading history.

    Example:
    >>> # Instantiating the Account class
    >>> account = Account()

    >>> # Getting account information
    >>> account_info = account.get_account_info()

    >>> # Printing account information
    >>> account.print_account_info()

    >>> # Getting terminal information
    >>> terminal_info = account.get_terminal_info()

    >>> # Retrieving and printing symbol information
    >>> symbol_info = account.show_symbol_info('EURUSD')

    >>> # Getting active orders
    >>> orders = account.get_orders()

    >>> # Fetching open positions
    >>> positions = account.get_positions()

    >>> # Accessing trade history
    >>> from_date = datetime(2020, 1, 1)
    >>> to_date = datetime.now()
    >>> trade_history = account.get_trade_history(from_date, to_date)
    """

    def __init__(self):
        if not mt5.initialize():
            raise_mt5_error(message=INIT_MSG)
        self._check_brokers()

    def _check_brokers(self):
        supported = _BROKERS_
        broker = self.get_terminal_info().company
        if broker not in supported.values():
            raise ValueError(
                f"{broker} is not currently supported broker for the Account() class\n"
                f"Currently Supported brokers are: {', '.join(supported)}\n"
                f"For {_BROKERS_['AMG']}, See {_ADMIRAL_MARKETS_URL_}\n"
                f"For {_BROKERS_['JGM']}, See {_JUST_MARKETS_URL_}"
            )

    def get_account_info(
        self,
        account:  Optional[int] = None,
        password: Optional[str] = None,
        server:   Optional[str] = None,
        timeout:  Optional[int] = 60_000
    ) -> Union[AccountInfo, None]:
        """
        Get info on the current trading account or a specific account .

        Args:
            account (int, optinal) : MT5 Trading account number.
            password (str, optinal): MT5 Trading account password.
            server (str, optinal)  : MT5 Trading account server 
                [Brokers or terminal server ["demo", "real"]]
                If no server is set, the last used server is applied automaticall
            timeout (int, optinal):
                 Connection timeout in milliseconds. Optional named parameter. 
                 If not specified, the value of 60 000 (60 seconds) is applied. 
                 If the connection is not established within the specified time, 
                 the call is forcibly terminated and the exception is generated.

        Returns:
        -   AccountInfo in the form of a Namedtuple structure. 
        -   None in case of an error

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        # connect to the trade account specifying a password and a server
        if (
            account is not None and
            password is not None and
            server is not None
        ):
            try:
                authorized = mt5.login(
                    account, password=password, server=server, timeout=timeout)
                if not authorized:
                    raise_mt5_error(
                        message=f"Failed to connect to  account #{account}")
                else:
                    info = mt5.account_info()
                    if info is None:
                        return None
                    else:
                        return AccountInfo(**info._asdict())
            except Exception as e:
                raise_mt5_error(e)
        else:
            try:
                info = mt5.account_info()
                if info is None:
                    return None
                else:
                    return AccountInfo(**info._asdict())
            except Exception as e:
                raise_mt5_error(e)

    def _show_info(self, info_getter, info_name, symbol=None):
        """
        Generic function to retrieve and print information.

        Args:
            info_getter (callable): Function to retrieve the information.
            info_name (str): Name of the information being retrieved.
            symbol (str, optional): Symbol name, required for some info types. 
                                     Defaults to None.

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """

        # Call the provided info retrieval function
        if symbol is not None:
            info = info_getter(symbol)
        else:
            info = info_getter()

        if info is not None:
            info_dict = info._asdict()
            df = pd.DataFrame(list(info_dict.items()),
                              columns=['PROPERTY', 'VALUE'])

            # Construct the print message based on whether a symbol is provided
            if symbol:
                print(
                    f"\n{info_name.upper()} INFO FOR {symbol} ({info.description})")
            else:
                print(f"\n{info_name.upper()} INFORMATIONS:")

            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(df.to_string())
        else:
            if symbol:
                msg = self._symbol_info_msg(symbol)
                raise_mt5_error(message=msg)
            else:
                raise_mt5_error()

    def show_account_info(self):
        """ Helper function to  print account info"""
        self._show_info(self.get_account_info, "account")

    def get_terminal_info(self, show=False) -> Union[TerminalInfo, None]:
        """
        Get the connected MetaTrader 5 client terminal status and settings.

        Args:
            show (bool): If True the Account information will be printed

        Returns:
        -   TerminalInfo in the form of NamedTuple Structure. 
        -   None in case of an error

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                return None
        except Exception as e:
            raise_mt5_error(e)

        terminal_info_dict = terminal_info._asdict()
        # convert the dictionary into DataFrame and print
        df = pd.DataFrame(list(terminal_info_dict.items()),
                          columns=['PROPERTY', 'VALUE'])
        if show:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(df.to_string())
        return TerminalInfo(**terminal_info_dict)

    def convert_currencies(self, qty: float, from_c: str, to_c: str) -> float:
        """Convert amount from a currency to another one.

        Args:
            qty (float): The amount of `currency` to convert.
            from_c (str): The currency to convert from.
            to_c (str): The currency to convert to.

        Returns:
        -   The value of `qty` in converted in `to_c`.

        Notes:
            If `from_c` or `to_co` are not supported, the `qty` will be return;
            check "https://www.ecb.europa.eu/stats/eurofxref/eurofxref.zip"
            for supported currencies or you can take a look at the `CurrencyConverter` project
            on Github https://github.com/alexprengere/currencyconverter .
        """
        filename = f"ecb_{datetime.now():%Y%m%d}.zip"
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(SINGLE_DAY_ECB_URL, filename)
            c = CurrencyConverter(filename)
        os.remove(filename)
        supported = c.currencies
        if (from_c not in supported or
                to_c not in supported
                ):
            rate = qty
        else:
            rate = c.convert(amount=qty, currency=from_c, new_currency=to_c)
        return rate

    def get_currency_rates(self, symbol: str) -> Dict[str, str]:
        """
        Args:
            symbol (str): The symbol for which to get currencies

        Returns:
            - `base currency` (bc)  
            - `margin currency` (mc)
            - `profit currency` (pc)
            - `account currency` (ac)

        Exemple:
        >>> account =  Account()
        >>> account.get_rates('EURUSD')
        {'bc': 'EUR', 'mc': 'EUR', 'pc': 'USD', 'ac': 'USD'}
        """
        info = self.get_symbol_info(symbol)
        bc = info.currency_base
        pc = info.currency_profit
        mc = info.currency_margin
        ac = self.get_account_info().currency
        return {'bc': bc, 'mc': mc, 'pc': pc, 'ac': ac}

    def get_symbols(self,
                    symbol_type="ALL",
                    check_etf=False,
                    save=False,
                    file_name="symbols",
                    include_desc=False,
                    display_total=False
                    ) -> List[str]:
        """ 
        Get all specified financial instruments from the MetaTrader 5 terminal.

        Args:
            symbol_type (str): The category of instrument to get. 
                - `ALL`: For all available symbols
                - `STK`: Stocks (e.g., 'GOOGL')
                - `ETF`: ETFs (e.g., 'QQQ')
                - `IDX`: Indices (e.g., 'SP500')
                - `FX`: Forex pairs (e.g., 'EURUSD')
                - `COMD`: Commodities (e.g., 'CRUDOIL', 'GOLD')
                - `FUT`: Futures (e.g., 'USTNote_U4')
                - `CRYPTO`: Cryptocurrencies (e.g., 'BTC', 'ETH')

            check_etf (bool): If True and symbol_type is 'etf', check if the 
                ETF description contains 'ETF'.

            save (bool): If True, save the symbols to a file.

            file_name (str): The name of the file to save the symbols to 
                (without the extension).

            include_desc (bool): If True, include the symbol's description 
                in the output and saved file.

        Returns:
            list: A list of symbols. 

        Raises:
            Exception: If there is an error connecting to MT5 or retrieving symbols.
        """
        symbols = mt5.symbols_get()
        if not symbols:
            raise_mt5_error()

        symbol_list = []
        patterns = _SYMBOLS_TYPE_

        if symbol_type != 'ALL':
            if symbol_type not in patterns:
                raise ValueError(f"Unsupported symbol type: {symbol_type}")

        if save:
            max_lengh = max([len(s.name) for s in symbols])
            file_path = f"{file_name}.txt"
            with open(file_path, mode='w', encoding='utf-8') as file:
                for s in symbols:
                    info = self.get_symbol_info(s.name)
                    if symbol_type == 'ALL':
                        self._write_symbol(file, info, include_desc, max_lengh)
                        symbol_list.append(s.name)
                    else:
                        pattern = re.compile(patterns[symbol_type])
                        match = re.search(pattern, info.path)
                        if match:
                            if symbol_type == "ETF" and check_etf and "ETF" not in info.description:
                                raise ValueError(
                                    f"{info.name} doesn't have 'ETF' in its description. "
                                    "If this is intended, set check_etf=False."
                                )
                            self._write_symbol(
                                file, info, include_desc, max_lengh)
                            symbol_list.append(s.name)

        else:  # If not saving to a file, just process the symbols
            for s in symbols:
                info = self.get_symbol_info(s.name)
                if symbol_type == 'ALL':
                    symbol_list.append(s.name)
                else:
                    pattern = re.compile(
                        patterns[symbol_type])  # , re.IGNORECASE
                    match = re.search(pattern, info.path)
                    if match:
                        if symbol_type == "ETF" and check_etf and "ETF" not in info.description:
                            raise ValueError(
                                f"{info.name} doesn't have 'ETF' in its description. "
                                "If this is intended, set check_etf=False."
                            )
                        symbol_list.append(s.name)

        # Print a summary of the retrieved symbols
        if display_total:
            names = {
                "ALL": 'Symbols',
                "STK": 'Stocks',
                "ETF": 'ETFs',
                "IDX": 'Indices',
                "FX":  'Forex Paires',
                "COMD": 'Commodities',
                "FUT": 'Futures',
                "CRYPTO": 'Cryptos Assets'
            }
            print(f"Total {names[symbol_type]}: {len(symbol_list)}")

        return symbol_list

    def _write_symbol(self, file, info, include_desc, max_lengh):
        """Helper function to write symbol information to a file."""
        if include_desc:
            space = " "*int(max_lengh-len(info.name))
            file.write(info.name + space + '|' +
                       info.description + '\n')
        else:
            file.write(info.name + '\n')

    def get_symbol_type(
            self,
            symbol: str
    ) -> Literal[
            "STK", "ETF", "IDX", "FX", "COMD", "FUT", "CRYPTO", "unknown"]:
        """
        Determines the type of a given financial instrument symbol.

        Args:
            symbol (str): The symbol of the financial instrument (e.g., `GOOGL`, `EURUSD`).

        Returns:
        -   symbol_type (str):
            - `STK`: For  Stocks (e.g., `GOOGL')
            - `ETF`: For ETFs (e.g., `QQQ`)
            - `IDX`: For Indices (e.g., `SP500')
            - `FX` : For Forex pairs (e.g., `EURUSD`)
            - `COMD`:For Commodities (e.g., `CRUDOIL`, `GOLD`)
            - `FUT` : For Futures (e.g., `USTNote_U4`)
            - `CRYPTO`: For Cryptocurrencies (e.g., `BTC`, `ETH`) 
                Returns `unknown` if the type cannot be determined. 
        """
        patterns = _SYMBOLS_TYPE_
        info = self.get_symbol_info(symbol)
        for symbol_type, pattern in patterns.items():
            match = re.search(pattern, info.path)  # , re.IGNORECASE
            if match:
                return symbol_type
        return "unknown"

    def _get_symbols_by_category(self, symbol_type, category, category_map):
        if category not in category_map:
            raise ValueError(
                f"Unsupported category: {category}. Choose from: {', '.join(category_map)}"
            )

        symbols = self.get_symbols(symbol_type=symbol_type)
        pattern = re.compile(category_map[category], re.IGNORECASE)

        symbol_list = []
        for s in symbols:
            info = self.get_symbol_info(s)
            match = re.search(pattern, info.path)
            if match:
                symbol_list.append(s)
        return symbol_list

    def get_fx_symbols(
            self,
            category: Literal["majors", "minors", "exotics"] = 'majors'
    ) -> List[str]:
        """
        Retrieves a list of forex symbols belonging to a specific category.

        Args:
            category (str, optional): The category of forex symbols to retrieve. 
                                        Possible values are 'majors', 'minors', 'exotics'. 
                                        Defaults to 'majors'.

        Returns:
            list: A list of forex symbol names matching the specified category.

        Raises:
            ValueError: If an unsupported category is provided.
        
        Notes:
            This mthods works primarly with Admirals Group AS products,
            For other brokers use `get_symbols()`
        """
        fx_categories = {
            "majors": r"\b(Majors?)\b",
            "minors": r"\b(Minors?)\b",
            "exotics": r"\b(Exotics?)\b",
        }
        return self._get_symbols_by_category('forex', category, fx_categories)

    def get_stocks_from(self, country_code: str = 'USA') -> List[str]:
        """
        Retrieves a list of stock symbols from a specific country.

        Supported countries are:
            * **Australia:** AUS
            * **Belgium:** BEL
            * **Denmark:** DNK 
            * **Finland:** FIN
            * **France:** FRA
            * **Germany:** DEU
            * **Netherlands:** NLD
            * **Norway:** NOR
            * **Portugal:** PRT
            * **Spain:** ESP
            * **Sweden:** SWE
            * **United Kingdom:** GBR
            * **United States:** USA
            * **Switzerland:** CHE

        Args:
            country (str, optional): The country code of stocks to retrieve.
                                    Defaults to 'USA'.

        Returns:
            list: A list of stock symbol names from the specified country.

        Raises:
            ValueError: If an unsupported country is provided.
        
        Notes:
            This mthods works primarly with Admirals Group AS products,
            For other brokers use `get_symbols()`
        """
        country_map = {
            "USA": r"\b(US)\b",
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
        }
        return self._get_symbols_by_category('stocks', country_code, country_map)

    def get_symbol_info(self, symbol: str) -> Union[SymbolInfo, None]:
        """Get symbol properties

        Args:
            symbol (str): Symbol name

        Returns:
        -   AccountInfo in the form of a NamedTuple().
        -   None in case of an error.

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            else:
                return SymbolInfo(**symbol_info._asdict())
        except Exception as e:
            msg = self._symbol_info_msg(symbol)
            raise_mt5_error(message=f"{e+msg}")

    def show_symbol_info(self, symbol: str):
        """
        Print symbol properties

        Args:
            symbol (str): Symbol name
        """
        self._show_info(self.get_symbol_info, "symbol", symbol=symbol)

    def _symbol_info_msg(self, symbol):
        return (
            f"No history found for {symbol} in Market Watch.\n"
            f"* Ensure {symbol} is selected and displayed in the Market Watch window.\n"
            f"* See https://www.metatrader5.com/en/terminal/help/trading/market_watch\n"
            f"* Ensure the symbol name is correct.\n"
        )

    def get_tick_info(self, symbol: str) -> Union[TickInfo, None]:
        """Get symbol tick properties

        Args:
            symbol (str): Symbol name

        Returns:
        -   AccountInfo in the form of a NamedTuple(). 
        -   None in case of an error.

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        try:
            tick_info = mt5.symbol_info_tick(symbol)
            if tick_info is None:
                return None
            else:
                return TickInfo(**tick_info._asdict())
        except Exception as e:
            msg = self._symbol_info_msg(symbol)
            raise_mt5_error(message=f"{e+msg}")

    def show_tick_info(self, symbol: str):
        """
        Print Tick properties

        Args:
            symbol (str): Symbol name
        """
        self._show_info(self.get_tick_info, "tick", symbol=symbol)

    def check_order(self,
                    request: TradeRequest) -> OrderCheckResult:
        """
        Check funds sufficiency for performing a required trading operation.

        Args:
            request (TradeRequest): `TradeRequest` type structure describing 
                a required trading action. Required unnamed parameter. 

        Returns:
        -   Check result as the `OrderCheckResult` structure. 
            The request field in the answer contains the structureof a trading request 
            passed to `check_order()`.

        Raises:
            MT5TerminalError: A specific exception based on the error code.

        Notes:
            Successful sending of a request does not entail that the requested trading 
            operation will be executed successfully. 
        """
        try:
            result = mt5.order_check(request)
            return OrderCheckResult(**result._asdict())
        except Exception as e:
            raise_mt5_error(e)

    def send_order(self,
                   request: TradeRequest) -> OrderSentResult:
        """
        Send a request to perform a trading operation from the terminal to the trade server. 

         Args:
            request (TradeRequest): `OrderSentResult` type structure describing a required 
            trading action. Required unnamed parameter.

        Returns:
        -   Execution result as the `OrderSentResult` structure. 
            The request field in the answer contains the structure of a trading request 
            passed to `send_order()`.

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        try:
            result = mt5.order_send(request)
            return OrderSentResult(**result._asdict())
        except Exception as e:
            raise_mt5_error(e)

    def get_positions(self,
                      symbol: Optional[str] = None,
                      group: Optional[str] = None,
                      ticket: Optional[int] = None,
                      to_df: bool = False
                      ) -> Union[pd.DataFrame, Tuple[TradePosition], None]:
        """
        Get open positions with the ability to filter by symbol or ticket. 
        There are four call options.

        * Call without parameters. Return open positions for all symbols.
        * Call specifying a symbol open positions should be received for.
        * Call specifying a group of symbols open positions should be received for.
        * Call specifying a position ticket.

        Args:
            symbol(str): Symbol name. Optional named parameter. 
                If a symbol is specified, the ticket parameter is ignored.

            group (str) The filter for arranging a group of necessary symbols. 
                Optional named parameter. If the group is specified, 
                the function returns only positions meeting a specified criteria 
                for a symbol name.

            ticket (int): Optional named parameter.
                Position ticket. Unique number assigned to each newly opened position. 
                It usually matches the ticket of an order used to open the position 
                except when the ticket is changed as a result of service operations on the server, 
                for example, when charging swaps with position re-opening.

            to_df (bool): If True, a DataFrame is returned.

        Returns:
        -   TradePosition in the form of a named tuple structure (namedtuple) or pd.DataFrame(). 
        -   None in case of an error.

        Notes:
            The method allows receiving all history orders within a specified period.
            The `group` parameter may contain several comma separated conditions. 
            A condition can be set as a mask using '*'. 
            The logical negation symbol '!' can be used for an exclusion. 
            All conditions are applied sequentially, which means conditions of including to a group 
            should be specified first followed by an exclusion condition. 
            For example, `group`="*, !EUR" means that deals for all symbols should be selected first 
            and the ones containing "EUR" in symbol names should be excluded afterwards.
        """
        if (symbol is not None) + (group is not None) + (ticket is not None) > 1:
            raise ValueError(
                "Only one of 'symbol', 'group', or 'ticket' can be specified as filter or None of them.")

        if symbol is not None:
            positions = mt5.positions_get(symbol=symbol)
        elif group is not None:
            positions = mt5.positions_get(group=group)
        elif ticket is not None:
            positions = mt5.positions_get(ticket=ticket)
        else:
            positions = mt5.positions_get()

        if len(positions) == 0:
            return None
        if to_df:
            df = pd.DataFrame(list(positions), columns=positions[0]._asdict())
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id'],
                    axis=1, inplace=True)
            return df
        else:
            trade_positions = [
                TradePosition(**p._asdict()) for p in positions]
            return tuple(trade_positions)

    def get_trades_history(
        self,
        date_from: datetime = datetime(2000, 1, 1),
        date_to: Optional[datetime] = None,
        group: Optional[str] = None,
        ticket: Optional[int] = None,   # TradeDeal.ticket
        position: Optional[int] = None,  # TradePosition.ticket
        to_df: bool = True,
        save: bool = False
    ) -> Union[pd.DataFrame, Tuple[TradeDeal], None]:
        """
        Get deals from trading history within the specified interval 
        with the ability to filter by `ticket` or `position`.

        * Call with a `time interval`. Return all deals falling within the specified interval.
        * Call specifying the `order ticket`. Return all deals having the specified `order ticket` 
            in the `DEAL_ORDER` property.
        * Call specifying the `position ticket`. Return all deals having the specified `position ticket` 
            in the `DEAL_POSITION_ID` property.

        Args:
            date_from (datetime): Date the bars are requested from. 
                Set by the `datetime` object or as a number of seconds elapsed since 1970.01.01. 
                Bars with the open time >= date_from are returned. Required unnamed parameter.

            date_to (datetime): Same as `date_from`

            group (str) The filter for arranging a group of necessary symbols. 
                Optional named parameter. If the group is specified, 
                the function returns only positions meeting a specified criteria 
                for a symbol name.

            ticket (int): Ticket of an order (stored in `DEAL_ORDER`) all deals should be received for. 
                Optional parameter. If not specified, the filter is not applied.

            position (int): Ticket of a position (stored in `DEAL_POSITION_ID`) all deals should be received for. 
                Optional parameter. If not specified, the filter is not applied.

            to_df (bool): If True, a DataFrame is returned.

            save (bool): If set to True, a csv file will be create a to save the history

        Returns:
        -   TradeDeal in the form of a named tuple structure (namedtuple) or pd.DataFrame().
        -   None in case of an error

        Notes:
            The method allows receiving all history orders within a specified period.
            The `group` parameter may contain several comma separated conditions. 
            A condition can be set as a mask using '*'. 
            The logical negation symbol '!' can be used for an exclusion. 
            All conditions are applied sequentially, which means conditions of including to a group 
            should be specified first followed by an exclusion condition. 
            For example, `group`="*, !EUR" means that deals for all symbols should be selected first 
            and the ones containing "EUR" in symbol names should be excluded afterwards.

        Example:
        >>> # get the number of deals in history
        >>> from_date = datetime(2020,1,1)
        >>> to_date = datetime.now()
        >>> account = Account()
        >>> history = account.get_trade_history(from_date, to_date)
        """
        if date_to is None:
            date_to = datetime.now()

        if (ticket is not None) + (group is not None) + (position is not None) > 1:
            raise ValueError(
                "Only one of 'position', 'group' or 'ticket' can be specified as filter or None of them .")
        if group is not None:
            position_deals = mt5.history_deals_get(
                date_from, date_to, group=group
            )
        elif ticket is not None:
            position_deals = mt5.history_deals_get(ticket=ticket)
        elif position is not None:
            position_deals = mt5.history_deals_get(position=position)
        else:
            position_deals = mt5.history_deals_get(date_from, date_to)

        if len(position_deals) == 0:
            return None

        df = pd.DataFrame(list(position_deals),
                          columns=position_deals[0]._asdict())
        df['time'] = pd.to_datetime(df['time'], unit='s')
        if save:
            file = "trade_history.csv"
            df.to_csv(file)
        if to_df:
            return df
        else:
            position_deals = [
                TradeDeal(**td._asdict())for td in position_deals]
            return tuple(position_deals)

    def get_orders(self,
                   symbol: Optional[str] = None,
                   group: Optional[str] = None,
                   ticket: Optional[int] = None,
                   to_df: bool = False
                   ) -> Union[pd.DataFrame, Tuple[TradeOrder], None]:
        """
        Get active orders with the ability to filter by symbol or ticket. 
        There are Four call options.

        * Call without parameters. Return open positions for all symbols.
        * Call specifying a symbol open positions should be received for.
        * Call specifying a group of symbols open positions should be received for.
        * Call specifying a position ticket.

        Args:
            symbol(str): Symbol name. Optional named parameter. 
                If a symbol is specified, the ticket parameter is ignored.

            group (str) The filter for arranging a group of necessary symbols. 
                Optional named parameter. If the group is specified, 
                the function returns only positions meeting a specified criteria 
                for a symbol name.

            ticket (int): Optional named parameter.
                Order ticket. Unique number assigned to each order.

            to_df (bool): If True, a DataFrame is returned.

        Returns:
        -   TradeOrder in the form of a named tuple structure (namedtuple) or pd.DataFrame(). 
        -   None in case of an error.

        Notes:
            The method allows receiving all history orders within a specified period.
            The `group` parameter may contain several comma separated conditions. 
            A condition can be set as a mask using '*'. 
            The logical negation symbol '!' can be used for an exclusion. 
            All conditions are applied sequentially, which means conditions of including to a group 
            should be specified first followed by an exclusion condition. 
            For example, `group`="*, !EUR" means that deals for all symbols should be selected first 
            and the ones containing "EUR" in symbol names should be excluded afterwards.
        """
        if (symbol is not None) + (group is not None) + (ticket is not None) > 1:
            raise ValueError(
                "Only one of 'symbol', 'group', or 'ticket' can be specified as filter or None of them.")

        if symbol is not None:
            orders = mt5.orders_get(symbol=symbol)
        elif group is not None:
            orders = mt5.orders_get(group=group)
        elif ticket is not None:
            orders = mt5.orders_get(ticket=ticket)
        else:
            orders = mt5.orders_get()

        if len(orders) == 0:
            return None

        if to_df:
            df = pd.DataFrame(list(orders), columns=orders[0]._asdict())
            df.drop(['time_expiration', 'type_time', 'state', 'position_by_id', 'reason',
                     'volume_current', 'price_stoplimit', 'sl', 'tp'], axis=1, inplace=True)
            df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
            df['time_done'] = pd.to_datetime(df['time_done'], unit='s')
            return df
        else:
            trade_orders = [TradeOrder(**o._asdict()) for o in orders]
            return tuple(trade_orders)

    def get_orders_history(
        self,
        date_from: datetime = datetime(2000, 1, 1),
        date_to: Optional[datetime] = None,
        group: Optional[str] = None,
        ticket: Optional[int] = None,  # order ticket
        position: Optional[int] = None,  # position ticket
        to_df: bool = True,
        save: bool = False
    ) -> Union[pd.DataFrame, Tuple[TradeOrder], None]:
        """
        Get orders from trading history within the specified interval 
        with the ability to filter by `ticket` or `position`.

        * Call with a `time interval`. Return all deals falling within the specified interval.
        * Call specifying the `order ticket`. Return all deals having the specified `order ticket` 
            in the `DEAL_ORDER` property.
        * Call specifying the `position ticket`. Return all deals having the specified `position ticket` 
            in the `DEAL_POSITION_ID` property.

        Args:
            date_from (datetime): Date the bars are requested from. 
                Set by the `datetime` object or as a number of seconds elapsed since 1970.01.01. 
                Bars with the open time >= date_from are returned. Required unnamed parameter.

            date_to (datetime): Same as `date_from`

            group (str) The filter for arranging a group of necessary symbols. 
                Optional named parameter. If the group is specified, 
                the function returns only positions meeting a specified criteria 
                for a symbol name.

            ticket (int): Order ticket that should be received. Optional parameter. 
                If not specified, the filter is not applied.

            position (int): Ticket of a position (stored in `DEAL_POSITION_ID`) all orders should be received for. 
                Optional parameter. If not specified, the filter is not applied.

            to_df (bool): If True, a DataFrame is returned.

            save (bool): If set to True, a csv file will be create a to save the history

        Returns:
        -   TradeOrder in the form of a named tuple structure (namedtuple) or pd.DataFrame().
        -   None in case of an error

        Notes:
            The method allows receiving all history orders within a specified period.
            The `group` parameter may contain several comma separated conditions. 
            A condition can be set as a mask using '*'. 
            The logical negation symbol '!' can be used for an exclusion. 
            All conditions are applied sequentially, which means conditions of including to a group 
            should be specified first followed by an exclusion condition. 
            For example, `group`="*, !EUR" means that deals for all symbols should be selected first 
            and the ones containing "EUR" in symbol names should be excluded afterwards.

        Example:
        >>> # get the number of deals in history
        >>> from_date = datetime(2020,1,1)
        >>> to_date = datetime.now()
        >>> account = Account()
        >>> history = account.get_order_history(from_date, to_date)
        """
        if date_to is None:
            date_to = datetime.now()

        if (group is not None) + (ticket is not None) + (position is not None) > 1:
            raise ValueError(
                "Only one of 'position', 'group' or 'ticket' can be specified or None of them as filter.")
        if group is not None:
            history_orders = mt5.history_orders_get(
                date_from, date_to, group=group
            )
        elif ticket is not None:
            history_orders = mt5.history_orders_get(ticket=ticket)
        elif position is not None:
            history_orders = mt5.history_orders_get(position=position)
        else:
            history_orders = mt5.history_orders_get(date_from, date_to)

        if len(history_orders) == 0:
            return None

        df = pd.DataFrame(list(history_orders),
                          columns=history_orders[0]._asdict())
        df.drop(['time_expiration', 'type_time', 'state', 'position_by_id', 'reason',
                 'volume_current', 'price_stoplimit', 'sl', 'tp'], axis=1, inplace=True)
        df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
        df['time_done'] = pd.to_datetime(df['time_done'], unit='s')

        if save:
            file = "trade_history.csv"
            df.to_csv(file)
        if to_df:
            return df
        else:
            history_orders = [
                TradeOrder(**td._asdict())for td in history_orders]
            return tuple(history_orders)
