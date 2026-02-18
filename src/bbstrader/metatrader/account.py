from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from bbstrader.api import (
    AccountInfo,
    SymbolInfo,
    TerminalInfo,
    TickInfo,
    TradeDeal,
    TradeOrder,
    TradePosition,
)
from bbstrader.api import Mt5client as client
from bbstrader.metatrader.broker import Broker, check_mt5_connection
from bbstrader.metatrader.utils import TIMEFRAMES, RateInfo, SymbolType, raise_mt5_error

__all__ = ["Account"]


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

        >>> # Getting terminal information
        >>> terminal_info = account.get_terminal_info()

        >>> # Getting active orders
        >>> orders = account.get_orders()

        >>> # Fetching open positions
        >>> positions = account.get_positions()

        >>> # Accessing trade history
        >>> from_date = datetime(2020, 1, 1)
        >>> to_date = datetime.now()
        >>> trade_history = account.get_trade_history(from_date, to_date)
    """

    def __init__(self, broker: Optional[Broker] = None, **kwargs):
        """
        Initialize the Account class.

        See `bbstrader.metatrader.broker.check_mt5_connection()`
        for more details on how to connect to MT5 terminal.

        """
        check_mt5_connection(**kwargs)
        self._info = client.account_info()
        terminal_info = self.get_terminal_info()
        self._broker = (
            broker
            if broker is not None
            else Broker(terminal_info.company if terminal_info else "Unknown")
        )

    @property
    def info(self) -> AccountInfo:
        return self._info

    @property
    def broker(self) -> Broker:
        return self._broker

    @property
    def timezone(self) -> Optional[str]:
        return self.broker.get_terminal_timezone()

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def number(self) -> int:
        return self._info.login

    @property
    def server(self) -> str:
        """The name of the trade server to which the client terminal is connected.
        (e.g., 'AdmiralsGroup-Demo')
        """
        return self._info.server

    @property
    def balance(self) -> float:
        return self._info.balance

    @property
    def leverage(self) -> int:
        return self._info.leverage

    @property
    def equity(self) -> float:
        return self._info.equity

    @property
    def currency(self) -> str:
        return self._info.currency

    def shutdown(self):
        """Close the connection to the MetaTrader 5 terminal."""
        client.shutdown()

    def get_account_info(
        self,
        account: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        timeout: Optional[int] = 60_000,
        path: Optional[str] = None,
    ) -> Union[AccountInfo, None]:
        """
        Get info on the current trading account or a specific account .

        Args:
            account (int, optinal) : MT5 Trading account number.
            password (str, optinal): MT5 Trading account password.

            server (str, optinal): MT5 Trading account server
                [Brokers or terminal server ["demo", "real"]]
                If no server is set, the last used server is applied automaticall

            timeout (int, optinal):
                 Connection timeout in milliseconds. Optional named parameter.
                 If not specified, the value of 60 000 (60 seconds) is applied.
                 If the connection is not established within the specified time,
                 the call is forcibly terminated and the exception is generated.
            path (str, optional): The path to the MetaTrader 5 terminal executable file.
                Defaults to None (e.g., "C:/Program Files/MetaTrader 5/terminal64.exe").

        Returns:
        -   AccountInfo
        -   None in case of an error

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        # connect to the trade account specifying a password and a server
        if account is not None and password is not None and server is not None:
            try:
                if path is not None:
                    self.broker.initialize_connection(
                        path=path,
                        login=account,
                        password=password,
                        server=server,
                        timeout=timeout,
                    )
                authorized = client.login(
                    account, password=password, server=server, timeout=timeout
                )
                if not authorized:
                    raise_mt5_error(f"Failed to connect to account #{account}")
                info = client.account_info()
                return info
            except Exception as e:
                raise_mt5_error(e)
        else:
            try:
                return client.account_info()
            except Exception as e:
                raise_mt5_error(e)

    def get_terminal_info(self) -> TerminalInfo | None:
        """
        Get the connected MetaTrader 5 client terminal status and settings.

        Returns:
        -   TerminalInfo
        -   None in case of an error

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        try:
            terminal_info = client.terminal_info()
            if terminal_info is None:
                return None
        except Exception as e:
            raise_mt5_error(e)
        return terminal_info

    def get_symbol_info(self, symbol: str) -> SymbolInfo | None:
        """Get symbol properties

        Args:
            symbol (str): Symbol name

        Returns:
        -   SymbolInfo.
        -   None in case of an error.

        Raises:
            MT5TerminalError: A specific exception based on the error code.

        """
        try:
            symbol_info = client.symbol_info(symbol)
            if symbol_info is None:
                return None
            else:
                return symbol_info
        except Exception as e:
            msg = self._symbol_info_msg(symbol)
            raise_mt5_error(message=f"{str(e)} {msg}")

    def _symbol_info_msg(self, symbol):
        return (
            f"No history found for {symbol} in Market Watch.\n"
            f"* Ensure {symbol} is selected and displayed in the Market Watch window.\n"
            f"* See https://www.metatrader5.com/en/terminal/help/trading/market_watch\n"
            f"* Ensure the symbol name is correct.\n"
        )

    def get_tick_info(self, symbol: str) -> TickInfo | None:
        """Get symbol tick properties

        Args:
            symbol (str): Symbol name

        Returns:
        -   TickInfo.
        -   None in case of an error.

        Raises:
            MT5TerminalError: A specific exception based on the error code.

        """
        try:
            tick_info = client.symbol_info_tick(symbol)
            if tick_info is None:
                return None
            else:
                return tick_info
        except Exception as e:
            msg = self._symbol_info_msg(symbol)
            raise_mt5_error(message=f"{str(e)} {msg}")

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
            >>> account.get_currency_rates('EURUSD')
            {'bc': 'EUR', 'mc': 'EUR', 'pc': 'USD', 'ac': 'USD'}
        """
        info = self.get_symbol_info(symbol)
        bc = info.currency_base
        pc = info.currency_profit
        mc = info.currency_margin
        ac = self._info.currency
        return {"bc": bc, "mc": mc, "pc": pc, "ac": ac}

    def get_symbols(
        self,
        symbol_type: SymbolType | str = "ALL",
        check_etf=False,
        save=False,
        file_name="symbols",
        include_desc=False,
        display_total=False,
    ) -> List[str]:
        """
        Get all specified financial instruments from the MetaTrader 5 terminal.

        Args:
            symbol_type (SymbolType | str): The type of financial instruments to retrieve.
            - `ALL`: For all available symbols
            - See `bbstrader.metatrader.utils.SymbolType` for more details.

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
        return self.broker.get_symbols(
            symbol_type=symbol_type,
            check_etf=check_etf,
            save=save,
            file_name=file_name,
            include_desc=include_desc,
            display_total=display_total,
        )

    def get_symbol_type(self, symbol: str) -> SymbolType:
        """
        Determines the type of a given financial instrument symbol.

        Args:
            symbol (str): The symbol of the financial instrument (e.g., `GOOGL`, `EURUSD`).

        Returns:
            SymbolType: The type of the financial instrument, one of the following:
            - `SymbolType.ETFs`
            - `SymbolType.BONDS`
            - `SymbolType.FOREX`
            - `SymbolType.FUTURES`
            - `SymbolType.STOCKS`
            - `SymbolType.INDICES`
            - `SymbolType.COMMODITIES`
            - `SymbolType.CRYPTO`
        - `SymbolType.unknown` if the type cannot be determined.

        """
        return self.broker.get_symbol_type(symbol)

    def _get_symbols_by_category(
        self, symbol_type: SymbolType | str, category: str, category_map: Dict[str, str]
    ) -> List[str]:
        return self.broker.get_symbols_by_category(symbol_type, category, category_map)

    def get_stocks_from_country(
        self, country_code: str = "USA", etf=False
    ) -> List[str]:
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
            * **Hong Kong:** HKG
            * **Ireland:** IRL
            * **Austria:** AUT

        Args:
            country (str, optional): The country code of stocks to retrieve.
                                    Defaults to 'USA'.

        Returns:
            list: A list of stock symbol names from the specified country.

        Raises:
            ValueError: If an unsupported country is provided.

        Notes:
            This mthods works primarly with brokers who specify the stock symbols type and exchanges,
            For other brokers use `get_symbols()` or this method will use it by default.
        """
        stocks = self._get_symbols_by_category(
            SymbolType.STOCKS, country_code, self.broker.countries_stocks
        )
        etfs = (
            self._get_symbols_by_category(
                SymbolType.ETFs, country_code, self.broker.countries_stocks
            )
            if etf
            else []
        )
        if not stocks and not etfs:
            stocks = self.get_symbols(symbol_type=SymbolType.STOCKS)
            etfs = self.get_symbols(symbol_type=SymbolType.ETFs) if etf else []
        return stocks + etfs

    def get_stocks_from_exchange(
        self, exchange_code: str = "XNYS", etf=True
    ) -> List[str]:
        """
        Get stock symbols from a specific exchange using the ISO Code for the exchange.

        Supported exchanges are from Admirals Group AS products:
        * **XASX:**        **Australian Securities Exchange**
        * **XBRU:**        **Euronext Brussels Exchange**
        * **XCSE:**        **Copenhagen Stock Exchange**
        * **XHEL:**        **NASDAQ OMX Helsinki**
        * **XPAR:**        **Euronext Paris**
        * **XETR:**        **Xetra Frankfurt**
        * **XOSL:**        **Oslo Stock Exchange**
        * **XLIS:**        **Euronext Lisbon**
        * **XMAD:**        **Bolsa de Madrid**
        * **XSTO:**        **NASDAQ OMX Stockholm**
        * **XLON:**        **London Stock Exchange**
        * **NYSE:**        **New York Stock Exchange**
        * **ARCA:**        **NYSE ARCA**
        * **AMEX:**        **NYSE AMEX**
        * **XNYS:**        **New York Stock Exchange (AMEX, ARCA, NYSE)**
        * **NASDAQ:**      **NASDAQ**
        * **BATS:**        **BATS Exchange**
        * **XSWX:**        **SWX Swiss Exchange**
        * **XAMS:**        **Euronext Amsterdam**

        Args:
            exchange_code (str, optional): The ISO code of the exchange.
            etf (bool, optional): If True, include ETFs from the exchange. Defaults to True.

        Returns:
            list: A list of stock symbol names from the specified exchange.

        Raises:
            ValueError: If an unsupported exchange is provided.

        Notes:
            This mthods works primarly with brokers who specify the stock symbols type and exchanges,
            For other brokers use `get_symbols()` or this method will use it by default.
        """
        stocks = self._get_symbols_by_category(
            SymbolType.STOCKS, exchange_code, self.broker.exchanges
        )
        etfs = (
            self._get_symbols_by_category(
                SymbolType.ETFs, exchange_code, self.broker.exchanges
            )
            if etf
            else []
        )
        if not stocks and not etfs:
            stocks = self.get_symbols(symbol_type=SymbolType.STOCKS)
            etfs = self.get_symbols(symbol_type=SymbolType.ETFs) if etf else []
        return stocks + etfs

    def get_rate_info(self, symbol: str, timeframe: str = "1m") -> RateInfo | None:
        """Get the most recent bar for a specified symbol and timeframe.

        Args:
            symbol (str): The symbol for which to get the rate information.
            timeframe (str): The timeframe for the rate information. Default is '1m'.
                            See ``bbstrader.metatrader.utils.TIMEFRAMES`` for supported timeframes.
        Returns:
            RateInfo: The most recent bar as a RateInfo named tuple.
            None: If no rates are found or an error occurs.
        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        rates = client.copy_rates_from_pos(symbol, TIMEFRAMES[timeframe], 0, 1)
        if rates is None or len(rates) == 0:
            return None
        rate = rates[0]
        return RateInfo(*rate)

    def get_positions(
        self,
        symbol: Optional[str] = None,
        group: Optional[str] = None,
        ticket: Optional[int] = None,
    ) -> Union[List[TradePosition] | None]:
        """
        Get open positions with the ability to filter by symbol or ticket.
        There are four call options:

        - Call without parameters. Returns open positions for all symbols.
        - Call specifying a symbol. Returns open positions for the specified symbol.
        - Call specifying a group of symbols. Returns open positions for the specified group of symbols.
        - Call specifying a position ticket. Returns the position corresponding to the specified ticket.

        Args:
            symbol (Optional[str]): Symbol name. Optional named parameter.
                If a symbol is specified, the `ticket` parameter is ignored.

            group (Optional[str]): The filter for arranging a group of necessary symbols.
                Optional named parameter. If the group is specified,
                the function returns only positions meeting specified criteria
                for a symbol name.

            ticket (Optional[int]): Position ticket. Optional named parameter.
                A unique number assigned to each newly opened position.
                It usually matches the ticket of the order used to open the position,
                except when the ticket is changed as a result of service operations on the server,
                for example, when charging swaps with position re-opening.


        Returns:
            [List[TradePosition] | None]:
            - List of `TradePosition`.

        Notes:
            The method allows receiving all open positions within a specified period.

            The `group` parameter may contain several comma-separated conditions.

            A condition can be set as a mask using '*'.

            The logical negation symbol '!' can be used for exclusion.

            All conditions are applied sequentially, which means conditions for inclusion
            in a group should be specified first, followed by an exclusion condition.

            For example, `group="*, !EUR"` means that deals for all symbols should be selected first,
            and those containing "EUR" in symbol names should be excluded afterward.
        """

        if (symbol is not None) + (group is not None) + (ticket is not None) > 1:
            raise ValueError(
                "Only one of 'symbol', 'group', or 'ticket' can be specified as filter or None of them."
            )

        if symbol is not None:
            positions = client.positions_get(symbol)
        elif group is not None:
            positions = client.positions_get_by_group(group)
        elif ticket is not None:
            positions = client.position_get_by_ticket(ticket)
        else:
            positions = client.positions_get()

        if positions is None:
            return None
        if isinstance(positions, TradePosition):
            return [positions]
        if len(positions) == 0:
            return None

        return positions

    def get_orders(
        self,
        symbol: Optional[str] = None,
        group: Optional[str] = None,
        ticket: Optional[int] = None,
    ) -> Union[List[TradeOrder] | None]:
        """
        Get active orders with the ability to filter by symbol or ticket.
        There are four call options:

        - Call without parameters. Returns open positions for all symbols.
        - Call specifying a symbol, open positions should be received for.
        - Call specifying a group of symbols, open positions should be received for.
        - Call specifying a position ticket.

        Args:
            symbol (Optional[str]): Symbol name. Optional named parameter.
                If a symbol is specified, the ticket parameter is ignored.

            group (Optional[str]): The filter for arranging a group of necessary symbols.
                Optional named parameter. If the group is specified,
                the function returns only positions meeting a specified criteria
                for a symbol name.

            ticket (Optional[int]): Order ticket. Optional named parameter.
                Unique number assigned to each order.

            to_df (bool): If True, a DataFrame is returned.

        Returns:
            [List[TradeOrder] | None]:
            - List of `TradeOrder` .

        Notes:
            The method allows receiving all history orders within a specified period.
            The `group` parameter may contain several comma-separated conditions.
            A condition can be set as a mask using '*'.

            The logical negation symbol '!' can be used for exclusion.
            All conditions are applied sequentially, which means conditions for inclusion
            in a group should be specified first, followed by an exclusion condition.

            For example, `group="*, !EUR"` means that deals for all symbols should be selected first
            and the ones containing "EUR" in symbol names should be excluded afterward.
        """

        if (symbol is not None) + (group is not None) + (ticket is not None) > 1:
            raise ValueError(
                "Only one of 'symbol', 'group', or 'ticket' can be specified as filter or None of them."
            )

        orders = None
        if symbol is not None:
            orders = client.orders_get(symbol)
        elif group is not None:
            orders = client.orders_get_by_group(group)
        elif ticket is not None:
            orders = client.order_get_by_ticket(ticket)
        else:
            orders = client.orders_get()

        if orders is None or len(orders) == 0:
            return None
        return orders

    def _fetch_history(
        self,
        fetch_type: str,  # "deals" or "orders"
        date_from: datetime,
        date_to: Optional[datetime],
        group: Optional[str],
        ticket: Optional[int],
        position: Optional[int],
        to_df: bool,
        drop_cols: List[str],
        time_cols: List[str],
    ) -> Any:
        from zoneinfo import ZoneInfo

        tz = self.broker.get_terminal_timezone()
        date_to = date_to or datetime.now(tz=ZoneInfo(tz))
        date_from = date_from.astimezone(tz=ZoneInfo(tz))

        filters = [group, ticket, position]
        if sum(f is not None for f in filters) > 1:
            raise ValueError(
                "Only one of 'position', 'group', or 'ticket' can be specified."
            )

        if fetch_type == "deals":
            client_func = client.history_deals_get
            pos_func = client.history_deals_get_by_pos
        else:
            client_func = client.history_orders_get
            pos_func = client.history_orders_get_by_pos

        data = None
        if ticket:
            data = client_func(ticket)
        elif position:
            data = pos_func(position)
        elif group:
            data = client_func(date_from, date_to, group)
        else:
            data = client_func(date_from, date_to)

        if not data:
            return None

        if to_df:
            from bbstrader.api import trade_object_to_df

            df = trade_object_to_df(data)
            for col in time_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], unit="s")

            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
            if fetch_type == "deals" and "time" in df.columns:
                df.set_index("time", inplace=True)
            return df

        return data

    def get_trades_history(
        self,
        date_from: datetime = datetime(2000, 1, 1),
        date_to: Optional[datetime] = None,
        group: Optional[str] = None,
        ticket: Optional[int] = None,  # TradeDeal.ticket
        position: Optional[int] = None,  # TradePosition.ticket
        to_df: bool = True,
    ) -> Union[pd.DataFrame, List[TradeDeal] | None]:
        """
        Get deals from trading history within the specified interval
        with the ability to filter by `ticket` or `position`.

        This method is useful if you need panda dataframe.

        You can call this method in the following ways:

        - Call with a `time interval`. Returns all deals falling within the specified interval.

        - Call specifying the `order ticket`. Returns all deals having the specified `order ticket` in the `DEAL_ORDER` property.

        - Call specifying the `position ticket`. Returns all deals having the specified `position ticket` in the `DEAL_POSITION_ID` property.

        Args:
            date_from (datetime): Date the bars are requested from.
                Set by the `datetime` object or as a number of seconds elapsed since 1970-01-01.
                Bars with the open time >= `date_from` are returned. Required unnamed parameter.

            date_to (Optional[datetime]): Same as `date_from`.

            group (Optional[str]): The filter for arranging a group of necessary symbols.
                Optional named parameter. If the group is specified,
                the function returns only positions meeting specified criteria
                for a symbol name.

            ticket (Optional[int]): Ticket of an order (stored in `DEAL_ORDER`) for which all deals should be received.
                Optional parameter. If not specified, the filter is not applied.

            position (Optional[int]): Ticket of a position (stored in `DEAL_POSITION_ID`) for which all deals should be received.
                Optional parameter. If not specified, the filter is not applied.

            to_df (bool): If True, a DataFrame is returned.

        Returns:
            Union[pd.DataFrame, Tuple[TradeDeal], None]:
            - `TradeDeal` in the form of a named tuple structure (namedtuple) or pd.DataFrame().

        Notes:
            The method allows receiving all history orders within a specified period.

            The `group` parameter may contain several comma-separated conditions.

            A condition can be set as a mask using '*'.

            The logical negation symbol '!' can be used for exclusion.

            All conditions are applied sequentially, which means conditions for inclusion
            in a group should be specified first, followed by an exclusion condition.

            For example, `group="*, !EUR"` means that deals for all symbols should be selected first
            and those containing "EUR" in symbol names should be excluded afterward.

        Example:
            >>> # Get the number of deals in history
            >>> from datetime import datetime
            >>> from_date = datetime(2020, 1, 1)
            >>> to_date = datetime.now()
            >>> account = Account()
            >>> history = account.get_trades_history(from_date, to_date)
        """
        return self._fetch_history(
            fetch_type="deals",
            drop_cols=["time_msc", "external_id"],
            time_cols=["time"],
            **dict(
                date_from=date_from,
                date_to=date_to,
                group=group,
                ticket=ticket,
                position=position,
                to_df=to_df,
            ),
        )

    def get_orders_history(
        self,
        date_from: datetime = datetime(2000, 1, 1),
        date_to: Optional[datetime] = None,
        group: Optional[str] = None,
        ticket: Optional[int] = None,  # order ticket
        position: Optional[int] = None,  # position ticket
        to_df: bool = True,
    ) -> Union[pd.DataFrame, List[TradeOrder] | None]:
        """
        Get orders from trading history within the specified interval
        with the ability to filter by `ticket` or `position`.

        You can call this method in the following ways:

        - Call with a `time interval`. Returns all deals falling within the specified interval.

        - Call specifying the `order ticket`. Returns all deals having the specified `order ticket` in the `DEAL_ORDER` property.

        - Call specifying the `position ticket`. Returns all deals having the specified `position ticket` in the `DEAL_POSITION_ID` property.

        Args:
            date_from (datetime): Date the bars are requested from.
                Set by the `datetime` object or as a number of seconds elapsed since 1970-01-01.
                Bars with the open time >= `date_from` are returned. Required unnamed parameter.

            date_to (Optional[datetime]): Same as `date_from`.

            group (Optional[str]): The filter for arranging a group of necessary symbols.
                Optional named parameter. If the group is specified,
                the function returns only positions meeting specified criteria
                for a symbol name.

            ticket (Optional[int]): Order ticket to filter results. Optional parameter.
                If not specified, the filter is not applied.

            position (Optional[int]): Ticket of a position (stored in `DEAL_POSITION_ID`) to filter results.
                Optional parameter. If not specified, the filter is not applied.

            to_df (bool): If True, a DataFrame is returned.

            save (bool): If True, a CSV file will be created to save the history.

        Returns:
            Union[pd.DataFrame, List[TradeOrder], None]
            - List of `TradeOrder` .

        Notes:
            The method allows receiving all history orders within a specified period.

            The `group` parameter may contain several comma-separated conditions.

            A condition can be set as a mask using '*'.

            The logical negation symbol '!' can be used for exclusion.

            All conditions are applied sequentially, which means conditions for inclusion
            in a group should be specified first, followed by an exclusion condition.

            For example, `group="*, !EUR"` means that deals for all symbols should be selected first
            and those containing "EUR" in symbol names should be excluded afterward.

        Example:
            >>> # Get the number of deals in history
            >>> from datetime import datetime
            >>> from_date = datetime(2020, 1, 1)
            >>> to_date = datetime.now()
            >>> account = Account()
            >>> history = account.get_orders_history(from_date, to_date)
        """
        return self._fetch_history(
            fetch_type="orders",
            drop_cols=[
                "time_expiration",
                "type_time",
                "state",
                "position_by_id",
                "reason",
                "volume_current",
                "price_stoplimit",
                "sl",
                "tp",
            ],
            time_cols=["time_setup", "time_done"],
            **dict(
                date_from=date_from,
                date_to=date_to,
                group=group,
                ticket=ticket,
                position=position,
                to_df=to_df,
            ),
        )

    def get_today_deals(self, id, group=None) -> List[TradeDeal]:
        """
        Get all today deals for a specific symbol or group of symbols

        Args:
            id (int): strategy or expert id
            group (str): Symbol or group or symbol
        Returns:
            List[TradeDeal]: List of today deals
        """

        from datetime import timedelta

        from_date = datetime.now() - timedelta(days=3)
        history = (
            self.get_trades_history(date_from=from_date, group=group, to_df=False) or []
        )
        positions_ids = set([deal.position_id for deal in history if deal.magic == id])
        today_deals = []
        for position in positions_ids:
            deal = self.get_trades_history(position=position, to_df=False) or []
            if deal is not None and len(deal) == 2:
                deal_time = datetime.fromtimestamp(deal[1].time)
                if deal_time.date() == datetime.now().date():
                    today_deals.append(deal[1])
        return today_deals
