from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd
import re
import os
import urllib.request
from currency_converter import SINGLE_DAY_ECB_URL, CurrencyConverter
from bbstrader.metatrader.utils import raise_mt5_error, INIT_MSG


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

    def get_account_info(
        self,
        account:  int = None,
        password: str = None,
        server:   str = None
    ):
        """
        Get info on the current trading account or a specific account .

        Args:
            account (int) : MT5 Account Number.
            password (str): MT5 Account Password.
            server (str)  : MT5 Account server 
                [Brokers or terminal server ["demo", "real"]].

        Returns:
        -   Info in the form of a namedtuple structure. 

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        # connect to the trade account specifying a password and a server
        if (
            account is not None and
            password is not None and
            server is not None
        ):
            authorized = mt5.login(account, password=password, server=server)
            if authorized:
                return mt5.account_info()
            else:
                raise_mt5_error(
                    message=f"Failed to connect to  account #{account}")
        else:
            try:
                return mt5.account_info()
            except Exception as e:
                raise_mt5_error(e)

    def print_account_info(self):
        """ helper function to  print account info"""

        account_info = mt5.account_info()
        if account_info != None:
            # set trading account data in the form of a dictionary
            account_info_dict = mt5.account_info()._asdict()
            # convert the dictionary into DataFrame and print
            df = pd.DataFrame(list(account_info_dict.items()),
                              columns=['PROPERTY', 'VALUE'])
            print("\nACCOUNT INFORMATIONS:")
            print(df)
        else:
            raise_mt5_error()

    def get_terminal_info(self):
        """
        Get the connected MetaTrader 5 client terminal status and settings.

        Returns:
        -   Info in the form of pd.DataFrame(). 

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        terminal_info = mt5.terminal_info()
        if terminal_info != None:
            # display data in the form of a list
            terminal_info_dict = mt5.terminal_info()._asdict()
            # convert the dictionary into DataFrame and print
            df = pd.DataFrame(list(terminal_info_dict.items()),
                              columns=['PROPERTY', 'VALUE'])
        else:
            raise_mt5_error()

        return df

    def convert_currencies(self, qty: float, from_c: str, to_c: str):
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

    def get_currency_rates(self, symbol: str):
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

    def check_cents(self, symbol, value):
        """
        Some symbols are quoted in cents for a given currency so 
        it is necessary that we convert them back the their base currency.

        The majority of UK companies listed on the London Stock Exchange are in GBX. 
        For instance, should a company be listed for 1,150 GBX, each share is worth £11.50. 
        Dividends can also be paid out in GBX.

        Args:
            symbol (str): The symbol for which to check currency type

        Returns:
        -    `value / 100 ` or `value`
        """
        rates = self.get_currency_rates(symbol)
        for rate in rates.values():
            if rate in ['USX', 'GBX']:
                # GBX / 100 = GBP, GBP * 100 = GBX
                # USX / 100 = USD, USD * 100 = USX
                new_value = value / 100
            else:
                new_value = value
        return new_value

    def get_symbols(self,
                    symbol_type="all",
                    check_etf=False,
                    save=False,
                    file_name="symbols",
                    include_desc=False
                    ):
        """ 
        Get all specified financial instruments from the MetaTrader 5 terminal.

        Args:
            symbol_type (str): The category of instrument to get. Possible values:
                - 'all': For all available symbols
                - 'stocks': Stocks (e.g., 'GOOGL')
                - 'etf': ETFs (e.g., 'QQQ')
                - 'indices': Indices (e.g., 'SP500')
                - 'forex': Forex pairs (e.g., 'EURUSD')
                - 'commodities': Commodities (e.g., 'CRUDOIL', 'GOLD')
                - 'futures': Futures (e.g., 'USTNote_U4')
                - 'cryptos': Cryptocurrencies (e.g., 'BTC', 'ETH')

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
        patterns = {
            "stocks": r'\b(Stocks?)\b',
            "etf": r'\b(ETFs?)\b',
            "indices": r'\b(Indices?)\b',
            "forex": r'\b(Forex)\b',
            "commodities": r'\b(Metals?|Agricultures?|Energies?)\b',
            "futures": r'\b(Futures?)\b',
            "cryptos": r'\b(Cryptos?)\b'
        }

        if symbol_type != 'all':
            if symbol_type not in patterns:
                raise ValueError(f"Unsupported symbol type: {symbol_type}")

        if save:
            max_lengh = max([len(s.name) for s in symbols])
            file_path = f"{file_name}.txt"
            with open(file_path, mode='w', encoding='utf-8') as file:
                for s in symbols:
                    info = self.get_symbol_info(s.name)
                    if symbol_type == 'all':
                        self._write_symbol(file, info, include_desc, max_lengh)
                        symbol_list.append(s.name)
                    else:
                        pattern = re.compile(
                            patterns[symbol_type], re.IGNORECASE)
                        match = re.search(pattern, info.path)
                        if match:
                            if symbol_type == "etf" and check_etf and "ETF" not in info.description:
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
                if symbol_type == 'all':
                    symbol_list.append(s.name)
                else:
                    pattern = re.compile(patterns[symbol_type], re.IGNORECASE)
                    match = re.search(pattern, info.path)
                    if match:
                        if symbol_type == "etf" and check_etf and "ETF" not in info.description:
                            raise ValueError(
                                f"{info.name} doesn't have 'ETF' in its description. "
                                "If this is intended, set check_etf=False."
                            )
                        symbol_list.append(s.name)

        # Print a summary of the retrieved symbols
        if symbol_type == 'etf':
            type_name = "ETFs"
        elif symbol_type == 'forex':
            type_name = 'Forex Pairs'
        else:
            type_name = symbol_type.capitalize()
        print(f"Total {type_name}: {len(symbol_list)}")

        return symbol_list

    def _write_symbol(self, file, info, include_desc, max_lengh):
        """Helper function to write symbol information to a file."""
        if include_desc:
            space = " "*int(max_lengh-len(info.name))
            file.write(info.name + space + '|' +
                       info.description + '\n')
        else:
            file.write(info.name + '\n')

    def get_symbol_type(self, symbol):
        """
        Determines the type of a given financial instrument symbol.

        Args:
            symbol (str): The symbol of the financial instrument (e.g., `GOOGL`, `EURUSD`).

        Returns:
            str: The type of the symbol:
                - `STK` for  Stocks (e.g., `GOOGL')
                - `ETF` for ETFs (e.g., `QQQ`)
                - `IDX` for Indices (e.g., `SP500')
                - `FX` for Forex pairs (e.g., `EURUSD`)
                - `COMD` for Commodities (e.g., `CRUDOIL`, `GOLD`)
                - `FUT` for Futures (e.g., `USTNote_U4`)
                - `CRYPTO` for Cryptocurrencies (e.g., `BTC`, `ETH`) 
                Returns `unknown` if the type cannot be determined. 
        """
        patterns = {
            "STK": r'\b(Stocks?)\b',
            "ETF": r'\b(ETFs?)\b',
            "IDX": r'\b(Indices?)\b',
            "FX": r'\b(Forex)\b',
            "COMD": r'\b(Metals?|Agricultures?|Energies?)\b',
            "FUT": r'\b(Futures?)\b',
            "CRYPTO": r'\b(Cryptos?)\b'
        }
        info = self.get_symbol_info(symbol)
        for symbol_type, pattern in patterns.items():
            match = re.search(pattern, info.path, re.IGNORECASE)
            if match:
                return symbol_type
        return "unknown"

    def show_symbol_info(self, symbol: str):
        """
        Print symbol properties

        Args:
            symbol (str): Symbol name

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info != None:
            # display data in the form of a list
            symbol_info_dict = mt5.symbol_info(symbol)._asdict()
            # convert the dictionary into DataFrame and print
            df = pd.DataFrame(list(symbol_info_dict.items()),
                              columns=['PROPERTY', 'VALUE'])
            print(f"\nSYMBOL INFO FOR {symbol} ({symbol_info.description})")
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            print(df)
        else:
            msg = self._symbol_info_msg(symbol)
            raise_mt5_error(message=msg)

    def _symbol_info_msg(self, symbol):
        return (
            f"No history found for {symbol} in Market Watch.\n"
            f"* Ensure {symbol} is selected and displayed in the Market Watch window.\n"
            f"* See https://www.metatrader5.com/en/terminal/help/trading/market_watch\n"
            f"* Ensure the symbol name is correct."
        )

    def get_symbol_info(self, symbol: str):
        """Get symbol properties

        Args:
            symbol (str): Symbol name

        Returns:
        -   Info in the form of a namedtuple(). 

        Raises:
            MT5TerminalError: A specific exception based on the error code.
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info != None:
            return symbol_info
        else:
            msg = self._symbol_info_msg(symbol)
            raise_mt5_error(message=msg)

    def get_orders(self) -> pd.DataFrame | None:
        """Get active orders 

        Returns:
        -   Info in the form of a pd.DataFrame() 
        -   None in case of an error. 
        """
        orders = mt5.orders_get()
        if len(orders) == 0:
            return None
        else:
            df = pd.DataFrame(list(orders), columns=orders[0]._asdict())
            df.drop([
                'time_done',
                'time_done_msc',
                'position_id',
                'position_by_id',
                'reason',
                'volume_initial',
                'price_stoplimit',
                'time_setup_msc',
                'time_expiration',
                'external_id'
            ], axis=1, inplace=True)
            df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
        return df

    def get_positions(self):
        """Get open positions

        Returns:
        -   Info in the form of a pd.DataFrame(). 
        -   None in case of an error.
        """
        positions = mt5.positions_get()
        if len(positions) == 0:
            return None
        elif len(positions) > 0:
            df = pd.DataFrame(list(positions), columns=positions[0]._asdict())
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.drop([
                'time_update',
                'time_msc',
                'time_update_msc',
                'external_id'
            ], axis=1, inplace=True)
        return df

    def get_trade_history(
        self,
        date_from: datetime = datetime(2000, 1, 1),
        date_to: datetime = None,
        group: str = None,
        save: bool = False
    ) -> pd.DataFrame | None:
        """
        Get deals from trading history within the specified interval

        Args:
            date_from (datetime): Date the bars are requested from. 
                Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. 
                Bars with the open time >= date_from are returned. Required unnamed parameter.

            date_to (datetime): Same as `date_from`
            save (bool): If set to True, a csv file will be create a to save the history

        Returns:
        -   Return info in the form of a a pd.DataFrame().
        -   None in case of an error

        Example:
        >>> # get the number of deals in history
        >>> from_date = datetime(2020,1,1)
        >>> to_date = datetime.now()
        >>> account = Account()
        >>> history = account.get_trade_history(from_date, to_date)
        """
        if date_to == None:
            date_to = datetime.now()
        if group is not None:
            g = group
            position_deals = mt5.history_deals_get(date_from, date_to, group=g)
        else:
            position_deals = mt5.history_deals_get(date_from, date_to)
        if len(position_deals) != 0:
            # display these deals as a table using pandas.DataFrame
            df = pd.DataFrame(list(position_deals),
                              columns=position_deals[0]._asdict())
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.drop(['time_msc', 'external_id', 'order'], axis=1, inplace=True)
            if save:
                file = "trade_history.csv"
                df.to_csv(file)
            return df
        else:
            return None
