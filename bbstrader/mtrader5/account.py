from datetime import datetime
import MetaTrader5 as mt5
import pandas as pd


class Account():
    """
    The Account class is used to get information about 
    the current trading account or a specific account.
    """
    def __init__(self):
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()

    def get_account_info(
        self,
        account:  int = None,
        password: str = None,
        server:   str = None
    ):
        """
        Get info on the current trading account or a specific account .

        Parameters
        ==========
        :param account (int) : MT5 Account Number
        :param password (str): MT5 Account Pasword
        :param server (str)  : MT5 Account server 
            [Brokers or terminal server ["demo", "real"]]
        
        Returns
            - Info in the form of a namedtuple structure. 
            - None in case of an error.
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
                raise Exception(
                    f"failed to connect to  account #{account}, error code = {mt5.last_error()}")                   
        else:
            try:
                return mt5.account_info()
            except Exception as e:
                print(e)

    def print_account_info(self):
        """ helper function to  print account info"""

        account_info = mt5.account_info()
        if account_info != None:
            # set trading account data in the form of a dictionary
            account_info_dict = mt5.account_info()._asdict()
            # convert the dictionary into DataFrame and print
            df = pd.DataFrame(list(account_info_dict.items()),
                              columns=['property', 'value'])
            print("ACCOUNT INFORMATIONS:")
            print(df)
        else:
            raise Exception(
                f"Sorry we can't access account info , error={mt5.last_error()}") 
                       

    def get_terminal_info(self):
        """
        Get the connected MetaTrader 5 client terminal status and settings.

        Returns
            -Info in the form of pd.DataFrame(). 
            - None in case of an error.
        """
        terminal_info = mt5.terminal_info()
        if terminal_info != None:
            # display data in the form of a list
            terminal_info_dict = mt5.terminal_info()._asdict()
            # convert the dictionary into DataFrame and print
            df = pd.DataFrame(list(terminal_info_dict.items()),
                              columns=['property', 'value'])
        else:
            raise Exception(
                f"Sorry we can't access temrninal info , error={mt5.last_error()}")
                
        return df

    def get_symbols(self):
        """ 
        Get all financial instruments from the MetaTrader 5 terminal

        Returns
            - symbols in the form of a List. 
            - None in case of an error.
        """
        # get all symbols
        symbols = mt5.symbols_get()
        symbol_list = []
        if symbols:
            for s in symbols:
                symbol_list.append(s.name)
        else:
            raise Exception(
                f"Sorry we can't get symbols , error={mt5.last_error()}")
        return symbol_list

    def show_symbol_info(self, symbol: str):
        """
        Print symbol properties

        Parameters
        ==========
        :param symbol (str): Symbol name

        Returns
            - Info in the form of a pd.DataFrame(). 
            - None in case of an error.
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info != None:
            # display data in the form of a list
            symbol_info_dict = mt5.symbol_info(symbol)._asdict()
            # convert the dictionary into DataFrame and print
            df = pd.DataFrame(list(symbol_info_dict.items()),
                              columns=['property', 'value'])
            print(f"SYMBOL INFO FOR {symbol}")
            pd.set_option('display.max_rows', None)  
            pd.set_option('display.max_columns', None)  
            print(df)
        else:
           raise Exception(
               f"Sorry we can't access symbol info , error={mt5.last_error()}")              

    def get_symbol_info(self, symbol: str):
        """Get symbol properties

        Parameters
        ==========
        :param symbol (str): Symbol name

        Returns
            - Info in the form of a namedtuple(). 
            - None in case of an error.
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info != None:
            return symbol_info
        else:
            raise Exception(
                f"Sorry we can't access symbol info , error={mt5.last_error()}"
            )               

    def get_orders(self) -> pd.DataFrame | None:
        """Get active orders 

        Returns
            - Info in the form of a pd.DataFrame() 
            - None in case of an error. 
        """
        orders = mt5.orders_get()
        if len(orders) == 0:
            print(f"No orders on the current account")
            quit()
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

        Returns
            - Info in the form of a pd.DataFrame(). 
            - None in case of an error.
        """
        positions = mt5.positions_get()
        if len(positions) == 0:
            print(f"No positions on current account")
            quit()
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

        Parameters
        ==========
        :param date_from (datetime) : Date the bars are requested from. 
            Set by the 'datetime' object or as a number of seconds elapsed since 1970.01.01. 
            Bars with the open time >= date_from are returned. Required unnamed parameter.
        :param date_to (datetime) : Same as date_from
        :save (bool) : Boolean value , if set to True ; 
            a csv file will be create a to save the history
        
        Returns
            - Return info in the form of a a pd.DataFrame().
            - None in case of an error

        Example:
        ```
        # get the number of deals in history
        from_date=datetime(2020,1,1)
        to_date=datetime.now()
        account = Account()
        history = account.get_trade_history(from_date, to_date)
        ```
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
            df = pd.DataFrame(list(position_deals), columns=position_deals[0]._asdict())                       
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.drop(['time_msc', 'external_id', 'order'], axis=1, inplace=True)
            if save:
                file = "trade_history.csv"
                df.to_csv(file)
            return df
        else:
            return None