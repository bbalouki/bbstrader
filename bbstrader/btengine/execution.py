from abc import ABCMeta, abstractmethod
from queue import Queue

from loguru import logger

from bbstrader.btengine.data import DataHandler
from bbstrader.btengine.event import FillEvent, OrderEvent
from bbstrader.config import BBSTRADER_DIR
from bbstrader.metatrader.account import Account

__all__ = ["ExecutionHandler", "SimExecutionHandler", "MT5ExecutionHandler"]


logger.add(
    f"{BBSTRADER_DIR}/logs/execution.log",
    enqueue=True,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
)


class ExecutionHandler(metaclass=ABCMeta):
    """
    The ExecutionHandler abstract class handles the interaction
    between a set of order objects generated by a Portfolio and
    the ultimate set of Fill objects that actually occur in the
    market.

    The handlers can be used to subclass simulated brokerages
    or live brokerages, with identical interfaces. This allows
    strategies to be backtested in a very similar manner to the
    live trading engine.

    The ExecutionHandler described here is exceedingly simple,
    since it fills all orders at the current market price.
    This is highly unrealistic, for other markets thant ``CFDs``
    but serves as a good baseline for improvement.
    """

    @abstractmethod
    def execute_order(self, event: OrderEvent):
        """
        Takes an Order event and executes it, producing
        a Fill event that gets placed onto the Events queue.

        Args:
            event (OrderEvent): Contains an Event object with order information.
        """
        pass


class SimExecutionHandler(ExecutionHandler):
    """
    The simulated execution handler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage or fill-ratio issues.

    This allows a straightforward "first go" test of any strategy,
    before implementation with a more sophisticated execution
    handler.
    """

    def __init__(self, events: Queue, data: DataHandler, **kwargs):
        """
        Initialises the handler, setting the event queues
        up internally.

        Args:
            events (Queue): The Queue of Event objects.
        """
        self.events = events
        self.bardata = data
        self.logger = kwargs.get("logger") or logger

    def execute_order(self, event: OrderEvent):
        """
        Simply converts Order objects into Fill objects naively,
        i.e. without any latency, slippage or fill ratio problems.

        Args:
            event (OrderEvent): Contains an Event object with order information.
        """
        if event.type == "ORDER":
            dtime = self.bardata.get_latest_bar_datetime(event.symbol)
            fill_event = FillEvent(
                timeindex=dtime,
                symbol=event.symbol,
                exchange="ARCA",
                quantity=event.quantity,
                direction=event.direction,
                fill_cost=None,
                commission=None,
                order=event.signal,
            )
            self.events.put(fill_event)
            self.logger.info(
                f"{event.direction} ORDER FILLED: SYMBOL={event.symbol}, "
                f"QUANTITY={event.quantity}, PRICE @{event.price} EXCHANGE={fill_event.exchange}",
                custom_time=fill_event.timeindex,
            )


class MT5ExecutionHandler(ExecutionHandler):
    """
    The main role of `MT5ExecutionHandler` class is to estimate the execution fees
    for different asset classes on the MT5 terminal.

    Generally we have four types of fees when we execute trades using the MT5 terminal
    (commissions, swap, spread and other fees). But most of these fees depend on the specifications
    of each instrument and the duration of the transaction for the swap for example.

    Calculating the exact fees for each instrument would be a bit complex because our Backtest engine
    and the Portfolio class do not take into account the duration of each trade to apply the appropriate
    rate for the swap for example. So we have to use only the model of calculating the commissions
    for each asset class and each instrument.

    The second thing that must be taken into account on MT5 is the type of account offered by the broker.
    Brokers have different account categories each with its specifications for each asset class and each instrument.
    Again considering all these conditions would make our class very complex. So we took the `Raw Spread`
    account fee calculation model from [Just Market](https://one.justmarkets.link/a/tufvj0xugm/registration/trader)
    for indicies, forex, commodities and crypto. We used the [Admiral Market](https://cabinet.a-partnership.com/visit/?bta=35537&brand=admiralmarkets)
    account fee calculation model from `Trade.MT5` account type for stocks and ETFs.

    NOTE:
        This class only works with `bbstrader.metatrader.data.MT5DataHandler` class.
    """

    def __init__(self, events: Queue, data: DataHandler, **kwargs):
        """
        Initialises the handler, setting the event queues up internally.

        Args:
            events (Queue): The Queue of Event objects.
        """
        self.events = events
        self.bardata = data
        self.logger = kwargs.get("logger") or logger
        self.__account = Account(**kwargs)

    def _calculate_lot(self, symbol, quantity, price):
        symbol_type = self.__account.get_symbol_type(symbol)
        symbol_info = self.__account.get_symbol_info(symbol)
        contract_size = symbol_info.trade_contract_size

        lot = (quantity * price) / (contract_size * price)
        if contract_size == 1:
            lot = quantity
        if symbol_type in ["COMD", "FUT", "CRYPTO"] and contract_size > 1:
            lot = quantity / contract_size
        if symbol_type == "FX":
            lot = quantity * price / contract_size
        return self._check_lot(symbol, lot)

    def _check_lot(self, symbol, lot):
        symbol_info = self.__account.get_symbol_info(symbol)
        if lot < symbol_info.volume_min:
            return symbol_info.volume_min
        elif lot > symbol_info.volume_max:
            return symbol_info.volume_max
        return round(lot, 2)

    def _estimate_total_fees(self, symbol, lot, qty, price):
        symbol_type = self.__account.get_symbol_type(symbol)
        if symbol_type in ["STK", "ETF"]:
            return self._estimate_stock_commission(symbol, qty, price)
        elif symbol_type == "FX":
            return self._estimate_forex_commission(lot)
        elif symbol_type == "COMD":
            return self._estimate_commodity_commission(lot)
        elif symbol_type == "IDX":
            return self._estimate_index_commission(lot)
        elif symbol_type == "FUT":
            return self._estimate_futures_commission()
        elif symbol_type == "CRYPTO":
            return self._estimate_crypto_commission()
        else:
            return 0.0

    def _estimate_stock_commission(self, symbol, qty, price):
        # https://admiralmarkets.com/start-trading/contract-specifications?regulator=jsc
        min_com = 1.0
        min_aud = 8.0
        min_dkk = 30.0
        min_nok = min_sek = 10.0
        us_com = 0.02  # per chare
        ger_fr_uk_cm = 0.001  # percent
        eu_asia_cm = 0.0015  # percent
        if (
            symbol in self.__account.get_stocks_from_country("USA")
            or self.__account.get_symbol_type(symbol) == "ETF"
            and self.__account.get_currency_rates(symbol)["mc"] == "USD"
        ):
            return max(min_com, qty * us_com)
        elif (
            symbol in self.__account.get_stocks_from_country("GBR")
            or symbol in self.__account.get_stocks_from_country("FRA")
            or symbol in self.__account.get_stocks_from_country("DEU")
            or self.__account.get_symbol_type(symbol) == "ETF"
            and self.__account.get_currency_rates(symbol)["mc"] in ["GBP", "EUR"]
        ):
            return max(min_com, qty * price * ger_fr_uk_cm)
        else:
            if self.__account.get_currency_rates(symbol)["mc"] == "AUD":
                return max(min_aud, qty * price * eu_asia_cm)
            elif self.__account.get_currency_rates(symbol)["mc"] == "DKK":
                return max(min_dkk, qty * price * eu_asia_cm)
            elif self.__account.get_currency_rates(symbol)["mc"] == "NOK":
                return max(min_nok, qty * price * eu_asia_cm)
            elif self.__account.get_currency_rates(symbol)["mc"] == "SEK":
                return max(min_sek, qty * price * eu_asia_cm)
            else:
                return max(min_com, qty * price * eu_asia_cm)

    def _estimate_forex_commission(self, lot):
        return 3.0 * lot

    def _estimate_commodity_commission(self, lot):
        return 3.0 * lot

    def _estimate_index_commission(self, lot):
        return 0.25 * lot

    def _estimate_futures_commission(self):
        return 0.0

    def _estimate_crypto_commission(self):
        return 0.0

    def execute_order(self, event: OrderEvent):
        """
        Executes an Order event by converting it into a Fill event.

        Args:
            event (OrderEvent): Contains an Event object with order information.
        """
        if event.type == "ORDER":
            symbol = event.symbol
            direction = event.direction
            quantity = event.quantity
            price = event.price
            lot = self._calculate_lot(symbol, quantity, price)
            fees = self._estimate_total_fees(symbol, lot, quantity, price)
            dtime = self.bardata.get_latest_bar_datetime(symbol)
            fill_event = FillEvent(
                timeindex=dtime,
                symbol=symbol,
                exchange="MT5",
                quantity=quantity,
                direction=direction,
                fill_cost=None,
                commission=fees,
                order=event.signal,
            )
            self.events.put(fill_event)
            self.logger.info(
                f"{direction} ORDER FILLED: SYMBOL={symbol}, QUANTITY={quantity}, "
                f"PRICE @{price} EXCHANGE={fill_event.exchange}",
                custom_time=fill_event.timeindex,
            )


class IBExecutionHandler(ExecutionHandler): ...
