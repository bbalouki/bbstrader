/// @file objects.hpp
/// @brief Declares the public API for the Metatrader5 objects in bbstrader library.
/// This is the primary file that you should \#include if
/// you want to use these objects.

#pragma once

#include <pybind11/numpy.h>

#include <concepts>
#include <string>
#include <type_traits>
#include <vector>

/// @brief Namespace containing MetaTrader 5 (MT5) data structures, enumerations, and helper
/// concepts.
namespace MT5 {

/// @brief Standard chart timeframes.
enum class Timeframe : int32_t {
    M1  = 1,            ///< 1 Minute
    M2  = 2,            ///< 2 Minutes
    M3  = 3,            ///< 3 Minutes
    M4  = 4,            ///< 4 Minutes
    M5  = 5,            ///< 5 Minutes
    M6  = 6,            ///< 6 Minutes
    M10 = 10,           ///< 10 Minutes
    M12 = 12,           ///< 12 Minutes
    M15 = 15,           ///< 15 Minutes
    M20 = 20,           ///< 20 Minutes
    M30 = 30,           ///< 30 Minutes
    H1  = 1 | 0x4000,   ///< 1 Hour
    H2  = 2 | 0x4000,   ///< 2 Hours
    H3  = 3 | 0x4000,   ///< 3 Hours
    H4  = 4 | 0x4000,   ///< 4 Hours
    H6  = 6 | 0x4000,   ///< 6 Hours
    H8  = 8 | 0x4000,   ///< 8 Hours
    H12 = 12 | 0x4000,  ///< 12 Hours
    D1  = 24 | 0x4000,  ///< 1 Day
    W1  = 1 | 0x8000,   ///< 1 Week
    MN1 = 1 | 0xC000    ///< 1 Month
};

/// @brief Order types for market and pending orders.
enum class OrderType : int32_t {
    BUY             = 0,  ///< Market Buy order
    SELL            = 1,  ///< Market Sell order
    BUY_LIMIT       = 2,  ///< Buy Limit pending order
    SELL_LIMIT      = 3,  ///< Sell Limit pending order
    BUY_STOP        = 4,  ///< Buy Stop pending order
    SELL_STOP       = 5,  ///< Sell Stop pending order
    BUY_STOP_LIMIT  = 6,  ///< Buy Stop Limit pending order
    SELL_STOP_LIMIT = 7,  ///< Sell Stop Limit pending order
    CLOSE_BY        = 8   ///< Order to close a position by an opposite one
};

/// @brief States of an order.
enum class OrderState : int32_t {
    STARTED        = 0,  ///< Order checked, but not yet accepted by broker
    PLACED         = 1,  ///< Order accepted
    CANCELED       = 2,  ///< Order canceled by client
    PARTIAL        = 3,  ///< Order partially executed
    FILLED         = 4,  ///< Order fully executed
    REJECTED       = 5,  ///< Order rejected
    EXPIRED        = 6,  ///< Order expired
    REQUEST_ADD    = 7,  ///< Order is being registered (internal)
    REQUEST_MODIFY = 8,  ///< Order is being modified (internal)
    REQUEST_CANCEL = 9   ///< Order is being deleted (internal)
};

/// @brief Order filling policies.
enum class OrderFilling : int32_t {
    FOK    = 0,  ///< Fill or Kill: Execute fully or cancel
    IOC    = 1,  ///< Immediate or Cancel: Execute available, cancel remainder
    RETURN = 2,  ///< Return: Execute available, keep remainder (standard)
    BOC    = 3   // Book Or Cancel
};

/// @brief Reason or source for the order placement.
enum class OrderReason : int32_t {
    CLIENT = 0,  ///< Manually placed by terminal
    MOBILE = 1,  ///< Placed via mobile app
    WEB    = 2,  ///< Placed via web platform
    EXPERT = 3,  ///< Placed by Expert Advisor (script)
    SL     = 4,  ///< Triggered by Stop Loss
    TP     = 5,  ///< Triggered by Take Profit
    SO     = 6   ///< Triggered by Stop Out
};

/// @brief Types of deals involved in trading.
enum class DealType : int32_t {
    BUY                      = 0,   ///< Buy deal
    SELL                     = 1,   ///< Sell deal
    BALANCE                  = 2,   ///< Balance operation
    CREDIT                   = 3,   ///< Credit operation
    CHARGE                   = 4,   ///< Additional charge
    CORRECTION               = 5,   ///< Correction deal
    BONUS                    = 6,   ///< Bonus deposit
    COMMISSION               = 7,   ///< Commission charge
    COMMISSION_DAILY         = 8,   ///< Daily commission
    COMMISSION_MONTHLY       = 9,   ///< Monthly commission
    COMMISSION_AGENT_DAILY   = 10,  ///< Daily agent commission
    COMMISSION_AGENT_MONTHLY = 11,  ///< Monthly agent commission
    INTEREST                 = 12,  ///< Interest rate charge
    BUY_CANCELED             = 13,  ///< Canceled buy deal
    SELL_CANCELED            = 14,  ///< Canceled sell deal
    DIVIDEND                 = 15,  ///< Dividend operation
    DIVIDEND_FRANKED         = 16,  ///< Franked dividend
    TAX                      = 17   ///< Tax charge
};

/// @brief Reason for the deal execution.
enum class DealReason : int32_t {
    CLIENT   = 0,  ///< Manually executed
    MOBILE   = 1,  ///< Executed via mobile
    WEB      = 2,  ///< Executed via web
    EXPERT   = 3,  ///< Executed by EA
    SL       = 4,  ///< Executed by Stop Loss
    TP       = 5,  ///< Executed by Take Profit
    SO       = 6,  ///< Executed by Stop Out
    ROLLOVER = 7,  ///< Rollover deal
    VMARGIN  = 8,  ///< Variation margin
    SPLIT    = 9   ///< Split
};

/// @brief Types of trade actions in a request.
enum class TradeAction : int32_t {
    DEAL     = 1,  ///< Place a market order (instant execution)
    PENDING  = 5,  ///< Place a pending order
    SLTP     = 6,  ///< Modify Stop Loss / Take Profit
    MODIFY   = 7,  ///< Modify pending order price/expiry
    REMOVE   = 8,  ///< Remove pending order
    CLOSE_BY = 10  ///< Close position by opposite position
};

/// @brief Mode of symbol margin and profit calculation.
enum class SymbolCalcMode : int32_t {
    FOREX               = 0,   ///< Forex mode
    FUTURES             = 1,   ///< Futures mode
    CFD                 = 2,   ///< CFD mode
    CFDINDEX            = 3,   ///< CFD Index mode
    CFDLEVERAGE         = 4,   ///< CFD Leverage mode
    FOREX_NO_LEVERAGE   = 5,   ///< Forex without leverage
    EXCH_STOCKS         = 32,  ///< Exchange Stocks
    EXCH_FUTURES        = 33,  ///< Exchange Futures
    EXCH_OPTIONS        = 34,  ///< Exchange Options
    EXCH_OPTIONS_MARGIN = 36,  ///< Exchange Options Margin
    EXCH_BONDS          = 37,  ///< Exchange Bonds
    EXCH_STOCKS_MOEX    = 38,  ///< MOEX Stocks
    EXCH_BONDS_MOEX     = 39,  ///< MOEX Bonds
    SERV_COLLATERAL     = 64   ///< Collateral
};

/// @brief Allowed trade modes for a symbol.
enum class SymbolTradeMode : int32_t {
    DISABLED  = 0,  ///< Trading disabled
    LONGONLY  = 1,  ///< Long only allowed (Note: likely maps to LONGONLY)
    SHORTONLY = 2,  ///< Short only allowed
    CLOSEONLY = 3,  ///< Only closing allowed
    FULL      = 4   ///< No restrictions
};

/// @brief Swap calculation modes.
enum class SymbolSwapMode : int32_t {
    DISABLED         = 0,  ///< No swaps
    POINTS           = 1,  ///< Swaps in points
    CURRENCY_SYMBOL  = 2,  ///< Swaps in symbol base currency
    CURRENCY_MARGIN  = 3,  ///< Swaps in margin currency
    CURRENCY_DEPOSIT = 4,  ///< Swaps in deposit currency
    INTEREST_CURRENT = 5,  ///< Interest current
    INTEREST_OPEN    = 6,  ///< Interest open
    REOPEN_CURRENT   = 7,  ///< Reopen current
    REOPEN_BID       = 8   ///< Reopen bid
};

/// @brief Days of the week.
enum class DayOfWeek : int32_t {
    SUNDAY    = 0,
    MONDAY    = 1,
    TUESDAY   = 2,
    WEDNESDAY = 3,
    THURSDAY  = 4,
    FRIDAY    = 5,
    SATURDAY  = 6
};

/// @brief Expiration modes for Good-Till-Canceled orders.
enum class SymbolGTCMode : int32_t { GTC = 0, DAILY = 1, DAILY_NO_STOPS = 2 };

/// @brief Option right type.
enum class OptionRight : int32_t { CALL = 0, PUT = 1 };

/// @brief Option style.
enum class OptionMode : int32_t { EUROPEAN = 0, AMERICAN = 1 };

/// @brief Account trade mode (Demo/Real).
enum class AccountTradeMode : int32_t { DEMO = 0, CONTEST = 1, REAL = 2 };

/// @brief Stop-out calculation mode.
enum class AccountStopoutMode : int32_t { PERCENT = 0, MONEY = 1 };

/// @brief Account margin calculation mode.
enum class AccountMarginMode : int32_t { RETAIL_NETTING = 0, EXCHANGE = 1, RETAIL_HEDGING = 2 };

/// @brief Order book entry type.
enum class BookType : int32_t { SELL = 1, BUY = 2, SELL_MARKET = 3, BUY_MARKET = 4 };

/// @brief Symbol execution mode.
enum class SymbolExecution : int32_t { REQUEST = 0, INSTANT = 1, MARKET = 2, EXCHANGE = 3 };

/// @brief Symbol chart price base.
enum class SymbolChartMode : int32_t { BID = 0, LAST = 1 };

/// @brief Deal entry direction.
enum class DealEntry : int32_t { IN = 0, OUT = 1, INOUT = 2, OUT_BY = 3 };

/// @brief Order expiration time flags.
enum class OrderTime : int32_t { GTC = 0, DAY = 1, SPECIFIED = 2, SPECIFIED_DAY = 3 };

/// @brief Position type.
enum class PositionType : int32_t { BUY = 0, SELL = 1 };

/// @brief Reason for position creation.
enum class PositionReason : int32_t { CLIENT = 0, MOBILE = 1, WEB = 2, EXPERT = 3 };

/// @brief Flags for copying ticks.
enum class CopyTicks : int32_t { ALL = -1, INFO = 1, TRADE = 2 };

/// @brief Flags describing tick content.
enum class TickFlag : int32_t {
    BID    = 0x02,  ///< Tick changed Bid price
    ASK    = 0x04,  ///< Tick changed Ask price
    LAST   = 0x08,  ///< Tick changed Last price
    VOLUME = 0x10,  ///< Tick changed Volume
    BUY    = 0x20,  ///< Tick is a result of a buy deal
    SELL   = 0x40   ///< Tick is a result of a sell deal
};

/// @brief Overload for TickFlag bitwise combination.
inline TickFlag operator|(TickFlag a, TickFlag b) {
    return static_cast<TickFlag>(static_cast<int32_t>(a) | static_cast<int32_t>(b));
}

/// @brief Return codes from the trade server.
enum class TradeRetcode : int32_t {
    REQUOTE              = 10004,  ///< Requote
    REJECT               = 10006,  ///< Request rejected
    CANCEL               = 10007,  ///< Request canceled by trader
    PLACED               = 10008,  ///< Order placed
    DONE                 = 10009,  ///< Request completed
    DONE_PARTIAL         = 10010,  ///< Only part of the request completed
    ERROR                = 10011,  ///< Request processing error
    TIMEOUT              = 10012,  ///< Request canceled by timeout
    INVALID              = 10013,  ///< Invalid request
    INVALID_VOLUME       = 10014,  ///< Invalid volume in the request
    INVALID_PRICE        = 10015,  ///< Invalid price in the request
    INVALID_STOPS        = 10016,  ///< Invalid stops in the request
    TRADE_DISABLED       = 10017,  ///< Trade is disabled
    MARKET_CLOSED        = 10018,  ///< Market is closed
    NO_MONEY             = 10019,  ///< There is not enough money to request
    PRICE_CHANGED        = 10020,  ///< Prices changed
    PRICE_OFF            = 10021,  ///< There are no quotes to process the request
    INVALID_EXPIRATION   = 10022,  ///< Invalid order expiration date in the request
    ORDER_CHANGED        = 10023,  ///< Order state changed
    TOO_MANY_REQUESTS    = 10024,  ///< Too many frequent requests
    NO_CHANGES           = 10025,  ///< No changes in request
    SERVER_DISABLES_AT   = 10026,  ///< Autotrading disabled by server
    CLIENT_DISABLES_AT   = 10027,  ///< Autotrading disabled by client terminal
    LOCKED               = 10028,  ///< Request locked for processing
    FROZEN               = 10029,  ///< Order or position frozen
    INVALID_FILL         = 10030,  ///< Invalid order filling type
    CONNECTION           = 10031,  ///< No connection with the trade server
    ONLY_REAL            = 10032,  ///< Operation is allowed only for live accounts
    LIMIT_ORDERS         = 10033,  ///< The number of pending orders has reached the limit
    LIMIT_VOLUME         = 10034,  ///< The volume for the symbol has reached the limit
    INVALID_ORDER        = 10035,  ///< Incorrect or prohibited order type
    POSITION_CLOSED      = 10036,  ///< Position with the specified ticket is already closed
    INVALID_CLOSE_VOLUME = 10038,  ///< A volume to close exceeds the current position volume
    CLOSE_ORDER_EXIST    = 10039,  ///< A close order already exists for the specified position
    LIMIT_POSITIONS      = 10040,  ///< The number of open positions has reached the limit
    REJECT_CANCEL = 10041,  ///< The pending order is currently being activated, cancel is rejected
    int64_t_ONLY  = 10042,  ///< The request is rejected, only long positions are allowed
    SHORT_ONLY    = 10043,  ///< The request is rejected, only short positions are allowed
    CLOSE_ONLY    = 10044,  ///< The request is rejected, only position closing is allowed
    FIFO_CLOSE    = 10045   ///< Position closing is allowed only by FIFO rule
};

/// @brief Internal library return codes.
enum class ReturnCode : int32_t {
    OK                    = 1,       ///< Success
    FAIL                  = -1,      ///< Generic Failure
    INVALID_PARAMS        = -2,      ///< Invalid Parameters
    NO_MEMORY             = -3,      ///< Memory Allocation Error
    NOT_FOUND             = -4,      ///< Not Found
    INVALID_VERSION       = -5,      ///< Version Mismatch
    AUTH_FAILED           = -6,      ///< Authentication Failed
    UNSUPPORTED           = -7,      ///< Unsupported Operation
    AUTO_TRADING_DISABLED = -8,      ///< AutoTrading Disabled
    INTERNAL_FAIL         = -10000,  ///< Internal Failure
    INTERNAL_FAIL_SEND    = -10001,  ///< Internal Send Failure
    INTERNAL_FAIL_RECEIVE = -10002,  ///< Internal Receive Failure
    INTERNAL_FAIL_INIT    = -10003,  ///< Internal Initialization Failure
    INTERNAL_FAIL_CONNECT = -10004,  ///< Internal Connection Failure
    INTERNAL_FAIL_TIMEOUT = -10005   ///< Internal Timeout
};

/// @brief Asset classes / Symbol types.
enum class SymbolType : int32_t {
    FOREX       = 0,  ///< Forex currency pairs
    FUTURES     = 1,  ///< Futures contracts
    STOCKS      = 2,  ///< Stocks and shares
    BONDS       = 3,  ///< Bonds
    CRYPTO      = 4,  ///< Cryptocurrencies
    ETFS        = 5,  ///< Exchange-Traded Funds
    INDICES     = 6,  ///< Market indices
    COMMODITIES = 7,  ///< Commodities
    OPTIONS     = 8,  ///< Options contracts
    UNKNOWN     = 9   ///< Unknown or unsupported type
};

/// @brief Information regarding the terminal and connection status.
struct TerminalInfo {
    bool        community_account;      ///< True if logged into MQL5 community
    bool        community_connection;   ///< True if connected to MQL5 community
    bool        connected;              ///< True if connected to trade server
    bool        dlls_allowed;           ///< True if DLLs are allowed
    bool        trade_allowed;          ///< True if trading is allowed
    bool        tradeapi_disabled;      ///< True if Trade API is disabled
    bool        email_enabled;          ///< True if email sending is enabled
    bool        ftp_enabled;            ///< True if FTP publishing is enabled
    bool        notifications_enabled;  ///< True if smartphone notifications are enabled
    bool        mqid;                   ///< MetaQuotes ID availability
    int         build;                  ///< Terminal build number
    int         maxbars;                ///< Max bars on chart
    int         codepage;               ///< Codepage number
    int         ping_last;              ///< Last ping value in microseconds
    double      community_balance;      ///< MQL5 community balance
    double      retransmission;         ///< Retransmission percentage
    std::string company;                ///< Broker company name
    std::string name;                   ///< Terminal name
    std::string language;               ///< Language
    std::string path;                   ///< Path to terminal installation
    std::string data_path;              ///< Path to terminal data folder
    std::string commondata_path;        ///< Path to common data folder
};

/// @brief Information about the trading account.
struct AccountInfo {
    int64_t     login;               ///< Account login number
    int32_t     trade_mode;          ///< Account trade mode (Demo, Real, Contest)
    int64_t     leverage;            ///< Account leverage
    int         limit_orders;        ///< Max allowed pending orders
    int32_t     margin_so_mode;      ///< Margin stop-out mode
    bool        trade_allowed;       ///< True if trading allowed for account
    bool        trade_expert;        ///< True if EAs are allowed to trade
    int32_t     margin_mode;         ///< Margin calculation mode (Hedging/Netting)
    int         currency_digits;     ///< Digits of account currency
    bool        fifo_close;          ///< True if FIFO closing is required
    double      balance;             ///< Account balance
    double      credit;              ///< Account credit
    double      profit;              ///< Current floating profit
    double      equity;              ///< Account equity
    double      margin;              ///< Used margin
    double      margin_free;         ///< Free margin
    double      margin_level;        ///< Margin level percentage
    double      margin_so_call;      ///< Margin call level
    double      margin_so_so;        ///< Stop-out level
    double      margin_initial;      ///< Initial margin requirement
    double      margin_maintenance;  ///< Maintenance margin requirement
    double      assets;              ///< Current assets
    double      liabilities;         ///< Current liabilities
    double      commission_blocked;  ///< Blocked commission
    std::string name;                ///< Client name
    std::string server;              ///< Trade server name
    std::string currency;            ///< Account currency
    std::string company;             ///< Broker company name
};

/// @brief Comprehensive information about a specific financial instrument (Symbol).
struct SymbolInfo {
    bool        custom;                      ///< Is custom symbol
    int32_t     chart_mode;                  ///< Price type used for bars generation
    bool        select;                      ///< Is symbol selected in Market Watch
    bool        visible;                     ///< Is symbol visible in Market Watch
    int64_t     session_deals;               ///< Number of deals in current session
    int64_t     session_buy_orders;          ///< Number of buy orders in current session
    int64_t     session_sell_orders;         ///< Number of sell orders in current session
    int64_t     volume;                      ///< Last deal volume
    int64_t     volumehigh;                  ///< Max volume of the day
    int64_t     volumelow;                   ///< Min volume of the day
    int64_t     time;                        ///< Time of the last quote
    int         digits;                      ///< Digits after decimal point
    int         spread;                      ///< Spread value in points
    bool        spread_float;                ///< Is spread floating
    int         ticks_bookdepth;             ///< Maximal depth of Depth of Market
    int32_t     trade_calc_mode;             ///< Calculation mode
    int32_t     trade_mode;                  ///< Trade mode
    int64_t     start_time;                  ///< Symbol trading start date
    int64_t     expiration_time;             ///< Symbol expiration date
    int         trade_stops_level;           ///< Minimal indention in points for stops
    int         trade_freeze_level;          ///< Distance to freeze trade operations
    int32_t     trade_exemode;               ///< Trade execution mode
    int32_t     swap_mode;                   ///< Swap calculation mode
    int         swap_rollover3days;          ///< Day of triple swap
    bool        margin_hedged_use_leg;       ///< Calculation of hedged margin using larger leg
    int32_t     expiration_mode;             ///< Flags of allowed order expiration modes
    int32_t     filling_mode;                ///< Flags of allowed order filling modes
    int32_t     order_mode;                  ///< Flags of allowed order types
    int32_t     order_gtc_mode;              ///< Expiration of Stop Loss and Take Profit
    int32_t     option_mode;                 ///< Option type
    int         option_right;                ///< Option right (Call/Put)
    double      bid;                         ///< Current Bid price
    double      bidhigh;                     ///< Max Bid of the day
    double      bidlow;                      ///< Min Bid of the day
    double      ask;                         ///< Current Ask price
    double      askhigh;                     ///< Max Ask of the day
    double      asklow;                      ///< Min Ask of the day
    double      last;                        ///< Price of the last deal
    double      lasthigh;                    ///< Max Last of the day
    double      lastlow;                     ///< Min Last of the day
    double      volume_real;                 ///< Last deal volume (double)
    double      volumehigh_real;             ///< Max volume (double)
    double      volumelow_real;              ///< Min volume (double)
    double      option_strike;               ///< Option strike price
    double      point;                       ///< Symbol point value
    double      trade_tick_value;            ///< Calculated tick value
    double      trade_tick_value_profit;     ///< Calculated tick value for profit
    double      trade_tick_value_loss;       ///< Calculated tick value for loss
    double      trade_tick_size;             ///< Minimal price change
    double      trade_contract_size;         ///< Trade contract size
    double      trade_accrued_interest;      ///< Accrued interest
    double      trade_face_value;            ///< Face value
    double      trade_liquidity_rate;        ///< Liquidity rate
    double      volume_min;                  ///< Minimal volume for a deal
    double      volume_max;                  ///< Maximal volume for a deal
    double      volume_step;                 ///< Minimal volume change step
    double      volume_limit;                ///< Max cumulative volume allowed
    double      swap_long;                   ///< Swap for buy positions
    double      swap_short;                  ///< Swap for sell positions
    double      margin_initial;              ///< Initial margin requirement
    double      margin_maintenance;          ///< Maintenance margin requirement
    double      session_volume;              ///< Volume of the current session
    double      session_turnover;            ///< Turnover of the current session
    double      session_interest;            ///< Open interest
    double      session_buy_orders_volume;   ///< Volume of buy orders
    double      session_sell_orders_volume;  ///< Volume of sell orders
    double      session_open;                ///< Session open price
    double      session_close;               ///< Session close price
    double      session_aw;                  ///< Average weighted session price
    double      session_price_settlement;    ///< Settlement price
    double      session_price_limit_min;     ///< Min price limit
    double      session_price_limit_max;     ///< Max price limit
    double      margin_hedged;               ///< Contract size or margin for hedged positions
    double      price_change;                ///< Price change percentage
    double      price_volatility;            ///< Price volatility
    double      price_theoretical;           ///< Theoretical price
    double      price_greeks_delta;          ///< Option Delta
    double      price_greeks_theta;          ///< Option Theta
    double      price_greeks_gamma;          ///< Option Gamma
    double      price_greeks_vega;           ///< Option Vega
    double      price_greeks_rho;            ///< Option Rho
    double      price_greeks_omega;          ///< Option Omega
    double      price_sensitivity;           ///< Sensitivity
    std::string basis;                       ///< Underlying asset name
    std::string category;                    ///< Symbol category
    std::string currency_base;               ///< Base currency
    std::string currency_profit;             ///< Profit currency
    std::string currency_margin;             ///< Margin currency
    std::string bank;                        ///< Feeder/Bank name
    std::string description;                 ///< Symbol description
    std::string exchange;                    ///< Exchange name
    std::string formula;                     ///< Pricing formula
    std::string isin;                        ///< ISIN code
    std::string name;                        ///< Symbol name
    std::string page;                        ///< Web page
    std::string path;                        ///< Path in symbol tree
};

// Pack structs to ensure binary compatibility with API/IPC
#pragma pack(push, 1)
/// @brief Represents a single tick (price update).
struct TickInfo {
    int64_t  time;         ///< Time of the last prices update
    double   bid;          ///< Current Bid price
    double   ask;          ///< Current Ask price
    double   last;         ///< Price of the last deal (Last)
    int64_t  volume;       ///< Volume for the current Last price
    int64_t  time_msc;     ///< Time of a price last update in milliseconds
    uint32_t flags;        ///< Tick flags (TickFlag enum)
    double   volume_real;  ///< Volume for the current Last price with greater accuracy
};

/// @brief Represents a bar of historical data (OHLCV).
struct RateInfo {
    int64_t  time;         ///< Period start time
    double   open;         ///< Open price
    double   high;         ///< The highest price of the period
    double   low;          ///< The lowest price of the period
    double   close;        ///< Close price
    uint64_t tick_volume;  ///< Tick volume
    int32_t  spread;       ///< Spread
    uint64_t real_volume;  ///< Trade volume
};
#pragma pack(pop)

/// @brief Represents an entry in the Depth of Market (Order Book).
struct BookInfo {
    int32_t  type;         ///< Order type from BookType enumeration
    double   price;        ///< Price
    uint64_t volume;       ///< Volume
    double   volume_real;  ///< Volume with greater accuracy
};

/// @brief Structure to send trade requests to the server.
struct TradeRequest {
    int32_t     action       = 0;    ///< Trade operation type (TradeAction)
    int64_t     magic        = 0;    ///< Expert Advisor ID (magic number)
    int64_t     order        = 0;    ///< Order ticket
    std::string symbol       = "";   ///< Trade symbol
    double      volume       = 0.0;  ///< Requested volume for a deal in lots
    double      price        = 0.0;  ///< Price
    double      stoplimit    = 0.0;  ///< StopLimit level of the order
    double      sl           = 0.0;  ///< Stop Loss level of the order
    double      tp           = 0.0;  ///< Take Profit level of the order
    int64_t     deviation    = 0;    ///< Maximal possible deviation from the requested price
    int32_t     type         = 0;    ///< Order type (OrderType)
    int32_t     type_filling = 0;    ///< Order execution type (OrderFilling)
    int32_t     type_time    = 0;    ///< Order expiration type (OrderTime)
    int64_t     expiration   = 0;    ///< Order expiration time
    std::string comment      = "";   ///< Order comment
    int64_t     position     = 0;    ///< Position ticket
    int64_t     position_by  = 0;    ///< The ticket of an opposite position
};

/// @brief Results of an order check (validation) operation.
struct OrderCheckResult {
    uint32_t     retcode;       ///< Reply code
    double       balance;       ///< Balance after the execution of the deal
    double       equity;        ///< Equity after the execution of the deal
    double       profit;        ///< Floating profit
    double       margin;        ///< Margin requirements
    double       margin_free;   ///< Free margin
    double       margin_level;  ///< Margin level
    std::string  comment;       ///< Comment to the reply code (description of the error)
    TradeRequest request;       ///< The request that was checked
};

/// @brief Results of a sent order.
struct OrderSentResult {
    uint32_t     retcode;           ///< Operation return code
    int64_t      deal;              ///< Deal ticket, if it is performed
    int64_t      order;             ///< Order ticket, if it is placed
    double       volume;            ///< Deal volume, confirmed by broker
    double       price;             ///< Deal price, confirmed by broker
    double       bid;               ///< Current Bid price
    double       ask;               ///< Current Ask price
    std::string  comment;           ///< Broker comment or description of return code
    uint32_t     request_id;        ///< Request ID set by the terminal during the dispatch
    int          retcode_external;  ///< Return code of an external trading system
    TradeRequest request;           ///< The original request
};

/// @brief Represents an active or historical order.
struct TradeOrder {
    int64_t     ticket;           ///< Order ticket
    int64_t     time_setup;       ///< Order setup time
    int64_t     time_setup_msc;   ///< Order setup time in msc
    int64_t     time_done;        ///< Order execution/cancellation time
    int64_t     time_done_msc;    ///< Order execution/cancellation time in msc
    int64_t     time_expiration;  ///< Order expiration time
    int32_t     type;             ///< Order type
    int32_t     type_time;        ///< Order lifetime
    int32_t     type_filling;     ///< Order filling type
    int32_t     state;            ///< Order state
    int64_t     magic;            ///< Expert Advisor ID
    int64_t     position_id;      ///< Position identifier
    int64_t     position_by_id;   ///< Identifier of opposite position
    int32_t     reason;           ///< Reason for order placement
    double      volume_initial;   ///< Initial volume
    double      volume_current;   ///< Current volume
    double      price_open;       ///< Price specified in the order
    double      sl;               ///< Stop Loss price
    double      tp;               ///< Take Profit price
    double      price_current;    ///< Current price of the symbol
    double      price_stoplimit;  ///< StopLimit price
    std::string symbol;           ///< Symbol name
    std::string comment;          ///< Order comment
    std::string external_id;      ///< External system ID
};

/// @brief Represents an open position.
struct TradePosition {
    int64_t     ticket;           ///< Position ticket
    int64_t     time;             ///< Position open time
    int64_t     time_msc;         ///< Position open time in msc
    int64_t     time_update;      ///< Position last update time
    int64_t     time_update_msc;  ///< Position last update time in msc
    int32_t     type;             ///< Position type
    int64_t     magic;            ///< Expert Advisor ID
    int64_t     identifier;       ///< Position identifier
    int32_t     reason;           ///< Reason for position opening
    double      volume;           ///< Position volume
    double      price_open;       ///< Position open price
    double      sl;               ///< Stop Loss price
    double      tp;               ///< Take Profit price
    double      price_current;    ///< Current symbol price
    double      swap;             ///< Accumulated swap
    double      profit;           ///< Current floating profit
    std::string symbol;           ///< Symbol name
    std::string comment;          ///< Position comment
    std::string external_id;      ///< External system ID
};

/// @brief Represents a historical deal (execution).
struct TradeDeal {
    int64_t     ticket;       ///< Deal ticket
    int64_t     order;        ///< Order ticket that generated the deal
    int64_t     time;         ///< Deal time
    int64_t     time_msc;     ///< Deal time in msc
    int32_t     type;         ///< Deal type
    int32_t     entry;        ///< Deal entry direction
    int64_t     magic;        ///< Expert Advisor ID
    int64_t     position_id;  ///< Position identifier
    int32_t     reason;       ///< Reason for deal execution
    double      volume;       ///< Deal volume
    double      price;        ///< Deal price
    double      commission;   ///< Commission
    double      swap;         ///< Swap
    double      profit;       ///< Profit
    double      fee;          ///< Fee
    std::string symbol;       ///< Symbol name
    std::string comment;      ///< Deal comment
    std::string external_id;  ///< External system ID
};

#if (defined(_MSVC_LANG) && _MSVC_LANG >= 202302L) || (__cplusplus >= 202302L)
/// @brief Concept to check if a type is a scoped enumeration (C++23 compliant).
template <typename T>
concept IsScopedEnum = std::is_scoped_enum_v<T>;
#else
/// @brief Concept to check if a type is a scoped enumeration (Legacy fallback).
template <typename T>
concept IsScopedEnum = std::is_enum_v<T> && !std::is_convertible_v<T, int>;
#endif

/// @brief Unary + operator to extract the underlying integer value of a scoped enum.
/// @tparam EnumClass A scoped enumeration type.
/// @param a The enum value.
/// @return The underlying integer value.
template <IsScopedEnum EnumClass>
constexpr auto operator+(EnumClass a) noexcept {
    return static_cast<std::underlying_type_t<EnumClass>>(a);
}

}  // namespace MT5
