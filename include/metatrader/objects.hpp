#pragma once

#include <pybind11/numpy.h>

#include <string>
#include <vector>

namespace MT5 {
enum class Timeframe : int32_t {
    M1  = 1,
    M2  = 2,
    M3  = 3,
    M4  = 4,
    M5  = 5,
    M6  = 6,
    M10 = 10,
    M12 = 12,
    M15 = 15,
    M20 = 20,
    M30 = 30,
    H1  = 1 | 0x4000,
    H2  = 2 | 0x4000,
    H3  = 3 | 0x4000,
    H4  = 4 | 0x4000,
    H6  = 6 | 0x4000,
    H8  = 8 | 0x4000,
    H12 = 12 | 0x4000,
    D1  = 24 | 0x4000,
    W1  = 1 | 0x8000,
    MN1 = 1 | 0xC000
};

enum class OrderType : int32_t {
    BUY             = 0,
    SELL            = 1,
    BUY_LIMIT       = 2,
    SELL_LIMIT      = 3,
    BUY_STOP        = 4,
    SELL_STOP       = 5,
    BUY_STOP_LIMIT  = 6,
    SELL_STOP_LIMIT = 7,
    CLOSE_BY        = 8
};

enum class OrderState : int32_t {
    STARTED        = 0,
    PLACED         = 1,
    CANCELED       = 2,
    PARTIAL        = 3,
    FILLED         = 4,
    REJECTED       = 5,
    EXPIRED        = 6,
    REQUEST_ADD    = 7,
    REQUEST_MODIFY = 8,
    REQUEST_CANCEL = 9
};

enum class OrderFilling : int32_t {
    FOK    = 0,  // Fill Or Kill
    IOC    = 1,  // Immediately Or Cancel
    RETURN = 2,  // Return remaining
    BOC    = 3   // Book Or Cancel
};

enum class OrderReason : int32_t {
    CLIENT = 0,
    MOBILE = 1,
    WEB    = 2,
    EXPERT = 3,
    SL     = 4,
    TP     = 5,
    SO     = 6
};

enum class DealType : int32_t {
    BUY                      = 0,
    SELL                     = 1,
    BALANCE                  = 2,
    CREDIT                   = 3,
    CHARGE                   = 4,
    CORRECTION               = 5,
    BONUS                    = 6,
    COMMISSION               = 7,
    COMMISSION_DAILY         = 8,
    COMMISSION_MONTHLY       = 9,
    COMMISSION_AGENT_DAILY   = 10,
    COMMISSION_AGENT_MONTHLY = 11,
    INTEREST                 = 12,
    BUY_CANCELED             = 13,
    SELL_CANCELED            = 14,
    DIVIDEND                 = 15,
    DIVIDEND_FRANKED         = 16,
    TAX                      = 17
};

enum class DealReason : int32_t {
    CLIENT   = 0,
    MOBILE   = 1,
    WEB      = 2,
    EXPERT   = 3,
    SL       = 4,
    TP       = 5,
    SO       = 6,
    ROLLOVER = 7,
    VMARGIN  = 8,
    SPLIT    = 9
};

enum class TradeAction : int32_t {
    DEAL     = 1,
    PENDING  = 5,
    SLTP     = 6,
    MODIFY   = 7,
    REMOVE   = 8,
    CLOSE_BY = 10
};

enum class SymbolCalcMode : int32_t {
    FOREX               = 0,
    FUTURES             = 1,
    CFD                 = 2,
    CFDINDEX            = 3,
    CFDLEVERAGE         = 4,
    FOREX_NO_LEVERAGE   = 5,
    EXCH_STOCKS         = 32,
    EXCH_FUTURES        = 33,
    EXCH_OPTIONS        = 34,
    EXCH_OPTIONS_MARGIN = 36,
    EXCH_BONDS          = 37,
    EXCH_STOCKS_MOEX    = 38,
    EXCH_BONDS_MOEX     = 39,
    SERV_COLLATERAL     = 64
};

enum class SymbolTradeMode : int32_t {
    DISABLED    = 0,
    int64_tONLY = 1,
    SHORTONLY   = 2,
    CLOSEONLY   = 3,
    FULL        = 4
};

enum class SymbolSwapMode : int32_t {
    DISABLED         = 0,
    POINTS           = 1,
    CURRENCY_SYMBOL  = 2,
    CURRENCY_MARGIN  = 3,
    CURRENCY_DEPOSIT = 4,
    INTEREST_CURRENT = 5,
    INTEREST_OPEN    = 6,
    REOPEN_CURRENT   = 7,
    REOPEN_BID       = 8
};

enum class DayOfWeek : int32_t {
    SUNDAY    = 0,
    MONDAY    = 1,
    TUESDAY   = 2,
    WEDNESDAY = 3,
    THURSDAY  = 4,
    FRIDAY    = 5,
    SATURDAY  = 6
};

enum class SymbolGTCMode : int32_t { GTC = 0, DAILY = 1, DAILY_NO_STOPS = 2 };
enum class OptionRight : int32_t { CALL = 0, PUT = 1 };
enum class OptionMode : int32_t { EUROPEAN = 0, AMERICAN = 1 };
enum class AccountTradeMode : int32_t { DEMO = 0, CONTEST = 1, REAL = 2 };
enum class AccountStopoutMode : int32_t { PERCENT = 0, MONEY = 1 };
enum class AccountMarginMode : int32_t { RETAIL_NETTING = 0, EXCHANGE = 1, RETAIL_HEDGING = 2 };
enum class BookType : int32_t { SELL = 1, BUY = 2, SELL_MARKET = 3, BUY_MARKET = 4 };
enum class SymbolExecution : int32_t { REQUEST = 0, INSTANT = 1, MARKET = 2, EXCHANGE = 3 };
enum class SymbolChartMode : int32_t { BID = 0, LAST = 1 };
enum class DealEntry : int32_t { IN = 0, OUT = 1, INOUT = 2, OUT_BY = 3 };
enum class OrderTime : int32_t { GTC = 0, DAY = 1, SPECIFIED = 2, SPECIFIED_DAY = 3 };
enum class PositionType : int32_t { BUY = 0, SELL = 1 };
enum class PositionReason : int32_t { CLIENT = 0, MOBILE = 1, WEB = 2, EXPERT = 3 };
enum class CopyTicks : int32_t { ALL = -1, INFO = 1, TRADE = 2 };

enum class TickFlag : uint32_t {
    BID    = 0x02,
    ASK    = 0x04,
    LAST   = 0x08,
    VOLUME = 0x10,
    BUY    = 0x20,
    SELL   = 0x40
};
inline TickFlag operator|(TickFlag a, TickFlag b) {
    return static_cast<TickFlag>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

enum class TradeRetcode : int32_t {
    REQUOTE              = 10004,
    REJECT               = 10006,
    CANCEL               = 10007,
    PLACED               = 10008,
    DONE                 = 10009,
    DONE_PARTIAL         = 10010,
    ERROR                = 10011,
    TIMEOUT              = 10012,
    INVALID              = 10013,
    INVALID_VOLUME       = 10014,
    INVALID_PRICE        = 10015,
    INVALID_STOPS        = 10016,
    TRADE_DISABLED       = 10017,
    MARKET_CLOSED        = 10018,
    NO_MONEY             = 10019,
    PRICE_CHANGED        = 10020,
    PRICE_OFF            = 10021,
    INVALID_EXPIRATION   = 10022,
    ORDER_CHANGED        = 10023,
    TOO_MANY_REQUESTS    = 10024,
    NO_CHANGES           = 10025,
    SERVER_DISABLES_AT   = 10026,
    CLIENT_DISABLES_AT   = 10027,
    LOCKED               = 10028,
    FROZEN               = 10029,
    INVALID_FILL         = 10030,
    CONNECTION           = 10031,
    ONLY_REAL            = 10032,
    LIMIT_ORDERS         = 10033,
    LIMIT_VOLUME         = 10034,
    INVALID_ORDER        = 10035,
    POSITION_CLOSED      = 10036,
    INVALID_CLOSE_VOLUME = 10038,
    CLOSE_ORDER_EXIST    = 10039,
    LIMIT_POSITIONS      = 10040,
    REJECT_CANCEL        = 10041,
    int64_t_ONLY         = 10042,
    SHORT_ONLY           = 10043,
    CLOSE_ONLY           = 10044,
    FIFO_CLOSE           = 10045
};

enum class ReturnCode : int32_t {
    OK                    = 1,
    FAIL                  = -1,
    INVALID_PARAMS        = -2,
    NO_MEMORY             = -3,
    NOT_FOUND             = -4,
    INVALID_VERSION       = -5,
    AUTH_FAILED           = -6,
    UNSUPPORTED           = -7,
    AUTO_TRADING_DISABLED = -8,
    INTERNAL_FAIL         = -10000,
    INTERNAL_FAIL_SEND    = -10001,
    INTERNAL_FAIL_RECEIVE = -10002,
    INTERNAL_FAIL_INIT    = -10003,
    INTERNAL_FAIL_CONNECT = -10004,
    INTERNAL_FAIL_TIMEOUT = -10005
};

enum class SymbolType : int32_t {
    FOREX       = 0,  // Forex currency pairs
    FUTURES     = 1,  // Futures contracts
    STOCKS      = 2,  // Stocks and shares
    BONDS       = 3,  // Bonds
    CRYPTO      = 4,  // Cryptocurrencies
    ETFS        = 5,  // Exchange-Traded Funds
    INDICES     = 6,  // Market indices
    COMMODITIES = 7,  // Commodities
    OPTIONS     = 8,  // Options contracts
    UNKNOWN     = 9   // Unknown or unsupported type
};

struct TerminalInfo {
    bool        community_account;
    bool        community_connection;
    bool        connected;
    bool        dlls_allowed;
    bool        trade_allowed;
    bool        tradeapi_disabled;
    bool        email_enabled;
    bool        ftp_enabled;
    bool        notifications_enabled;
    bool        mqid;
    int         build;
    int         maxbars;
    int         codepage;
    int         ping_last;
    double      community_balance;
    double      retransmission;
    std::string company;
    std::string name;
    std::string language;
    std::string path;
    std::string data_path;
    std::string commondata_path;
};

struct AccountInfo {
    int64_t     login;
    int32_t     trade_mode;
    int64_t     leverage;
    int         limit_orders;
    int32_t     margin_so_mode;
    bool        trade_allowed;
    bool        trade_expert;
    int32_t     margin_mode;
    int         currency_digits;
    bool        fifo_close;
    double      balance;
    double      credit;
    double      profit;
    double      equity;
    double      margin;
    double      margin_free;
    double      margin_level;
    double      margin_so_call;
    double      margin_so_so;
    double      margin_initial;
    double      margin_maintenance;
    double      assets;
    double      liabilities;
    double      commission_blocked;
    std::string name;
    std::string server;
    std::string currency;
    std::string company;
};

struct SymbolInfo {
    bool        custom;
    int32_t     chart_mode;
    bool        select;
    bool        visible;
    int64_t     session_deals;
    int64_t     session_buy_orders;
    int64_t     session_sell_orders;
    int64_t     volume;
    int64_t     volumehigh;
    int64_t     volumelow;
    int64_t     time;
    int         digits;
    int         spread;
    bool        spread_float;
    int         ticks_bookdepth;
    int32_t     trade_calc_mode;
    int32_t     trade_mode;
    int64_t     start_time;
    int64_t     expiration_time;
    int         trade_stops_level;
    int         trade_freeze_level;
    int32_t     trade_exemode;
    int32_t     swap_mode;
    int         swap_rollover3days;
    bool        margin_hedged_use_leg;
    int32_t     expiration_mode;
    int32_t     filling_mode;
    int32_t     order_mode;
    int32_t     order_gtc_mode;
    int32_t     option_mode;
    int         option_right;
    double      bid;
    double      bidhigh;
    double      bidlow;
    double      ask;
    double      askhigh;
    double      asklow;
    double      last;
    double      lasthigh;
    double      lastlow;
    double      volume_real;
    double      volumehigh_real;
    double      volumelow_real;
    double      option_strike;
    double      point;
    double      trade_tick_value;
    double      trade_tick_value_profit;
    double      trade_tick_value_loss;
    double      trade_tick_size;
    double      trade_contract_size;
    double      trade_accrued_interest;
    double      trade_face_value;
    double      trade_liquidity_rate;
    double      volume_min;
    double      volume_max;
    double      volume_step;
    double      volume_limit;
    double      swap_long;
    double      swap_short;
    double      margin_initial;
    double      margin_maintenance;
    double      session_volume;
    double      session_turnover;
    double      session_interest;
    double      session_buy_orders_volume;
    double      session_sell_orders_volume;
    double      session_open;
    double      session_close;
    double      session_aw;
    double      session_price_settlement;
    double      session_price_limit_min;
    double      session_price_limit_max;
    double      margin_hedged;
    double      price_change;
    double      price_volatility;
    double      price_theoretical;
    double      price_greeks_delta;
    double      price_greeks_theta;
    double      price_greeks_gamma;
    double      price_greeks_vega;
    double      price_greeks_rho;
    double      price_greeks_omega;
    double      price_sensitivity;
    std::string basis;
    std::string category;
    std::string currency_base;
    std::string currency_profit;
    std::string currency_margin;
    std::string bank;
    std::string description;
    std::string exchange;
    std::string formula;
    std::string isin;
    std::string name;
    std::string page;
    std::string path;
};

#pragma pack(push, 1)
struct TickInfo {
    int64_t  time;         // Time of the last prices update
    double   bid;          // Current Bid price
    double   ask;          // Current Ask price
    double   last;         // Price of the last deal (Last)
    int64_t  volume;       // Volume for the current Last price
    int64_t  time_msc;     // Time of a price last update in milliseconds
    uint32_t flags;        // Tick flags
    double   volume_real;  // Volume for the current Last price with greater accuracy
};

struct RateInfo {
    int64_t  time;         // Period start time
    double   open;         // Open price
    double   high;         // The highest price of the period
    double   low;          // The lowest price of the period
    double   close;        // Close price
    uint64_t tick_volume;  // Tick volume
    int32_t  spread;       // Spread
    uint64_t real_volume;  // Trade volume
};
#pragma pack(pop)

struct BookInfo {
    int32_t  type;         // Order type from ENUM_BOOK_TYPE enumeration
    double   price;        // Price
    uint64_t volume;       // Volume
    double   volume_real;  // Volume with greater accuracy
};

struct TradeRequest {
    int32_t     action       = 0;    // Trade operation type
    int64_t     magic        = 0;    // Expert Advisor ID (magic number)
    int64_t     order        = 0;    // Order ticket
    std::string symbol       = "";   // Trade symbol
    double      volume       = 0.0;  // Requested volume for a deal in lots
    double      price        = 0.0;  // Price
    double      stoplimit    = 0.0;  // StopLimit level of the order
    double      sl           = 0.0;  // Stop Loss level of the order
    double      tp           = 0.0;  // Take Profit level of the order
    int64_t     deviation    = 0;    // Maximal possible deviation from the requested price
    int32_t     type         = 0;    // Order type
    int32_t     type_filling = 0;    // Order execution type
    int32_t     type_time    = 0;    // Order expiration type
    int64_t     expiration   = 0;    // Order expiration time
    std::string comment      = "";   // Order comment
    int64_t     position     = 0;    // Position ticket
    int64_t     position_by  = 0;    // The ticket of an opposite position
};

struct OrderCheckResult {
    uint32_t     retcode;       // Reply code
    double       balance;       // Balance after the execution of the deal
    double       equity;        // Equity after the execution of the deal
    double       profit;        // Floating profit
    double       margin;        // Margin requirements
    double       margin_free;   // Free margin
    double       margin_level;  // Margin level
    std::string  comment;       // Comment to the reply code (description of the error)
    TradeRequest request;
};

struct OrderSentResult {
    uint32_t    retcode;  // Operation return code
    int64_t     deal;     // Deal ticket, if it is performed
    int64_t     order;    // Order ticket, if it is placed
    double      volume;   // Deal volume, confirmed by broker
    double      price;    // Deal price, confirmed by broker
    double      bid;      // Current Bid price
    double      ask;      // Current Ask price
    std::string comment;  // Broker comment to operation (by default it is filled by description of
                          // trade server return code)
    uint32_t     request_id;        // Request ID set by the terminal during the dispatch
    int          retcode_external;  // Return code of an external trading system
    TradeRequest request;
};

struct TradeOrder {
    int64_t     ticket;
    int64_t     time_setup;
    int64_t     time_setup_msc;
    int64_t     time_done;
    int64_t     time_done_msc;
    int64_t     time_expiration;
    int32_t     type;
    int32_t     type_time;
    int32_t     type_filling;
    int32_t     state;
    int64_t     magic;
    int64_t     position_id;
    int64_t     position_by_id;
    int32_t     reason;
    double      volume_initial;
    double      volume_current;
    double      price_open;
    double      sl;
    double      tp;
    double      price_current;
    double      price_stoplimit;
    std::string symbol;
    std::string comment;
    std::string external_id;
};

struct TradePosition {
    int64_t     ticket;
    int64_t     time;
    int64_t     time_msc;
    int64_t     time_update;
    int64_t     time_update_msc;
    int32_t     type;
    int64_t     magic;
    int64_t     identifier;
    int32_t     reason;
    double      volume;
    double      price_open;
    double      sl;
    double      tp;
    double      price_current;
    double      swap;
    double      profit;
    std::string symbol;
    std::string comment;
    std::string external_id;
};

struct TradeDeal {
    int64_t     ticket;
    int64_t     order;
    int64_t     time;
    int64_t     time_msc;
    int32_t     type;
    int32_t     entry;
    int64_t     magic;
    int64_t     position_id;
    int32_t     reason;
    double      volume;
    double      price;
    double      commission;
    double      swap;
    double      profit;
    double      fee;
    std::string symbol;
    std::string comment;
    std::string external_id;
};
}  // namespace MT5
