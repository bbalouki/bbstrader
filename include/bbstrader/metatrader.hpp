/// @file metatrader.hpp
/// @brief Declares the public API for the bbstrader library.
/// This is the primary file that you should \#include if
/// you want to use the library.

#pragma once

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "bbstrader/objects.hpp"

namespace py = pybind11;

/// @brief Namespace containing MetaTrader 5 (MT5) client definitions and wrapper classes.
namespace MT5 {

/// @brief Alias for const std::string to simplify function signatures.
using str = const std::string;

/// @brief Tuple representing version info: <major version, minor version, build date string>.
using VersionInfo = std::tuple<int32_t, int32_t, std::string>;

/// @brief Tuple representing the last error: <error code, error description>.
using LastErrorResult = std::tuple<int32_t, std::string>;

/// @brief Optional vector of TradeDeal objects.
using DealsData = std::optional<std::vector<TradeDeal>>;

/// @brief Optional vector of SymbolInfo objects.
using SymbolsData = std::optional<std::vector<SymbolInfo>>;

/// @brief Optional vector of BookInfo (Market Depth) objects.
using BookData = std::optional<std::vector<BookInfo>>;

/// @brief Optional vector of TradeOrder objects.
using OrdersData = std::optional<std::vector<TradeOrder>>;

/// @brief Optional vector of TradePosition objects.
using PositionsData = std::optional<std::vector<TradePosition>>;

/// @brief Optional Numpy array of RateInfo structures (OHLCV data).
using RateInfoType = std::optional<py::array_t<RateInfo>>;

/// @brief Optional Numpy array of TickInfo structures.
using TickInfoType = std::optional<py::array_t<TickInfo>>;

/// @brief System clock time point for date/time operations.
using DateTime = std::chrono::system_clock::time_point;

// Function Type Definitions (Callbacks/Handlers)

/// @brief Callback to initialize the MT5 terminal automatically.
using InitializeAuto = std::function<bool()>;

/// @brief Callback to initialize the MT5 terminal with a specific path.
using InitializeWithPath = std::function<bool(str&)>;

/// @brief Callback to initialize the MT5 terminal with full configuration options.
using InitializeFull = std::function<bool(str&, uint64_t, str&, str&, uint32_t, bool)>;

/// @brief Callback to login to an account.
using Login = std::function<bool(uint64_t, str&, str&, uint32_t)>;

/// @brief Callback to shut down the initialized terminal.
using Shutdown = std::function<void()>;

/// @brief Callback to retrieve the terminal version.
using GetVersion = std::function<std::optional<VersionInfo>()>;

/// @brief Callback to retrieve the last error code and description.
using GetLastError = std::function<std::optional<LastErrorResult>()>;

/// @brief Callback to retrieve static terminal information.
using GetTerminalInfo = std::function<std::optional<TerminalInfo>()>;

/// @brief Callback to retrieve account information.
using GetAccountInfo = std::function<std::optional<AccountInfo>()>;

/// @brief Callback to retrieve information for a specific symbol.
using GetSymbolInfo = std::function<std::optional<SymbolInfo>(str&)>;

/// @brief Callback to retrieve the last tick for a specific symbol.
using GetTickInfo = std::function<std::optional<TickInfo>(str&)>;

/// @brief Callback to get the total count of available symbols.
using GetTotalSymbols = std::function<std::optional<int32_t>()>;

/// @brief Callback to retrieve all available symbols.
using GetSymbolsAll = std::function<SymbolsData()>;

/// @brief Callback to select or deselect a symbol in Market Watch.
using SelectSymbol = std::function<bool(str&, bool)>;

/// @brief Callback to retrieve symbols belonging to a specific group.
using GetSymbolsByGroup = std::function<SymbolsData(str&)>;

/// @brief Callback to subscribe to the Market Book (Depth of Market) for a symbol.
using SubscribeBook = std::function<bool(str&)>;

/// @brief Callback to unsubscribe from the Market Book for a symbol.
using UnsubscribeBook = std::function<bool(str&)>;

/// @brief Callback to get the current Market Book state for a symbol.
using GetBookInfo = std::function<BookData(str&)>;

// Rates & Ticks Handlers

/// @brief Callback to get rates starting from a specific date.
using GetRatesByDate = std::function<RateInfoType(str&, int32_t, int64_t, int32_t)>;

/// @brief Callback to get rates starting from a specific index position.
using GetRatesByPos = std::function<RateInfoType(str&, int32_t, int32_t, int32_t)>;

/// @brief Callback to get rates within a time range.
using GetRatesByRange = std::function<RateInfoType(str&, int32_t, int64_t, int64_t)>;

/// @brief Callback to get ticks starting from a specific date.
using GetTicksByDate = std::function<TickInfoType(str&, int64_t, int32_t, int32_t)>;

/// @brief Callback to get ticks within a time range.
using GetTicksByRange = std::function<TickInfoType(str&, int64_t, int64_t, int32_t)>;

/// @defgroup OrderHandlers Order Management Callbacks
/// @brief Functional interfaces for retrieving trade order data.
///  @{

/// @brief Callback to retrieve all current trade orders.
using GetOrdersAll = std::function<OrdersData()>;

/// @brief Callback to retrieve trade orders filtered by symbol.
using GetOrdersBySymbol = std::function<OrdersData(str&)>;

/// @brief Callback to retrieve trade orders filtered by a specific group.
using GetOrdersByGroup = std::function<OrdersData(str&)>;

/// @brief Callback to retrieve a specific order by its unique ticket ID.
using GetOrderByTicket = std::function<std::optional<TradeOrder>(uint64_t)>;

/// @brief Callback to retrieve the total count of active orders.
using GetTotalOrders = std::function<std::optional<int32_t>()>;
/// @}

/// @defgroup PositionHandlers Active Position Handlers
/// @brief Functional interfaces for retrieving active trade positions.
/// @{

/// @brief Callback to retrieve all active trade positions.
using GetPositionsAll = std::function<PositionsData()>;

/// @brief Callback to retrieve active positions filtered by symbol.
using GetPositionsBySymbol = std::function<PositionsData(str&)>;

/// @brief Callback to retrieve active positions filtered by group.
using GetPositionsByGroup = std::function<PositionsData(str&)>;

/// @brief Callback to retrieve a specific position by its unique ticket ID.
using GetPositionByTicket = std::function<std::optional<TradePosition>(uint64_t)>;

/// @brief Callback to retrieve the total count of active positions.
using GetTotalPositions = std::function<std::optional<int32_t>()>;

/// @}

// Trading Handlers

/// @brief Callback to calculate margin.
using CalculateMargin = std::function<std::optional<double>(int32_t, str&, double, double)>;

/// @brief Callback to calculate profit.
using CalculateProfit = std::function<std::optional<double>(int32_t, str&, double, double, double)>;

/// @brief Callback to check an order request before sending.
using CheckOrder = std::function<std::optional<OrderCheckResult>(const TradeRequest&)>;

/// @brief Callback to send an order to the market.
using SendOrder = std::function<std::optional<OrderSentResult>(const TradeRequest&)>;

// History Handlers

/// @defgroup HistoryHandlers Order and Deal History Callbacks
/// @brief Functional interfaces for retrieving historical trade data (Closed Orders and Deals).
/// @{

/// @brief Callback to retrieve historical orders within a specific time range.
using GetHistoryOrdersByRange = std::function<OrdersData(int64_t, int64_t)>;

/// @brief Callback to retrieve historical orders within a specific time range and filter.
using GetHistoryOrdersByGroup = std::function<OrdersData(int64_t, int64_t, str&)>;

/// @brief Callback to retrieve a specific historical order by its ticket ID.
using GetHistoryOrderByTicket = std::function<std::optional<TradeOrder>(uint64_t)>;

/// @brief Callback to retrieve all historical orders associated with a specific Position ID.
using GetHistoryOrdersByPosId = std::function<OrdersData(uint64_t)>;

/// @brief Callback to retrieve the total count of historical orders within a time range.
using GetHistoryOrdersTotal = std::function<std::optional<int32_t>(int64_t, int64_t)>;

/// @brief Callback to retrieve historical deals (executions) within a time range.
using GetHistoryDealsByRange = std::function<DealsData(int64_t, int64_t)>;

/// @brief Callback to retrieve historical deals (executions) within a time range buy asset.
using GetHistoryDealsByGroup = std::function<DealsData(int64_t, int64_t, str&)>;

/// @brief Callback to retrieve historical deals associated with a specific Order Ticket.
using GetHistoryDealsByTicket = std::function<DealsData(uint64_t)>;

/// @brief Callback to retrieve historical deals associated with a specific Position ID.
using GetHistoryDealsByPosId = std::function<DealsData(uint64_t)>;

/// @brief Callback to retrieve the total count of historical deals within a time range.
using GetHistoryDealsTotal = std::function<std::optional<int32_t>(int64_t, int64_t)>;
/// @}

/// @class MetaTraderClient
/// @brief A client class that abstracts MetaTrader 5 functionality using injected handlers.
/// @details This class acts as a facade, delegating actual API calls to a set of std::function
/// handlers provided at construction.
class MetaTraderClient {
   public:
    /// @struct Handlers
    /// @brief Container for all function callbacks required by the client.
    struct Handlers {
        // System
        InitializeAuto     init_auto;          ///< Auto-initialization handler
        InitializeWithPath init_path;          ///< Path-based initialization handler
        InitializeFull     init_full;          ///< Full-parameter initialization handler
        Login              login;              ///< Account login handler
        Shutdown           shutdown;           ///< Shutdown handler
        GetVersion         get_version;        ///< Version retrieval handler
        GetLastError       get_last_error;     ///< Error retrieval handler
        GetTerminalInfo    get_terminal_info;  ///< Terminal info handler
        GetAccountInfo     get_account_info;   ///< Account info handler

        // Symbols
        GetTotalSymbols   get_total_symbols;     ///< Symbol count handler
        GetSymbolsAll     get_symbols_all;       ///< Get all symbols handler
        GetSymbolInfo     get_symbol_info;       ///< Get symbol details handler
        SelectSymbol      select_symbol;         ///< Symbol selection handler
        GetSymbolsByGroup get_symbols_by_group;  ///< Get symbols by group handler

        // Market Depth
        SubscribeBook   subscribe_book;    ///< Subscribe to DOM handler
        UnsubscribeBook unsubscribe_book;  ///< Unsubscribe from DOM handler
        GetBookInfo     get_book_info;     ///< Get DOM info handler

        // Rates & Ticks
        GetRatesByDate  get_rates_by_date;   ///< Get rates by date handler
        GetRatesByPos   get_rates_by_pos;    ///< Get rates by position handler
        GetRatesByRange get_rates_by_range;  ///< Get rates by range handler
        GetTicksByDate  get_ticks_by_date;   ///< Get ticks by date handler
        GetTicksByRange get_ticks_by_range;  ///< Get ticks by range handler
        GetTickInfo     get_tick_info;       ///< Get current tick info handler

        // Active Orders
        GetOrdersAll      get_orders_all;        ///< Get all active orders handler
        GetOrdersBySymbol get_orders_by_symbol;  ///< Get orders by symbol handler
        GetOrdersByGroup  get_orders_by_group;   ///< Get orders by group handler
        GetOrderByTicket  get_order_by_ticket;   ///< Get order by ticket handler
        GetTotalOrders    get_total_orders;      ///< Get total active orders handler

        // Active Positions
        GetPositionsAll      get_positions_all;     ///< Get all positions handler
        GetPositionsBySymbol get_positions_symbol;  ///< Get positions by symbol handler
        GetPositionsByGroup  get_positions_group;   ///< Get positions by group handler
        GetPositionByTicket  get_position_ticket;   ///< Get position by ticket handler
        GetTotalPositions    get_total_positions;   ///< Get total positions handler

        // Trading
        CheckOrder      check_order;  ///< Order check handler
        SendOrder       send_order;   ///< Order send handler
        CalculateMargin calc_margin;  ///< Margin calculation handler
        CalculateProfit calc_profit;  ///< Profit calculation handler

        // History Orders
        GetHistoryOrdersByRange get_hist_orders_range;  ///< Get history orders by range handler
        GetHistoryOrdersByGroup get_hist_orders_group;  ///< Get history orders by group handler
        GetHistoryOrderByTicket get_hist_order_ticket;  ///< Get history order by ticket handler
        GetHistoryOrdersByPosId get_hist_orders_pos;  ///< Get history orders by position ID handler
        GetHistoryOrdersTotal   get_hist_orders_total;  ///< Get history orders count handler

        // History Deals
        GetHistoryDealsByRange  get_hist_deals_range;   ///< Get history deals by range handler
        GetHistoryDealsByGroup  get_hist_deals_group;   ///< Get history deals by group handler
        GetHistoryDealsByTicket get_hist_deals_ticket;  ///< Get history deal by ticket handler
        GetHistoryDealsByPosId  get_hist_deals_pos;    ///< Get history deals by position ID handler
        GetHistoryDealsTotal    get_hist_deals_total;  ///< Get history deals count handler
    };

    /// @brief Default constructor.
    MetaTraderClient()                                   = delete;
    MetaTraderClient(const MetaTraderClient&)            = delete;
    MetaTraderClient& operator=(const MetaTraderClient&) = delete;

    /// @brief Virtual destructor.
    virtual ~MetaTraderClient() = default;

    /// @brief Constructor with handlers.
    /// @param handlers A struct containing the function implementations.
    explicit MetaTraderClient(Handlers handlers) : this_handlers(std::move(handlers)) {}

    // System Methods

    /// @brief Initializes the MT5 terminal automatically.
    /// @return True if successful, false otherwise.
    virtual auto initialize() -> bool {
        return this_handlers.init_auto ? this_handlers.init_auto() : false;
    }

    /// @brief Initializes the MT5 terminal at a specific path.
    /// @param path Path to the MT5 terminal executable or directory.
    /// @return True if successful, false otherwise.
    virtual auto initialize(str& path) -> bool {
        return this_handlers.init_path ? this_handlers.init_path(path) : false;
    }

    /// @brief Initializes the MT5 terminal with full configuration.
    /// @param path Path to MT5.
    /// @param account Account number.
    /// @param pw Password.
    /// @param srv Server name.
    /// @param timeout Connection timeout in milliseconds.
    /// @param portable Boolean flag for portable mode.
    /// @return True if successful, false otherwise.
    virtual auto initialize(
        str& path, uint64_t account, str& pw, str& srv, uint32_t timeout, bool portable
    ) -> bool {
        return this_handlers.init_full
                   ? this_handlers.init_full(path, account, pw, srv, timeout, portable)
                   : false;
    }

    /// @brief Connects to a trading account.
    /// @param account Account number.
    /// @param pw Password.
    /// @param srv Server name.
    /// @param timeout Connection timeout.
    /// @return True if login successful, false otherwise.
    virtual auto login(uint64_t account, str& pw, str& srv, uint32_t timeout) -> bool {
        return this_handlers.login ? this_handlers.login(account, pw, srv, timeout) : false;
    }

    /// @brief Shuts down the MT5 connection/terminal.
    virtual auto shutdown() -> void {
        if (this_handlers.shutdown)
            this_handlers.shutdown();
    }

    /// @brief Gets the MT5 version information.
    /// @return Optional VersionInfo tuple.
    virtual auto version() -> std::optional<VersionInfo> {
        return this_handlers.get_version ? this_handlers.get_version() : std::nullopt;
    }

    /// @brief Gets the last error occurred.
    /// @return Optional LastErrorResult tuple (code, description). Returns (-1, "fail") if handler
    /// missing.
    virtual auto last_error() -> std::optional<LastErrorResult> {
        return this_handlers.get_last_error ? this_handlers.get_last_error()
                                            : std::make_tuple(-1, std::string("fail"));
    }

    /// @brief Gets terminal status and settings.
    /// @return Optional TerminalInfo struct.
    virtual auto terminal_info() -> std::optional<TerminalInfo> {
        return this_handlers.get_terminal_info ? this_handlers.get_terminal_info() : std::nullopt;
    }

    /// @brief Gets current account information.
    /// @return Optional AccountInfo struct.
    virtual auto account_info() -> std::optional<AccountInfo> {
        return this_handlers.get_account_info ? this_handlers.get_account_info() : std::nullopt;
    }

    // Symbol Methods

    /// @brief Gets the total number of available symbols.
    /// @return Number of symbols or 0 if failed.
    virtual auto symbols_total() -> std::optional<int32_t> {
        return this_handlers.get_total_symbols ? this_handlers.get_total_symbols() : 0;
    }

    /// @brief Gets all available symbols.
    /// @return Optional vector of SymbolInfo.
    virtual auto symbols_get() -> SymbolsData {
        return this_handlers.get_symbols_all ? this_handlers.get_symbols_all() : std::nullopt;
    }

    /// @brief Gets symbols for a specific group (e.g., "EUR*").
    /// @param group Group name pattern.
    /// @return Optional vector of SymbolInfo.
    virtual auto symbols_get(str& group) -> SymbolsData {
        return this_handlers.get_symbols_by_group ? this_handlers.get_symbols_by_group(group)
                                                  : std::nullopt;
    }

    /// @brief Gets information for a specific symbol.
    /// @param symbol Symbol name.
    /// @return Optional SymbolInfo struct.
    virtual auto symbol_info(str& symbol) -> std::optional<SymbolInfo> {
        return this_handlers.get_symbol_info ? this_handlers.get_symbol_info(symbol) : std::nullopt;
    }

    /// @brief Selects or deselects a symbol in the Market Watch window.
    /// @param symbol Symbol name.
    /// @param enable True to select, false to deselect.
    /// @return True if successful.
    virtual auto symbol_select(str& symbol, bool enable) -> bool {
        return this_handlers.select_symbol ? this_handlers.select_symbol(symbol, enable) : false;
    }

    /// @brief Gets the current tick info for a symbol.
    /// @param symbol Symbol name.
    /// @return Optional TickInfo struct.
    virtual auto symbol_info_tick(str& symbol) -> std::optional<TickInfo> {
        return this_handlers.get_tick_info ? this_handlers.get_tick_info(symbol) : std::nullopt;
    }

    // Market Depth Methods

    /// @brief Subscribes to the Market Book (Depth of Market) for a symbol.
    /// @param symbol Symbol name.
    /// @return True if subscription successful.
    virtual auto market_book_add(str& symbol) -> bool {
        return this_handlers.subscribe_book ? this_handlers.subscribe_book(symbol) : false;
    }

    /// @brief Unsubscribes from the Market Book for a symbol.
    /// @param symbol Symbol name.
    /// @return True if unsubscription successful.
    virtual auto market_book_release(str& symbol) -> bool {
        return this_handlers.unsubscribe_book ? this_handlers.unsubscribe_book(symbol) : false;
    }

    /// @brief Gets the current Market Book state.
    /// @param symbol Symbol name.
    /// @return Optional vector of BookInfo.
    virtual auto market_book_get(str& symbol) -> BookData {
        return this_handlers.get_book_info ? this_handlers.get_book_info(symbol) : std::nullopt;
    }

    // Market Data (Copy Rates/Ticks) Methods

    /// @brief Copies rates from the MT5 terminal starting at a specific DateTime.
    /// @param s Symbol name.
    /// @param t Timeframe.
    /// @param from Start DateTime (converted to timestamp).
    /// @param count Number of bars to retrieve.
    /// @return Numpy array of RateInfo.
    virtual auto copy_rates_from(str& s, int32_t t, DateTime from, int32_t count) -> RateInfoType {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        return copy_rates_from(s, t, from_ts, count);
    }

    /// @brief Copies rates starting at a specific timestamp.
    /// @param s Symbol name.
    /// @param t Timeframe.
    /// @param from Start timestamp (seconds).
    /// @param count Number of bars.
    /// @return Numpy array of RateInfo.
    virtual auto copy_rates_from(str& s, int32_t t, int64_t from, int32_t count) -> RateInfoType {
        return this_handlers.get_rates_by_date ? this_handlers.get_rates_by_date(s, t, from, count)
                                               : py::array_t<RateInfo>();
    }

    /// @brief Copies rates relative to the current position (index).
    /// @param s Symbol name.
    /// @param t Timeframe.
    /// @param start Start index.
    /// @param count Number of bars.
    /// @return Numpy array of RateInfo.
    virtual auto copy_rates_from_pos(str& s, int32_t t, int32_t start, int32_t count)
        -> RateInfoType {
        return this_handlers.get_rates_by_pos ? this_handlers.get_rates_by_pos(s, t, start, count)
                                              : py::array_t<RateInfo>();
    }

    /// @brief Copies rates within a specific DateTime range.
    /// @param s Symbol name.
    /// @param t Timeframe.
    /// @param from Start DateTime.
    /// @param to End DateTime.
    /// @return Numpy array of RateInfo.
    virtual auto copy_rates_range(str& s, int32_t t, DateTime from, DateTime to) -> RateInfoType {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return copy_rates_range(s, t, from_ts, to_ts);
    }

    /// @brief Copies rates within a specific timestamp range.
    /// @param s Symbol name.
    /// @param t Timeframe.
    /// @param from Start timestamp.
    /// @param to End timestamp.
    /// @return Numpy array of RateInfo.
    virtual auto copy_rates_range(str& s, int32_t t, int64_t from, int64_t to) -> RateInfoType {
        return this_handlers.get_rates_by_range ? this_handlers.get_rates_by_range(s, t, from, to)
                                                : py::array_t<RateInfo>();
    }

    /// @brief Copies ticks starting from a specific DateTime.
    /// @param s Symbol name.
    /// @param from Start DateTime.
    /// @param count Number of ticks.
    /// @param flags Tick flags (info/trade/etc).
    /// @return Numpy array of TickInfo.
    virtual auto copy_ticks_from(str& s, DateTime from, int32_t count, int32_t flags)
        -> TickInfoType {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        return copy_ticks_from(s, from_ts, count, flags);
    }

    /// @brief Copies ticks starting from a specific timestamp.
    virtual auto copy_ticks_from(str& s, int64_t from, int32_t count, int32_t flags)
        -> TickInfoType {
        return this_handlers.get_ticks_by_date
                   ? this_handlers.get_ticks_by_date(s, from, count, flags)
                   : py::array_t<TickInfo>();
    }

    /// @brief Copies ticks within a specific DateTime range.
    virtual auto copy_ticks_range(str& s, DateTime from, DateTime to, int32_t flags)
        -> TickInfoType {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return copy_ticks_range(s, from_ts, to_ts, flags);
    }

    /// @brief Copies ticks within a specific timestamp range.
    virtual auto copy_ticks_range(str& s, int64_t from, int64_t to, int32_t flags) -> TickInfoType {
        return this_handlers.get_ticks_by_range
                   ? this_handlers.get_ticks_by_range(s, from, to, flags)
                   : py::array_t<TickInfo>();
    }

    // Active Order Methods

    /// @brief Gets all active orders.
    /// @return Optional vector of TradeOrder.
    virtual auto orders_get() -> OrdersData {
        return this_handlers.get_orders_all ? this_handlers.get_orders_all() : std::nullopt;
    }

    /// @brief Gets active orders for a specific symbol.
    virtual auto orders_get(str& symbol) -> OrdersData {
        return this_handlers.get_orders_by_symbol ? this_handlers.get_orders_by_symbol(symbol)
                                                  : std::nullopt;
    }

    /// @brief Gets active orders for a specific symbol group.
    virtual auto orders_get_by_group(str& group) -> OrdersData {
        return this_handlers.get_orders_by_group ? this_handlers.get_orders_by_group(group)
                                                 : std::nullopt;
    }

    /// @brief Gets a specific order by its ticket number.
    virtual auto order_get_by_ticket(uint64_t ticket) -> std::optional<TradeOrder> {
        return this_handlers.get_order_by_ticket ? this_handlers.get_order_by_ticket(ticket)
                                                 : std::nullopt;
    }

    /// @brief Gets total number of active orders.
    virtual auto orders_total() -> std::optional<int32_t> {
        return this_handlers.get_total_orders ? this_handlers.get_total_orders() : 0;
    }

    // Active Position Methods

    /// @brief Gets all open positions.
    /// @return Optional vector of TradePosition.
    virtual auto positions_get() -> PositionsData {
        return this_handlers.get_positions_all ? this_handlers.get_positions_all() : std::nullopt;
    }

    /// @brief Gets open positions for a specific symbol.
    virtual auto positions_get(str& symbol) -> PositionsData {
        return this_handlers.get_positions_symbol ? this_handlers.get_positions_symbol(symbol)
                                                  : std::nullopt;
    }

    /// @brief Gets open positions for a specific group.
    virtual auto positions_get_by_group(str& group) -> PositionsData {
        return this_handlers.get_positions_group ? this_handlers.get_positions_group(group)
                                                 : std::nullopt;
    }

    /// @brief Gets a specific position by its ticket.
    virtual auto position_get_by_ticket(uint64_t ticket) -> std::optional<TradePosition> {
        return this_handlers.get_position_ticket ? this_handlers.get_position_ticket(ticket)
                                                 : std::nullopt;
    }

    /// @brief Gets total number of open positions.
    virtual auto positions_total() -> std::optional<int32_t> {
        return this_handlers.get_total_positions ? this_handlers.get_total_positions() : 0;
    }

    // Trading Methods

    /// @brief Checks if a trade request is valid (Python dict overload).
    /// @param dict Python dictionary representing the trade request.
    /// @return Optional OrderCheckResult.
    virtual auto order_check(const py::dict& dict) -> std::optional<OrderCheckResult> {
        return order_check(dict.cast<TradeRequest>());
    }

    /// @brief Checks if a trade request is valid.
    /// @param req TradeRequest structure.
    /// @return Optional OrderCheckResult.
    virtual auto order_check(const TradeRequest& req) -> std::optional<OrderCheckResult> {
        return this_handlers.check_order ? this_handlers.check_order(req) : OrderCheckResult{};
    }

    /// @brief Sends a trade request to the server (Python dict overload).
    /// @param dict Python dictionary representing the trade request.
    /// @return Optional OrderSentResult.
    virtual auto order_send(const py::dict& dict) -> std::optional<OrderSentResult> {
        return order_send(dict.cast<TradeRequest>());
    }

    /// @brief Sends a trade request to the server.
    /// @param req TradeRequest structure.
    /// @return Optional OrderSentResult.
    virtual auto order_send(const TradeRequest& req) -> std::optional<OrderSentResult> {
        return this_handlers.send_order ? this_handlers.send_order(req) : OrderSentResult{};
    }

    /// @brief Calculates the margin required for an order.
    /// @param action Order action (buy/sell).
    /// @param sym Symbol name.
    /// @param vol Volume.
    /// @param prc Price.
    /// @return Required margin or nullopt.
    virtual auto order_calc_margin(int32_t action, str& sym, double vol, double prc)
        -> std::optional<double> {
        return this_handlers.calc_margin ? this_handlers.calc_margin(action, sym, vol, prc)
                                         : std::nullopt;
    }

    /// @brief Calculates the potential profit for an order.
    /// @param action Order action.
    /// @param sym Symbol name.
    /// @param vol Volume.
    /// @param open Open price.
    /// @param close Close price.
    /// @return Profit or nullopt.
    virtual auto order_calc_profit(int32_t action, str& sym, double vol, double open, double close)
        -> std::optional<double> {
        return this_handlers.calc_profit ? this_handlers.calc_profit(action, sym, vol, open, close)
                                         : std::nullopt;
    }

    // History Order Methods

    /// @brief Gets historical orders within a timestamp range.
    /// @param from Start timestamp.
    /// @param to End timestamp.
    /// @param group Group name filter
    /// @return Optional vector of TradeOrder.
    virtual auto history_orders_get(int64_t from, int64_t to, str& group) -> OrdersData {
        return this_handlers.get_hist_orders_group
                   ? this_handlers.get_hist_orders_group(from, to, group)
                   : std::nullopt;
    }

    /// @brief Gets historical orders within a DateTime range.
    virtual auto history_orders_get(DateTime from, DateTime to, str& group) -> OrdersData {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_orders_get(from_ts, to_ts, group);
    }

    /// @brief Gets historical orders within a timestamp range.
    /// @param from Start timestamp.
    /// @param to End timestamp.
    /// @return Optional vector of TradeOrder.
    virtual auto history_orders_get(int64_t from, int64_t to) -> OrdersData {
        return this_handlers.get_hist_orders_range ? this_handlers.get_hist_orders_range(from, to)
                                                   : std::nullopt;
    }

    /// @brief Gets historical orders within a DateTime range.
    virtual auto history_orders_get(DateTime from, DateTime to) -> OrdersData {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_orders_get(from_ts, to_ts);
    }

    /// @brief Gets a historical order by its ticket.
    virtual auto history_orders_get(uint64_t ticket) -> std::optional<TradeOrder> {
        return this_handlers.get_hist_order_ticket ? this_handlers.get_hist_order_ticket(ticket)
                                                   : std::nullopt;
    }

    /// @brief Gets historical orders associated with a position ID.
    virtual auto history_orders_get_by_pos(uint64_t pos_id) -> OrdersData {
        return this_handlers.get_hist_orders_pos ? this_handlers.get_hist_orders_pos(pos_id)
                                                 : std::nullopt;
    }

    /// @brief Gets the count of historical orders in a timestamp range.
    virtual auto history_orders_total(int64_t from, int64_t to) -> std::optional<int32_t> {
        return this_handlers.get_hist_orders_total ? this_handlers.get_hist_orders_total(from, to)
                                                   : 0;
    }

    /// @brief Gets the count of historical orders in a DateTime range.
    virtual auto history_orders_total(DateTime from, DateTime to) -> std::optional<int32_t> {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_orders_total(from_ts, to_ts);
    }

    // History Deal Methods

    /// @brief Gets historical deals within a timestamp range.
    /// @param from Start timestamp.
    /// @param to End timestamp.
    /// @param group Group name filter
    /// @return Optional vector of TradeDeal.
    virtual auto history_deals_get(int64_t from, int64_t to, str& group) -> DealsData {
        return this_handlers.get_hist_deals_group
                   ? this_handlers.get_hist_deals_group(from, to, group)
                   : std::nullopt;
    }

    /// @brief Gets historical deals within a timestamp range.
    /// @param from Start timestamp.
    /// @param to End timestamp.
    /// @return Optional vector of TradeDeal.
    virtual auto history_deals_get(int64_t from, int64_t to) -> DealsData {
        return this_handlers.get_hist_deals_range ? this_handlers.get_hist_deals_range(from, to)
                                                  : std::nullopt;
    }

    /// @brief Gets historical deals within a DateTime range.
    virtual auto history_deals_get(DateTime from, DateTime to, str& group) -> DealsData {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_deals_get(from_ts, to_ts, group);
    }

    /// @brief Gets historical deals within a DateTime range.
    virtual auto history_deals_get(DateTime from, DateTime to) -> DealsData {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_deals_get(from_ts, to_ts);
    }

    /// @brief Gets a historical deal by its ticket.
    virtual auto history_deals_get(uint64_t ticket) -> DealsData {
        return this_handlers.get_hist_deals_ticket ? this_handlers.get_hist_deals_ticket(ticket)
                                                   : std::nullopt;
    }

    /// @brief Gets historical deals associated with a position ID.
    virtual auto history_deals_get_by_pos(uint64_t pos_id) -> DealsData {
        return this_handlers.get_hist_deals_pos ? this_handlers.get_hist_deals_pos(pos_id)
                                                : std::nullopt;
    }

    /// @brief Gets the count of historical deals in a timestamp range.
    virtual auto history_deals_total(int64_t from, int64_t to) -> std::optional<int32_t> {
        return this_handlers.get_hist_deals_total ? this_handlers.get_hist_deals_total(from, to)
                                                  : 0;
    }

    /// @brief Gets the count of historical deals in a DateTime range.
    virtual auto history_deals_total(DateTime from, DateTime to) -> std::optional<int32_t> {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_deals_total(from_ts, to_ts);
    }

   private:
    Handlers this_handlers;  ///< The struct containing all callback function implementations.
};

}  // namespace MT5
