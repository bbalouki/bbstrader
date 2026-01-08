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

#include "metatrader/objects.hpp"

namespace py = pybind11;

namespace MT5 {
using str             = const std::string;
using VersionInfo     = std::tuple<int32_t, int32_t, std::string>;
using LastErrorResult = std::tuple<int32_t, std::string>;
using DealsData       = std::optional<std::vector<TradeDeal>>;
using SymbolsData     = std::optional<std::vector<SymbolInfo>>;
using BookData        = std::optional<std::vector<BookInfo>>;
using OrdersData      = std::optional<std::vector<TradeOrder>>;
using PositionsData   = std::optional<std::vector<TradePosition>>;
using RateInfoType    = std::optional<py::array_t<RateInfo>>;
using TickInfoType    = std::optional<py::array_t<TickInfo>>;
using DateTime        = std::chrono::system_clock::time_point;

using InitializeAuto     = std::function<bool()>;
using InitializeWithPath = std::function<bool(str&)>;
using InitializeFull     = std::function<bool(str&, uint64_t, str&, str&, uint32_t, bool)>;
using Login              = std::function<bool(uint64_t, str&, str&, uint32_t)>;
using Shutdown           = std::function<void()>;
using GetVersion         = std::function<std::optional<VersionInfo>()>;
using GetLastError       = std::function<std::optional<LastErrorResult>()>;

using GetTerminalInfo = std::function<std::optional<TerminalInfo>()>;
using GetAccountInfo  = std::function<std::optional<AccountInfo>()>;
using GetSymbolInfo   = std::function<std::optional<SymbolInfo>(str&)>;
using GetTickInfo     = std::function<std::optional<TickInfo>(str&)>;

using GetTotalSymbols   = std::function<std::optional<int32_t>()>;
using GetSymbolsAll     = std::function<SymbolsData()>;
using SelectSymbol      = std::function<bool(str&, bool)>;
using GetSymbolsByGroup = std::function<SymbolsData(str&)>;

using SubscribeBook   = std::function<bool(str&)>;
using UnsubscribeBook = std::function<bool(str&)>;
using GetBookInfo     = std::function<BookData(str&)>;

using GetRatesByDate  = std::function<RateInfoType(str&, int32_t, int64_t, int32_t)>;
using GetRatesByPos   = std::function<RateInfoType(str&, int32_t, int32_t, int32_t)>;
using GetRatesByRange = std::function<RateInfoType(str&, int32_t, int64_t, int64_t)>;

using GetTicksByDate  = std::function<TickInfoType(str&, int64_t, int32_t, int32_t)>;
using GetTicksByRange = std::function<TickInfoType(str&, int64_t, int64_t, int32_t)>;

using GetOrdersAll      = std::function<OrdersData()>;
using GetOrdersBySymbol = std::function<OrdersData(str&)>;
using GetOrdersByGroup  = std::function<OrdersData(str&)>;
using GetOrderByTicket  = std::function<std::optional<TradeOrder>(uint64_t)>;
using GetTotalOrders    = std::function<std::optional<int32_t>()>;

using GetPositionsAll      = std::function<PositionsData()>;
using GetPositionsBySymbol = std::function<PositionsData(str&)>;
using GetPositionsByGroup  = std::function<PositionsData(str&)>;
using GetPositionByTicket  = std::function<std::optional<TradePosition>(uint64_t)>;
using GetTotalPositions    = std::function<std::optional<int32_t>()>;

using CheckOrder = std::function<std::optional<OrderCheckResult>(const TradeRequest&)>;
using SendOrder  = std::function<std::optional<OrderSentResult>(const TradeRequest&)>;

using CalculateMargin = std::function<std::optional<double>(int32_t, str&, double, double)>;
using CalculateProfit = std::function<std::optional<double>(int32_t, str&, double, double, double)>;

using GetHistoryOrdersByRange = std::function<OrdersData(int64_t, int64_t, str&)>;
using GetHistoryOrderByTicket = std::function<std::optional<TradeOrder>(uint64_t)>;
using GetHistoryOrdersByPosId = std::function<OrdersData(uint64_t)>;
using GetHistoryOrdersTotal   = std::function<std::optional<int32_t>(int64_t, int64_t)>;

using GetHistoryDealsByRange  = std::function<DealsData(int64_t, int64_t, str&)>;
using GetHistoryDealsByTicket = std::function<DealsData(uint64_t)>;
using GetHistoryDealsByPosId  = std::function<DealsData(uint64_t)>;
using GetHistoryDealsTotal    = std::function<std::optional<int32_t>(int64_t, int64_t)>;

class MetaTraderClient {
   public:
    struct Handlers {
        // System
        InitializeAuto     init_auto;
        InitializeWithPath init_path;
        InitializeFull     init_full;
        Login              login;
        Shutdown           shutdown;
        GetVersion         get_version;
        GetLastError       get_last_error;
        GetTerminalInfo    get_terminal_info;
        GetAccountInfo     get_account_info;

        // Symbols
        GetTotalSymbols   get_total_symbols;
        GetSymbolsAll     get_symbols_all;
        GetSymbolInfo     get_symbol_info;
        SelectSymbol      select_symbol;
        GetSymbolsByGroup get_symbols_by_group;

        // Market Depth
        SubscribeBook   subscribe_book;
        UnsubscribeBook unsubscribe_book;
        GetBookInfo     get_book_info;

        // Rates & Ticks
        GetRatesByDate  get_rates_by_date;
        GetRatesByPos   get_rates_by_pos;
        GetRatesByRange get_rates_by_range;
        GetTicksByDate  get_ticks_by_date;
        GetTicksByRange get_ticks_by_range;
        GetTickInfo     get_tick_info;

        // Active Orders
        GetOrdersAll      get_orders_all;
        GetOrdersBySymbol get_orders_by_symbol;
        GetOrdersByGroup  get_orders_by_group;
        GetOrderByTicket  get_order_by_ticket;
        GetTotalOrders    get_total_orders;

        // Active Positions
        GetPositionsAll      get_positions_all;
        GetPositionsBySymbol get_positions_symbol;
        GetPositionsByGroup  get_positions_group;
        GetPositionByTicket  get_position_ticket;
        GetTotalPositions    get_total_positions;

        // Trading
        CheckOrder      check_order;
        SendOrder       send_order;
        CalculateMargin calc_margin;
        CalculateProfit calc_profit;

        // History Orders
        GetHistoryOrdersByRange get_hist_orders_range;
        GetHistoryOrderByTicket get_hist_order_ticket;
        GetHistoryOrdersByPosId get_hist_orders_pos;
        GetHistoryOrdersTotal   get_hist_orders_total;

        // History Deals
        GetHistoryDealsByRange  get_hist_deals_range;
        GetHistoryDealsByTicket get_hist_deals_ticket;
        GetHistoryDealsByPosId  get_hist_deals_pos;
        GetHistoryDealsTotal    get_hist_deals_total;
    };
    MetaTraderClient()          = default;
    virtual ~MetaTraderClient() = default;

    explicit MetaTraderClient(Handlers handlers) : h(std::move(handlers)) {}

    // System
    virtual bool initialize() { return h.init_auto ? h.init_auto() : false; }
    virtual bool initialize(str& path) { return h.init_path ? h.init_path(path) : false; }
    virtual bool initialize(
        str& path, uint64_t account, str& pw, str& srv, uint32_t timeout, bool portable
    ) {
        return h.init_full ? h.init_full(path, account, pw, srv, timeout, portable) : false;
    }
    virtual bool login(uint64_t account, str& pw, str& srv, uint32_t timeout) {
        return h.login ? h.login(account, pw, srv, timeout) : false;
    }

    virtual void shutdown() {
        if (h.shutdown)
            h.shutdown();
    }
    virtual std::optional<VersionInfo> version() {
        return h.get_version ? h.get_version() : std::nullopt;
    }
    virtual std::optional<LastErrorResult> last_error() {
        return h.get_last_error ? h.get_last_error() : std::make_tuple(-1, std::string("fail"));
    }
    virtual std::optional<TerminalInfo> terminal_info() {
        return h.get_terminal_info ? h.get_terminal_info() : std::nullopt;
    }
    virtual std::optional<AccountInfo> account_info() {
        return h.get_account_info ? h.get_account_info() : std::nullopt;
    }

    // Symbols
    virtual std::optional<int32_t> symbols_total() {
        return h.get_total_symbols ? h.get_total_symbols() : 0;
    }
    virtual SymbolsData symbols_get() {
        return h.get_symbols_all ? h.get_symbols_all() : std::nullopt;
    }
    virtual SymbolsData symbols_get(str& group) {
        return h.get_symbols_by_group ? h.get_symbols_by_group(group) : std::nullopt;
    }
    virtual std::optional<SymbolInfo> symbol_info(str& symbol) {
        return h.get_symbol_info ? h.get_symbol_info(symbol) : std::nullopt;
    }
    virtual bool symbol_select(str& symbol, bool enable) {
        return h.select_symbol ? h.select_symbol(symbol, enable) : false;
    }
    virtual std::optional<TickInfo> symbol_info_tick(str& symbol) {
        return h.get_tick_info ? h.get_tick_info(symbol) : std::nullopt;
    }

    // Market Depth
    virtual bool market_book_add(str& symbol) {
        return h.subscribe_book ? h.subscribe_book(symbol) : false;
    }
    virtual bool market_book_release(str& symbol) {
        return h.unsubscribe_book ? h.unsubscribe_book(symbol) : false;
    }
    virtual BookData market_book_get(str& symbol) {
        return h.get_book_info ? h.get_book_info(symbol) : std::nullopt;
    }

    // Market Data
    virtual RateInfoType copy_rates_from(str& s, int32_t t, DateTime from, int32_t count) {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        return copy_rates_from(s, t, from_ts, count);
    }
    virtual RateInfoType copy_rates_from(str& s, int32_t t, int64_t from, int32_t count) {
        return h.get_rates_by_date ? h.get_rates_by_date(s, t, from, count)
                                   : py::array_t<RateInfo>();
    }
    virtual RateInfoType copy_rates_from_pos(str& s, int32_t t, int32_t start, int32_t count) {
        return h.get_rates_by_pos ? h.get_rates_by_pos(s, t, start, count)
                                  : py::array_t<RateInfo>();
    }
    virtual RateInfoType copy_rates_range(str& s, int32_t t, DateTime from, DateTime to) {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return copy_rates_range(s, t, from_ts, to_ts);
    }
    virtual RateInfoType copy_rates_range(str& s, int32_t t, int64_t from, int64_t to) {
        return h.get_rates_by_range ? h.get_rates_by_range(s, t, from, to)
                                    : py::array_t<RateInfo>();
    }
    virtual TickInfoType copy_ticks_from(str& s, DateTime from, int32_t count, int32_t flags) {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        return copy_ticks_from(s, from_ts, count, flags);
    }
    virtual TickInfoType copy_ticks_from(str& s, int64_t from, int32_t count, int32_t flags) {
        return h.get_ticks_by_date ? h.get_ticks_by_date(s, from, count, flags)
                                   : py::array_t<TickInfo>();
    }
    virtual TickInfoType copy_ticks_range(str& s, DateTime from, DateTime to, int32_t flags) {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return copy_ticks_range(s, from_ts, to_ts, flags);
    }
    virtual TickInfoType copy_ticks_range(str& s, int64_t from, int64_t to, int32_t flags) {
        return h.get_ticks_by_range ? h.get_ticks_by_range(s, from, to, flags)
                                    : py::array_t<TickInfo>();
    }

    // Active Orders
    virtual OrdersData orders_get() { return h.get_orders_all ? h.get_orders_all() : std::nullopt; }
    virtual OrdersData orders_get(str& symbol) {
        return h.get_orders_by_symbol ? h.get_orders_by_symbol(symbol) : std::nullopt;
    }
    virtual OrdersData orders_get_by_group(str& group) {
        return h.get_orders_by_group ? h.get_orders_by_group(group) : std::nullopt;
    }
    virtual std::optional<TradeOrder> order_get_by_ticket(uint64_t ticket) {
        return h.get_order_by_ticket ? h.get_order_by_ticket(ticket) : std::nullopt;
    }
    virtual std::optional<int32_t> orders_total() {
        return h.get_total_orders ? h.get_total_orders() : 0;
    }

    // Active Positions
    virtual PositionsData positions_get() {
        return h.get_positions_all ? h.get_positions_all() : std::nullopt;
    }
    virtual PositionsData positions_get(str& symbol) {
        return h.get_positions_symbol ? h.get_positions_symbol(symbol) : std::nullopt;
    }
    virtual PositionsData positions_get_by_group(str& group) {
        return h.get_positions_group ? h.get_positions_group(group) : std::nullopt;
    }
    virtual std::optional<TradePosition> position_get_by_ticket(uint64_t ticket) {
        return h.get_position_ticket ? h.get_position_ticket(ticket) : std::nullopt;
    }
    virtual std::optional<int32_t> positions_total() {
        return h.get_total_positions ? h.get_total_positions() : 0;
    }

    // Trading
    virtual std::optional<OrderCheckResult> order_check(const py::dict& dict) {
        return order_check(dict.cast<TradeRequest>());
    }
    virtual std::optional<OrderCheckResult> order_check(const TradeRequest& req) {
        return h.check_order ? h.check_order(req) : OrderCheckResult{};
    }
    virtual std::optional<OrderSentResult> order_send(const py::dict& dict) {
        return order_send(dict.cast<TradeRequest>());
    }
    virtual std::optional<OrderSentResult> order_send(const TradeRequest& req) {
        return h.send_order ? h.send_order(req) : OrderSentResult{};
    }
    virtual std::optional<double> order_calc_margin(
        int32_t action, str& sym, double vol, double prc
    ) {
        return h.calc_margin ? h.calc_margin(action, sym, vol, prc) : std::nullopt;
    }
    virtual std::optional<double> order_calc_profit(
        int32_t action, str& sym, double vol, double open, double close
    ) {
        return h.calc_profit ? h.calc_profit(action, sym, vol, open, close) : std::nullopt;
    }

    // History Orders
    virtual OrdersData history_orders_get(int64_t from, int64_t to, str& group) {
        return h.get_hist_orders_range ? h.get_hist_orders_range(from, to, group) : std::nullopt;
    }
    virtual OrdersData history_orders_get(DateTime from, DateTime to, str& group) {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_orders_get(from_ts, to_ts, group);
    }
    virtual std::optional<TradeOrder> history_orders_get(uint64_t ticket) {
        return h.get_hist_order_ticket ? h.get_hist_order_ticket(ticket) : std::nullopt;
    }
    virtual OrdersData history_orders_get_by_pos(uint64_t pos_id) {
        return h.get_hist_orders_pos ? h.get_hist_orders_pos(pos_id) : std::nullopt;
    }
    virtual std::optional<int32_t> history_orders_total(int64_t from, int64_t to) {
        return h.get_hist_orders_total ? h.get_hist_orders_total(from, to) : 0;
    }
    virtual std::optional<int32_t> history_orders_total(DateTime from, DateTime to) {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_orders_total(from_ts, to_ts);
    }

    // History Deals
    virtual DealsData history_deals_get(int64_t from, int64_t to, str& group) {
        return h.get_hist_deals_range ? h.get_hist_deals_range(from, to, group) : std::nullopt;
    }
    virtual DealsData history_deals_get(DateTime from, DateTime to, str& group) {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_deals_get(from_ts, to_ts, group);
    }
    virtual DealsData history_deals_get(uint64_t ticket) {
        return h.get_hist_deals_ticket ? h.get_hist_deals_ticket(ticket) : std::nullopt;
    }
    virtual DealsData history_deals_get_by_pos(uint64_t pos_id) {
        return h.get_hist_deals_pos ? h.get_hist_deals_pos(pos_id) : std::nullopt;
    }
    virtual std::optional<int32_t> history_deals_total(int64_t from, int64_t to) {
        return h.get_hist_deals_total ? h.get_hist_deals_total(from, to) : 0;
    }
    virtual std::optional<int32_t> history_deals_total(DateTime from, DateTime to) {
        auto from_ts = static_cast<int64_t>(std::chrono::system_clock::to_time_t(from));
        auto to_ts   = static_cast<int64_t>(std::chrono::system_clock::to_time_t(to));
        return history_deals_total(from_ts, to_ts);
    }

   private:
    Handlers h;
};

}  // namespace MT5
