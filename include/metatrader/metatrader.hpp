#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "metatrader/objects.hpp"

namespace py = pybind11;
namespace MT5 {
using InitializeAuto     = std::function<bool()>;
using InitializeWithPath = std::function<bool(const std::string& path)>;
using InitializeFull     = std::function<bool(
    const std::string& path,
    uint64_t           login,
    const std::string& password,
    const std::string& server,
    uint32_t           timeout,
    bool               portable
)>;

using Login = std::function<
    bool(uint64_t login, const std::string& password, const std::string& server, uint32_t timeout)>;

using Shutdown        = std::function<void()>;
using VersionInfo     = std::tuple<int32_t, int32_t, std::string>;
using GetVersion      = std::function<std::optional<VersionInfo>()>;
using LastErrorResult = std::tuple<int32_t, std::string>;
using GetLastError    = std::function<LastErrorResult()>;

using GetTerminalInfo = std::function<std::optional<TerminalInfo>()>;
using GetAccountInfo  = std::function<std::optional<AccountInfo>()>;

using GetTotalSymbols = std::function<int32_t()>;
using GetSymbolsAll   = std::function<std::optional<std::vector<SymbolInfo>>()>;
using GetSymbolInfo   = std::function<std::optional<SymbolInfo>(const std::string& symbol)>;
using SelectSymbol    = std::function<bool(const std::string& symbol, bool enable)>;
using GetSymbolsByGroup =
    std::function<std::optional<std::vector<SymbolInfo>>(const std::string& group)>;

using SubscribeBook   = std::function<bool(const std::string& symbol)>;
using UnsubscribeBook = std::function<bool(const std::string& symbol)>;
using GetBookInfo = std::function<std::optional<std::vector<BookInfo>>(const std::string& symbol)>;

using GetRatesByDate =
    std::function<py::array_t<RateInfo>(const std::string&, int32_t, int64_t, int32_t)>;
using GetRatesByPos =
    std::function<py::array_t<RateInfo>(const std::string&, int32_t, int32_t, int32_t)>;
using GetRatesByRange =
    std::function<py::array_t<RateInfo>(const std::string&, int32_t, int64_t, int64_t)>;

using GetTicksByDate =
    std::function<py::array_t<TickInfo>(const std::string&, int64_t, int32_t, uint32_t)>;
using GetTicksByRange =
    std::function<py::array_t<TickInfo>(const std::string&, int64_t, int64_t, uint32_t)>;

using GetTickInfo = std::function<std::optional<TickInfo>(const std::string& symbol)>;

using GetOrdersAll = std::function<std::optional<std::vector<TradeOrder>>()>;
using GetOrdersBySymbol =
    std::function<std::optional<std::vector<TradeOrder>>(const std::string& symbol)>;
using GetOrdersByGroup =
    std::function<std::optional<std::vector<TradeOrder>>(const std::string& group)>;
using GetOrderByTicket = std::function<std::optional<TradeOrder>(uint64_t ticket)>;
using GetTotalOrders   = std::function<int32_t()>;

using GetPositionsAll = std::function<std::optional<std::vector<TradePosition>>()>;
using GetPositionsBySymbol =
    std::function<std::optional<std::vector<TradePosition>>(const std::string& symbol)>;
using GetPositionsByGroup =
    std::function<std::optional<std::vector<TradePosition>>(const std::string& group)>;
using GetPositionByTicket = std::function<std::optional<TradePosition>(uint64_t ticket)>;
using GetTotalPositions   = std::function<int32_t()>;

using CheckOrder = std::function<OrderCheckResult(const TradeRequest& request)>;
using SendOrder  = std::function<OrderSentResult(const TradeRequest& request)>;

using CalculateMargin = std::function<
    std::optional<double>(int32_t action, const std::string& symbol, double volume, double price)>;
using CalculateProfit = std::function<std::optional<double>(
    int32_t action, const std::string& symbol, double volume, double open, double close
)>;

using GetHistoryOrdersByRange = std::function<
    std::optional<std::vector<TradeOrder>>(int64_t from, int64_t to, const std::string& group)>;
using GetHistoryOrderByTicket = std::function<std::optional<TradeOrder>(uint64_t ticket)>;
using GetHistoryOrdersByPosId =
    std::function<std::optional<std::vector<TradeOrder>>(uint64_t position_id)>;
using GetHistoryOrdersTotal = std::function<int32_t(int64_t from, int64_t to)>;

using GetHistoryDealsByRange = std::function<
    std::optional<std::vector<TradeDeal>>(int64_t from, int64_t to, const std::string& group)>;
using GetHistoryDealsByTicket =
    std::function<std::optional<std::vector<TradeDeal>>(uint64_t order_ticket)>;
using GetHistoryDealsByPosId =
    std::function<std::optional<std::vector<TradeDeal>>(uint64_t position_id)>;
using GetHistoryDealsTotal = std::function<int32_t(int64_t from, int64_t to)>;

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

    // --- System ---
    virtual bool initialize() { return h.init_auto ? h.init_auto() : false; }
    virtual bool initialize(const std::string& path) {
        return h.init_path ? h.init_path(path) : false;
    }
    virtual bool initialize(
        const std::string& path,
        uint64_t           account,
        const std::string& pw,
        const std::string& srv,
        uint32_t           timeout,
        bool               portable
    ) {
        return h.init_full ? h.init_full(path, account, pw, srv, timeout, portable) : false;
    }
    virtual bool login(
        uint64_t account, const std::string& pw, const std::string& srv, uint32_t timeout
    ) {
        return h.login ? h.login(account, pw, srv, timeout) : false;
    }

    virtual void shutdown() {
        if (h.shutdown)
            h.shutdown();
    }
    virtual std::optional<VersionInfo> version() {
        return h.get_version ? h.get_version() : std::nullopt;
    }
    virtual LastErrorResult last_error() {
        return h.get_last_error ? h.get_last_error() : std::make_tuple(-1, std::string("fail"));
    }
    virtual std::optional<TerminalInfo> terminal_info() {
        return h.get_terminal_info ? h.get_terminal_info() : std::nullopt;
    }
    virtual std::optional<AccountInfo> account_info() {
        return h.get_account_info ? h.get_account_info() : std::nullopt;
    }

    // --- Symbols ---
    virtual int32_t symbols_total() { return h.get_total_symbols ? h.get_total_symbols() : 0; }
    virtual std::optional<std::vector<SymbolInfo>> symbols_get() {
        return h.get_symbols_all ? h.get_symbols_all() : std::nullopt;
    }
    virtual std::optional<std::vector<SymbolInfo>> symbols_get(const std::string& group) {
        return h.get_symbols_by_group ? h.get_symbols_by_group(group) : std::nullopt;
    }
    virtual std::optional<SymbolInfo> symbol_info(const std::string& symbol) {
        return h.get_symbol_info ? h.get_symbol_info(symbol) : std::nullopt;
    }
    virtual bool symbol_select(const std::string& symbol, bool enable) {
        return h.select_symbol ? h.select_symbol(symbol, enable) : false;
    }

    // --- Market Depth ---
    virtual bool market_book_add(const std::string& symbol) {
        return h.subscribe_book ? h.subscribe_book(symbol) : false;
    }
    virtual bool market_book_release(const std::string& symbol) {
        return h.unsubscribe_book ? h.unsubscribe_book(symbol) : false;
    }
    virtual std::optional<std::vector<BookInfo>> market_book_get(const std::string& symbol) {
        return h.get_book_info ? h.get_book_info(symbol) : std::nullopt;
    }

    // --- Market Data ---
    virtual py::array_t<RateInfo> copy_rates_from(
        const std::string& s, int32_t t, int64_t from, int32_t count
    ) {
        return h.get_rates_by_date ? h.get_rates_by_date(s, t, from, count)
                                   : py::array_t<RateInfo>();
    }
    virtual py::array_t<RateInfo> copy_rates_from_pos(
        const std::string& s, int32_t t, int32_t start, int32_t count
    ) {
        return h.get_rates_by_pos ? h.get_rates_by_pos(s, t, start, count)
                                  : py::array_t<RateInfo>();
    }
    virtual py::array_t<RateInfo> copy_rates_range(
        const std::string& s, int32_t t, int64_t from, int64_t to
    ) {
        return h.get_rates_by_range ? h.get_rates_by_range(s, t, from, to)
                                    : py::array_t<RateInfo>();
    }
    virtual py::array_t<TickInfo> copy_ticks_from(
        const std::string& s, int64_t from, int32_t count, uint32_t flags
    ) {
        return h.get_ticks_by_date ? h.get_ticks_by_date(s, from, count, flags)
                                   : py::array_t<TickInfo>();
    }
    virtual py::array_t<TickInfo> copy_ticks_range(
        const std::string& s, int64_t from, int64_t to, uint32_t flags
    ) {
        return h.get_ticks_by_range ? h.get_ticks_by_range(s, from, to, flags)
                                    : py::array_t<TickInfo>();
    }
    virtual std::optional<TickInfo> symbol_info_tick(const std::string& symbol) {
        return h.get_tick_info ? h.get_tick_info(symbol) : std::nullopt;
    }

    // --- Active Orders ---
    virtual std::optional<std::vector<TradeOrder>> orders_get() {
        return h.get_orders_all ? h.get_orders_all() : std::nullopt;
    }
    virtual std::optional<std::vector<TradeOrder>> orders_get(const std::string& symbol) {
        return h.get_orders_by_symbol ? h.get_orders_by_symbol(symbol) : std::nullopt;
    }
    virtual std::optional<std::vector<TradeOrder>> orders_get_by_group(const std::string& group) {
        return h.get_orders_by_group ? h.get_orders_by_group(group) : std::nullopt;
    }
    virtual std::optional<TradeOrder> order_get_by_ticket(uint64_t ticket) {
        return h.get_order_by_ticket ? h.get_order_by_ticket(ticket) : std::nullopt;
    }
    virtual int32_t orders_total() { return h.get_total_orders ? h.get_total_orders() : 0; }

    // --- Active Positions ---
    virtual std::optional<std::vector<TradePosition>> positions_get() {
        return h.get_positions_all ? h.get_positions_all() : std::nullopt;
    }
    virtual std::optional<std::vector<TradePosition>> positions_get(const std::string& symbol) {
        return h.get_positions_symbol ? h.get_positions_symbol(symbol) : std::nullopt;
    }
    virtual std::optional<std::vector<TradePosition>> positions_get_by_group(
        const std::string& group
    ) {
        return h.get_positions_group ? h.get_positions_group(group) : std::nullopt;
    }
    virtual std::optional<TradePosition> position_get_by_ticket(uint64_t ticket) {
        return h.get_position_ticket ? h.get_position_ticket(ticket) : std::nullopt;
    }
    virtual int32_t positions_total() {
        return h.get_total_positions ? h.get_total_positions() : 0;
    }

    // --- Trading ---
    virtual OrderCheckResult order_check(py::object req) {
        return h.check_order ? h.check_order(req.cast<const TradeRequest&>())
                             : OrderCheckResult{};
    }
    virtual OrderSentResult order_send(py::object req) {
        return h.send_order ? h.send_order(req.cast<const TradeRequest&>())
                            : OrderSentResult{};
    }
    virtual std::optional<double> order_calc_margin(
        int32_t act, const std::string& sym, double vol, double prc
    ) {
        return h.calc_margin ? h.calc_margin(act, sym, vol, prc) : std::nullopt;
    }
    virtual std::optional<double> order_calc_profit(
        int32_t act, const std::string& sym, double vol, double open, double close
    ) {
        return h.calc_profit ? h.calc_profit(act, sym, vol, open, close) : std::nullopt;
    }

    // --- History Orders ---
    virtual std::optional<std::vector<TradeOrder>> history_orders_get(
        int64_t from, int64_t to, const std::string& group
    ) {
        return h.get_hist_orders_range ? h.get_hist_orders_range(from, to, group) : std::nullopt;
    }
    virtual std::optional<TradeOrder> history_orders_get(uint64_t ticket) {
        return h.get_hist_order_ticket ? h.get_hist_order_ticket(ticket) : std::nullopt;
    }
    virtual std::optional<std::vector<TradeOrder>> history_orders_get_by_pos(uint64_t pos_id) {
        return h.get_hist_orders_pos ? h.get_hist_orders_pos(pos_id) : std::nullopt;
    }
    virtual int32_t history_orders_total(int64_t from, int64_t to) {
        return h.get_hist_orders_total ? h.get_hist_orders_total(from, to) : 0;
    }

    // --- History Deals ---
    virtual std::optional<std::vector<TradeDeal>> history_deals_get(
        int64_t from, int64_t to, const std::string& group
    ) {
        return h.get_hist_deals_range ? h.get_hist_deals_range(from, to, group) : std::nullopt;
    }
    virtual std::optional<std::vector<TradeDeal>> history_deals_get(uint64_t ticket) {
        return h.get_hist_deals_ticket ? h.get_hist_deals_ticket(ticket) : std::nullopt;
    }
    virtual std::optional<std::vector<TradeDeal>> history_deals_get_by_pos(uint64_t pos_id) {
        return h.get_hist_deals_pos ? h.get_hist_deals_pos(pos_id) : std::nullopt;
    }
    virtual int32_t history_deals_total(int64_t from, int64_t to) {
        return h.get_hist_deals_total ? h.get_hist_deals_total(from, to) : 0;
    }

   private:
    Handlers h;
};

}  // namespace MT5

// namespace MT5
