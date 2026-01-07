#include "metatrader/metatrader.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace MT5;

// ===================================================================================
// Conversion Helpers to NamedTuple
// ===================================================================================

namespace {  // Anonymous namespace for helpers

// Forward declaration for nested types
py::object to_namedtuple(const TradeRequest& req);

py::object to_namedtuple(const TerminalInfo& info) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "TerminalInfo",
        std::vector<std::string>{
            "community_account", "community_connection", "connected",       "dlls_allowed",
            "trade_allowed",     "tradeapi_disabled",    "email_enabled",   "ftp_enabled",
            "notifications_enabled", "mqid",             "build",           "maxbars",
            "codepage",          "ping_last",            "community_balance", "retransmission",
            "company",           "name",                 "language",        "path",
            "data_path",         "commondata_path"}
    );
    return nt(
        info.community_account, info.community_connection, info.connected, info.dlls_allowed,
        info.trade_allowed, info.tradeapi_disabled, info.email_enabled, info.ftp_enabled,
        info.notifications_enabled, info.mqid, info.build, info.maxbars, info.codepage,
        info.ping_last, info.community_balance, info.retransmission, info.company, info.name,
        info.language, info.path, info.data_path, info.commondata_path
    );
}

py::object to_namedtuple(const AccountInfo& info) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "AccountInfo",
        std::vector<std::string>{
            "login",          "trade_mode",     "leverage",       "limit_orders", "margin_so_mode",
            "trade_allowed",  "trade_expert",   "margin_mode",    "currency_digits", "fifo_close",
            "balance",        "credit",         "profit",         "equity",       "margin",
            "margin_free",    "margin_level",   "margin_so_call", "margin_so_so", "margin_initial",
            "margin_maintenance", "assets",     "liabilities",    "commission_blocked", "name",
            "server",         "currency",       "company"}
    );
    return nt(
        info.login, info.trade_mode, info.leverage, info.limit_orders, info.margin_so_mode,
        info.trade_allowed, info.trade_expert, info.margin_mode, info.currency_digits,
        info.fifo_close, info.balance, info.credit, info.profit, info.equity, info.margin,
        info.margin_free, info.margin_level, info.margin_so_call, info.margin_so_so,
        info.margin_initial, info.margin_maintenance, info.assets, info.liabilities,
        info.commission_blocked, info.name, info.server, info.currency, info.company
    );
}

py::object to_namedtuple(const SymbolInfo& info) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "SymbolInfo",
        std::vector<std::string>{
            "custom", "chart_mode", "select", "visible", "session_deals",
            "session_buy_orders", "session_sell_orders", "volume", "volumehigh", "volumelow",
            "time", "digits", "spread", "spread_float", "ticks_bookdepth", "trade_calc_mode",
            "trade_mode", "start_time", "expiration_time", "trade_stops_level",
            "trade_freeze_level", "trade_exemode", "swap_mode", "swap_rollover3days",
            "margin_hedged_use_leg", "expiration_mode", "filling_mode", "order_mode",
            "order_gtc_mode", "option_mode", "option_right", "bid", "bidhigh", "bidlow", "ask",
            "askhigh", "asklow", "last", "lasthigh", "lastlow", "volume_real",
            "volumehigh_real", "volumelow_real", "option_strike", "point", "trade_tick_value",
            "trade_tick_value_profit", "trade_tick_value_loss", "trade_tick_size",
            "trade_contract_size", "trade_accrued_interest", "trade_face_value",
            "trade_liquidity_rate", "volume_min", "volume_max", "volume_step", "volume_limit",
            "swap_long", "swap_short", "margin_initial", "margin_maintenance", "session_volume",
            "session_turnover", "session_interest", "session_buy_orders_volume",
            "session_sell_orders_volume", "session_open", "session_close", "session_aw",
            "session_price_settlement", "session_price_limit_min", "session_price_limit_max",
            "margin_hedged", "price_change", "price_volatility", "price_theoretical",
            "price_greeks_delta", "price_greeks_theta", "price_greeks_gamma",
            "price_greeks_vega", "price_greeks_rho", "price_greeks_omega", "price_sensitivity",
            "basis", "category", "currency_base", "currency_profit", "currency_margin", "bank",
            "description", "exchange", "formula", "isin", "name", "page", "path"}
    );
    return nt(
        info.custom, info.chart_mode, info.select, info.visible, info.session_deals,
        info.session_buy_orders, info.session_sell_orders, info.volume, info.volumehigh,
        info.volumelow, info.time, info.digits, info.spread, info.spread_float,
        info.ticks_bookdepth, info.trade_calc_mode, info.trade_mode, info.start_time,
        info.expiration_time, info.trade_stops_level, info.trade_freeze_level,
        info.trade_exemode, info.swap_mode, info.swap_rollover3days,
        info.margin_hedged_use_leg, info.expiration_mode, info.filling_mode, info.order_mode,
        info.order_gtc_mode, info.option_mode, info.option_right, info.bid, info.bidhigh,
        info.bidlow, info.ask, info.askhigh, info.asklow, info.last, info.lasthigh,
        info.lastlow, info.volume_real, info.volumehigh_real, info.volumelow_real,
        info.option_strike, info.point, info.trade_tick_value, info.trade_tick_value_profit,
        info.trade_tick_value_loss, info.trade_tick_size, info.trade_contract_size,
        info.trade_accrued_interest, info.trade_face_value, info.trade_liquidity_rate,
        info.volume_min, info.volume_max, info.volume_step, info.volume_limit, info.swap_long,
        info.swap_short, info.margin_initial, info.margin_maintenance, info.session_volume,
        info.session_turnover, info.session_interest, info.session_buy_orders_volume,
        info.session_sell_orders_volume, info.session_open, info.session_close, info.session_aw,
        info.session_price_settlement, info.session_price_limit_min,
        info.session_price_limit_max, info.margin_hedged, info.price_change,
        info.price_volatility, info.price_theoretical, info.price_greeks_delta,
        info.price_greeks_theta, info.price_greeks_gamma, info.price_greeks_vega,
        info.price_greeks_rho, info.price_greeks_omega, info.price_sensitivity, info.basis,
        info.category, info.currency_base, info.currency_profit, info.currency_margin,
        info.bank, info.description, info.exchange, info.formula, info.isin, info.name,
        info.page, info.path
    );
}

py::object to_namedtuple(const TickInfo& info) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "TickInfo",
        std::vector<std::string>{
            "time", "bid", "ask", "last", "volume", "time_msc", "flags", "volume_real"}
    );
    return nt(
        info.time, info.bid, info.ask, info.last, info.volume, info.time_msc, info.flags,
        info.volume_real
    );
}

py::object to_namedtuple(const RateInfo& info) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "RateInfo",
        std::vector<std::string>{
            "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"}
    );
    return nt(
        info.time, info.open, info.high, info.low, info.close, info.tick_volume, info.spread,
        info.real_volume
    );
}

py::object to_namedtuple(const BookInfo& info) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "BookInfo", std::vector<std::string>{"type", "price", "volume", "volume_real"}
    );
    return nt(info.type, info.price, info.volume, info.volume_real);
}

py::object to_namedtuple(const TradeRequest& req) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "TradeRequest",
        std::vector<std::string>{
            "action", "magic", "order", "symbol", "volume", "price", "stoplimit", "sl", "tp",
            "deviation", "type", "type_filling", "type_time", "expiration", "comment",
            "position", "position_by"}
    );
    return nt(
        req.action, req.magic, req.order, req.symbol, req.volume, req.price, req.stoplimit,
        req.sl, req.tp, req.deviation, req.type, req.type_filling, req.type_time,
        req.expiration, req.comment, req.position, req.position_by
    );
}

py::object to_namedtuple(const OrderCheckResult& res) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "OrderCheckResult",
        std::vector<std::string>{
            "retcode", "balance", "equity", "profit", "margin", "margin_free", "margin_level",
            "comment", "request"}
    );
    return nt(
        res.retcode, res.balance, res.equity, res.profit, res.margin, res.margin_free,
        res.margin_level, res.comment, to_namedtuple(res.request)
    );
}

py::object to_namedtuple(const OrderSentResult& res) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "OrderSentResult",
        std::vector<std::string>{
            "retcode", "deal", "order", "volume", "price", "bid", "ask", "comment",
            "request_id", "retcode_external", "request"}
    );
    return nt(
        res.retcode, res.deal, res.order, res.volume, res.price, res.bid, res.ask, res.comment,
        res.request_id, res.retcode_external, to_namedtuple(res.request)
    );
}

py::object to_namedtuple(const TradeOrder& order) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "TradeOrder",
        std::vector<std::string>{
            "ticket", "time_setup", "time_setup_msc", "time_done", "time_done_msc",
            "time_expiration", "type", "type_time", "type_filling", "state", "magic",
            "position_id", "position_by_id", "reason", "volume_initial", "volume_current",
            "price_open", "sl", "tp", "price_current", "price_stoplimit", "symbol", "comment",
            "external_id"}
    );
    return nt(
        order.ticket, order.time_setup, order.time_setup_msc, order.time_done,
        order.time_done_msc, order.time_expiration, order.type, order.type_time,
        order.type_filling, order.state, order.magic, order.position_id, order.position_by_id,
        order.reason, order.volume_initial, order.volume_current, order.price_open, order.sl,
        order.tp, order.price_current, order.price_stoplimit, order.symbol, order.comment,
        order.external_id
    );
}

py::object to_namedtuple(const TradePosition& pos) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "TradePosition",
        std::vector<std::string>{
            "ticket", "time", "time_msc", "time_update", "time_update_msc", "type", "magic",
            "identifier", "reason", "volume", "price_open", "sl", "tp", "price_current",
            "swap", "profit", "symbol", "comment", "external_id"}
    );
    return nt(
        pos.ticket, pos.time, pos.time_msc, pos.time_update, pos.time_update_msc, pos.type,
        pos.magic, pos.identifier, pos.reason, pos.volume, pos.price_open, pos.sl, pos.tp,
        pos.price_current, pos.swap, pos.profit, pos.symbol, pos.comment, pos.external_id
    );
}

py::object to_namedtuple(const TradeDeal& deal) {
    static auto nt = py::module_::import("collections").attr("namedtuple")(
        "TradeDeal",
        std::vector<std::string>{
            "ticket", "order", "time", "time_msc", "type", "entry", "magic", "position_id",
            "reason", "volume", "price", "commission", "swap", "profit", "fee", "symbol",
            "comment", "external_id"}
    );
    return nt(
        deal.ticket, deal.order, deal.time, deal.time_msc, deal.type, deal.entry, deal.magic,
        deal.position_id, deal.reason, deal.volume, deal.price, deal.commission, deal.swap,
        deal.profit, deal.fee, deal.symbol, deal.comment, deal.external_id
    );
}

// Generic list converter
template <typename T> py::object to_list_of_namedtuples(const std::optional<std::vector<T>>& vec_opt) {
    if (!vec_opt) {
        return py::none();
    }
    py::list list;
    for (const auto& item : *vec_opt) {
        list.append(to_namedtuple(item));
    }
    return list;
}

}  // anonymous namespace

// ===================================================================================
// Trampoline class for virtual functions
// ===================================================================================

class PyMetaTraderClient : public MetaTraderClient {
public:
    using MetaTraderClient::MetaTraderClient;

    // --- System ---
    bool initialize() override { PYBIND11_OVERRIDE(bool, MetaTraderClient, initialize); }
    bool initialize(const std::string& path) override { PYBIND11_OVERRIDE(bool, MetaTraderClient, initialize, path); }
    bool initialize(const std::string& path, uint64_t login, const std::string& pw, const std::string& srv, uint32_t to, bool port) override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, initialize, path, login, pw, srv, to, port);
    }
    bool login(uint64_t login, const std::string& pw, const std::string& srv, uint32_t timeout) override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, login, login, pw, srv, timeout);
    }
    void shutdown() override { PYBIND11_OVERRIDE(void, MetaTraderClient, shutdown); }
    std::optional<std::string> version() override { PYBIND11_OVERRIDE(std::optional<std::string>, MetaTraderClient, version); }
    LastErrorResult last_error() override { PYBIND11_OVERRIDE(LastErrorResult, MetaTraderClient, last_error); }
    std::optional<TerminalInfo> terminal_info() override { PYBIND11_OVERRIDE(std::optional<TerminalInfo>, MetaTraderClient, terminal_info); }
    std::optional<AccountInfo> account_info() override { PYBIND11_OVERRIDE(std::optional<AccountInfo>, MetaTraderClient, account_info); }

    // --- Symbols ---
    int32_t symbols_total() override { PYBIND11_OVERRIDE(int32_t, MetaTraderClient, symbols_total); }
    std::optional<std::vector<SymbolInfo>> symbols_get() override { PYBIND11_OVERRIDE(std::optional<std::vector<SymbolInfo>>, MetaTraderClient, symbols_get); }
    std::optional<std::vector<SymbolInfo>> symbols_get(const std::string& group) override { PYBIND11_OVERRIDE(std::optional<std::vector<SymbolInfo>>, MetaTraderClient, symbols_get, group); }
    std::optional<SymbolInfo> symbol_info(const std::string& symbol) override { PYBIND11_OVERRIDE(std::optional<SymbolInfo>, MetaTraderClient, symbol_info, symbol); }
    bool symbol_select(const std::string& symbol, bool enable) override { PYBIND11_OVERRIDE(bool, MetaTraderClient, symbol_select, symbol, enable); }

    // --- Market Depth ---
    bool market_book_add(const std::string& symbol) override { PYBIND11_OVERRIDE(bool, MetaTraderClient, market_book_add, symbol); }
    bool market_book_release(const std::string& symbol) override { PYBIND11_OVERRIDE(bool, MetaTraderClient, market_book_release, symbol); }
    std::optional<std::vector<BookInfo>> market_book_get(const std::string& symbol) override { PYBIND11_OVERRIDE(std::optional<std::vector<BookInfo>>, MetaTraderClient, market_book_get, symbol); }

    // --- Market Data ---
    std::optional<std::vector<RateInfo>> copy_rates_from(const std::string& s, int32_t t, int64_t from, int32_t count) override {
        PYBIND11_OVERRIDE(std::optional<std::vector<RateInfo>>, MetaTraderClient, copy_rates_from, s, t, from, count);
    }
    std::optional<std::vector<RateInfo>> copy_rates_from_pos(const std::string& s, int32_t t, int32_t start, int32_t count) override {
        PYBIND11_OVERRIDE(std::optional<std::vector<RateInfo>>, MetaTraderClient, copy_rates_from_pos, s, t, start, count);
    }
    std::optional<std::vector<RateInfo>> copy_rates_range(const std::string& s, int32_t t, int64_t from, int64_t to) override {
        PYBIND11_OVERRIDE(std::optional<std::vector<RateInfo>>, MetaTraderClient, copy_rates_range, s, t, from, to);
    }
    std::optional<std::vector<TickInfo>> copy_ticks_from(const std::string& s, int64_t from, int32_t count, uint32_t flags) override {
        PYBIND11_OVERRIDE(std::optional<std::vector<TickInfo>>, MetaTraderClient, copy_ticks_from, s, from, count, flags);
    }
    std::optional<std::vector<TickInfo>> copy_ticks_range(const std::string& s, int64_t from, int64_t to, uint32_t flags) override {
        PYBIND11_OVERRIDE(std::optional<std::vector<TickInfo>>, MetaTraderClient, copy_ticks_range, s, from, to, flags);
    }
    std::optional<TickInfo> symbol_info_tick(const std::string& symbol) override { PYBIND11_OVERRIDE(std::optional<TickInfo>, MetaTraderClient, symbol_info_tick, symbol); }

    // --- Trading ---
    OrderCheckResult order_check(const TradeRequest& req) override { PYBIND11_OVERRIDE(OrderCheckResult, MetaTraderClient, order_check, req); }
    OrderSentResult order_send(const TradeRequest& req) override { PYBIND11_OVERRIDE(OrderSentResult, MetaTraderClient, order_send, req); }
    std::optional<double> order_calc_margin(int32_t act, const std::string& sym, double vol, double prc) override {
        PYBIND11_OVERRIDE(std::optional<double>, MetaTraderClient, order_calc_margin, act, sym, vol, prc);
    }
    std::optional<double> order_calc_profit(int32_t act, const std::string& sym, double vol, double open, double close) override {
        PYBIND11_OVERRIDE(std::optional<double>, MetaTraderClient, order_calc_profit, act, sym, vol, open, close);
    }

    // --- Active Orders & Positions ---
    std::optional<std::vector<TradeOrder>> orders_get() override { PYBIND11_OVERRIDE(std::optional<std::vector<TradeOrder>>, MetaTraderClient, orders_get); }
    std::optional<std::vector<TradeOrder>> orders_get(const std::string& symbol) override { PYBIND11_OVERRIDE(std::optional<std::vector<TradeOrder>>, MetaTraderClient, orders_get, symbol); }
    std::optional<std::vector<TradeOrder>> orders_get_by_group(const std::string& group) override { PYBIND11_OVERRIDE(std::optional<std::vector<TradeOrder>>, MetaTraderClient, orders_get_by_group, group); }
    std::optional<TradeOrder> order_get_by_ticket(uint64_t ticket) override { PYBIND11_OVERRIDE(std::optional<TradeOrder>, MetaTraderClient, order_get_by_ticket, ticket); }
    int32_t orders_total() override { PYBIND11_OVERRIDE(int32_t, MetaTraderClient, orders_total); }

    std::optional<std::vector<TradePosition>> positions_get() override { PYBIND11_OVERRIDE(std::optional<std::vector<TradePosition>>, MetaTraderClient, positions_get); }
    std::optional<std::vector<TradePosition>> positions_get(const std::string& symbol) override { PYBIND11_OVERRIDE(std::optional<std::vector<TradePosition>>, MetaTraderClient, positions_get, symbol); }
    std::optional<std::vector<TradePosition>> positions_get_by_group(const std::string& group) override { PYBIND11_OVERRIDE(std::optional<std::vector<TradePosition>>, MetaTraderClient, positions_get_by_group, group); }
    std::optional<TradePosition> position_get_by_ticket(uint64_t ticket) override { PYBIND11_OVERRIDE(std::optional<TradePosition>, MetaTraderClient, position_get_by_ticket, ticket); }
    int32_t positions_total() override { PYBIND11_OVERRIDE(int32_t, MetaTraderClient, positions_total); }

    // --- History ---
    std::optional<std::vector<TradeOrder>> history_orders_get(int64_t from, int64_t to, const std::string& group) override { PYBIND11_OVERRIDE(std::optional<std::vector<TradeOrder>>, MetaTraderClient, history_orders_get, from, to, group); }
    std::optional<TradeOrder> history_orders_get(uint64_t ticket) override { PYBIND11_OVERRIDE(std::optional<TradeOrder>, MetaTraderClient, history_orders_get, ticket); }
    std::optional<std::vector<TradeOrder>> history_orders_get_by_pos(uint64_t pos_id) override { PYBIND11_OVERRIDE(std::optional<std::vector<TradeOrder>>, MetaTraderClient, history_orders_get_by_pos, pos_id); }
    int32_t history_orders_total(int64_t from, int64_t to) override { PYBIND11_OVERRIDE(int32_t, MetaTraderClient, history_orders_total, from, to); }

    std::optional<std::vector<TradeDeal>> history_deals_get(int64_t from, int64_t to, const std::string& group) override { PYBIND11_OVERRIDE(std::optional<std::vector<TradeDeal>>, MetaTraderClient, history_deals_get, from, to, group); }
    std::optional<std::vector<TradeDeal>> history_deals_get(uint64_t ticket) override { PYBIND11_OVERRIDE(std::optional<std::vector<TradeDeal>>, MetaTraderClient, history_deals_get, ticket); }
    std::optional<std::vector<TradeDeal>> history_deals_get_by_pos(uint64_t pos_id) override { PYBIND11_OVERRIDE(std::optional<std::vector<TradeDeal>>, MetaTraderClient, history_deals_get_by_pos, pos_id); }
    int32_t history_deals_total(int64_t from, int64_t to) override { PYBIND11_OVERRIDE(int32_t, MetaTraderClient, history_deals_total, from, to); }
};

// ===================================================================================
// PYBIND11_MODULE
// ===================================================================================

PYBIND11_MODULE(metatrader_client, m) {
    m.doc() = "High-performance MetaTrader 5 C++/Python Bridge";

    py::class_<MetaTraderClient::Handlers>(m, "Handlers")
        .def(py::init<>())
        // System & Session
        .def_readwrite("init_auto", &MetaTraderClient::Handlers::init_auto)
        .def_readwrite("init_path", &MetaTraderClient::Handlers::init_path)
        .def_readwrite("init_full", &MetaTraderClient::Handlers::init_full)
        .def_readwrite("login", &MetaTraderClient::Handlers::login)
        .def_readwrite("shutdown", &MetaTraderClient::Handlers::shutdown)
        .def_readwrite("get_version", &MetaTraderClient::Handlers::get_version)
        .def_readwrite("get_last_error", &MetaTraderClient::Handlers::get_last_error)
        .def_readwrite("get_terminal_info", &MetaTraderClient::Handlers::get_terminal_info)
        .def_readwrite("get_account_info", &MetaTraderClient::Handlers::get_account_info)
        // Symbols & Market Depth
        .def_readwrite("get_total_symbols", &MetaTraderClient::Handlers::get_total_symbols)
        .def_readwrite("get_symbols_all", &MetaTraderClient::Handlers::get_symbols_all)
        .def_readwrite("get_symbol_info", &MetaTraderClient::Handlers::get_symbol_info)
        .def_readwrite("select_symbol", &MetaTraderClient::Handlers::select_symbol)
        .def_readwrite("get_symbols_by_group", &MetaTraderClient::Handlers::get_symbols_by_group)
        .def_readwrite("subscribe_book", &MetaTraderClient::Handlers::subscribe_book)
        .def_readwrite("unsubscribe_book", &MetaTraderClient::Handlers::unsubscribe_book)
        .def_readwrite("get_book_info", &MetaTraderClient::Handlers::get_book_info)
        // Market Data (Rates & Ticks)
        .def_readwrite("get_rates_by_date", &MetaTraderClient::Handlers::get_rates_by_date)
        .def_readwrite("get_rates_by_pos", &MetaTraderClient::Handlers::get_rates_by_pos)
        .def_readwrite("get_rates_by_range", &MetaTraderClient::Handlers::get_rates_by_range)
        .def_readwrite("get_ticks_by_date", &MetaTraderClient::Handlers::get_ticks_by_date)
        .def_readwrite("get_ticks_by_range", &MetaTraderClient::Handlers::get_ticks_by_range)
        .def_readwrite("get_tick_info", &MetaTraderClient::Handlers::get_tick_info)
        // Orders & Positions (Active)
        .def_readwrite("get_orders_all", &MetaTraderClient::Handlers::get_orders_all)
        .def_readwrite("get_orders_by_symbol", &MetaTraderClient::Handlers::get_orders_by_symbol)
        .def_readwrite("get_orders_by_group", &MetaTraderClient::Handlers::get_orders_by_group)
        .def_readwrite("get_order_by_ticket", &MetaTraderClient::Handlers::get_order_by_ticket)
        .def_readwrite("get_total_orders", &MetaTraderClient::Handlers::get_total_orders)
        .def_readwrite("get_positions_all", &MetaTraderClient::Handlers::get_positions_all)
        .def_readwrite("get_positions_symbol", &MetaTraderClient::Handlers::get_positions_symbol)
        .def_readwrite("get_positions_group", &MetaTraderClient::Handlers::get_positions_group)
        .def_readwrite("get_position_ticket", &MetaTraderClient::Handlers::get_position_ticket)
        .def_readwrite("get_total_positions", &MetaTraderClient::Handlers::get_total_positions)
        // Trading Operations
        .def_readwrite("check_order", &MetaTraderClient::Handlers::check_order)
        .def_readwrite("send_order", &MetaTraderClient::Handlers::send_order)
        .def_readwrite("calc_margin", &MetaTraderClient::Handlers::calc_margin)
        .def_readwrite("calc_profit", &MetaTraderClient::Handlers::calc_profit)
        // History (Orders & Deals)
        .def_readwrite("get_hist_orders_range", &MetaTraderClient::Handlers::get_hist_orders_range)
        .def_readwrite("get_hist_order_ticket", &MetaTraderClient::Handlers::get_hist_order_ticket)
        .def_readwrite("get_hist_orders_pos", &MetaTraderClient::Handlers::get_hist_orders_pos)
        .def_readwrite("get_hist_orders_total", &MetaTraderClient::Handlers::get_hist_orders_total)
        .def_readwrite("get_hist_deals_range", &MetaTraderClient::Handlers::get_hist_deals_range)
        .def_readwrite("get_hist_deals_ticket", &MetaTraderClient::Handlers::get_hist_deals_ticket)
        .def_readwrite("get_hist_deals_pos", &MetaTraderClient::Handlers::get_hist_deals_pos)
        .def_readwrite("get_hist_deals_total", &MetaTraderClient::Handlers::get_hist_deals_total);

    py::class_<MetaTraderClient, PyMetaTraderClient>(m, "MetaTraderClient")
        .def(py::init<>())
        .def(py::init<MetaTraderClient::Handlers>())

        // --- System Methods ---
        .def("initialize", py::overload_cast<>(&MetaTraderClient::initialize))
        .def("initialize", py::overload_cast<const std::string&>(&MetaTraderClient::initialize))
        .def("initialize", py::overload_cast<const std::string&, uint64_t, const std::string&, const std::string&, uint32_t, bool>(&MetaTraderClient::initialize))
        .def("login", &MetaTraderClient::login)
        .def("shutdown", &MetaTraderClient::shutdown)
        .def("version", &MetaTraderClient::version)
        .def("last_error", &MetaTraderClient::last_error)
        .def("terminal_info", [](MetaTraderClient& self) {
            auto result = self.terminal_info();
            return result ? to_namedtuple(*result) : py::none();
        })
        .def("account_info", [](MetaTraderClient& self) {
            auto result = self.account_info();
            return result ? to_namedtuple(*result) : py::none();
        })

        // --- Symbol & Market Depth Methods ---
        .def("symbols_total", &MetaTraderClient::symbols_total)
        .def("symbols_get", [](MetaTraderClient& self) { return to_list_of_namedtuples(self.symbols_get()); })
        .def("symbols_get", [](MetaTraderClient& self, const std::string& group) { return to_list_of_namedtuples(self.symbols_get(group)); })
        .def("symbol_info", [](MetaTraderClient& self, const std::string& symbol) {
            auto result = self.symbol_info(symbol);
            return result ? to_namedtuple(*result) : py::none();
        })
        .def("symbol_select", &MetaTraderClient::symbol_select)
        .def("market_book_add", &MetaTraderClient::market_book_add)
        .def("market_book_release", &MetaTraderClient::market_book_release)
        .def("market_book_get", [](MetaTraderClient& self, const std::string& symbol) { return to_list_of_namedtuples(self.market_book_get(symbol)); })

        // --- Market Data Methods ---
        .def("copy_rates_from", [](MetaTraderClient& self, const std::string& s, int32_t t, int64_t from, int32_t count) { return to_list_of_namedtuples(self.copy_rates_from(s, t, from, count)); })
        .def("copy_rates_from_pos", [](MetaTraderClient& self, const std::string& s, int32_t t, int32_t start, int32_t count) { return to_list_of_namedtuples(self.copy_rates_from_pos(s, t, start, count)); })
        .def("copy_rates_range", [](MetaTraderClient& self, const std::string& s, int32_t t, int64_t from, int64_t to) { return to_list_of_namedtuples(self.copy_rates_range(s, t, from, to)); })
        .def("copy_ticks_from", [](MetaTraderClient& self, const std::string& s, int64_t from, int32_t count, uint32_t flags) { return to_list_of_namedtuples(self.copy_ticks_from(s, from, count, flags)); })
        .def("copy_ticks_range", [](MetaTraderClient& self, const std::string& s, int64_t from, int64_t to, uint32_t flags) { return to_list_of_namedtuples(self.copy_ticks_range(s, from, to, flags)); })
        .def("symbol_info_tick", [](MetaTraderClient& self, const std::string& symbol) {
            auto result = self.symbol_info_tick(symbol);
            return result ? to_namedtuple(*result) : py::none();
        })

        // --- Trading Methods ---
        .def("order_check", [](MetaTraderClient& self, const TradeRequest& req) { return to_namedtuple(self.order_check(req)); })
        .def("order_send", [](MetaTraderClient& self, const TradeRequest& req) { return to_namedtuple(self.order_send(req)); })
        .def("order_calc_margin", &MetaTraderClient::order_calc_margin)
        .def("order_calc_profit", &MetaTraderClient::order_calc_profit)

        // --- Active Orders & Positions Methods ---
        .def("orders_get", [](MetaTraderClient& self) { return to_list_of_namedtuples(self.orders_get()); })
        .def("orders_get", [](MetaTraderClient& self, const std::string& symbol) { return to_list_of_namedtuples(self.orders_get(symbol)); })
        .def("orders_get_by_group", [](MetaTraderClient& self, const std::string& group) { return to_list_of_namedtuples(self.orders_get_by_group(group)); })
        .def("order_get_by_ticket", [](MetaTraderClient& self, uint64_t ticket) {
            auto result = self.order_get_by_ticket(ticket);
            return result ? to_namedtuple(*result) : py::none();
        })
        .def("orders_total", &MetaTraderClient::orders_total)
        .def("positions_get", [](MetaTraderClient& self) { return to_list_of_namedtuples(self.positions_get()); })
        .def("positions_get", [](MetaTraderClient& self, const std::string& symbol) { return to_list_of_namedtuples(self.positions_get(symbol)); })
        .def("positions_get_by_group", [](MetaTraderClient& self, const std::string& group) { return to_list_of_namedtuples(self.positions_get_by_group(group)); })
        .def("position_get_by_ticket", [](MetaTraderClient& self, uint64_t ticket) {
            auto result = self.position_get_by_ticket(ticket);
            return result ? to_namedtuple(*result) : py::none();
        })
        .def("positions_total", &MetaTraderClient::positions_total)

        // --- History Methods ---
        .def("history_orders_get", [](MetaTraderClient& self, int64_t from, int64_t to, const std::string& group) { return to_list_of_namedtuples(self.history_orders_get(from, to, group)); })
        .def("history_orders_get", [](MetaTraderClient& self, uint64_t ticket) {
            auto result = self.history_orders_get(ticket);
            return result ? to_namedtuple(*result) : py::none();
        })
        .def("history_orders_get_by_pos", [](MetaTraderClient& self, uint64_t pos_id) { return to_list_of_namedtuples(self.history_orders_get_by_pos(pos_id)); })
        .def("history_orders_total", &MetaTraderClient::history_orders_total)
        .def("history_deals_get", [](MetaTraderClient& self, int64_t from, int64_t to, const std::string& group) { return to_list_of_namedtuples(self.history_deals_get(from, to, group)); })
        .def("history_deals_get", [](MetaTraderClient& self, uint64_t ticket) { return to_list_of_namedtuples(self.history_deals_get(ticket)); })
        .def("history_deals_get_by_pos", [](MetaTraderClient& self, uint64_t pos_id) { return to_list_of_namedtuples(self.history_deals_get_by_pos(pos_id)); })
        .def("history_deals_total", &MetaTraderClient::history_deals_total);
}
