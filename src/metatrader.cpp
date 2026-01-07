#include "metatrader/metatrader.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace MT5;

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
        .def("initialize", py::overload_cast<>(&MetaTraderClient::initialize))
        .def("initialize", py::overload_cast<const std::string&>(&MetaTraderClient::initialize))
        .def("initialize", py::overload_cast<const std::string&, uint64_t, const std::string&, const std::string&, uint32_t, bool>(&MetaTraderClient::initialize))
        .def("login", &MetaTraderClient::login)
        .def("shutdown", &MetaTraderClient::shutdown)
        .def("version", &MetaTraderClient::version)
        .def("last_error", &MetaTraderClient::last_error)
        .def("terminal_info", &MetaTraderClient::terminal_info, py::return_value_policy::move)
        .def("account_info", &MetaTraderClient::account_info, py::return_value_policy::move)
        .def("symbols_total", &MetaTraderClient::symbols_total)
        .def("symbols_get", py::overload_cast<>(&MetaTraderClient::symbols_get), py::return_value_policy::move)
        .def("symbols_get", py::overload_cast<const std::string&>(&MetaTraderClient::symbols_get), py::return_value_policy::move)
        .def("symbol_info", &MetaTraderClient::symbol_info, py::return_value_policy::move)
        .def("symbol_select", &MetaTraderClient::symbol_select)
        .def("market_book_add", &MetaTraderClient::market_book_add)
        .def("market_book_release", &MetaTraderClient::market_book_release)
        .def("market_book_get", &MetaTraderClient::market_book_get, py::return_value_policy::move)
        .def("copy_rates_from", &MetaTraderClient::copy_rates_from, py::return_value_policy::move)
        .def("copy_rates_from_pos", &MetaTraderClient::copy_rates_from_pos, py::return_value_policy::move)
        .def("copy_rates_range", &MetaTraderClient::copy_rates_range, py::return_value_policy::move)
        .def("copy_ticks_from", &MetaTraderClient::copy_ticks_from, py::return_value_policy::move)
        .def("copy_ticks_range", &MetaTraderClient::copy_ticks_range, py::return_value_policy::move)
        .def("symbol_info_tick", &MetaTraderClient::symbol_info_tick, py::return_value_policy::move)
        .def("order_check", &MetaTraderClient::order_check, py::return_value_policy::move)
        .def("order_send", &MetaTraderClient::order_send, py::return_value_policy::move)
        .def("order_calc_margin", &MetaTraderClient::order_calc_margin)
        .def("order_calc_profit", &MetaTraderClient::order_calc_profit)
        .def("orders_get", py::overload_cast<>(&MetaTraderClient::orders_get), py::return_value_policy::move)
        .def("orders_get", py::overload_cast<const std::string&>(&MetaTraderClient::orders_get), py::return_value_policy::move)
        .def("orders_get_by_group", &MetaTraderClient::orders_get_by_group, py::return_value_policy::move)
        .def("order_get_by_ticket", &MetaTraderClient::order_get_by_ticket, py::return_value_policy::move)
        .def("orders_total", &MetaTraderClient::orders_total)
        .def("positions_get", py::overload_cast<>(&MetaTraderClient::positions_get), py::return_value_policy::move)
        .def("positions_get", py::overload_cast<const std::string&>(&MetaTraderClient::positions_get), py::return_value_policy::move)
        .def("positions_get_by_group", &MetaTraderClient::positions_get_by_group, py::return_value_policy::move)
        .def("position_get_by_ticket", &MetaTraderClient::position_get_by_ticket, py::return_value_policy::move)
        .def("positions_total", &MetaTraderClient::positions_total)
        .def("history_orders_get", py::overload_cast<int64_t, int64_t, const std::string&>(&MetaTraderClient::history_orders_get), py::return_value_policy::move)
        .def("history_orders_get", py::overload_cast<uint64_t>(&MetaTraderClient::history_orders_get), py::return_value_policy::move)
        .def("history_orders_get_by_pos", &MetaTraderClient::history_orders_get_by_pos, py::return_value_policy::move)
        .def("history_orders_total", &MetaTraderClient::history_orders_total)
        .def("history_deals_get", py::overload_cast<int64_t, int64_t, const std::string&>(&MetaTraderClient::history_deals_get), py::return_value_policy::move)
        .def("history_deals_get", py::overload_cast<uint64_t>(&MetaTraderClient::history_deals_get), py::return_value_policy::move)
        .def("history_deals_get_by_pos", &MetaTraderClient::history_deals_get_by_pos, py::return_value_policy::move)
        .def("history_deals_total", &MetaTraderClient::history_deals_total);

    py::class_<TerminalInfo>(m, "TerminalInfo")
        .def(py::init<>())
        .def_readonly("community_account", &TerminalInfo::community_account)
        .def_readonly("community_connection", &TerminalInfo::community_connection)
        .def_readonly("connected", &TerminalInfo::connected)
        .def_readonly("dlls_allowed", &TerminalInfo::dlls_allowed)
        .def_readonly("trade_allowed", &TerminalInfo::trade_allowed)
        .def_readonly("tradeapi_disabled", &TerminalInfo::tradeapi_disabled)
        .def_readonly("email_enabled", &TerminalInfo::email_enabled)
        .def_readonly("ftp_enabled", &TerminalInfo::ftp_enabled)
        .def_readonly("notifications_enabled", &TerminalInfo::notifications_enabled)
        .def_readonly("mqid", &TerminalInfo::mqid)
        .def_readonly("build", &TerminalInfo::build)
        .def_readonly("maxbars", &TerminalInfo::maxbars)
        .def_readonly("codepage", &TerminalInfo::codepage)
        .def_readonly("ping_last", &TerminalInfo::ping_last)
        .def_readonly("community_balance", &TerminalInfo::community_balance)
        .def_readonly("retransmission", &TerminalInfo::retransmission)
        .def_readonly("company", &TerminalInfo::company)
        .def_readonly("name", &TerminalInfo::name)
        .def_readonly("language", &TerminalInfo::language)
        .def_readonly("path", &TerminalInfo::path)
        .def_readonly("data_path", &TerminalInfo::data_path)
        .def_readonly("commondata_path", &TerminalInfo::commondata_path)
        .def("__repr__", [](const TerminalInfo &self) {
            return py::str("TerminalInfo(name='{}', company='{}', connected={})").format(self.name, self.company, self.connected);
        });

    py::class_<AccountInfo>(m, "AccountInfo")
        .def(py::init<>())
        .def_readonly("login", &AccountInfo::login)
        .def_readonly("trade_mode", &AccountInfo::trade_mode)
        .def_readonly("leverage", &AccountInfo::leverage)
        .def_readonly("limit_orders", &AccountInfo::limit_orders)
        .def_readonly("margin_so_mode", &AccountInfo::margin_so_mode)
        .def_readonly("trade_allowed", &AccountInfo::trade_allowed)
        .def_readonly("trade_expert", &AccountInfo::trade_expert)
        .def_readonly("margin_mode", &AccountInfo::margin_mode)
        .def_readonly("currency_digits", &AccountInfo::currency_digits)
        .def_readonly("fifo_close", &AccountInfo::fifo_close)
        .def_readonly("balance", &AccountInfo::balance)
        .def_readonly("credit", &AccountInfo::credit)
        .def_readonly("profit", &AccountInfo::profit)
        .def_readonly("equity", &AccountInfo::equity)
        .def_readonly("margin", &AccountInfo::margin)
        .def_readonly("margin_free", &AccountInfo::margin_free)
        .def_readonly("margin_level", &AccountInfo::margin_level)
        .def_readonly("margin_so_call", &AccountInfo::margin_so_call)
        .def_readonly("margin_so_so", &AccountInfo::margin_so_so)
        .def_readonly("margin_initial", &AccountInfo::margin_initial)
        .def_readonly("margin_maintenance", &AccountInfo::margin_maintenance)
        .def_readonly("assets", &AccountInfo::assets)
        .def_readonly("liabilities", &AccountInfo::liabilities)
        .def_readonly("commission_blocked", &AccountInfo::commission_blocked)
        .def_readonly("name", &AccountInfo::name)
        .def_readonly("server", &AccountInfo::server)
        .def_readonly("currency", &AccountInfo::currency)
        .def_readonly("company", &AccountInfo::company)
        .def("__repr__", [](const AccountInfo &self) {
            return py::str("AccountInfo(login={}, name='{}', balance={})").format(self.login, self.name, self.balance);
        });

    py::class_<SymbolInfo>(m, "SymbolInfo")
        .def(py::init<>())
        .def_readonly("custom", &SymbolInfo::custom).def_readonly("chart_mode", &SymbolInfo::chart_mode).def_readonly("select", &SymbolInfo::select)
        .def_readonly("visible", &SymbolInfo::visible).def_readonly("session_deals", &SymbolInfo::session_deals).def_readonly("session_buy_orders", &SymbolInfo::session_buy_orders)
        .def_readonly("session_sell_orders", &SymbolInfo::session_sell_orders).def_readonly("volume", &SymbolInfo::volume).def_readonly("volumehigh", &SymbolInfo::volumehigh)
        .def_readonly("volumelow", &SymbolInfo::volumelow).def_readonly("time", &SymbolInfo::time).def_readonly("digits", &SymbolInfo::digits)
        .def_readonly("spread", &SymbolInfo::spread).def_readonly("spread_float", &SymbolInfo::spread_float).def_readonly("ticks_bookdepth", &SymbolInfo::ticks_bookdepth)
        .def_readonly("trade_calc_mode", &SymbolInfo::trade_calc_mode).def_readonly("trade_mode", &SymbolInfo::trade_mode).def_readonly("start_time", &SymbolInfo::start_time)
        .def_readonly("expiration_time", &SymbolInfo::expiration_time).def_readonly("trade_stops_level", &SymbolInfo::trade_stops_level).def_readonly("trade_freeze_level", &SymbolInfo::trade_freeze_level)
        .def_readonly("trade_exemode", &SymbolInfo::trade_exemode).def_readonly("swap_mode", &SymbolInfo::swap_mode).def_readonly("swap_rollover3days", &SymbolInfo::swap_rollover3days)
        .def_readonly("margin_hedged_use_leg", &SymbolInfo::margin_hedged_use_leg).def_readonly("expiration_mode", &SymbolInfo::expiration_mode).def_readonly("filling_mode", &SymbolInfo::filling_mode)
        .def_readonly("order_mode", &SymbolInfo::order_mode).def_readonly("order_gtc_mode", &SymbolInfo::order_gtc_mode).def_readonly("option_mode", &SymbolInfo::option_mode)
        .def_readonly("option_right", &SymbolInfo::option_right).def_readonly("bid", &SymbolInfo::bid).def_readonly("bidhigh", &SymbolInfo::bidhigh)
        .def_readonly("bidlow", &SymbolInfo::bidlow).def_readonly("ask", &SymbolInfo::ask).def_readonly("askhigh", &SymbolInfo::askhigh)
        .def_readonly("asklow", &SymbolInfo::asklow).def_readonly("last", &SymbolInfo::last).def_readonly("lasthigh", &SymbolInfo::lasthigh)
        .def_readonly("lastlow", &SymbolInfo::lastlow).def_readonly("volume_real", &SymbolInfo::volume_real).def_readonly("volumehigh_real", &SymbolInfo::volumehigh_real)
        .def_readonly("volumelow_real", &SymbolInfo::volumelow_real).def_readonly("option_strike", &SymbolInfo::option_strike).def_readonly("point", &SymbolInfo::point)
        .def_readonly("trade_tick_value", &SymbolInfo::trade_tick_value).def_readonly("trade_tick_value_profit", &SymbolInfo::trade_tick_value_profit).def_readonly("trade_tick_value_loss", &SymbolInfo::trade_tick_value_loss)
        .def_readonly("trade_tick_size", &SymbolInfo::trade_tick_size).def_readonly("trade_contract_size", &SymbolInfo::trade_contract_size).def_readonly("trade_accrued_interest", &SymbolInfo::trade_accrued_interest)
        .def_readonly("trade_face_value", &SymbolInfo::trade_face_value).def_readonly("trade_liquidity_rate", &SymbolInfo::trade_liquidity_rate).def_readonly("volume_min", &SymbolInfo::volume_min)
        .def_readonly("volume_max", &SymbolInfo::volume_max).def_readonly("volume_step", &SymbolInfo::volume_step).def_readonly("volume_limit", &SymbolInfo::volume_limit)
        .def_readonly("swap_long", &SymbolInfo::swap_long).def_readonly("swap_short", &SymbolInfo::swap_short).def_readonly("margin_initial", &SymbolInfo::margin_initial)
        .def_readonly("margin_maintenance", &SymbolInfo::margin_maintenance).def_readonly("session_volume", &SymbolInfo::session_volume).def_readonly("session_turnover", &SymbolInfo::session_turnover)
        .def_readonly("session_interest", &SymbolInfo::session_interest).def_readonly("session_buy_orders_volume", &SymbolInfo::session_buy_orders_volume).def_readonly("session_sell_orders_volume", &SymbolInfo::session_sell_orders_volume)
        .def_readonly("session_open", &SymbolInfo::session_open).def_readonly("session_close", &SymbolInfo::session_close).def_readonly("session_aw", &SymbolInfo::session_aw)
        .def_readonly("session_price_settlement", &SymbolInfo::session_price_settlement).def_readonly("session_price_limit_min", &SymbolInfo::session_price_limit_min).def_readonly("session_price_limit_max", &SymbolInfo::session_price_limit_max)
        .def_readonly("margin_hedged", &SymbolInfo::margin_hedged).def_readonly("price_change", &SymbolInfo::price_change).def_readonly("price_volatility", &SymbolInfo::price_volatility)
        .def_readonly("price_theoretical", &SymbolInfo::price_theoretical).def_readonly("price_greeks_delta", &SymbolInfo::price_greeks_delta).def_readonly("price_greeks_theta", &SymbolInfo::price_greeks_theta)
        .def_readonly("price_greeks_gamma", &SymbolInfo::price_greeks_gamma).def_readonly("price_greeks_vega", &SymbolInfo::price_greeks_vega).def_readonly("price_greeks_rho", &SymbolInfo::price_greeks_rho)
        .def_readonly("price_greeks_omega", &SymbolInfo::price_greeks_omega).def_readonly("price_sensitivity", &SymbolInfo::price_sensitivity).def_readonly("basis", &SymbolInfo::basis)
        .def_readonly("category", &SymbolInfo::category).def_readonly("currency_base", &SymbolInfo::currency_base).def_readonly("currency_profit", &SymbolInfo::currency_profit)
        .def_readonly("currency_margin", &SymbolInfo::currency_margin).def_readonly("bank", &SymbolInfo::bank).def_readonly("description", &SymbolInfo::description)
        .def_readonly("exchange", &SymbolInfo::exchange).def_readonly("formula", &SymbolInfo::formula).def_readonly("isin", &SymbolInfo::isin)
        .def_readonly("name", &SymbolInfo::name).def_readonly("page", &SymbolInfo::page).def_readonly("path", &SymbolInfo::path)
        .def("__repr__", [](const SymbolInfo &self) {
            return py::str("SymbolInfo(name='{}', description='{}')").format(self.name, self.description);
        });

    py::class_<TickInfo>(m, "TickInfo")
        .def(py::init<>())
        .def_readonly("time", &TickInfo::time).def_readonly("bid", &TickInfo::bid).def_readonly("ask", &TickInfo::ask)
        .def_readonly("last", &TickInfo::last).def_readonly("volume", &TickInfo::volume).def_readonly("time_msc", &TickInfo::time_msc)
        .def_readonly("flags", &TickInfo::flags).def_readonly("volume_real", &TickInfo::volume_real)
        .def("__repr__", [](const TickInfo &self) {
            return py::str("TickInfo(time={}, bid={}, ask={})").format(self.time, self.bid, self.ask);
        });

    py::class_<RateInfo>(m, "RateInfo")
        .def(py::init<>())
        .def_readonly("time", &RateInfo::time).def_readonly("open", &RateInfo::open).def_readonly("high", &RateInfo::high)
        .def_readonly("low", &RateInfo::low).def_readonly("close", &RateInfo::close).def_readonly("tick_volume", &RateInfo::tick_volume)
        .def_readonly("spread", &RateInfo::spread).def_readonly("real_volume", &RateInfo::real_volume)
        .def("__repr__", [](const RateInfo &self) {
            return py::str("RateInfo(time={}, open={}, high={}, low={}, close={})").format(self.time, self.open, self.high, self.low, self.close);
        });

    py::class_<BookInfo>(m, "BookInfo")
        .def(py::init<>())
        .def_readonly("type", &BookInfo::type).def_readonly("price", &BookInfo::price).def_readonly("volume", &BookInfo::volume)
        .def_readonly("volume_real", &BookInfo::volume_real)
        .def("__repr__", [](const BookInfo &self) {
            return py::str("BookInfo(type={}, price={}, volume={})").format(self.type, self.price, self.volume);
        });

    py::class_<TradeRequest>(m, "TradeRequest")
        .def(py::init<>())
        .def_readonly("action", &TradeRequest::action).def_readonly("magic", &TradeRequest::magic).def_readonly("order", &TradeRequest::order)
        .def_readonly("symbol", &TradeRequest::symbol).def_readonly("volume", &TradeRequest::volume).def_readonly("price", &TradeRequest::price)
        .def_readonly("stoplimit", &TradeRequest::stoplimit).def_readonly("sl", &TradeRequest::sl).def_readonly("tp", &TradeRequest::tp)
        .def_readonly("deviation", &TradeRequest::deviation).def_readonly("type", &TradeRequest::type).def_readonly("type_filling", &TradeRequest::type_filling)
        .def_readonly("type_time", &TradeRequest::type_time).def_readonly("expiration", &TradeRequest::expiration).def_readonly("comment", &TradeRequest::comment)
        .def_readonly("position", &TradeRequest::position).def_readonly("position_by", &TradeRequest::position_by)
        .def("__repr__", [](const TradeRequest &self) {
            return py::str("TradeRequest(action={}, symbol='{}', volume={})").format(self.action, self.symbol, self.volume);
        });

    py::class_<OrderCheckResult>(m, "OrderCheckResult")
        .def(py::init<>())
        .def_readonly("retcode", &OrderCheckResult::retcode).def_readonly("balance", &OrderCheckResult::balance).def_readonly("equity", &OrderCheckResult::equity)
        .def_readonly("profit", &OrderCheckResult::profit).def_readonly("margin", &OrderCheckResult::margin).def_readonly("margin_free", &OrderCheckResult::margin_free)
        .def_readonly("margin_level", &OrderCheckResult::margin_level).def_readonly("comment", &OrderCheckResult::comment).def_readonly("request", &OrderCheckResult::request)
        .def("__repr__", [](const OrderCheckResult &self) {
            return py::str("OrderCheckResult(retcode={}, comment='{}')").format(self.retcode, self.comment);
        });

    py::class_<OrderSentResult>(m, "OrderSentResult")
        .def(py::init<>())
        .def_readonly("retcode", &OrderSentResult::retcode).def_readonly("deal", &OrderSentResult::deal).def_readonly("order", &OrderSentResult::order)
        .def_readonly("volume", &OrderSentResult::volume).def_readonly("price", &OrderSentResult::price).def_readonly("bid", &OrderSentResult::bid)
        .def_readonly("ask", &OrderSentResult::ask).def_readonly("comment", &OrderSentResult::comment).def_readonly("request_id", &OrderSentResult::request_id)
        .def_readonly("retcode_external", &OrderSentResult::retcode_external).def_readonly("request", &OrderSentResult::request)
        .def("__repr__", [](const OrderSentResult &self) {
            return py::str("OrderSentResult(retcode={}, order={}, deal={})").format(self.retcode, self.order, self.deal);
        });

    py::class_<TradeOrder>(m, "TradeOrder")
        .def(py::init<>())
        .def_readonly("ticket", &TradeOrder::ticket).def_readonly("time_setup", &TradeOrder::time_setup).def_readonly("time_setup_msc", &TradeOrder::time_setup_msc)
        .def_readonly("time_done", &TradeOrder::time_done).def_readonly("time_done_msc", &TradeOrder::time_done_msc).def_readonly("time_expiration", &TradeOrder::time_expiration)
        .def_readonly("type", &TradeOrder::type).def_readonly("type_time", &TradeOrder::type_time).def_readonly("type_filling", &TradeOrder::type_filling)
        .def_readonly("state", &TradeOrder::state).def_readonly("magic", &TradeOrder::magic).def_readonly("position_id", &TradeOrder::position_id)
        .def_readonly("position_by_id", &TradeOrder::position_by_id).def_readonly("reason", &TradeOrder::reason).def_readonly("volume_initial", &TradeOrder::volume_initial)
        .def_readonly("volume_current", &TradeOrder::volume_current).def_readonly("price_open", &TradeOrder::price_open).def_readonly("sl", &TradeOrder::sl)
        .def_readonly("tp", &TradeOrder::tp).def_readonly("price_current", &TradeOrder::price_current).def_readonly("price_stoplimit", &TradeOrder::price_stoplimit)
        .def_readonly("symbol", &TradeOrder::symbol).def_readonly("comment", &TradeOrder::comment).def_readonly("external_id", &TradeOrder::external_id)
        .def("__repr__", [](const TradeOrder &self) {
            return py::str("TradeOrder(ticket={}, symbol='{}', price_open={})").format(self.ticket, self.symbol, self.price_open);
        });

    py::class_<TradePosition>(m, "TradePosition")
        .def(py::init<>())
        .def_readonly("ticket", &TradePosition::ticket).def_readonly("time", &TradePosition::time).def_readonly("time_msc", &TradePosition::time_msc)
        .def_readonly("time_update", &TradePosition::time_update).def_readonly("time_update_msc", &TradePosition::time_update_msc).def_readonly("type", &TradePosition::type)
        .def_readonly("magic", &TradePosition::magic).def_readonly("identifier", &TradePosition::identifier).def_readonly("reason", &TradePosition::reason)
        .def_readonly("volume", &TradePosition::volume).def_readonly("price_open", &TradePosition::price_open).def_readonly("sl", &TradePosition::sl)
        .def_readonly("tp", &TradePosition::tp).def_readonly("price_current", &TradePosition::price_current).def_readonly("swap", &TradePosition::swap)
        .def_readonly("profit", &TradePosition::profit).def_readonly("symbol", &TradePosition::symbol).def_readonly("comment", &TradePosition::comment)
        .def_readonly("external_id", &TradePosition::external_id)
        .def("__repr__", [](const TradePosition &self) {
            return py::str("TradePosition(ticket={}, symbol='{}', volume={}, price_open={})").format(self.ticket, self.symbol, self.volume, self.price_open);
        });

    py::class_<TradeDeal>(m, "TradeDeal")
        .def(py::init<>())
        .def_readonly("ticket", &TradeDeal::ticket).def_readonly("order", &TradeDeal::order).def_readonly("time", &TradeDeal::time)
        .def_readonly("time_msc", &TradeDeal::time_msc).def_readonly("type", &TradeDeal::type).def_readonly("entry", &TradeDeal::entry)
        .def_readonly("magic", &TradeDeal::magic).def_readonly("position_id", &TradeDeal::position_id).def_readonly("reason", &TradeDeal::reason)
        .def_readonly("volume", &TradeDeal::volume).def_readonly("price", &TradeDeal::price).def_readonly("commission", &TradeDeal::commission)
        .def_readonly("swap", &TradeDeal::swap).def_readonly("profit", &TradeDeal::profit).def_readonly("fee", &TradeDeal::fee)
        .def_readonly("symbol", &TradeDeal::symbol).def_readonly("comment", &TradeDeal::comment).def_readonly("external_id", &TradeDeal::external_id)
        .def("__repr__", [](const TradeDeal &self) {
            return py::str("TradeDeal(ticket={}, order={}, symbol='{}', volume={}, price={})").format(self.ticket, self.order, self.symbol, self.volume, self.price);
        });
}
