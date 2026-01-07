#include "metatrader/metatrader.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace MT5;

class PyMetaTraderClient : public MetaTraderClient {
   public:
    using MetaTraderClient::MetaTraderClient;

    // --- System ---
    bool initialize() override { PYBIND11_OVERRIDE(bool, MetaTraderClient, initialize); }
    bool initialize(const std::string& path) override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, initialize, path);
    }
    bool initialize(
        const std::string& path,
        uint64_t           account,
        const std::string& pw,
        const std::string& srv,
        uint32_t           to,
        bool               port
    ) override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, initialize, path, account, pw, srv, to, port);
    }
    bool login(
        uint64_t account, const std::string& pw, const std::string& srv, uint32_t timeout
    ) override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, login, account, pw, srv, timeout);
    }
    void shutdown() override { PYBIND11_OVERRIDE(void, MetaTraderClient, shutdown); }
    std::optional<VersionInfo> version() override {
        PYBIND11_OVERRIDE(std::optional<VersionInfo>, MetaTraderClient, version);
    }
    LastErrorResult last_error() override {
        PYBIND11_OVERRIDE(LastErrorResult, MetaTraderClient, last_error);
    }
    std::optional<TerminalInfo> terminal_info() override {
        PYBIND11_OVERRIDE(std::optional<TerminalInfo>, MetaTraderClient, terminal_info);
    }
    std::optional<AccountInfo> account_info() override {
        PYBIND11_OVERRIDE(std::optional<AccountInfo>, MetaTraderClient, account_info);
    }

    // --- Symbols ---
    int32_t symbols_total() override {
        PYBIND11_OVERRIDE(int32_t, MetaTraderClient, symbols_total);
    }
    std::optional<std::vector<SymbolInfo>> symbols_get() override {
        PYBIND11_OVERRIDE(std::optional<std::vector<SymbolInfo>>, MetaTraderClient, symbols_get);
    }
    std::optional<std::vector<SymbolInfo>> symbols_get(const std::string& group) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<SymbolInfo>>, MetaTraderClient, symbols_get, group
        );
    }
    std::optional<SymbolInfo> symbol_info(const std::string& symbol) override {
        PYBIND11_OVERRIDE(std::optional<SymbolInfo>, MetaTraderClient, symbol_info, symbol);
    }
    bool symbol_select(const std::string& symbol, bool enable) override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, symbol_select, symbol, enable);
    }

    // --- Market Depth ---
    bool market_book_add(const std::string& symbol) override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, market_book_add, symbol);
    }
    bool market_book_release(const std::string& symbol) override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, market_book_release, symbol);
    }
    std::optional<std::vector<BookInfo>> market_book_get(const std::string& symbol) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<BookInfo>>, MetaTraderClient, market_book_get, symbol
        );
    }

    // --- Market Data ---
    std::optional<std::vector<RateInfo>> copy_rates_from(
        const std::string& s, int32_t t, int64_t from, int32_t count
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<RateInfo>>,
            MetaTraderClient,
            copy_rates_from,
            s,
            t,
            from,
            count
        );
    }
    std::optional<std::vector<RateInfo>> copy_rates_from_pos(
        const std::string& s, int32_t t, int32_t start, int32_t count
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<RateInfo>>,
            MetaTraderClient,
            copy_rates_from_pos,
            s,
            t,
            start,
            count
        );
    }
    std::optional<std::vector<RateInfo>> copy_rates_range(
        const std::string& s, int32_t t, int64_t from, int64_t to
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<RateInfo>>, MetaTraderClient, copy_rates_range, s, t, from, to
        );
    }
    std::optional<std::vector<TickInfo>> copy_ticks_from(
        const std::string& s, int64_t from, int32_t count, uint32_t flags
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TickInfo>>,
            MetaTraderClient,
            copy_ticks_from,
            s,
            from,
            count,
            flags
        );
    }
    std::optional<std::vector<TickInfo>> copy_ticks_range(
        const std::string& s, int64_t from, int64_t to, uint32_t flags
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TickInfo>>,
            MetaTraderClient,
            copy_ticks_range,
            s,
            from,
            to,
            flags
        );
    }
    std::optional<TickInfo> symbol_info_tick(const std::string& symbol) override {
        PYBIND11_OVERRIDE(std::optional<TickInfo>, MetaTraderClient, symbol_info_tick, symbol);
    }

    // --- Trading ---
    OrderCheckResult order_check(const TradeRequest& req) override {
        PYBIND11_OVERRIDE(OrderCheckResult, MetaTraderClient, order_check, req);
    }
    OrderSentResult order_send(const TradeRequest& req) override {
        PYBIND11_OVERRIDE(OrderSentResult, MetaTraderClient, order_send, req);
    }
    std::optional<double> order_calc_margin(
        int32_t act, const std::string& sym, double vol, double prc
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<double>, MetaTraderClient, order_calc_margin, act, sym, vol, prc
        );
    }
    std::optional<double> order_calc_profit(
        int32_t act, const std::string& sym, double vol, double open, double close
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<double>, MetaTraderClient, order_calc_profit, act, sym, vol, open, close
        );
    }

    // --- Active Orders & Positions ---
    std::optional<std::vector<TradeOrder>> orders_get() override {
        PYBIND11_OVERRIDE(std::optional<std::vector<TradeOrder>>, MetaTraderClient, orders_get);
    }
    std::optional<std::vector<TradeOrder>> orders_get(const std::string& symbol) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradeOrder>>, MetaTraderClient, orders_get, symbol
        );
    }
    std::optional<std::vector<TradeOrder>> orders_get_by_group(const std::string& group) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradeOrder>>, MetaTraderClient, orders_get_by_group, group
        );
    }
    std::optional<TradeOrder> order_get_by_ticket(uint64_t ticket) override {
        PYBIND11_OVERRIDE(std::optional<TradeOrder>, MetaTraderClient, order_get_by_ticket, ticket);
    }
    int32_t orders_total() override { PYBIND11_OVERRIDE(int32_t, MetaTraderClient, orders_total); }

    std::optional<std::vector<TradePosition>> positions_get() override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradePosition>>, MetaTraderClient, positions_get
        );
    }
    std::optional<std::vector<TradePosition>> positions_get(const std::string& symbol) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradePosition>>, MetaTraderClient, positions_get, symbol
        );
    }
    std::optional<std::vector<TradePosition>> positions_get_by_group(
        const std::string& group
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradePosition>>,
            MetaTraderClient,
            positions_get_by_group,
            group
        );
    }
    std::optional<TradePosition> position_get_by_ticket(uint64_t ticket) override {
        PYBIND11_OVERRIDE(
            std::optional<TradePosition>, MetaTraderClient, position_get_by_ticket, ticket
        );
    }
    int32_t positions_total() override {
        PYBIND11_OVERRIDE(int32_t, MetaTraderClient, positions_total);
    }

    // --- History ---
    std::optional<std::vector<TradeOrder>> history_orders_get(
        int64_t from, int64_t to, const std::string& group
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradeOrder>>,
            MetaTraderClient,
            history_orders_get,
            from,
            to,
            group
        );
    }
    std::optional<TradeOrder> history_orders_get(uint64_t ticket) override {
        PYBIND11_OVERRIDE(std::optional<TradeOrder>, MetaTraderClient, history_orders_get, ticket);
    }
    std::optional<std::vector<TradeOrder>> history_orders_get_by_pos(uint64_t pos_id) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradeOrder>>,
            MetaTraderClient,
            history_orders_get_by_pos,
            pos_id
        );
    }
    int32_t history_orders_total(int64_t from, int64_t to) override {
        PYBIND11_OVERRIDE(int32_t, MetaTraderClient, history_orders_total, from, to);
    }

    std::optional<std::vector<TradeDeal>> history_deals_get(
        int64_t from, int64_t to, const std::string& group
    ) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradeDeal>>,
            MetaTraderClient,
            history_deals_get,
            from,
            to,
            group
        );
    }
    std::optional<std::vector<TradeDeal>> history_deals_get(uint64_t ticket) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradeDeal>>, MetaTraderClient, history_deals_get, ticket
        );
    }
    std::optional<std::vector<TradeDeal>> history_deals_get_by_pos(uint64_t pos_id) override {
        PYBIND11_OVERRIDE(
            std::optional<std::vector<TradeDeal>>,
            MetaTraderClient,
            history_deals_get_by_pos,
            pos_id
        );
    }
    int32_t history_deals_total(int64_t from, int64_t to) override {
        PYBIND11_OVERRIDE(int32_t, MetaTraderClient, history_deals_total, from, to);
    }
};


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

    // 2. Main MetaTraderClient Binding
    py::class_<MetaTraderClient, PyMetaTraderClient>(m, "MetaTraderClient")
        .def(py::init<>())
        .def(py::init<MetaTraderClient::Handlers>())

        // --- System Methods ---
        .def("initialize", py::overload_cast<>(&MetaTraderClient::initialize))
        .def("initialize", py::overload_cast<const std::string&>(&MetaTraderClient::initialize))
        .def(
            "initialize",
            py::overload_cast<
                const std::string&,
                uint64_t,
                const std::string&,
                const std::string&,
                uint32_t,
                bool>(&MetaTraderClient::initialize)
        )
        .def("login", &MetaTraderClient::login)
        .def("shutdown", &MetaTraderClient::shutdown)
        .def("version", &MetaTraderClient::version)
        .def("last_error", &MetaTraderClient::last_error)
        .def("terminal_info", &MetaTraderClient::terminal_info)
        .def("account_info", &MetaTraderClient::account_info)

        // --- Symbol & Market Depth Methods ---
        .def("symbols_total", &MetaTraderClient::symbols_total)
        .def(
            "symbols_get",
            py::overload_cast<>(&MetaTraderClient::symbols_get),
            py::return_value_policy::move
        )
        .def(
            "symbols_get",
            py::overload_cast<const std::string&>(&MetaTraderClient::symbols_get),
            py::return_value_policy::move
        )
        .def("symbol_info", &MetaTraderClient::symbol_info)
        .def("symbol_select", &MetaTraderClient::symbol_select)
        .def("market_book_add", &MetaTraderClient::market_book_add)
        .def("market_book_release", &MetaTraderClient::market_book_release)
        .def("market_book_get", &MetaTraderClient::market_book_get, py::return_value_policy::move)

        // --- Market Data Methods (Optimized Policy) ---
        .def("copy_rates_from", &MetaTraderClient::copy_rates_from, py::return_value_policy::move)
        .def(
            "copy_rates_from_pos",
            &MetaTraderClient::copy_rates_from_pos,
            py::return_value_policy::move
        )
        .def("copy_rates_range", &MetaTraderClient::copy_rates_range, py::return_value_policy::move)
        .def("copy_ticks_from", &MetaTraderClient::copy_ticks_from, py::return_value_policy::move)
        .def("copy_ticks_range", &MetaTraderClient::copy_ticks_range, py::return_value_policy::move)
        .def("symbol_info_tick", &MetaTraderClient::symbol_info_tick, py::return_value_policy::move)

        // --- Active Orders & Positions Methods ---
        .def(
            "orders_get",
            py::overload_cast<>(&MetaTraderClient::orders_get),
            py::return_value_policy::move
        )
        .def(
            "orders_get",
            py::overload_cast<const std::string&>(&MetaTraderClient::orders_get),
            py::return_value_policy::move
        )
        .def(
            "orders_get_by_group",
            &MetaTraderClient::orders_get_by_group,
            py::return_value_policy::move
        )
        .def("order_get_by_ticket", &MetaTraderClient::order_get_by_ticket)
        .def("orders_total", &MetaTraderClient::orders_total)

        .def(
            "positions_get",
            py::overload_cast<>(&MetaTraderClient::positions_get),
            py::return_value_policy::move
        )
        .def(
            "positions_get",
            py::overload_cast<const std::string&>(&MetaTraderClient::positions_get),
            py::return_value_policy::move
        )
        .def(
            "positions_get_by_group",
            &MetaTraderClient::positions_get_by_group,
            py::return_value_policy::move
        )
        .def("position_get_by_ticket", &MetaTraderClient::position_get_by_ticket)
        .def("positions_total", &MetaTraderClient::positions_total)

        // --- Trading Methods ---
        .def("order_send", &MetaTraderClient::order_send)
        .def("order_check", &MetaTraderClient::order_check)
        .def("order_calc_margin", &MetaTraderClient::order_calc_margin)
        .def("order_calc_profit", &MetaTraderClient::order_calc_profit)

        // --- History Methods (Optimized Policy) ---
        .def(
            "history_orders_get",
            py::overload_cast<int64_t, int64_t, const std::string&>(
                &MetaTraderClient::history_orders_get
            ),
            py::return_value_policy::move
        )
        .def(
            "history_orders_get", py::overload_cast<uint64_t>(&MetaTraderClient::history_orders_get)
        )
        .def(
            "history_orders_get_by_pos",
            &MetaTraderClient::history_orders_get_by_pos,
            py::return_value_policy::move
        )
        .def("history_orders_total", &MetaTraderClient::history_orders_total)

        .def(
            "history_deals_get",
            py::overload_cast<int64_t, int64_t, const std::string&>(
                &MetaTraderClient::history_deals_get
            ),
            py::return_value_policy::move
        )
        .def("history_deals_get", py::overload_cast<uint64_t>(&MetaTraderClient::history_deals_get))
        .def(
            "history_deals_get_by_pos",
            &MetaTraderClient::history_deals_get_by_pos,
            py::return_value_policy::move
        )
        .def("history_deals_total", &MetaTraderClient::history_deals_total);

    // 1. Terminal Info
    py::class_<TerminalInfo>(m, "TerminalInfo")
        .def(py::init<>())
        .def_readwrite("community_account", &TerminalInfo::community_account)
        .def_readwrite("community_connection", &TerminalInfo::community_connection)
        .def_readwrite("connected", &TerminalInfo::connected)
        .def_readwrite("dlls_allowed", &TerminalInfo::dlls_allowed)
        .def_readwrite("trade_allowed", &TerminalInfo::trade_allowed)
        .def_readwrite("tradeapi_disabled", &TerminalInfo::tradeapi_disabled)
        .def_readwrite("email_enabled", &TerminalInfo::email_enabled)
        .def_readwrite("ftp_enabled", &TerminalInfo::ftp_enabled)
        .def_readwrite("notifications_enabled", &TerminalInfo::notifications_enabled)
        .def_readwrite("mqid", &TerminalInfo::mqid)
        .def_readwrite("build", &TerminalInfo::build)
        .def_readwrite("maxbars", &TerminalInfo::maxbars)
        .def_readwrite("codepage", &TerminalInfo::codepage)
        .def_readwrite("ping_last", &TerminalInfo::ping_last)
        .def_readwrite("community_balance", &TerminalInfo::community_balance)
        .def_readwrite("retransmission", &TerminalInfo::retransmission)
        .def_readwrite("company", &TerminalInfo::company)
        .def_readwrite("name", &TerminalInfo::name)
        .def_readwrite("language", &TerminalInfo::language)
        .def_readwrite("path", &TerminalInfo::path)
        .def_readwrite("data_path", &TerminalInfo::data_path)
        .def_readwrite("commondata_path", &TerminalInfo::commondata_path);

    // 2. Account Info
    py::class_<AccountInfo>(m, "AccountInfo")
        .def(py::init<>())
        .def_readwrite("login", &AccountInfo::login)
        .def_readwrite("trade_mode", &AccountInfo::trade_mode)
        .def_readwrite("leverage", &AccountInfo::leverage)
        .def_readwrite("limit_orders", &AccountInfo::limit_orders)
        .def_readwrite("margin_so_mode", &AccountInfo::margin_so_mode)
        .def_readwrite("trade_allowed", &AccountInfo::trade_allowed)
        .def_readwrite("trade_expert", &AccountInfo::trade_expert)
        .def_readwrite("margin_mode", &AccountInfo::margin_mode)
        .def_readwrite("currency_digits", &AccountInfo::currency_digits)
        .def_readwrite("fifo_close", &AccountInfo::fifo_close)
        .def_readwrite("balance", &AccountInfo::balance)
        .def_readwrite("credit", &AccountInfo::credit)
        .def_readwrite("profit", &AccountInfo::profit)
        .def_readwrite("equity", &AccountInfo::equity)
        .def_readwrite("margin", &AccountInfo::margin)
        .def_readwrite("margin_free", &AccountInfo::margin_free)
        .def_readwrite("margin_level", &AccountInfo::margin_level)
        .def_readwrite("margin_so_call", &AccountInfo::margin_so_call)
        .def_readwrite("margin_so_so", &AccountInfo::margin_so_so)
        .def_readwrite("margin_initial", &AccountInfo::margin_initial)
        .def_readwrite("margin_maintenance", &AccountInfo::margin_maintenance)
        .def_readwrite("assets", &AccountInfo::assets)
        .def_readwrite("liabilities", &AccountInfo::liabilities)
        .def_readwrite("commission_blocked", &AccountInfo::commission_blocked)
        .def_readwrite("name", &AccountInfo::name)
        .def_readwrite("server", &AccountInfo::server)
        .def_readwrite("currency", &AccountInfo::currency)
        .def_readwrite("company", &AccountInfo::company);

    // 3. Symbol Info (Extensive Mapping)
    py::class_<SymbolInfo>(m, "SymbolInfo")
        .def(py::init<>())
        .def_readwrite("custom", &SymbolInfo::custom)
        .def_readwrite("chart_mode", &SymbolInfo::chart_mode)
        .def_readwrite("select", &SymbolInfo::select)
        .def_readwrite("visible", &SymbolInfo::visible)
        .def_readwrite("session_deals", &SymbolInfo::session_deals)
        .def_readwrite("session_buy_orders", &SymbolInfo::session_buy_orders)
        .def_readwrite("session_sell_orders", &SymbolInfo::session_sell_orders)
        .def_readwrite("volume", &SymbolInfo::volume)
        .def_readwrite("volumehigh", &SymbolInfo::volumehigh)
        .def_readwrite("volumelow", &SymbolInfo::volumelow)
        .def_readwrite("time", &SymbolInfo::time)
        .def_readwrite("digits", &SymbolInfo::digits)
        .def_readwrite("spread", &SymbolInfo::spread)
        .def_readwrite("spread_float", &SymbolInfo::spread_float)
        .def_readwrite("ticks_bookdepth", &SymbolInfo::ticks_bookdepth)
        .def_readwrite("trade_calc_mode", &SymbolInfo::trade_calc_mode)
        .def_readwrite("trade_mode", &SymbolInfo::trade_mode)
        .def_readwrite("start_time", &SymbolInfo::start_time)
        .def_readwrite("expiration_time", &SymbolInfo::expiration_time)
        .def_readwrite("trade_stops_level", &SymbolInfo::trade_stops_level)
        .def_readwrite("trade_freeze_level", &SymbolInfo::trade_freeze_level)
        .def_readwrite("trade_exemode", &SymbolInfo::trade_exemode)
        .def_readwrite("swap_mode", &SymbolInfo::swap_mode)
        .def_readwrite("swap_rollover3days", &SymbolInfo::swap_rollover3days)
        .def_readwrite("margin_hedged_use_leg", &SymbolInfo::margin_hedged_use_leg)
        .def_readwrite("expiration_mode", &SymbolInfo::expiration_mode)
        .def_readwrite("filling_mode", &SymbolInfo::filling_mode)
        .def_readwrite("order_mode", &SymbolInfo::order_mode)
        .def_readwrite("order_gtc_mode", &SymbolInfo::order_gtc_mode)
        .def_readwrite("option_mode", &SymbolInfo::option_mode)
        .def_readwrite("option_right", &SymbolInfo::option_right)
        .def_readwrite("bid", &SymbolInfo::bid)
        .def_readwrite("bidhigh", &SymbolInfo::bidhigh)
        .def_readwrite("bidlow", &SymbolInfo::bidlow)
        .def_readwrite("ask", &SymbolInfo::ask)
        .def_readwrite("askhigh", &SymbolInfo::askhigh)
        .def_readwrite("asklow", &SymbolInfo::asklow)
        .def_readwrite("last", &SymbolInfo::last)
        .def_readwrite("lasthigh", &SymbolInfo::lasthigh)
        .def_readwrite("lastlow", &SymbolInfo::lastlow)
        .def_readwrite("volume_real", &SymbolInfo::volume_real)
        .def_readwrite("volumehigh_real", &SymbolInfo::volumehigh_real)
        .def_readwrite("volumelow_real", &SymbolInfo::volumelow_real)
        .def_readwrite("option_strike", &SymbolInfo::option_strike)
        .def_readwrite("point", &SymbolInfo::point)
        .def_readwrite("trade_tick_value", &SymbolInfo::trade_tick_value)
        .def_readwrite("trade_tick_value_profit", &SymbolInfo::trade_tick_value_profit)
        .def_readwrite("trade_tick_value_loss", &SymbolInfo::trade_tick_value_loss)
        .def_readwrite("trade_tick_size", &SymbolInfo::trade_tick_size)
        .def_readwrite("trade_contract_size", &SymbolInfo::trade_contract_size)
        .def_readwrite("trade_accrued_interest", &SymbolInfo::trade_accrued_interest)
        .def_readwrite("trade_face_value", &SymbolInfo::trade_face_value)
        .def_readwrite("trade_liquidity_rate", &SymbolInfo::trade_liquidity_rate)
        .def_readwrite("volume_min", &SymbolInfo::volume_min)
        .def_readwrite("volume_max", &SymbolInfo::volume_max)
        .def_readwrite("volume_step", &SymbolInfo::volume_step)
        .def_readwrite("volume_limit", &SymbolInfo::volume_limit)
        .def_readwrite("swap_long", &SymbolInfo::swap_long)
        .def_readwrite("swap_short", &SymbolInfo::swap_short)
        .def_readwrite("margin_initial", &SymbolInfo::margin_initial)
        .def_readwrite("margin_maintenance", &SymbolInfo::margin_maintenance)
        .def_readwrite("session_volume", &SymbolInfo::session_volume)
        .def_readwrite("session_turnover", &SymbolInfo::session_turnover)
        .def_readwrite("session_interest", &SymbolInfo::session_interest)
        .def_readwrite("session_buy_orders_volume", &SymbolInfo::session_buy_orders_volume)
        .def_readwrite("session_sell_orders_volume", &SymbolInfo::session_sell_orders_volume)
        .def_readwrite("session_open", &SymbolInfo::session_open)
        .def_readwrite("session_close", &SymbolInfo::session_close)
        .def_readwrite("session_aw", &SymbolInfo::session_aw)
        .def_readwrite("session_price_settlement", &SymbolInfo::session_price_settlement)
        .def_readwrite("session_price_limit_min", &SymbolInfo::session_price_limit_min)
        .def_readwrite("session_price_limit_max", &SymbolInfo::session_price_limit_max)
        .def_readwrite("margin_hedged", &SymbolInfo::margin_hedged)
        .def_readwrite("price_change", &SymbolInfo::price_change)
        .def_readwrite("price_volatility", &SymbolInfo::price_volatility)
        .def_readwrite("price_theoretical", &SymbolInfo::price_theoretical)
        .def_readwrite("price_greeks_delta", &SymbolInfo::price_greeks_delta)
        .def_readwrite("price_greeks_theta", &SymbolInfo::price_greeks_theta)
        .def_readwrite("price_greeks_gamma", &SymbolInfo::price_greeks_gamma)
        .def_readwrite("price_greeks_vega", &SymbolInfo::price_greeks_vega)
        .def_readwrite("price_greeks_rho", &SymbolInfo::price_greeks_rho)
        .def_readwrite("price_greeks_omega", &SymbolInfo::price_greeks_omega)
        .def_readwrite("price_sensitivity", &SymbolInfo::price_sensitivity)
        .def_readwrite("basis", &SymbolInfo::basis)
        .def_readwrite("category", &SymbolInfo::category)
        .def_readwrite("currency_base", &SymbolInfo::currency_base)
        .def_readwrite("currency_profit", &SymbolInfo::currency_profit)
        .def_readwrite("currency_margin", &SymbolInfo::currency_margin)
        .def_readwrite("bank", &SymbolInfo::bank)
        .def_readwrite("description", &SymbolInfo::description)
        .def_readwrite("exchange", &SymbolInfo::exchange)
        .def_readwrite("formula", &SymbolInfo::formula)
        .def_readwrite("isin", &SymbolInfo::isin)
        .def_readwrite("name", &SymbolInfo::name)
        .def_readwrite("page", &SymbolInfo::page)
        .def_readwrite("path", &SymbolInfo::path);

    // 4. Market Data Structs
    py::class_<TickInfo>(m, "TickInfo")
        .def(py::init<>())
        .def_readwrite("time", &TickInfo::time)
        .def_readwrite("bid", &TickInfo::bid)
        .def_readwrite("ask", &TickInfo::ask)
        .def_readwrite("last", &TickInfo::last)
        .def_readwrite("volume", &TickInfo::volume)
        .def_readwrite("time_msc", &TickInfo::time_msc)
        .def_readwrite("flags", &TickInfo::flags)
        .def_readwrite("volume_real", &TickInfo::volume_real);

    py::class_<RateInfo>(m, "RateInfo")
        .def(py::init<>())
        .def_readwrite("time", &RateInfo::time)
        .def_readwrite("open", &RateInfo::open)
        .def_readwrite("high", &RateInfo::high)
        .def_readwrite("low", &RateInfo::low)
        .def_readwrite("close", &RateInfo::close)
        .def_readwrite("tick_volume", &RateInfo::tick_volume)
        .def_readwrite("spread", &RateInfo::spread)
        .def_readwrite("real_volume", &RateInfo::real_volume);

    py::class_<BookInfo>(m, "BookInfo")
        .def(py::init<>())
        .def_readwrite("type", &BookInfo::type)
        .def_readwrite("price", &BookInfo::price)
        .def_readwrite("volume", &BookInfo::volume)
        .def_readwrite("volume_real", &BookInfo::volume_real);

    // 5. Trade Request & Results
    py::class_<TradeRequest>(m, "TradeRequest")
        .def(py::init<>())
        .def_readwrite("action", &TradeRequest::action)
        .def_readwrite("magic", &TradeRequest::magic)
        .def_readwrite("order", &TradeRequest::order)
        .def_readwrite("symbol", &TradeRequest::symbol)
        .def_readwrite("volume", &TradeRequest::volume)
        .def_readwrite("price", &TradeRequest::price)
        .def_readwrite("stoplimit", &TradeRequest::stoplimit)
        .def_readwrite("sl", &TradeRequest::sl)
        .def_readwrite("tp", &TradeRequest::tp)
        .def_readwrite("deviation", &TradeRequest::deviation)
        .def_readwrite("type", &TradeRequest::type)
        .def_readwrite("type_filling", &TradeRequest::type_filling)
        .def_readwrite("type_time", &TradeRequest::type_time)
        .def_readwrite("expiration", &TradeRequest::expiration)
        .def_readwrite("comment", &TradeRequest::comment)
        .def_readwrite("position", &TradeRequest::position)
        .def_readwrite("position_by", &TradeRequest::position_by);

    py::class_<OrderCheckResult>(m, "OrderCheckResult")
        .def(py::init<>())
        .def_readwrite("retcode", &OrderCheckResult::retcode)
        .def_readwrite("balance", &OrderCheckResult::balance)
        .def_readwrite("equity", &OrderCheckResult::equity)
        .def_readwrite("profit", &OrderCheckResult::profit)
        .def_readwrite("margin", &OrderCheckResult::margin)
        .def_readwrite("margin_free", &OrderCheckResult::margin_free)
        .def_readwrite("margin_level", &OrderCheckResult::margin_level)
        .def_readwrite("comment", &OrderCheckResult::comment)
        .def_readwrite("request", &OrderCheckResult::request);

    py::class_<OrderSentResult>(m, "OrderSentResult")
        .def(py::init<>())
        .def_readwrite("retcode", &OrderSentResult::retcode)
        .def_readwrite("deal", &OrderSentResult::deal)
        .def_readwrite("order", &OrderSentResult::order)
        .def_readwrite("volume", &OrderSentResult::volume)
        .def_readwrite("price", &OrderSentResult::price)
        .def_readwrite("bid", &OrderSentResult::bid)
        .def_readwrite("ask", &OrderSentResult::ask)
        .def_readwrite("comment", &OrderSentResult::comment)
        .def_readwrite("request_id", &OrderSentResult::request_id)
        .def_readwrite("retcode_external", &OrderSentResult::retcode_external)
        .def_readwrite("request", &OrderSentResult::request);

    // 6. Execution Objects (Orders, Positions, Deals)
    py::class_<TradeOrder>(m, "TradeOrder")
        .def(py::init<>())
        .def_readwrite("ticket", &TradeOrder::ticket)
        .def_readwrite("time_setup", &TradeOrder::time_setup)
        .def_readwrite("time_setup_msc", &TradeOrder::time_setup_msc)
        .def_readwrite("time_done", &TradeOrder::time_done)
        .def_readwrite("time_done_msc", &TradeOrder::time_done_msc)
        .def_readwrite("time_expiration", &TradeOrder::time_expiration)
        .def_readwrite("type", &TradeOrder::type)
        .def_readwrite("type_time", &TradeOrder::type_time)
        .def_readwrite("type_filling", &TradeOrder::type_filling)
        .def_readwrite("state", &TradeOrder::state)
        .def_readwrite("magic", &TradeOrder::magic)
        .def_readwrite("position_id", &TradeOrder::position_id)
        .def_readwrite("position_by_id", &TradeOrder::position_by_id)
        .def_readwrite("reason", &TradeOrder::reason)
        .def_readwrite("volume_initial", &TradeOrder::volume_initial)
        .def_readwrite("volume_current", &TradeOrder::volume_current)
        .def_readwrite("price_open", &TradeOrder::price_open)
        .def_readwrite("sl", &TradeOrder::sl)
        .def_readwrite("tp", &TradeOrder::tp)
        .def_readwrite("price_current", &TradeOrder::price_current)
        .def_readwrite("price_stoplimit", &TradeOrder::price_stoplimit)
        .def_readwrite("symbol", &TradeOrder::symbol)
        .def_readwrite("comment", &TradeOrder::comment)
        .def_readwrite("external_id", &TradeOrder::external_id);

    py::class_<TradePosition>(m, "TradePosition")
        .def(py::init<>())
        .def_readwrite("ticket", &TradePosition::ticket)
        .def_readwrite("time", &TradePosition::time)
        .def_readwrite("time_msc", &TradePosition::time_msc)
        .def_readwrite("time_update", &TradePosition::time_update)
        .def_readwrite("time_update_msc", &TradePosition::time_update_msc)
        .def_readwrite("type", &TradePosition::type)
        .def_readwrite("magic", &TradePosition::magic)
        .def_readwrite("identifier", &TradePosition::identifier)
        .def_readwrite("reason", &TradePosition::reason)
        .def_readwrite("volume", &TradePosition::volume)
        .def_readwrite("price_open", &TradePosition::price_open)
        .def_readwrite("sl", &TradePosition::sl)
        .def_readwrite("tp", &TradePosition::tp)
        .def_readwrite("price_current", &TradePosition::price_current)
        .def_readwrite("swap", &TradePosition::swap)
        .def_readwrite("profit", &TradePosition::profit)
        .def_readwrite("symbol", &TradePosition::symbol)
        .def_readwrite("comment", &TradePosition::comment)
        .def_readwrite("external_id", &TradePosition::external_id);

    py::class_<TradeDeal>(m, "TradeDeal")
        .def(py::init<>())
        .def_readwrite("ticket", &TradeDeal::ticket)
        .def_readwrite("order", &TradeDeal::order)
        .def_readwrite("time", &TradeDeal::time)
        .def_readwrite("time_msc", &TradeDeal::time_msc)
        .def_readwrite("type", &TradeDeal::type)
        .def_readwrite("entry", &TradeDeal::entry)
        .def_readwrite("magic", &TradeDeal::magic)
        .def_readwrite("position_id", &TradeDeal::position_id)
        .def_readwrite("reason", &TradeDeal::reason)
        .def_readwrite("volume", &TradeDeal::volume)
        .def_readwrite("price", &TradeDeal::price)
        .def_readwrite("commission", &TradeDeal::commission)
        .def_readwrite("swap", &TradeDeal::swap)
        .def_readwrite("profit", &TradeDeal::profit)
        .def_readwrite("fee", &TradeDeal::fee)
        .def_readwrite("symbol", &TradeDeal::symbol)
        .def_readwrite("comment", &TradeDeal::comment)
        .def_readwrite("external_id", &TradeDeal::external_id);
}
