#include "bbstrader/metatrader.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace MT5;

class PyMetaTraderClient : public MetaTraderClient {
   public:
    using MetaTraderClient::MetaTraderClient;

    // System
    auto initialize() -> bool override { PYBIND11_OVERRIDE(bool, MetaTraderClient, initialize); }
    auto initialize(str& path) -> bool override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, initialize, path);
    }
    auto initialize(str& path, uint64_t account, str& pw, str& srv, uint32_t to, bool port)
        -> bool override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, initialize, path, account, pw, srv, to, port);
    }
    auto login(uint64_t account, str& pw, str& srv, uint32_t timeout) -> bool override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, login, account, pw, srv, timeout);
    }
    auto shutdown() -> void override { PYBIND11_OVERRIDE(void, MetaTraderClient, shutdown); }
    auto version() -> std::optional<VersionInfo> override {
        PYBIND11_OVERRIDE(std::optional<VersionInfo>, MetaTraderClient, version);
    }
    auto last_error() -> std::optional<LastErrorResult> override {
        PYBIND11_OVERRIDE(std::optional<LastErrorResult>, MetaTraderClient, last_error);
    }
    auto terminal_info() -> std::optional<TerminalInfo> override {
        PYBIND11_OVERRIDE(std::optional<TerminalInfo>, MetaTraderClient, terminal_info);
    }
    auto account_info() -> std::optional<AccountInfo> override {
        PYBIND11_OVERRIDE(std::optional<AccountInfo>, MetaTraderClient, account_info);
    }

    // Symbols
    auto symbols_total() -> std::optional<int32_t> override {
        PYBIND11_OVERRIDE(std::optional<int32_t>, MetaTraderClient, symbols_total);
    }
    auto symbols_get() -> SymbolsData override {
        PYBIND11_OVERRIDE(SymbolsData, MetaTraderClient, symbols_get);
    }
    auto symbols_get(str& group) -> SymbolsData override {
        PYBIND11_OVERRIDE(SymbolsData, MetaTraderClient, symbols_get, group);
    }
    auto symbol_info(str& symbol) -> std::optional<SymbolInfo> override {
        PYBIND11_OVERRIDE(std::optional<SymbolInfo>, MetaTraderClient, symbol_info, symbol);
    }
    auto symbol_select(str& symbol, bool enable) -> bool override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, symbol_select, symbol, enable);
    }

    // Market Depth
    auto market_book_add(str& symbol) -> bool override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, market_book_add, symbol);
    }
    auto market_book_release(str& symbol) -> bool override {
        PYBIND11_OVERRIDE(bool, MetaTraderClient, market_book_release, symbol);
    }
    auto market_book_get(str& symbol) -> BookData override {
        PYBIND11_OVERRIDE(BookData, MetaTraderClient, market_book_get, symbol);
    }

    // Market Data
    auto copy_rates_from(str& s, int32_t tf, DateTime from, int32_t count)
        -> RateInfoType override {
        PYBIND11_OVERRIDE(RateInfoType, MetaTraderClient, copy_rates_from, s, tf, from, count);
    }
    auto copy_rates_from(str& s, int32_t tf, int64_t from, int32_t count) -> RateInfoType override {
        PYBIND11_OVERRIDE(RateInfoType, MetaTraderClient, copy_rates_from, s, tf, from, count);
    }
    auto copy_rates_from_pos(str& s, int32_t tf, int32_t start, int32_t count)
        -> RateInfoType override {
        PYBIND11_OVERRIDE(RateInfoType, MetaTraderClient, copy_rates_from_pos, s, tf, start, count);
    }
    auto copy_rates_range(str& s, int32_t tf, DateTime from, DateTime to) -> RateInfoType override {
        PYBIND11_OVERRIDE(RateInfoType, MetaTraderClient, copy_rates_range, s, tf, from, to);
    }
    auto copy_rates_range(str& s, int32_t tf, int64_t from, int64_t to) -> RateInfoType override {
        PYBIND11_OVERRIDE(RateInfoType, MetaTraderClient, copy_rates_range, s, tf, from, to);
    }
    auto copy_ticks_from(str& s, DateTime from, int32_t count, int32_t flags)
        -> TickInfoType override {
        PYBIND11_OVERRIDE(TickInfoType, MetaTraderClient, copy_ticks_from, s, from, count, flags);
    }
    auto copy_ticks_from(str& s, int64_t from, int32_t count, int32_t flags)
        -> TickInfoType override {
        PYBIND11_OVERRIDE(TickInfoType, MetaTraderClient, copy_ticks_from, s, from, count, flags);
    }
    auto copy_ticks_range(str& s, DateTime from, DateTime to, int32_t flags)
        -> TickInfoType override {
        PYBIND11_OVERRIDE(TickInfoType, MetaTraderClient, copy_ticks_range, s, from, to, flags);
    }
    auto copy_ticks_range(str& s, int64_t from, int64_t to, int32_t flags)
        -> TickInfoType override {
        PYBIND11_OVERRIDE(TickInfoType, MetaTraderClient, copy_ticks_range, s, from, to, flags);
    }
    auto symbol_info_tick(str& symbol) -> std::optional<TickInfo> override {
        PYBIND11_OVERRIDE(std::optional<TickInfo>, MetaTraderClient, symbol_info_tick, symbol);
    }

    // Trading
    auto order_check(const py::dict& dict) -> std::optional<OrderCheckResult> override {
        PYBIND11_OVERRIDE(std::optional<OrderCheckResult>, MetaTraderClient, order_check, dict);
    }
    auto order_check(const TradeRequest& req) -> std::optional<OrderCheckResult> override {
        PYBIND11_OVERRIDE(std::optional<OrderCheckResult>, MetaTraderClient, order_check, req);
    }
    auto order_send(const py::dict& dict) -> std::optional<OrderSentResult> override {
        PYBIND11_OVERRIDE(std::optional<OrderSentResult>, MetaTraderClient, order_send, dict);
    }
    auto order_send(const TradeRequest& req) -> std::optional<OrderSentResult> override {
        PYBIND11_OVERRIDE(std::optional<OrderSentResult>, MetaTraderClient, order_send, req);
    }
    auto order_calc_margin(int32_t action, str& sym, double vol, double prc)
        -> std::optional<double> override {
        PYBIND11_OVERRIDE(
            std::optional<double>, MetaTraderClient, order_calc_margin, action, sym, vol, prc
        );
    }
    auto order_calc_profit(int32_t action, str& sym, double vol, double open, double close)
        -> std::optional<double> override {
        PYBIND11_OVERRIDE(
            std::optional<double>,
            MetaTraderClient,
            order_calc_profit,
            action,
            sym,
            vol,
            open,
            close
        );
    }

    // Active Orders & Positions
    auto orders_get() -> OrdersData override {
        PYBIND11_OVERRIDE(OrdersData, MetaTraderClient, orders_get);
    }
    auto orders_get(str& symbol) -> OrdersData override {
        PYBIND11_OVERRIDE(OrdersData, MetaTraderClient, orders_get, symbol);
    }
    auto orders_get_by_group(str& group) -> OrdersData override {
        PYBIND11_OVERRIDE(OrdersData, MetaTraderClient, orders_get_by_group, group);
    }
    auto order_get_by_ticket(uint64_t ticket) -> std::optional<TradeOrder> override {
        PYBIND11_OVERRIDE(std::optional<TradeOrder>, MetaTraderClient, order_get_by_ticket, ticket);
    }
    auto orders_total() -> std::optional<int32_t> override {
        PYBIND11_OVERRIDE(std::optional<int32_t>, MetaTraderClient, orders_total);
    }

    auto positions_get() -> PositionsData override {
        PYBIND11_OVERRIDE(PositionsData, MetaTraderClient, positions_get);
    }
    auto positions_get(str& symbol) -> PositionsData override {
        PYBIND11_OVERRIDE(PositionsData, MetaTraderClient, positions_get, symbol);
    }
    auto positions_get_by_group(str& group) -> PositionsData override {
        PYBIND11_OVERRIDE(PositionsData, MetaTraderClient, positions_get_by_group, group);
    }
    auto position_get_by_ticket(uint64_t ticket) -> std::optional<TradePosition> override {
        PYBIND11_OVERRIDE(
            std::optional<TradePosition>, MetaTraderClient, position_get_by_ticket, ticket
        );
    }
    auto positions_total() -> std::optional<int32_t> override {
        PYBIND11_OVERRIDE(std::optional<int32_t>, MetaTraderClient, positions_total);
    }

    // History
    auto history_orders_get(int64_t from, int64_t to, str& group) -> OrdersData override {
        PYBIND11_OVERRIDE(OrdersData, MetaTraderClient, history_orders_get, from, to, group);
    }
    auto history_orders_get(int64_t from, int64_t to) -> OrdersData override {
        PYBIND11_OVERRIDE(OrdersData, MetaTraderClient, history_orders_get, from, to);
    }
    auto history_orders_get(uint64_t ticket) -> std::optional<TradeOrder> override {
        PYBIND11_OVERRIDE(std::optional<TradeOrder>, MetaTraderClient, history_orders_get, ticket);
    }
    auto history_orders_get_by_pos(uint64_t pos_id) -> OrdersData override {
        PYBIND11_OVERRIDE(OrdersData, MetaTraderClient, history_orders_get_by_pos, pos_id);
    }
    auto history_orders_total(int64_t from, int64_t to) -> std::optional<int32_t> override {
        PYBIND11_OVERRIDE(std::optional<int32_t>, MetaTraderClient, history_orders_total, from, to);
    }

    auto history_deals_get(int64_t from, int64_t to, str& group) -> DealsData override {
        PYBIND11_OVERRIDE(DealsData, MetaTraderClient, history_deals_get, from, to, group);
    }
    auto history_deals_get(int64_t from, int64_t to) -> DealsData override {
        PYBIND11_OVERRIDE(DealsData, MetaTraderClient, history_deals_get, from, to);
    }
    auto history_deals_get(uint64_t ticket) -> DealsData override {
        PYBIND11_OVERRIDE(DealsData, MetaTraderClient, history_deals_get, ticket);
    }
    auto history_deals_get_by_pos(uint64_t pos_id) -> DealsData override {
        PYBIND11_OVERRIDE(DealsData, MetaTraderClient, history_deals_get_by_pos, pos_id);
    }
    auto history_deals_total(int64_t from, int64_t to) -> std::optional<int32_t> override {
        PYBIND11_OVERRIDE(std::optional<int32_t>, MetaTraderClient, history_deals_total, from, to);
    }
};

auto register_rate_info() -> void {
    using T = RateInfo;
    py::detail::register_structured_dtype(
        {PYBIND11_FIELD_DESCRIPTOR(T, time),
         PYBIND11_FIELD_DESCRIPTOR(T, open),
         PYBIND11_FIELD_DESCRIPTOR(T, high),
         PYBIND11_FIELD_DESCRIPTOR(T, low),
         PYBIND11_FIELD_DESCRIPTOR(T, close),
         PYBIND11_FIELD_DESCRIPTOR(T, tick_volume),
         PYBIND11_FIELD_DESCRIPTOR(T, spread),
         PYBIND11_FIELD_DESCRIPTOR(T, real_volume)},
        typeid(T),
        sizeof(T),
        nullptr
    );
}
auto register_tick_info() -> void {
    using T = TickInfo;
    py::detail::register_structured_dtype(
        {PYBIND11_FIELD_DESCRIPTOR(T, time),
         PYBIND11_FIELD_DESCRIPTOR(T, bid),
         PYBIND11_FIELD_DESCRIPTOR(T, ask),
         PYBIND11_FIELD_DESCRIPTOR(T, last),
         PYBIND11_FIELD_DESCRIPTOR(T, volume),
         PYBIND11_FIELD_DESCRIPTOR(T, time_msc),
         PYBIND11_FIELD_DESCRIPTOR(T, flags),
         PYBIND11_FIELD_DESCRIPTOR(T, volume_real)},
        typeid(T),
        sizeof(T),
        nullptr
    );
}

PYBIND11_MODULE(client, m) {
    m.doc() = "High-performance MetaTrader 5 C++/Python Bridge";
    register_rate_info();
    register_tick_info();

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

    // 3. Symbol Info
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

    // 5. Trading Request & Result
    py::class_<TradeRequest>(m, "TradeRequest")
        .def(py::init<>())
        .def(py::init([](const py::dict& dict) {
            auto req = std::make_unique<TradeRequest>();
            if (dict.contains("action"))
                req->action = dict["action"].cast<int32_t>();
            if (dict.contains("magic"))
                req->magic = dict["magic"].cast<int64_t>();
            if (dict.contains("order"))
                req->order = dict["order"].cast<int64_t>();
            if (dict.contains("symbol"))
                req->symbol = dict["symbol"].cast<std::string>();
            if (dict.contains("volume"))
                req->volume = dict["volume"].cast<double>();
            if (dict.contains("price"))
                req->price = dict["price"].cast<double>();
            if (dict.contains("stoplimit"))
                req->stoplimit = dict["stoplimit"].cast<double>();
            if (dict.contains("sl"))
                req->sl = dict["sl"].cast<double>();
            if (dict.contains("tp"))
                req->tp = dict["tp"].cast<double>();
            if (dict.contains("deviation"))
                req->deviation = dict["deviation"].cast<int64_t>();
            if (dict.contains("type"))
                req->type = dict["type"].cast<int32_t>();
            if (dict.contains("type_filling"))
                req->type_filling = dict["type_filling"].cast<int32_t>();
            if (dict.contains("type_time"))
                req->type_time = dict["type_time"].cast<int32_t>();
            if (dict.contains("expiration"))
                req->expiration = dict["expiration"].cast<int64_t>();
            if (dict.contains("comment"))
                req->comment = dict["comment"].cast<std::string>();
            if (dict.contains("position"))
                req->position = dict["position"].cast<int64_t>();
            if (dict.contains("position_by"))
                req->position_by = dict["position_by"].cast<int64_t>();
            return req;
        }))
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

    // 3. Enable implicit conversion from Python dict to C++ TradeRequest
    py::implicitly_convertible<py::dict, TradeRequest>();

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

    // 6. Records (Order, Position, Deal)
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

    // 7. Handlers Struct
    using Handlers = MetaTraderClient::Handlers;
    py::class_<MetaTraderClient::Handlers>(m, "MetaTraderHandlers")
        .def(py::init<>())
        // System & Session
        .def_readwrite("init_auto", &Handlers::init_auto)
        .def_readwrite("init_path", &Handlers::init_path)
        .def_readwrite("init_full", &Handlers::init_full)
        .def_readwrite("login", &Handlers::login)
        .def_readwrite("shutdown", &Handlers::shutdown)
        .def_readwrite("get_version", &Handlers::get_version)
        .def_readwrite("get_last_error", &Handlers::get_last_error)
        .def_readwrite("get_terminal_info", &Handlers::get_terminal_info)
        .def_readwrite("get_account_info", &Handlers::get_account_info)
        // Symbols & Market Depth
        .def_readwrite("get_total_symbols", &Handlers::get_total_symbols)
        .def_readwrite("get_symbols_all", &Handlers::get_symbols_all)
        .def_readwrite("get_symbol_info", &Handlers::get_symbol_info)
        .def_readwrite("select_symbol", &Handlers::select_symbol)
        .def_readwrite("get_symbols_by_group", &Handlers::get_symbols_by_group)
        .def_readwrite("subscribe_book", &Handlers::subscribe_book)
        .def_readwrite("unsubscribe_book", &Handlers::unsubscribe_book)
        .def_readwrite("get_book_info", &Handlers::get_book_info)
        // Market Data (Rates & Ticks)
        .def_readwrite("get_rates_by_date", &Handlers::get_rates_by_date)
        .def_readwrite("get_rates_by_pos", &Handlers::get_rates_by_pos)
        .def_readwrite("get_rates_by_range", &Handlers::get_rates_by_range)
        .def_readwrite("get_ticks_by_date", &Handlers::get_ticks_by_date)
        .def_readwrite("get_ticks_by_range", &Handlers::get_ticks_by_range)
        .def_readwrite("get_tick_info", &Handlers::get_tick_info)
        // Orders & Positions (Active)
        .def_readwrite("get_orders_all", &Handlers::get_orders_all)
        .def_readwrite("get_orders_by_symbol", &Handlers::get_orders_by_symbol)
        .def_readwrite("get_orders_by_group", &Handlers::get_orders_by_group)
        .def_readwrite("get_order_by_ticket", &Handlers::get_order_by_ticket)
        .def_readwrite("get_total_orders", &Handlers::get_total_orders)
        .def_readwrite("get_positions_all", &Handlers::get_positions_all)
        .def_readwrite("get_positions_symbol", &Handlers::get_positions_symbol)
        .def_readwrite("get_positions_group", &Handlers::get_positions_group)
        .def_readwrite("get_position_ticket", &Handlers::get_position_ticket)
        .def_readwrite("get_total_positions", &Handlers::get_total_positions)
        // Trading Operations
        .def_readwrite("check_order", &Handlers::check_order)
        .def_readwrite("send_order", &Handlers::send_order)
        .def_readwrite("calc_margin", &Handlers::calc_margin)
        .def_readwrite("calc_profit", &Handlers::calc_profit)
        // History (Orders & Deals)
        .def_readwrite("get_hist_orders_range", &Handlers::get_hist_orders_range)
        .def_readwrite("get_hist_orders_group", &Handlers::get_hist_orders_group)
        .def_readwrite("get_hist_order_ticket", &Handlers::get_hist_order_ticket)
        .def_readwrite("get_hist_orders_pos", &Handlers::get_hist_orders_pos)
        .def_readwrite("get_hist_orders_total", &Handlers::get_hist_orders_total)
        .def_readwrite("get_hist_deals_range", &Handlers::get_hist_deals_range)
        .def_readwrite("get_hist_deals_group", &Handlers::get_hist_deals_group)
        .def_readwrite("get_hist_deals_ticket", &Handlers::get_hist_deals_ticket)
        .def_readwrite("get_hist_deals_pos", &Handlers::get_hist_deals_pos)
        .def_readwrite("get_hist_deals_total", &Handlers::get_hist_deals_total);

    // 8. Main Client Class
    py::class_<MetaTraderClient, PyMetaTraderClient>(m, "MetaTraderClient")
        .def(py::init<MetaTraderClient::Handlers>(), py::arg("handlers"))

        // System
        .def("initialize", py::overload_cast<>(&MetaTraderClient::initialize))
        .def("initialize", py::overload_cast<str&>(&MetaTraderClient::initialize), py::arg("path"))
        .def(
            "initialize",
            py::overload_cast<str&, uint64_t, str&, str&, uint32_t, bool>(
                &MetaTraderClient::initialize
            ),
            py::arg("path"),
            py::arg("login"),
            py::arg("password"),
            py::arg("server"),
            py::arg("timeout"),
            py::arg("portable")
        )
        .def(
            "login",
            &MetaTraderClient::login,
            py::arg("login"),
            py::arg("password"),
            py::arg("server"),
            py::arg("timeout")
        )
        .def("shutdown", &MetaTraderClient::shutdown)
        .def("version", &MetaTraderClient::version)
        .def("last_error", &MetaTraderClient::last_error)
        .def("terminal_info", &MetaTraderClient::terminal_info)
        .def("account_info", &MetaTraderClient::account_info)

        // Symbols
        .def("symbols_total", &MetaTraderClient::symbols_total)
        .def(
            "symbols_get",
            py::overload_cast<>(&MetaTraderClient::symbols_get),
            py::return_value_policy::move
        )
        .def(
            "symbols_get",
            py::overload_cast<str&>(&MetaTraderClient::symbols_get),
            py::arg("group"),
            py::return_value_policy::move
        )
        .def("symbol_info", &MetaTraderClient::symbol_info, py::arg("symbol"))
        .def(
            "symbol_select", &MetaTraderClient::symbol_select, py::arg("symbol"), py::arg("enable")
        )

        // Market Depth
        .def("market_book_add", &MetaTraderClient::market_book_add, py::arg("symbol"))
        .def("market_book_release", &MetaTraderClient::market_book_release, py::arg("symbol"))
        .def(
            "market_book_get",
            &MetaTraderClient::market_book_get,
            py::arg("symbol"),
            py::return_value_policy::move
        )

        // Market Data
        .def(
            "copy_rates_from",
            py::overload_cast<str&, int32_t, int64_t, int32_t>(&MetaTraderClient::copy_rates_from),
            py::arg("symbol"),
            py::arg("timeframe"),
            py::arg("date_from"),
            py::arg("count"),
            py::return_value_policy::move
        )
        .def(
            "copy_rates_from",
            py::overload_cast<str&, int32_t, DateTime, int32_t>(&MetaTraderClient::copy_rates_from),
            py::arg("symbol"),
            py::arg("timeframe"),
            py::arg("date_from"),
            py::arg("count"),
            py::return_value_policy::move
        )
        .def(
            "copy_rates_from_pos",
            &MetaTraderClient::copy_rates_from_pos,
            py::arg("symbol"),
            py::arg("timeframe"),
            py::arg("start_pos"),
            py::arg("count"),
            py::return_value_policy::move
        )
        .def(
            "copy_rates_range",
            py::overload_cast<str&, int32_t, int64_t, int64_t>(&MetaTraderClient::copy_rates_range),
            py::arg("symbol"),
            py::arg("timeframe"),
            py::arg("date_from"),
            py::arg("date_to"),
            py::return_value_policy::move
        )
        .def(
            "copy_rates_range",
            py::overload_cast<str&, int32_t, DateTime, DateTime>(
                &MetaTraderClient::copy_rates_range
            ),
            py::arg("symbol"),
            py::arg("timeframe"),
            py::arg("date_from"),
            py::arg("date_to"),
            py::return_value_policy::move
        )
        .def(
            "copy_ticks_from",
            py::overload_cast<str&, int64_t, int32_t, int32_t>(&MetaTraderClient::copy_ticks_from),
            py::arg("symbol"),
            py::arg("date_from"),
            py::arg("count"),
            py::arg("flags"),
            py::return_value_policy::move
        )
        .def(
            "copy_ticks_from",
            py::overload_cast<str&, DateTime, int32_t, int32_t>(&MetaTraderClient::copy_ticks_from),
            py::arg("symbol"),
            py::arg("date_from"),
            py::arg("count"),
            py::arg("flags"),
            py::return_value_policy::move
        )
        .def(
            "copy_ticks_range",
            py::overload_cast<str&, int64_t, int64_t, int32_t>(&MetaTraderClient::copy_ticks_range),
            py::arg("symbol"),
            py::arg("date_from"),
            py::arg("date_to"),
            py::arg("flags"),
            py::return_value_policy::move
        )
        .def(
            "copy_ticks_range",
            py::overload_cast<str&, DateTime, DateTime, int32_t>(
                &MetaTraderClient::copy_ticks_range
            ),
            py::arg("symbol"),
            py::arg("date_from"),
            py::arg("date_to"),
            py::arg("flags"),
            py::return_value_policy::move
        )
        .def(
            "symbol_info_tick",
            &MetaTraderClient::symbol_info_tick,
            py::arg("symbol"),
            py::return_value_policy::move
        )

        // Active Orders
        .def(
            "orders_get",
            py::overload_cast<>(&MetaTraderClient::orders_get),
            py::return_value_policy::move
        )
        .def(
            "orders_get",
            py::overload_cast<str&>(&MetaTraderClient::orders_get),
            py::arg("symbol"),
            py::return_value_policy::move
        )
        .def(
            "orders_get_by_group",
            &MetaTraderClient::orders_get_by_group,
            py::arg("group"),
            py::return_value_policy::move
        )
        .def("order_get_by_ticket", &MetaTraderClient::order_get_by_ticket, py::arg("ticket"))
        .def("orders_total", &MetaTraderClient::orders_total)

        // Active Positions
        .def(
            "positions_get",
            py::overload_cast<>(&MetaTraderClient::positions_get),
            py::return_value_policy::move
        )
        .def(
            "positions_get",
            py::overload_cast<str&>(&MetaTraderClient::positions_get),
            py::arg("symbol"),
            py::return_value_policy::move
        )
        .def(
            "positions_get_by_group",
            &MetaTraderClient::positions_get_by_group,
            py::arg("group"),
            py::return_value_policy::move
        )
        .def("position_get_by_ticket", &MetaTraderClient::position_get_by_ticket, py::arg("ticket"))
        .def("positions_total", &MetaTraderClient::positions_total)

        // Trading Operations
        .def(
            "order_send",
            py::overload_cast<const TradeRequest&>(&MetaTraderClient::order_send),
            py::arg("request")
        )
        .def(
            "order_send",
            py::overload_cast<const py::dict&>(&MetaTraderClient::order_send),
            py::arg("request")
        )
        .def(
            "order_check",
            py::overload_cast<const TradeRequest&>(&MetaTraderClient::order_check),
            py::arg("request")
        )
        .def(
            "order_check",
            py::overload_cast<const py::dict&>(&MetaTraderClient::order_check),
            py::arg("request")
        )
        .def(
            "order_calc_margin",
            &MetaTraderClient::order_calc_margin,
            py::arg("action"),
            py::arg("symbol"),
            py::arg("volume"),
            py::arg("price")
        )
        .def(
            "order_calc_profit",
            &MetaTraderClient::order_calc_profit,
            py::arg("action"),
            py::arg("symbol"),
            py::arg("volume"),
            py::arg("price_open"),
            py::arg("price_close")
        )

        // History
        .def(
            "history_orders_get",
            py::overload_cast<int64_t, int64_t, str&>(&MetaTraderClient::history_orders_get),
            py::arg("date_from"),
            py::arg("date_to"),
            py::arg("group"),
            py::return_value_policy::move
        )
        .def(
            "history_orders_get",
            py::overload_cast<DateTime, DateTime, str&>(&MetaTraderClient::history_orders_get),
            py::arg("date_from"),
            py::arg("date_to"),
            py::arg("group"),
            py::return_value_policy::move
        )
        .def(
            "history_orders_get",
            py::overload_cast<int64_t, int64_t>(&MetaTraderClient::history_orders_get),
            py::arg("date_from"),
            py::arg("date_to"),
            py::return_value_policy::move
        )
        .def(
            "history_orders_get",
            py::overload_cast<DateTime, DateTime>(&MetaTraderClient::history_orders_get),
            py::arg("date_from"),
            py::arg("date_to"),
            py::return_value_policy::move
        )
        .def(
            "history_orders_get",
            py::overload_cast<uint64_t>(&MetaTraderClient::history_orders_get),
            py::arg("ticket"),
            py::return_value_policy::move
        )
        .def(
            "history_orders_get_by_pos",
            &MetaTraderClient::history_orders_get_by_pos,
            py::arg("position_id"),
            py::return_value_policy::move
        )
        .def(
            "history_orders_total",
            py::overload_cast<int64_t, int64_t>(&MetaTraderClient::history_orders_total),
            py::arg("date_from"),
            py::arg("date_to")
        )
        .def(
            "history_orders_total",
            py::overload_cast<DateTime, DateTime>(&MetaTraderClient::history_orders_total),
            py::arg("date_from"),
            py::arg("date_to")
        )
        .def(
            "history_deals_get",
            py::overload_cast<int64_t, int64_t, str&>(&MetaTraderClient::history_deals_get),
            py::arg("date_from"),
            py::arg("date_to"),
            py::arg("group"),
            py::return_value_policy::move
        )
        .def(
            "history_deals_get",
            py::overload_cast<DateTime, DateTime, str&>(&MetaTraderClient::history_deals_get),
            py::arg("date_from"),
            py::arg("date_to"),
            py::arg("group"),
            py::return_value_policy::move
        )
        .def(
            "history_deals_get",
            py::overload_cast<int64_t, int64_t>(&MetaTraderClient::history_deals_get),
            py::arg("date_from"),
            py::arg("date_to"),
            py::return_value_policy::move
        )
        .def(
            "history_deals_get",
            py::overload_cast<DateTime, DateTime>(&MetaTraderClient::history_deals_get),
            py::arg("date_from"),
            py::arg("date_to"),
            py::return_value_policy::move
        )
        .def(
            "history_deals_get",
            py::overload_cast<uint64_t>(&MetaTraderClient::history_deals_get),
            py::arg("ticket"),
            py::return_value_policy::move
        )
        .def(
            "history_deals_get_by_pos",
            &MetaTraderClient::history_deals_get_by_pos,
            py::arg("position_id"),
            py::return_value_policy::move
        )
        .def(
            "history_deals_total",
            py::overload_cast<int64_t, int64_t>(&MetaTraderClient::history_deals_total),
            py::arg("date_from"),
            py::arg("date_to")
        )
        .def(
            "history_deals_total",
            py::overload_cast<DateTime, DateTime>(&MetaTraderClient::history_deals_total),
            py::arg("date_from"),
            py::arg("date_to")
        );
}
