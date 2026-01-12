#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "bbstrader/metatrader.hpp"

// Test fixture for MetaTraderClient tests
class MetaTraderClientTest : public ::testing::Test {
   protected:
    MT5::MetaTraderClient::Handlers        handlers;
    std::unique_ptr<MT5::MetaTraderClient> client;

    // Flags to verify handler calls
    bool generic_handler_called = false;

    void SetUp() override {
        // Reset flags before each test
        generic_handler_called = false;
        // All handlers are initially null
        handlers = {};
        client   = std::make_unique<MT5::MetaTraderClient>(handlers);
    }

    void SetClientWithHandlers() { client = std::make_unique<MT5::MetaTraderClient>(handlers); }
};

// System Methods
TEST_F(MetaTraderClientTest, ForwardsInitialize) {
    handlers.init_auto = [this]() {
        generic_handler_called = true;
        return true;
    };
    SetClientWithHandlers();
    EXPECT_TRUE(client->initialize());
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsInitializeWithPath) {
    handlers.init_path = [&](auto&) {
        generic_handler_called = true;
        return true;
    };
    SetClientWithHandlers();
    std::string path = "path";
    EXPECT_TRUE(client->initialize(path));
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsInitializeWithFullArgs) {
    handlers.init_full = [&](auto&, auto, auto&, auto&, auto, auto) {
        generic_handler_called = true;
        return true;
    };
    SetClientWithHandlers();
    std::string path = "path", pw = "pw", srv = "srv";
    EXPECT_TRUE(client->initialize(path, 123, pw, srv, 1000, false));
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsLogin) {
    handlers.login = [&](auto, auto&, auto&, auto) {
        generic_handler_called = true;
        return true;
    };
    SetClientWithHandlers();
    std::string pw = "pw", srv = "srv";
    EXPECT_TRUE(client->login(123, pw, srv, 1000));
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsShutdown) {
    handlers.shutdown = [&]() {
        generic_handler_called = true;
    };
    SetClientWithHandlers();
    client->shutdown();
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsVersion) {
    handlers.get_version = [&]() {
        generic_handler_called = true;
        return MT5::VersionInfo{};
    };
    SetClientWithHandlers();
    EXPECT_NE(client->version(), std::nullopt);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsLastError) {
    handlers.get_last_error = [&]() {
        generic_handler_called = true;
        return MT5::LastErrorResult{};
    };
    SetClientWithHandlers();
    EXPECT_NE(client->last_error(), std::nullopt);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsTerminalInfo) {
    handlers.get_terminal_info = [this]() -> std::optional<MT5::TerminalInfo> {
        generic_handler_called = true;
        return MT5::TerminalInfo{};
    };
    SetClientWithHandlers();
    EXPECT_NE(client->terminal_info(), std::nullopt);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsAccountInfo) {
    handlers.get_account_info = [&]() {
        generic_handler_called = true;
        return MT5::AccountInfo{};
    };
    SetClientWithHandlers();
    EXPECT_NE(client->account_info(), std::nullopt);
    EXPECT_TRUE(generic_handler_called);
}

// Symbol Methods
TEST_F(MetaTraderClientTest, ForwardsSymbolsTotal) {
    handlers.get_total_symbols = [this]() {
        generic_handler_called = true;
        return 42;
    };
    SetClientWithHandlers();
    EXPECT_EQ(client->symbols_total(), 42);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsSymbolsGet) {
    handlers.get_symbols_all = [&]() {
        generic_handler_called = true;
        return MT5::SymbolsData{};
    };
    SetClientWithHandlers();
    client->symbols_get();
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsSymbolsGetWithGroup) {
    handlers.get_symbols_by_group = [&](auto&) {
        generic_handler_called = true;
        return MT5::SymbolsData{};
    };
    SetClientWithHandlers();
    std::string group = "group";
    client->symbols_get(group);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsSymbolInfo) {
    handlers.get_symbol_info = [&](auto&) {
        generic_handler_called = true;
        return MT5::SymbolInfo{};
    };
    SetClientWithHandlers();
    std::string symbol = "symbol";
    client->symbol_info(symbol);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsSymbolSelect) {
    std::string symbol_arg;
    bool        enable_arg = false;
    handlers.select_symbol = [&](const std::string& symbol, bool enable) {
        generic_handler_called = true;
        symbol_arg             = symbol;
        enable_arg             = enable;
        return true;
    };
    SetClientWithHandlers();

    std::string symbol = "EURUSD";
    EXPECT_TRUE(client->symbol_select(symbol, true));
    EXPECT_TRUE(generic_handler_called);
    EXPECT_EQ(symbol_arg, "EURUSD");
    EXPECT_TRUE(enable_arg);
}

TEST_F(MetaTraderClientTest, ForwardsSymbolInfoTick) {
    handlers.get_tick_info = [&](auto&) {
        generic_handler_called = true;
        return MT5::TickInfo{};
    };
    SetClientWithHandlers();
    std::string symbol = "symbol";
    client->symbol_info_tick(symbol);
    EXPECT_TRUE(generic_handler_called);
}

// Market Depth Methods
TEST_F(MetaTraderClientTest, ForwardsMarketBookAdd) {
    handlers.subscribe_book = [&](auto&) {
        generic_handler_called = true;
        return true;
    };
    SetClientWithHandlers();
    std::string symbol = "symbol";
    client->market_book_add(symbol);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsMarketBookRelease) {
    handlers.unsubscribe_book = [&](auto&) {
        generic_handler_called = true;
        return true;
    };
    SetClientWithHandlers();
    std::string symbol = "symbol";
    client->market_book_release(symbol);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsMarketBookGet) {
    handlers.get_book_info = [&](auto&) {
        generic_handler_called = true;
        return MT5::BookData{};
    };
    SetClientWithHandlers();
    std::string symbol = "symbol";
    client->market_book_get(symbol);
    EXPECT_TRUE(generic_handler_called);
}

// Market Data Methods
TEST_F(MetaTraderClientTest, ForwardsCopyRatesFrom) {
    handlers.get_rates_by_date = [&](auto&, auto, auto, auto) {
        generic_handler_called = true;
        return MT5::RateInfoType{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->copy_rates_from(symbol, 1, 0, 10);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsCopyRatesFromWithDateTime) {
    handlers.get_rates_by_date = [&](auto&, auto, auto, auto) {
        generic_handler_called = true;
        return MT5::RateInfoType{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->copy_rates_from(symbol, 1, std::chrono::system_clock::now(), 10);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsCopyRatesFromPos) {
    handlers.get_rates_by_pos = [&](auto&, auto, auto, auto) {
        generic_handler_called = true;
        return MT5::RateInfoType{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->copy_rates_from_pos(symbol, 1, 0, 10);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsCopyRatesRange) {
    handlers.get_rates_by_range = [&](auto&, auto, auto, auto) {
        generic_handler_called = true;
        return MT5::RateInfoType{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->copy_rates_range(symbol, 1, 0, 1);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsCopyRatesRangeWithDateTime) {
    handlers.get_rates_by_range = [&](auto&, auto, auto, auto) {
        generic_handler_called = true;
        return MT5::RateInfoType{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->copy_rates_range(symbol, 1, std::chrono::system_clock::now(), std::chrono::system_clock::now());
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsCopyTicksFrom) {
    handlers.get_ticks_by_date = [&](auto&, auto, auto, auto) {
        generic_handler_called = true;
        return MT5::TickInfoType{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->copy_ticks_from(symbol, 0, 10, 0);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsCopyTicksFromWithDateTime) {
    handlers.get_ticks_by_date = [&](auto&, auto, auto, auto) {
        generic_handler_called = true;
        return MT5::TickInfoType{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->copy_ticks_from(symbol, std::chrono::system_clock::now(), 10, 0);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsCopyTicksRange) {
    handlers.get_ticks_by_range = [&](auto&, auto, auto, auto) {
        generic_handler_called = true;
        return MT5::TickInfoType{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->copy_ticks_range(symbol, 0, 1, 0);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsCopyTicksRangeWithDateTime) {
    handlers.get_ticks_by_range = [&](auto&, auto, auto, auto) {
        generic_handler_called = true;
        return MT5::TickInfoType{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->copy_ticks_range(symbol, std::chrono::system_clock::now(), std::chrono::system_clock::now(), 0);
    EXPECT_TRUE(generic_handler_called);
}

// Active Order Methods
TEST_F(MetaTraderClientTest, ForwardsOrdersGet) {
    handlers.get_orders_all = [this]() {
        generic_handler_called = true;
        return MT5::OrdersData{};
    };
    SetClientWithHandlers();
    client->orders_get();
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsOrdersGetBySymbol) {
    std::string symbol_arg;
    handlers.get_orders_by_symbol = [&](const std::string& symbol) {
        generic_handler_called = true;
        symbol_arg             = symbol;
        return MT5::OrdersData{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->orders_get(symbol);
    EXPECT_TRUE(generic_handler_called);
    EXPECT_EQ(symbol_arg, "EURUSD");
}

TEST_F(MetaTraderClientTest, ForwardsOrdersGetByGroup) {
    handlers.get_orders_by_group = [&](auto&) {
        generic_handler_called = true;
        return MT5::OrdersData{};
    };
    SetClientWithHandlers();
    std::string group = "group";
    client->orders_get_by_group(group);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsOrderGetByTicket) {
    handlers.get_order_by_ticket = [&](auto) {
        generic_handler_called = true;
        return MT5::TradeOrder{};
    };
    SetClientWithHandlers();
    client->order_get_by_ticket(123);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsOrdersTotal) {
    handlers.get_total_orders = [&]() {
        generic_handler_called = true;
        return 10;
    };
    SetClientWithHandlers();
    EXPECT_EQ(client->orders_total(), 10);
    EXPECT_TRUE(generic_handler_called);
}


// Active Position Methods
TEST_F(MetaTraderClientTest, ForwardsPositionsGet) {
    handlers.get_positions_all = [&]() {
        generic_handler_called = true;
        return MT5::PositionsData{};
    };
    SetClientWithHandlers();
    client->positions_get();
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsPositionsGetBySymbol) {
    handlers.get_positions_symbol = [&](auto&) {
        generic_handler_called = true;
        return MT5::PositionsData{};
    };
    SetClientWithHandlers();
    std::string symbol = "symbol";
    client->positions_get(symbol);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsPositionsGetByGroup) {
    handlers.get_positions_group = [&](auto&) {
        generic_handler_called = true;
        return MT5::PositionsData{};
    };
    SetClientWithHandlers();
    std::string group = "group";
    client->positions_get_by_group(group);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsPositionsTotal) {
    handlers.get_total_positions = [this]() {
        generic_handler_called = true;
        return 10;
    };
    SetClientWithHandlers();
    EXPECT_EQ(client->positions_total(), 10);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsPositionGetByTicket) {
    uint64_t ticket_arg          = 0;
    handlers.get_position_ticket = [&](uint64_t ticket) {
        generic_handler_called = true;
        ticket_arg             = ticket;
        return MT5::TradePosition{};
    };
    SetClientWithHandlers();
    client->position_get_by_ticket(12345);
    EXPECT_TRUE(generic_handler_called);
    EXPECT_EQ(ticket_arg, 12345);
}

// Trading Methods
TEST_F(MetaTraderClientTest, ForwardsOrderCalcMargin) {
    handlers.calc_margin = [&](auto, auto&, auto, auto) {
        generic_handler_called = true;
        return 150.5;
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    EXPECT_EQ(client->order_calc_margin(0, symbol, 0.1, 1.2), 150.5);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsOrderCheck) {
    handlers.check_order = [&](const MT5::TradeRequest&) {
        generic_handler_called = true;
        return MT5::OrderCheckResult{};
    };
    SetClientWithHandlers();
    MT5::TradeRequest request{};
    client->order_check(request);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsOrderSend) {
    handlers.send_order = [&](const MT5::TradeRequest&) {
        generic_handler_called = true;
        return MT5::OrderSentResult{};
    };
    SetClientWithHandlers();
    MT5::TradeRequest request{};
    client->order_send(request);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsOrderCalcProfit) {
    handlers.calc_profit = [&](auto, auto&, auto, auto, auto) {
        generic_handler_called = true;
        return 200.5;
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    EXPECT_EQ(client->order_calc_profit(0, symbol, 0.1, 1.2, 1.3), 200.5);
    EXPECT_TRUE(generic_handler_called);
}

// History Order Methods
TEST_F(MetaTraderClientTest, ForwardsHistoryOrdersGet) {
    handlers.get_hist_orders_range = [&](auto, auto) {
        generic_handler_called = true;
        return MT5::OrdersData{};
    };
    SetClientWithHandlers();
    client->history_orders_get(0, 1);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsHistoryOrdersGetWithGroup) {
    handlers.get_hist_orders_group = [&](auto, auto, auto&) {
        generic_handler_called = true;
        return MT5::OrdersData{};
    };
    SetClientWithHandlers();
    std::string group = "group";
    client->history_orders_get(0, 1, group);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsHistoryOrdersGetByTicket) {
    handlers.get_hist_order_ticket = [&](auto) {
        generic_handler_called = true;
        return MT5::TradeOrder{};
    };
    SetClientWithHandlers();
    client->history_orders_get(123);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsHistoryOrdersGetByPos) {
    uint64_t pos_id_arg          = 0;
    handlers.get_hist_orders_pos = [&](uint64_t pos_id) {
        generic_handler_called = true;
        pos_id_arg             = pos_id;
        return MT5::OrdersData{};
    };
    SetClientWithHandlers();
    client->history_orders_get_by_pos(54321);
    EXPECT_TRUE(generic_handler_called);
    EXPECT_EQ(pos_id_arg, 54321);
}

TEST_F(MetaTraderClientTest, ForwardsHistoryOrdersTotal) {
    handlers.get_hist_orders_total = [&](auto, auto) {
        generic_handler_called = true;
        return 20;
    };
    SetClientWithHandlers();
    EXPECT_EQ(client->history_orders_total(0, 1), 20);
    EXPECT_TRUE(generic_handler_called);
}

// History Deal Methods
TEST_F(MetaTraderClientTest, ForwardsHistoryDealsGet) {
    handlers.get_hist_deals_range = [&](auto, auto) {
        generic_handler_called = true;
        return MT5::DealsData{};
    };
    SetClientWithHandlers();
    client->history_deals_get(0, 1);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsHistoryDealsGetWithGroup) {
    handlers.get_hist_deals_group = [&](auto, auto, auto&) {
        generic_handler_called = true;
        return MT5::DealsData{};
    };
    SetClientWithHandlers();
    std::string group = "group";
    client->history_deals_get(0, 1, group);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsHistoryDealsGetByTicket) {
    handlers.get_hist_deals_ticket = [&](auto) {
        generic_handler_called = true;
        return MT5::DealsData{};
    };
    SetClientWithHandlers();
    client->history_deals_get(123);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsHistoryDealsGetByPos) {
    handlers.get_hist_deals_pos = [&](auto) {
        generic_handler_called = true;
        return MT5::DealsData{};
    };
    SetClientWithHandlers();
    client->history_deals_get_by_pos(123);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsHistoryDealsTotal) {
    handlers.get_hist_deals_total = [&](auto, auto) {
        generic_handler_called = true;
        return 50;
    };
    SetClientWithHandlers();
    EXPECT_EQ(client->history_deals_total(0, 1), 50);
    EXPECT_TRUE(generic_handler_called);
}
