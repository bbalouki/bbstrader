#include <gtest/gtest.h>
#include "bbstrader/metatrader.hpp"
#include <string>
#include <vector>
#include <memory>

// Test fixture for MetaTraderClient tests
class MetaTraderClientTest : public ::testing::Test {
protected:
    MT5::MetaTraderClient::Handlers handlers;
    std::unique_ptr<MT5::MetaTraderClient> client;

    // Flags to verify handler calls
    bool generic_handler_called = false;

    void SetUp() override {
        // Reset flags before each test
        generic_handler_called = false;
        // All handlers are initially null
        handlers = {};
        client = std::make_unique<MT5::MetaTraderClient>(handlers);
    }

    void SetClientWithHandlers() {
        client = std::make_unique<MT5::MetaTraderClient>(handlers);
    }
};

// Test default constructor and behavior with no handlers
TEST_F(MetaTraderClientTest, NoHandlersDefaultBehavior) {
    MT5::MetaTraderClient default_client;
    std::string path = "path";
    std::string pw = "pw";
    std::string srv = "srv";
    std::string group = "group";
    uint64_t ticket = 123;

    // System
    EXPECT_FALSE(default_client.initialize());
    EXPECT_FALSE(default_client.initialize(path));
    EXPECT_FALSE(default_client.initialize(path, 123, pw, srv, 1000, false));
    EXPECT_FALSE(default_client.login(123, pw, srv, 1000));
    EXPECT_EQ(default_client.version(), std::nullopt);
    auto last_error = default_client.last_error();
    EXPECT_TRUE(last_error.has_value());
    EXPECT_EQ(std::get<0>(last_error.value()), -1);
    EXPECT_EQ(std::get<1>(last_error.value()), "fail");
    EXPECT_EQ(default_client.terminal_info(), std::nullopt);
    EXPECT_EQ(default_client.account_info(), std::nullopt);

    // Symbols
    EXPECT_EQ(default_client.symbols_total(), 0);
    EXPECT_EQ(default_client.symbols_get(), std::nullopt);
    EXPECT_EQ(default_client.symbols_get(group), std::nullopt);
    EXPECT_EQ(default_client.symbol_info(path), std::nullopt);
    EXPECT_FALSE(default_client.symbol_select(path, true));
    EXPECT_EQ(default_client.symbol_info_tick(path), std::nullopt);

    // Market Depth
    EXPECT_FALSE(default_client.market_book_add(path));
    EXPECT_FALSE(default_client.market_book_release(path));
    EXPECT_EQ(default_client.market_book_get(path), std::nullopt);

    // Active Orders
    EXPECT_EQ(default_client.orders_get(), std::nullopt);
    EXPECT_EQ(default_client.orders_get(path), std::nullopt);
    EXPECT_EQ(default_client.orders_get_by_group(group), std::nullopt);
    EXPECT_EQ(default_client.order_get_by_ticket(ticket), std::nullopt);
    EXPECT_EQ(default_client.orders_total(), 0);

    // Active Positions
    EXPECT_EQ(default_client.positions_get(), std::nullopt);
    EXPECT_EQ(default_client.positions_get(path), std::nullopt);
    EXPECT_EQ(default_client.positions_get_by_group(group), std::nullopt);
    EXPECT_EQ(default_client.position_get_by_ticket(ticket), std::nullopt);
    EXPECT_EQ(default_client.positions_total(), 0);

    // Trading
    EXPECT_EQ(default_client.order_calc_margin(0, path, 1.0, 1.0), std::nullopt);
    EXPECT_EQ(default_client.order_calc_profit(0, path, 1.0, 1.0, 2.0), std::nullopt);
}

// --- System Methods ---
TEST_F(MetaTraderClientTest, ForwardsInitialize) {
    handlers.init_auto = [this]() { generic_handler_called = true; return true; };
    SetClientWithHandlers();
    EXPECT_TRUE(client->initialize());
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

// --- Symbol Methods ---
TEST_F(MetaTraderClientTest, ForwardsSymbolsTotal) {
    handlers.get_total_symbols = [this]() { generic_handler_called = true; return 42; };
    SetClientWithHandlers();
    EXPECT_EQ(client->symbols_total(), 42);
    EXPECT_TRUE(generic_handler_called);
}

TEST_F(MetaTraderClientTest, ForwardsSymbolSelect) {
    std::string symbol_arg;
    bool enable_arg = false;
    handlers.select_symbol = [&](const std::string& symbol, bool enable) {
        generic_handler_called = true;
        symbol_arg = symbol;
        enable_arg = enable;
        return true;
    };
    SetClientWithHandlers();

    std::string symbol = "EURUSD";
    EXPECT_TRUE(client->symbol_select(symbol, true));
    EXPECT_TRUE(generic_handler_called);
    EXPECT_EQ(symbol_arg, "EURUSD");
    EXPECT_TRUE(enable_arg);
}


// --- Market Data Methods ---
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


// --- Active Order Methods ---
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
        symbol_arg = symbol;
        return MT5::OrdersData{};
    };
    SetClientWithHandlers();
    std::string symbol = "EURUSD";
    client->orders_get(symbol);
    EXPECT_TRUE(generic_handler_called);
    EXPECT_EQ(symbol_arg, "EURUSD");
}

// --- Active Position Methods ---
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
    uint64_t ticket_arg = 0;
    handlers.get_position_ticket = [&](uint64_t ticket) {
        generic_handler_called = true;
        ticket_arg = ticket;
        return MT5::TradePosition{};
    };
    SetClientWithHandlers();
    client->position_get_by_ticket(12345);
    EXPECT_TRUE(generic_handler_called);
    EXPECT_EQ(ticket_arg, 12345);
}

// --- Trading Methods ---
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
    handlers.check_order = [&](const MT5::TradeRequest& req) {
        generic_handler_called = true;
        return MT5::OrderCheckResult{};
    };
    SetClientWithHandlers();
    MT5::TradeRequest request{};
    client->order_check(request);
    EXPECT_TRUE(generic_handler_called);
}


// --- History Order Methods ---
TEST_F(MetaTraderClientTest, ForwardsHistoryOrdersGetByPos) {
    uint64_t pos_id_arg = 0;
    handlers.get_hist_orders_pos = [&](uint64_t pos_id) {
        generic_handler_called = true;
        pos_id_arg = pos_id;
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

// --- History Deal Methods ---
TEST_F(MetaTraderClientTest, ForwardsHistoryDealsGet) {
    handlers.get_hist_deals_range = [&](auto, auto, auto&) {
        generic_handler_called = true;
        return MT5::DealsData{};
    };
    SetClientWithHandlers();
    std::string group = "group";
    client->history_deals_get(0, 1, group);
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
