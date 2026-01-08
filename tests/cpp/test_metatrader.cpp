#include <gtest/gtest.h>
#include "bbstrader/metatrader.hpp"
#include <string>
#include <vector>

// Test fixture for MetaTraderClient tests
class MetaTraderClientTest : public ::testing::Test {
protected:
    MT5::MetaTraderClient::Handlers handlers;
    std::unique_ptr<MT5::MetaTraderClient> client;

    // Flags to verify handler calls
    bool init_auto_called = false;
    bool shutdown_called = false;
    bool get_version_called = false;

    void SetUp() override {
        // Reset flags before each test
        init_auto_called = false;
        shutdown_called = false;
        get_version_called = false;

        // Initialize handlers with mock implementations
        handlers.init_auto = [this]() {
            init_auto_called = true;
            return true;
        };
        handlers.shutdown = [this]() {
            shutdown_called = true;
        };
        handlers.get_version = [this]() -> std::optional<MT5::VersionInfo> {
            get_version_called = true;
            return std::make_tuple(5, 123, "2023-01-01");
        };

        client = std::make_unique<MT5::MetaTraderClient>(handlers);
    }
};

// Test default constructor and behavior with no handlers
TEST(MetaTraderClientDefaultTest, NoHandlers) {
    MT5::MetaTraderClient default_client;
    EXPECT_FALSE(default_client.initialize());
    EXPECT_EQ(default_client.version(), std::nullopt);
    auto last_error = default_client.last_error();
    EXPECT_TRUE(last_error.has_value());
    EXPECT_EQ(std::get<0>(last_error.value()), -1);
    EXPECT_EQ(std::get<1>(last_error.value()), "fail");
}

// Test that the client correctly forwards calls to the handlers
TEST_F(MetaTraderClientTest, ForwardsInitialize) {
    EXPECT_TRUE(client->initialize());
    EXPECT_TRUE(init_auto_called);
}

TEST_F(MetaTraderClientTest, ForwardsShutdown) {
    client->shutdown();
    EXPECT_TRUE(shutdown_called);
}

TEST_F(MetaTraderClientTest, ForwardsGetVersion) {
    auto version = client->version();
    EXPECT_TRUE(get_version_called);
    EXPECT_TRUE(version.has_value());
    EXPECT_EQ(std::get<0>(version.value()), 5);
}

// Test DateTime to timestamp conversion
TEST_F(MetaTraderClientTest, DateTimeConversion) {
    bool handler_called = false;
    int64_t received_timestamp = 0;

    handlers.get_rates_by_date = [&](MT5::str&, int32_t, int64_t from, int32_t) -> MT5::RateInfoType {
        handler_called = true;
        received_timestamp = from;
        return std::nullopt;
    };

    client = std::make_unique<MT5::MetaTraderClient>(handlers);

    auto now = std::chrono::system_clock::now();
    auto expected_timestamp = std::chrono::system_clock::to_time_t(now);
    std::string symbol = "EURUSD";

    client->copy_rates_from(symbol, 0, now, 1);

    EXPECT_TRUE(handler_called);
    EXPECT_EQ(received_timestamp, expected_timestamp);
}

// Test that a method returns a default value if its handler is missing
TEST_F(MetaTraderClientTest, MissingHandler) {
    handlers.get_account_info = nullptr;
    client = std::make_unique<MT5::MetaTraderClient>(handlers);
    EXPECT_EQ(client->account_info(), std::nullopt);
}
