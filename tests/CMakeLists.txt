cmake_minimum_required(VERSION 3.20)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

add_executable(
    metatrader_test
    cpp/test_metatrader.cpp
)

target_link_libraries(
    metatrader_test
    PRIVATE
    GTest::gtest_main
    bbstrader
    Python::Python
)

include(GoogleTest)
gtest_discover_tests(metatrader_test)
