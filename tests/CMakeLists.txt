cmake_minimum_required(VERSION 3.20)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cache/_deps")
    include(FetchContent)
    FetchContent_Declare(
        GTest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG "v${GTEST_VERSION}"
    )
    FetchContent_MakeAvailable(GTest)
endif()

add_executable(
    metatrader_client_test
    cpp/test_metatrader_client.cpp
)

target_link_libraries(
    metatrader_client_test
    PRIVATE
    Python::Python
    GTest::gtest_main
    bbstrader::bbstrader
)

include(GoogleTest)
gtest_discover_tests(
    metatrader_client_test
    PROPERTIES LABELS "library"
)
