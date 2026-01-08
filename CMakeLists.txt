cmake_minimum_required(VERSION 3.20)

file(READ "VERSION.txt" RAW_VERSION)
string(STRIP "${RAW_VERSION}" APP_VERSION)

project(BBSTRADER VERSION ${APP_VERSION})

enable_testing()

include(${PROJECT_SOURCE_DIR}/cmake/Helpers.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Versions.cmake)

if(${PROJECT_NAME}_APPLY_FORMATING)
    apply_formating(${PROJECT_NAME}_USE_PER_FILE_FORMATTING)
endif()

if(${PROJECT_NAME}_APPLY_CLANG_TIDY_GLOBALY)
    apply_clang_tidy_globaly()
endif()
if(${PROJECT_NAME}_BUILD_DOCUMENTATION)
    build_documenation()
endif()
if(${PROJECT_NAME}_ENABLE_ADDRESS_SANITIZER)
    anable_address_sanitizer()
endif()

add_subdirectory(src)
add_subdirectory(tests)
