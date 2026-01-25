cmake_minimum_required(VERSION 3.20)

include(GNUInstallDirs)

# 1. Core C++ Library 
add_library(bbstrader 
    cpp/bbstrader.cpp
)
add_library(bbstrader::bbstrader ALIAS bbstrader)
configure_visibility(bbstrader)

target_include_directories(
    bbstrader PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(MSVC)
    target_compile_options(bbstrader PRIVATE /bigobj)
endif()

if (NOT ANDROID)
    target_link_libraries(
        bbstrader
        PRIVATE warnings::strict
    )
endif()


# 2. Python Bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    # Find Dependencies
    find_package(Python COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        include(FetchContent)
        set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cache/_deps")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        "v${PYBIND11_VERSION}"
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    target_link_libraries(bbstrader PRIVATE pybind11::headers)
    pybind11_add_module(client MODULE cpp/metatrader.cpp)
    target_include_directories(
        client PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    )
    target_link_libraries(client PRIVATE bbstrader)
    target_link_libraries(client PRIVATE pybind11::module)

    configure_visibility(client)

    if(MSVC)
        target_compile_options(client PRIVATE /bigobj)
    endif()
    if (NOT ANDROID)
        target_link_libraries(
            client
            PRIVATE warnings::strict
        )
    endif()

    # INSTALLATION INSTRUCTIONS FOR PYTHON
    install(TARGETS client DESTINATION bbstrader/api)

endif()

if(NOT SKBUILD)
    # 3: Output Destinations
    install(
        TARGETS bbstrader
        EXPORT BbstraderTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        TARGETS warnings
        EXPORT BbstraderTargets
    )

    # 4: Header Destinations
    install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/bbstrader
    )

    # 5: Target Information
    install(
        EXPORT BbstraderTargets
        FILE BbstraderTargets.cmake 
        NAMESPACE bbstrader::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bbstrader
    )

    # 6: Main Package Configuration File
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BbstraderConfig.cmake.in"
        "BbstraderConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/bbstrader"
    )

    # 7: Package Version File
    write_basic_package_version_file(
        "BbstraderConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    # 8: Generated Package Files
    install(
        FILES 
        "${CMAKE_CURRENT_BINARY_DIR}/BbstraderConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/BbstraderConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/bbstrader"
    )
endif()
