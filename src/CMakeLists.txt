# https://pybind11.readthedocs.io/en/stable/compiling.html#advanced-interface-library-targets

find_package(pybind11 REQUIRED)


add_library(bbstrader
    metatrader.cpp 
)

target_include_directories(bbstrader PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(bbstrader PRIVATE pybind11::headers)


