cmake_minimum_required(VERSION 3.12)
project(CandyCrush)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找pybind11
find_package(pybind11 REQUIRED)

# 添加库
pybind11_add_module(candy_crush_cpp
    src/pybind_wrapper.cpp
    src/candy_crush.cpp
)

# 设置输出目录
set_target_properties(candy_crush_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)