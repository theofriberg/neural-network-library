# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(neural-network-lib VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

# Add the executable
add_executable(neural-network-lib main.cpp)

target_include_directories(neural-network-lib PRIVATE include/Eigen)
