# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(neural-network-lib VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Collect all source files in the src directory
file(GLOB SOURCES "src/*.cpp")

# Define the library
add_library(neural_network STATIC ${SOURCES})

# Specify include directories for the library
target_include_directories(neural_network PUBLIC include)

# Define the main executable
add_executable(neural_network_lib src/main.cpp)

# Link the library to the main executable
target_link_libraries(neural_network_lib PRIVATE neural_network)

target_include_directories(neural_network_lib PRIVATE include/Eigen)

# Add the tests subdirectory
add_subdirectory(tests)

