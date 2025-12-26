cmake_minimum_required(VERSION 3.10)
project(neural-network-lib VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(Python3_EXECUTABLE "/Users/theodorfriberg/opt/anaconda3/envs/neuralnet/bin/python3")
set(Python3_FIND_VIRTUALENV ONLY)
set(Python3_FIND_REGISTRY NEVER)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Fetch Eigen
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1  # använd senaste stabila
)
FetchContent_MakeAvailable(pybind11)


# Collect source files (EXCLUDING main.cpp)
file(GLOB_RECURSE LIB_SOURCES
    "src/*.cpp"
)

# Remove main.cpp from the library sources
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main.cpp$")

# Create the library
add_library(neural_network STATIC ${LIB_SOURCES})
target_include_directories(neural_network PUBLIC include)
target_link_libraries(neural_network PUBLIC Eigen3::Eigen)

# Main executable
add_executable(neural_network_lib src/main.cpp)
target_link_libraries(neural_network_lib PRIVATE neural_network)


# Create python module
pybind11_add_module(mynn bindings/bindings.cpp)
target_link_libraries(mynn PRIVATE neural_network)
target_include_directories(mynn PRIVATE include)


# Add test + example subdirectories
add_subdirectory(tests)
add_subdirectory(examples)
