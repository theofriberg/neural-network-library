cmake_minimum_required(VERSION 3.10)
project(neural-network-lib VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch Eigen
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

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


# Add test + example subdirectories
add_subdirectory(tests)
add_subdirectory(examples)
