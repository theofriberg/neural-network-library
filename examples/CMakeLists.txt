# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(XOR XOR_train.cpp)

target_include_directories(XOR PRIVATE ../neural-network-library/include)
target_link_libraries(XOR PRIVATE neural_network)
add_dependencies(XOR neural_network)

set_target_properties(XOR PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)