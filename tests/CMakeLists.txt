include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Function to add a test executable for a single test file
function(add_gtest_executable test_name test_file)
  # Create an executable for the test file
  add_executable(${test_name} ${test_file})

  # Link the neural network library and GTest
  target_link_libraries(${test_name} PRIVATE neural_network gtest gtest_main pthread)

  # Ensure the library is built before the tests
  add_dependencies(${test_name} neural_network)

  # Add include directories for tests
  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/Eigen)

  # Register the test as a CTest entry
  add_test(NAME ${test_name} COMMAND ${test_name} --gtest_color=yes --gtest_fail_fast)
endfunction()

enable_testing()

add_gtest_executable(test-layers test_layers.cpp)
