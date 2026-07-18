# LEARNER: Each *_test.cpp becomes an executable registered with CTest.
# Run all tests from the build tree:
#   ctest --test-dir build --output-on-failure

include(GoogleTest)

function(eduort_add_test test_name)
  add_executable(${test_name} ${ARGN})
  target_link_libraries(${test_name} PRIVATE eduort::eduort GTest::gtest_main)
  eduort_set_project_warnings(${test_name})
  gtest_discover_tests(${test_name})
endfunction()

eduort_add_test(status_test status_test.cpp)
eduort_add_test(allocator_test allocator_test.cpp)
eduort_add_test(tensor_test tensor_test.cpp)
