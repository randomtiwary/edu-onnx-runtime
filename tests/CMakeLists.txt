# LEARNER: Each *_test.cpp becomes an executable registered with CTest.
# Run all tests from the build tree:
#   ctest --test-dir build --output-on-failure

include(GoogleTest)

set(EDUORT_TESTDATA_DIR "${PROJECT_SOURCE_DIR}/testdata")

function(eduort_add_test test_name)
  add_executable(${test_name} ${ARGN})
  target_link_libraries(${test_name} PRIVATE eduort::eduort GTest::gtest_main)
  target_compile_definitions(${test_name} PRIVATE
    "EDUORT_TESTDATA_DIR=\"${EDUORT_TESTDATA_DIR}\"")
  eduort_set_project_warnings(${test_name})
  gtest_discover_tests(${test_name})
endfunction()

eduort_add_test(status_test status_test.cpp)
eduort_add_test(allocator_test allocator_test.cpp)
eduort_add_test(tensor_test tensor_test.cpp)
eduort_add_test(macros_test macros_test.cpp)
eduort_add_test(proto_smoke_test proto_smoke_test.cpp)
eduort_add_test(loader_test loader_test.cpp)
eduort_add_test(topo_sort_test topo_sort_test.cpp)
