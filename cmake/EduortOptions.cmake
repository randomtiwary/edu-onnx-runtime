# LEARNER: Project options live in one place so CMakeLists stays readable.
# Each option is a compile-time switch you can flip with -D... on the cmake line.

option(EDUORT_ENABLE_CUDA
  "Build CUDA execution provider (requires nvcc + working toolkit). Default OFF until PR10."
  OFF)

option(EDUORT_BUILD_TESTS
  "Build unit tests (GoogleTest via FetchContent)."
  ON)

option(EDUORT_BUILD_TOOLS
  "Build CLI tools (eduort-run). Enabled in PR9b."
  OFF)

option(EDUORT_WARNINGS_AS_ERRORS
  "Treat compiler warnings as errors (strict mode for clean PRs)."
  OFF)

# LEARNER: We keep C++17 as the default standard (design Key Decision K1).
# C++20 can be experimented with later without forcing it on learners' toolchains.
set(EDUORT_CXX_STANDARD "17" CACHE STRING "C++ standard for eduort (17 recommended)")
set_property(CACHE EDUORT_CXX_STANDARD PROPERTY STRINGS "17" "20")
