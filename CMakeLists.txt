# =============================================================================
# eduort — Educational ONNX Runtime (C++)
# =============================================================================
# LEARNER: CMake is the build system. This file answers three questions:
#   1) What do we build?  (a static library named `eduort`)
#   2) How do we compile it? (C++ standard, warnings, include paths)
#   3) What options exist? (CUDA, tests, tools, protobuf)
#
# Generate + build (from repo root):
#   cmake -G Ninja -B build -DEDUORT_ENABLE_CUDA=OFF
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# Design reference: docs/design.md  |  Pedagogy: docs/comment-style.md
# =============================================================================

cmake_minimum_required(VERSION 3.24)

project(eduort
  VERSION 0.0.1
  DESCRIPTION "Educational ONNX inference runtime in C++"
  LANGUAGES CXX
)

# ---- Options ----------------------------------------------------------------
include(cmake/EduortOptions.cmake)
include(cmake/CompilerWarnings.cmake)

# ---- Global C++ settings ----------------------------------------------------
set(CMAKE_CXX_STANDARD ${EDUORT_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for clangd / IDEs (nice for learning).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Third-party: Protobuf + ONNX schema codegen (PR3a) ----------------------
include(cmake/Dependencies.cmake)

# ---- Library target ---------------------------------------------------------
# LEARNER: We use a STATIC library by default (Key Decision K18).
add_library(eduort STATIC
  src/version.cpp
  src/status.cpp
  src/allocator.cpp
  src/tensor.cpp
  src/file_utils.cpp
  src/model_proto.cpp
)
add_library(eduort::eduort ALIAS eduort)

target_include_directories(eduort
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Propagate version macros to dependents as well.
target_compile_definitions(eduort
  PUBLIC
    EDUORT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    EDUORT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    EDUORT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# LEARNER: Link the generated onnx::ModelProto library so callers can
# #include "onnx.pb.h" via eduort headers (model_proto.h).
target_link_libraries(eduort PUBLIC eduort::onnx_proto)

if(EDUORT_ENABLE_CUDA)
  message(WARNING
    "EDUORT_ENABLE_CUDA=ON is accepted but CUDA sources are not implemented until PR10. "
    "Building CPU-only library; BuildInfo will still report CUDA=ON.")
  target_compile_definitions(eduort PUBLIC EDUORT_BUILT_WITH_CUDA=1)
endif()

eduort_set_project_warnings(eduort)

# ---- Smoke executable -------------------------------------------------------
add_executable(eduort_version_smoke
  tools/version_smoke.cpp
)
target_link_libraries(eduort_version_smoke PRIVATE eduort::eduort)
eduort_set_project_warnings(eduort_version_smoke)

# ---- Unit tests -------------------------------------------------------------
if(EDUORT_BUILD_TESTS)
  enable_testing()
  include(cmake/GoogleTest.cmake)
  add_subdirectory(tests)
endif()

# ---- Status summary ---------------------------------------------------------
message(STATUS "eduort ${PROJECT_VERSION}")
message(STATUS "  C++ standard     : ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA enabled     : ${EDUORT_ENABLE_CUDA}")
message(STATUS "  Build tests      : ${EDUORT_BUILD_TESTS}")
message(STATUS "  Build tools      : ${EDUORT_BUILD_TOOLS} (CLI in PR9b)")
message(STATUS "  Warnings as errs : ${EDUORT_WARNINGS_AS_ERRORS}")
