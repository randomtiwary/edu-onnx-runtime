# LEARNER: Third-party deps for eduort. PR3a adds Protocol Buffers + ONNX schema.
#
# Design (K3): full official onnx.proto pin v1.14.1 (vendored under third_party/onnx).
# System libprotobuf + protoc preferred; optional FetchContent if missing.

include(FetchContent)

option(EDUORT_FETCH_PROTOBUF
  "Download protobuf via FetchContent when find_package fails."
  ON)

# ---- Protobuf ---------------------------------------------------------------
find_package(Protobuf QUIET)

if(NOT Protobuf_FOUND)
  if(NOT EDUORT_FETCH_PROTOBUF)
    message(FATAL_ERROR
      "Protobuf not found. Install libprotobuf-dev and protobuf-compiler,\n"
      "  or reconfigure with -DEDUORT_FETCH_PROTOBUF=ON.")
  endif()
  message(STATUS "Protobuf not found; FetchContent google/protobuf v21.12 (slow first time)...")
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    protobuf
    GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
    GIT_TAG        v21.12
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(protobuf)
else()
  message(STATUS "Found Protobuf ${Protobuf_VERSION}")
  message(STATUS "  protoc           : ${Protobuf_PROTOC_EXECUTABLE}")
endif()

if(NOT TARGET protobuf::libprotobuf)
  # Older FindProtobuf modules only set Protobuf_LIBRARIES.
  if(Protobuf_FOUND AND Protobuf_LIBRARIES)
    add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
    set_target_properties(protobuf::libprotobuf PROPERTIES
      IMPORTED_LOCATION "${Protobuf_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR "protobuf::libprotobuf target missing after Protobuf setup")
  endif()
endif()

# Resolve a concrete protoc path for add_custom_command (no genex).
if(Protobuf_PROTOC_EXECUTABLE)
  set(EDUORT_PROTOC "${Protobuf_PROTOC_EXECUTABLE}")
elseif(TARGET protobuf::protoc)
  # Configure-time path for FetchContent builds after MakeAvailable.
  set(EDUORT_PROTOC "$<TARGET_FILE:protobuf::protoc>")
else()
  find_program(EDUORT_PROTOC protoc REQUIRED)
endif()

# ---- Generate C++ from vendored onnx.proto ----------------------------------
# LEARNER: protoc reads the schema and emits onnx.pb.h / onnx.pb.cc with classes
# like onnx::ModelProto. Those files live in the *build* tree only.

set(EDUORT_ONNX_PROTO_FILE "${PROJECT_SOURCE_DIR}/third_party/onnx/onnx.proto")
if(NOT EXISTS "${EDUORT_ONNX_PROTO_FILE}")
  message(FATAL_ERROR "ONNX schema missing: ${EDUORT_ONNX_PROTO_FILE}")
endif()

set(EDUORT_PROTO_GEN_DIR "${CMAKE_BINARY_DIR}/generated/onnx")
file(MAKE_DIRECTORY "${EDUORT_PROTO_GEN_DIR}")

set(EDUORT_ONNX_PB_CC "${EDUORT_PROTO_GEN_DIR}/onnx.pb.cc")
set(EDUORT_ONNX_PB_H  "${EDUORT_PROTO_GEN_DIR}/onnx.pb.h")

add_custom_command(
  OUTPUT "${EDUORT_ONNX_PB_CC}" "${EDUORT_ONNX_PB_H}"
  COMMAND ${EDUORT_PROTOC}
          --cpp_out=${EDUORT_PROTO_GEN_DIR}
          -I ${PROJECT_SOURCE_DIR}/third_party/onnx
          ${EDUORT_ONNX_PROTO_FILE}
  DEPENDS "${EDUORT_ONNX_PROTO_FILE}"
  COMMENT "Generating C++ from third_party/onnx/onnx.proto (ONNX v1.14.1 pin)"
  VERBATIM
)

add_library(eduort_onnx_proto STATIC
  "${EDUORT_ONNX_PB_CC}"
  "${EDUORT_ONNX_PB_H}"
)
add_library(eduort::onnx_proto ALIAS eduort_onnx_proto)

target_link_libraries(eduort_onnx_proto PUBLIC protobuf::libprotobuf)
target_include_directories(eduort_onnx_proto PUBLIC
  "$<BUILD_INTERFACE:${EDUORT_PROTO_GEN_DIR}>"
)
target_compile_features(eduort_onnx_proto PUBLIC cxx_std_17)

# LEARNER: Generated code trips -Wall/-Werror; it is not ours to fix.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(eduort_onnx_proto PRIVATE -w)
elseif(MSVC)
  target_compile_options(eduort_onnx_proto PRIVATE /w)
endif()

message(STATUS "  ONNX schema      : third_party/onnx/onnx.proto (pin v1.14.1)")
message(STATUS "  Generated C++    : ${EDUORT_PROTO_GEN_DIR}")
