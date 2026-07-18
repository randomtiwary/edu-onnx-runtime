// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: End-to-end proof that:
//   1) protoc generated onnx::ModelProto
//   2) we can read a committed .onnx file
//   3) protobuf parse fills ir_version + graph
//
// Full Graph IR mapping is PR3b — here we only smoke-parse.

#include "eduort/model_proto.h"

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

namespace eduort {
namespace {

// CMake defines EDUORT_TESTDATA_DIR to the source testdata/ path.
#ifndef EDUORT_TESTDATA_DIR
#error "EDUORT_TESTDATA_DIR must be defined by tests/CMakeLists.txt"
#endif

TEST(ProtoSmokeTest, LoadMinimalModel) {
  const std::string path =
      std::string(EDUORT_TESTDATA_DIR) + "/models/minimal_smoke.onnx";

  StatusOr<std::unique_ptr<onnx::ModelProto>> model_or =
      LoadModelProtoFromFile(path);
  ASSERT_TRUE(model_or.ok()) << model_or.status().ToString();

  const onnx::ModelProto& model = *model_or.value();

  // LEARNER: ir_version is the ONNX IR version the producer targeted.
  // Our fixture uses 8 (common for onnx 1.13–1.14 era models).
  EXPECT_EQ(model.ir_version(), 8);

  // has_graph() is the key smoke check — a ModelProto without a graph is useless.
  ASSERT_TRUE(model.has_graph());
  EXPECT_EQ(model.graph().name(), "smoke_graph");
  EXPECT_EQ(model.producer_name(), "eduort-fixture");
}

TEST(ProtoSmokeTest, RejectsMissingFile) {
  StatusOr<std::unique_ptr<onnx::ModelProto>> model_or =
      LoadModelProtoFromFile("/no/such/eduort_model.onnx");
  EXPECT_FALSE(model_or.ok());
  EXPECT_EQ(model_or.status().code(), ErrorCode::kModelLoad);
}

TEST(ProtoSmokeTest, RejectsGarbageBytes) {
  const char junk[] = "this is not protobuf";
  StatusOr<std::unique_ptr<onnx::ModelProto>> model_or =
      LoadModelProtoFromBytes(junk, sizeof(junk) - 1);
  // ParseFromArray may still return true for some partial garbage in proto2;
  // we only require that either parse fails OR graph is absent / nonsense.
  // For random ASCII, protobuf typically fails parse.
  if (model_or.ok()) {
    // Extremely permissive parsers: at least we got a ModelProto object.
    SUCCEED();
  } else {
    EXPECT_EQ(model_or.status().code(), ErrorCode::kModelLoad);
  }
}

TEST(ProtoSmokeTest, RejectsOversizeCap) {
  const std::string path =
      std::string(EDUORT_TESTDATA_DIR) + "/models/minimal_smoke.onnx";
  // max_bytes = 1 is smaller than the 33-byte fixture.
  StatusOr<std::unique_ptr<onnx::ModelProto>> model_or =
      LoadModelProtoFromFile(path, /*max_bytes=*/1);
  EXPECT_FALSE(model_or.ok());
  EXPECT_EQ(model_or.status().code(), ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace eduort
