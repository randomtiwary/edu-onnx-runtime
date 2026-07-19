// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Tests for ModelProto → Graph IR mapping (PR3b).

#include "eduort/onnx_loader.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace eduort {
namespace {

#ifndef EDUORT_TESTDATA_DIR
#error "EDUORT_TESTDATA_DIR must be defined"
#endif

std::string Td(const char* rel) {
  return std::string(EDUORT_TESTDATA_DIR) + "/" + rel;
}

TEST(LoaderTest, LoadsAddTwoNodesGraph) {
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/add_two_nodes.onnx"));
  ASSERT_TRUE(g_or.ok()) << g_or.status().ToString();
  const Graph& g = g_or.value();

  EXPECT_EQ(g.name, "identity_add_graph");
  EXPECT_EQ(g.ir_version, 8);
  EXPECT_EQ(g.opset_version, 13);

  // Domain ai.onnx on first node must canonicalize to "".
  ASSERT_EQ(g.nodes.size(), 2u);
  EXPECT_EQ(g.nodes[0].op_type, "Add");
  EXPECT_EQ(g.nodes[0].domain, "");
  EXPECT_EQ(g.nodes[0].name, "add1");
  ASSERT_EQ(g.nodes[0].inputs.size(), 2u);
  EXPECT_EQ(g.nodes[0].inputs[0], "X");
  EXPECT_EQ(g.nodes[0].inputs[1], "W");
  EXPECT_EQ(g.nodes[0].outputs[0], "T");

  EXPECT_EQ(g.nodes[1].op_type, "Add");
  EXPECT_EQ(g.nodes[1].inputs[0], "T");
  EXPECT_EQ(g.nodes[1].inputs[1], "B");

  // Initializers W and B
  ASSERT_EQ(g.initializers.count("W"), 1u);
  ASSERT_EQ(g.initializers.count("B"), 1u);
  EXPECT_EQ(g.initializers.at("W").shape().NumElements(), 2u);
  EXPECT_FLOAT_EQ(g.initializers.at("W").data_f32()[0], 10.f);
  EXPECT_FLOAT_EQ(g.initializers.at("B").data_f32()[1], 2.f);

  // Graph I/O
  ASSERT_EQ(g.graph_inputs.size(), 2u);
  EXPECT_EQ(g.graph_inputs[0], "X");
  EXPECT_EQ(g.graph_inputs[1], "W");
  ASSERT_EQ(g.graph_outputs.size(), 1u);
  EXPECT_EQ(g.graph_outputs[0], "Y");

  // Shapes from ValueInfo
  ASSERT_EQ(g.value_shapes.count("X"), 1u);
  EXPECT_EQ(g.value_shapes.at("X").Dims(), (std::vector<uint64_t>{1, 2}));
}

TEST(LoaderTest, ConstantAttributeTensor) {
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/const_identity.onnx"));
  ASSERT_TRUE(g_or.ok()) << g_or.status().ToString();
  const Graph& g = g_or.value();
  EXPECT_EQ(g.opset_version, 11);
  ASSERT_GE(g.nodes.size(), 1u);
  EXPECT_EQ(g.nodes[0].op_type, "Constant");
  ASSERT_EQ(g.nodes[0].attributes.size(), 1u);
  EXPECT_EQ(g.nodes[0].attributes[0].name, "value");
  EXPECT_EQ(g.nodes[0].attributes[0].kind, Attribute::Kind::kTensor);
  ASSERT_TRUE(g.nodes[0].attributes[0].tensor.has_value());
  EXPECT_FLOAT_EQ(g.nodes[0].attributes[0].tensor->data_f32()[0], 3.f);
}

TEST(LoaderTest, RejectsUnsupportedDomain) {
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/bad_domain.onnx"));
  ASSERT_FALSE(g_or.ok());
  EXPECT_EQ(g_or.status().code(), ErrorCode::kModelLoad);
  EXPECT_NE(g_or.status().message().find("domain"), std::string::npos);
}

TEST(LoaderTest, RejectsOpsetOutOfRange) {
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/bad_opset.onnx"));
  ASSERT_FALSE(g_or.ok());
  EXPECT_EQ(g_or.status().code(), ErrorCode::kModelLoad);
  EXPECT_NE(g_or.status().message().find("opset"), std::string::npos);
}

TEST(LoaderTest, MinimalSmokeStillLoads) {
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/minimal_smoke.onnx"));
  ASSERT_TRUE(g_or.ok()) << g_or.status().ToString();
  EXPECT_EQ(g_or->name, "smoke_graph");
  EXPECT_EQ(g_or->opset_version, 13);
  EXPECT_TRUE(g_or->nodes.empty());
}

TEST(CanonicalizeDomainTest, AiOnnxAndEmpty) {
  EXPECT_EQ(CanonicalizeDomain(""), "");
  EXPECT_EQ(CanonicalizeDomain("ai.onnx"), "");
  EXPECT_EQ(CanonicalizeDomain("ai.onnx.ml"), "ai.onnx.ml");
}

}  // namespace
}  // namespace eduort
