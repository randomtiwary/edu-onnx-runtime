// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Structural validation + Kahn topo sort (PR4).

#include "eduort/graph.h"
#include "eduort/onnx_loader.h"
#include "eduort/topo_sort.h"
#include "eduort/validate.h"

#include <algorithm>
#include <string>
#include <unordered_map>
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

Graph MakeNodeGraph(std::vector<Node> nodes,
                    std::vector<std::string> inputs = {},
                    std::vector<std::string> outputs = {},
                    std::unordered_map<std::string, Tensor> inits = {}) {
  Graph g;
  g.name = "test";
  g.opset_version = 13;
  g.nodes = std::move(nodes);
  g.graph_inputs = std::move(inputs);
  g.graph_outputs = std::move(outputs);
  g.initializers = std::move(inits);
  return g;
}

TEST(ValidateTest, AddFixtureIsStructuralOk) {
  StatusOr<Graph> g = LoadGraphFromFile(Td("models/add_two_nodes.onnx"));
  ASSERT_TRUE(g.ok()) << g.status().ToString();
  EXPECT_TRUE(ValidateStructure(*g).ok()) << ValidateStructure(*g).ToString();
}

TEST(ValidateTest, DanglingInputFails) {
  Node n;
  n.name = "bad";
  n.op_type = "Identity";  // unknown ops are still allowed structurally
  n.inputs = {"missing"};
  n.outputs = {"Y"};
  Graph g = MakeNodeGraph({n}, /*inputs=*/{}, /*outputs=*/{"Y"});
  Status st = ValidateStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(st.message().find("undefined"), std::string::npos);
}

TEST(ValidateTest, EmptyOptionalInputOk) {
  Node n;
  n.op_type = "Gemm";
  n.inputs = {"X", "W", ""};  // optional C omitted
  n.outputs = {"Y"};
  StatusOr<Tensor> x =
      Tensor::Create(DataType::kFloat32, TensorShape({1, 1}));
  ASSERT_TRUE(x.ok());
  Graph g = MakeNodeGraph({n}, {"X", "W"}, {"Y"});
  g.initializers.emplace("W", std::move(x).value());
  // X is graph input without initializer
  EXPECT_TRUE(ValidateStructure(g).ok()) << ValidateStructure(g).ToString();
}

TEST(ValidateTest, DuplicateProducerFails) {
  Node a;
  a.op_type = "Identity";
  a.inputs = {"X"};
  a.outputs = {"Y"};
  Node b;
  b.op_type = "Identity";
  b.inputs = {"X"};
  b.outputs = {"Y"};  // same output name
  Graph g = MakeNodeGraph({a, b}, {"X"}, {"Y"});
  Status st = ValidateStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_NE(st.message().find("multiple nodes"), std::string::npos);
}

TEST(ValidateTest, UnknownOpTypeStillOk) {
  // LEARNER: PR4 must not reject unknown ops — kernels come later.
  Node n;
  n.op_type = "TotallyFakeOp";
  n.inputs = {"X"};
  n.outputs = {"Y"};
  Graph g = MakeNodeGraph({n}, {"X"}, {"Y"});
  EXPECT_TRUE(ValidateStructure(g).ok());
}

TEST(TopoSortTest, OrdersProducerBeforeConsumer) {
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/add_two_nodes.onnx"));
  ASSERT_TRUE(g_or.ok());
  Graph g = std::move(g_or).value();
  Status st = PrepareGraphStructure(g);
  ASSERT_TRUE(st.ok()) << st.ToString();
  ASSERT_EQ(g.topo_order.size(), 2u);
  // add1 produces T; add2 consumes T → add1 before add2
  EXPECT_EQ(g.nodes[static_cast<std::size_t>(g.topo_order[0])].name, "add1");
  EXPECT_EQ(g.nodes[static_cast<std::size_t>(g.topo_order[1])].name, "add2");
}

TEST(TopoSortTest, EmptyGraph) {
  Graph g;
  g.opset_version = 13;
  StatusOr<std::vector<int>> order = ComputeTopoOrder(g);
  ASSERT_TRUE(order.ok());
  EXPECT_TRUE(order->empty());
}

TEST(TopoSortTest, CycleDetected) {
  // A: in T2 out T1 ; B: in T1 out T2  → cycle
  Node a;
  a.name = "A";
  a.op_type = "Identity";
  a.inputs = {"T2"};
  a.outputs = {"T1"};
  Node b;
  b.name = "B";
  b.op_type = "Identity";
  b.inputs = {"T1"};
  b.outputs = {"T2"};
  // Seeds: neither T1 nor T2 is a graph input — both only node outputs.
  // Structural validate: T2 is produced by B, T1 by A — both defined as outputs.
  Graph g = MakeNodeGraph({a, b}, /*inputs=*/{}, /*outputs=*/{"T1"});
  ASSERT_TRUE(ValidateStructure(g).ok()) << ValidateStructure(g).ToString();
  StatusOr<std::vector<int>> order = ComputeTopoOrder(g);
  ASSERT_FALSE(order.ok());
  EXPECT_NE(order.status().message().find("cycle"), std::string::npos);
}

TEST(TopoSortTest, IndependentNodesAnyOrder) {
  Node a;
  a.name = "A";
  a.op_type = "Identity";
  a.inputs = {"X"};
  a.outputs = {"Y1"};
  Node b;
  b.name = "B";
  b.op_type = "Identity";
  b.inputs = {"X"};
  b.outputs = {"Y2"};
  Graph g = MakeNodeGraph({a, b}, {"X"}, {"Y1", "Y2"});
  StatusOr<std::vector<int>> order = ComputeTopoOrder(g);
  ASSERT_TRUE(order.ok()) << order.status().ToString();
  ASSERT_EQ(order->size(), 2u);
  // Both indices 0 and 1 present
  std::vector<int> sorted = *order;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(sorted[0], 0);
  EXPECT_EQ(sorted[1], 1);
}

TEST(PrepareGraphTest, ConstIdentity) {
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/const_identity.onnx"));
  ASSERT_TRUE(g_or.ok());
  Graph g = std::move(g_or).value();
  ASSERT_TRUE(PrepareGraphStructure(g).ok());
  ASSERT_EQ(g.topo_order.size(), 2u);
  EXPECT_EQ(g.nodes[static_cast<std::size_t>(g.topo_order[0])].op_type,
            "Constant");
  EXPECT_EQ(g.nodes[static_cast<std::size_t>(g.topo_order[1])].op_type,
            "Identity");
}

}  // namespace
}  // namespace eduort
