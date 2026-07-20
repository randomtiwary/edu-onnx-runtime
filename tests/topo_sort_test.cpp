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

StatusOr<Tensor> TinyF32() {
  return Tensor::Create(DataType::kFloat32, TensorShape({1}));
}

TEST(ValidateTest, AddFixtureIsStructuralOk) {
  StatusOr<Graph> g = LoadGraphFromFile(Td("models/add_two_nodes.onnx"));
  ASSERT_TRUE(g.ok()) << g.status().ToString();
  Status st = ValidateStructure(*g);
  EXPECT_TRUE(st.ok()) << st.ToString();
  EXPECT_EQ(st.code(), ErrorCode::kOk);
}

TEST(ValidateTest, DanglingInputFails) {
  Node n;
  n.name = "bad";
  n.op_type = "Identity";
  n.inputs = {"missing"};
  n.outputs = {"Y"};
  Graph g = MakeNodeGraph({n}, /*inputs=*/{}, /*outputs=*/{"Y"});
  Status st = ValidateStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), ErrorCode::kModelLoad);
  EXPECT_NE(st.message().find("undefined"), std::string::npos);
}

TEST(ValidateTest, EmptyOptionalInputOk) {
  Node n;
  n.op_type = "Gemm";
  n.inputs = {"X", "W", ""};  // optional C omitted
  n.outputs = {"Y"};
  StatusOr<Tensor> w = TinyF32();
  ASSERT_TRUE(w.ok());
  Graph g = MakeNodeGraph({n}, {"X", "W"}, {"Y"});
  g.initializers.emplace("W", std::move(w).value());
  Status st = ValidateStructure(g);
  EXPECT_TRUE(st.ok()) << st.ToString();
}

TEST(ValidateTest, DuplicateProducerFails) {
  Node a;
  a.op_type = "Identity";
  a.inputs = {"X"};
  a.outputs = {"Y"};
  Node b;
  b.op_type = "Identity";
  b.inputs = {"X"};
  b.outputs = {"Y"};
  Graph g = MakeNodeGraph({a, b}, {"X"}, {"Y"});
  Status st = ValidateStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), ErrorCode::kModelLoad);
  EXPECT_NE(st.message().find("multiple nodes"), std::string::npos);
}

TEST(ValidateTest, UnknownOpTypeStillOk) {
  Node n;
  n.op_type = "TotallyFakeOp";
  n.inputs = {"X"};
  n.outputs = {"Y"};
  Graph g = MakeNodeGraph({n}, {"X"}, {"Y"});
  EXPECT_TRUE(ValidateStructure(g).ok());
}

TEST(ValidateTest, PureGraphInputRedefinitionFails) {
  Node n;
  n.op_type = "Identity";
  n.inputs = {"X"};
  n.outputs = {"X"};  // redefines pure feed
  Graph g = MakeNodeGraph({n}, {"X"}, {"X"});
  Status st = ValidateStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), ErrorCode::kModelLoad);
  EXPECT_NE(st.message().find("redefines"), std::string::npos);
}

TEST(ValidateTest, InitializerOnlyRedefinitionFails) {
  Node n;
  n.op_type = "Identity";
  n.inputs = {"W"};
  n.outputs = {"W"};  // redefines initializer-only name
  StatusOr<Tensor> w = TinyF32();
  ASSERT_TRUE(w.ok());
  Graph g = MakeNodeGraph({n}, /*inputs=*/{}, /*outputs=*/{"W"});
  g.initializers.emplace("W", std::move(w).value());
  Status st = ValidateStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), ErrorCode::kModelLoad);
  EXPECT_NE(st.message().find("redefines"), std::string::npos);
}

TEST(ValidateTest, InitializerPlusInputRedefinitionFails) {
  // W is both graph input and initializer (ONNX default-weight pattern).
  // A node must still not write W.
  Node n;
  n.op_type = "Identity";
  n.inputs = {"X"};
  n.outputs = {"W"};
  StatusOr<Tensor> w = TinyF32();
  ASSERT_TRUE(w.ok());
  Graph g = MakeNodeGraph({n}, {"X", "W"}, {"W"});
  g.initializers.emplace("W", std::move(w).value());
  Status st = ValidateStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), ErrorCode::kModelLoad);
  EXPECT_NE(st.message().find("redefines"), std::string::npos);
}

TEST(ValidateTest, EmptyOutputNameFails) {
  Node n;
  n.op_type = "Identity";
  n.inputs = {"X"};
  n.outputs = {""};
  Graph g = MakeNodeGraph({n}, {"X"}, {"Y"});
  Status st = ValidateStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), ErrorCode::kModelLoad);
  EXPECT_NE(st.message().find("empty output"), std::string::npos);
}

TEST(ValidateTest, EmptyOpTypeFails) {
  Node n;
  n.op_type = "";
  n.inputs = {"X"};
  n.outputs = {"Y"};
  Graph g = MakeNodeGraph({n}, {"X"}, {"Y"});
  Status st = ValidateStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), ErrorCode::kModelLoad);
  EXPECT_NE(st.message().find("empty op_type"), std::string::npos);
}

TEST(TopoSortTest, OrdersProducerBeforeConsumer) {
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/add_two_nodes.onnx"));
  ASSERT_TRUE(g_or.ok());
  Graph g = std::move(g_or).value();
  Status st = PrepareGraphStructure(g);
  ASSERT_TRUE(st.ok()) << st.ToString();
  ASSERT_EQ(g.topo_order.size(), 2u);
  EXPECT_EQ(g.nodes[g.topo_order[0]].name, "add1");
  EXPECT_EQ(g.nodes[g.topo_order[1]].name, "add2");
}

TEST(TopoSortTest, EmptyGraph) {
  Graph g;
  g.opset_version = 13;
  StatusOr<std::vector<std::size_t>> order = ComputeTopoOrder(g);
  ASSERT_TRUE(order.ok());
  EXPECT_TRUE(order->empty());
}

TEST(TopoSortTest, CycleDetected) {
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
  Graph g = MakeNodeGraph({a, b}, /*inputs=*/{}, /*outputs=*/{"T1"});
  ASSERT_TRUE(ValidateStructure(g).ok()) << ValidateStructure(g).ToString();
  StatusOr<std::vector<std::size_t>> order = ComputeTopoOrder(g);
  ASSERT_FALSE(order.ok());
  EXPECT_EQ(order.status().code(), ErrorCode::kModelLoad);
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
  StatusOr<std::vector<std::size_t>> order = ComputeTopoOrder(g);
  ASSERT_TRUE(order.ok()) << order.status().ToString();
  ASSERT_EQ(order->size(), 2u);
  std::vector<std::size_t> sorted = *order;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(sorted[0], 0u);
  EXPECT_EQ(sorted[1], 1u);
}

TEST(TopoSortTest, SelfLoopFails) {
  Node n;
  n.name = "loop";
  n.op_type = "Identity";
  n.inputs = {"Y"};
  n.outputs = {"Y"};
  // LEARNER: Y is only a node output (not a seed), so ValidateStructure is OK;
  // Kahn still rejects the A→A self-loop on value Y.
  Graph g = MakeNodeGraph({n}, /*inputs=*/{}, /*outputs=*/{"Y"});
  ASSERT_TRUE(ValidateStructure(g).ok()) << ValidateStructure(g).ToString();
  StatusOr<std::vector<std::size_t>> order = ComputeTopoOrder(g);
  ASSERT_FALSE(order.ok());
  EXPECT_EQ(order.status().code(), ErrorCode::kModelLoad);
  EXPECT_NE(order.status().message().find("self-loop"), std::string::npos);
}

TEST(TopoSortTest, MultiInputSameProducerCountedOnce) {
  // A produces T; B consumes T twice → only one edge A→B (indegree 1).
  Node a;
  a.name = "A";
  a.op_type = "Identity";
  a.inputs = {"X"};
  a.outputs = {"T"};
  Node b;
  b.name = "B";
  b.op_type = "Add";  // fake; structure only
  b.inputs = {"T", "T"};
  b.outputs = {"Y"};
  Graph g = MakeNodeGraph({a, b}, {"X"}, {"Y"});
  StatusOr<std::vector<std::size_t>> order = ComputeTopoOrder(g);
  ASSERT_TRUE(order.ok()) << order.status().ToString();
  ASSERT_EQ(order->size(), 2u);
  EXPECT_EQ(order->at(0), 0u);  // A first
  EXPECT_EQ(order->at(1), 1u);  // B second
}

TEST(TopoSortTest, StandaloneSsaEnforced) {
  // ComputeTopoOrder without ValidateStructure still rejects multi-producer.
  Node a;
  a.op_type = "Identity";
  a.inputs = {"X"};
  a.outputs = {"Y"};
  Node b;
  b.op_type = "Identity";
  b.inputs = {"X"};
  b.outputs = {"Y"};
  Graph g = MakeNodeGraph({a, b}, {"X"}, {"Y"});
  StatusOr<std::vector<std::size_t>> order = ComputeTopoOrder(g);
  ASSERT_FALSE(order.ok());
  EXPECT_EQ(order.status().code(), ErrorCode::kModelLoad);
  EXPECT_NE(order.status().message().find("multiple nodes"), std::string::npos);
}

TEST(PrepareGraphTest, ConstIdentity) {
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/const_identity.onnx"));
  ASSERT_TRUE(g_or.ok());
  Graph g = std::move(g_or).value();
  ASSERT_TRUE(PrepareGraphStructure(g).ok());
  ASSERT_EQ(g.topo_order.size(), 2u);
  EXPECT_EQ(g.nodes[g.topo_order[0]].op_type, "Constant");
  EXPECT_EQ(g.nodes[g.topo_order[1]].op_type, "Identity");
}

TEST(PrepareGraphTest, ClearsStaleTopoOrderOnFailure) {
  // First prepare succeeds.
  StatusOr<Graph> g_or = LoadGraphFromFile(Td("models/add_two_nodes.onnx"));
  ASSERT_TRUE(g_or.ok());
  Graph g = std::move(g_or).value();
  ASSERT_TRUE(PrepareGraphStructure(g).ok());
  ASSERT_EQ(g.topo_order.size(), 2u);

  // Mutate into a cycle; prepare must fail and clear topo_order.
  g.nodes.clear();
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
  g.nodes = {a, b};
  g.graph_inputs.clear();
  g.initializers.clear();
  g.graph_outputs = {"T1"};

  Status st = PrepareGraphStructure(g);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), ErrorCode::kModelLoad);
  EXPECT_TRUE(g.topo_order.empty())
      << "failed Prepare must not leave a stale topo_order";
}

}  // namespace
}  // namespace eduort
