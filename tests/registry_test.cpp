// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Registry lookup + Identity kernel smoke (PR5).

#include "eduort/cpu_provider.h"
#include "eduort/kernel.h"
#include "eduort/planner.h"
#include "eduort/registry.h"
#include "eduort/topo_sort.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

namespace eduort {
namespace {

Graph MakeIdentityGraph() {
  Graph g;
  g.name = "id";
  g.opset_version = 13;
  g.graph_inputs = {"X"};
  g.graph_outputs = {"Y"};
  Node n;
  n.name = "id0";
  n.op_type = "Identity";
  n.domain = "";
  n.inputs = {"X"};
  n.outputs = {"Y"};
  g.nodes.push_back(std::move(n));
  EXPECT_TRUE(PrepareGraphStructure(g).ok());
  return g;
}

TEST(RegistryTest, MvpListContainsIdentity) {
  const auto& ops = MvpOpTypeList();
  EXPECT_NE(std::find(ops.begin(), ops.end(), "Identity"), ops.end());
}

TEST(RegistryTest, LookupPicksHighestSinceVersion) {
  KernelRegistry reg;
  int called = 0;
  KernelRegistration old_r;
  old_r.domain = "";
  old_r.op_type = "Identity";
  old_r.since_version = 1;
  old_r.ep_name = "CPU";
  old_r.factory = [&](const Node&) -> StatusOr<std::unique_ptr<IKernel>> {
    called = 1;
    return Status::Error(ErrorCode::kRuntime, "old");
  };
  reg.Register(std::move(old_r));

  KernelRegistration new_r;
  new_r.domain = "";
  new_r.op_type = "Identity";
  new_r.since_version = 10;
  new_r.ep_name = "CPU";
  new_r.factory = [&](const Node&) -> StatusOr<std::unique_ptr<IKernel>> {
    called = 10;
    // Return a real Identity via RegisterCpuKernels path — just mark version.
    class Dummy : public IKernel {
     public:
      Status Compute(OpKernelContext&) override { return Status::OK(); }
    };
    return std::unique_ptr<IKernel>(new Dummy());
  };
  reg.Register(std::move(new_r));

  Node n;
  n.op_type = "Identity";
  n.domain = "";
  auto k = reg.Create(n, /*model_opset=*/13, "CPU");
  ASSERT_TRUE(k.ok()) << k.status().ToString();
  EXPECT_EQ(called, 10);
}

TEST(RegistryTest, RejectsSinceAboveModelOpset) {
  KernelRegistry reg;
  KernelRegistration r;
  r.domain = "";
  r.op_type = "Identity";
  r.since_version = 14;  // only valid for opset ≥ 14
  r.ep_name = "CPU";
  r.factory = [](const Node&) -> StatusOr<std::unique_ptr<IKernel>> {
    return Status::Error(ErrorCode::kRuntime, "should not call");
  };
  reg.Register(std::move(r));

  Node n;
  n.op_type = "Identity";
  auto k = reg.Create(n, /*model_opset=*/13, "CPU");
  EXPECT_FALSE(k.ok());
  EXPECT_EQ(k.status().code(), ErrorCode::kUnsupportedOperator);
}

TEST(RegistryTest, CanonicalizesAiOnnxDomain) {
  KernelRegistry reg;
  RegisterCpuKernels(reg);
  Node n;
  n.op_type = "Identity";
  n.domain = "ai.onnx";  // should match registration on ""
  EXPECT_TRUE(reg.HasKernel(n, 13, "CPU"));
  auto k = reg.Create(n, 13, "CPU");
  EXPECT_TRUE(k.ok()) << k.status().ToString();
}

TEST(IdentityKernelTest, CopiesTensor) {
  KernelRegistry reg;
  RegisterCpuKernels(reg);
  CpuExecutionProvider cpu(&reg);

  Node n;
  n.op_type = "Identity";
  n.inputs = {"X"};
  n.outputs = {"Y"};

  StatusOr<std::unique_ptr<IKernel>> k = cpu.CreateKernel(n, 13);
  ASSERT_TRUE(k.ok()) << k.status().ToString();

  StatusOr<Tensor> x = Tensor::Create(DataType::kFloat32, TensorShape({2, 2}));
  ASSERT_TRUE(x.ok());
  float* p = x->mutable_data_f32();
  p[0] = 1.f;
  p[1] = 2.f;
  p[2] = 3.f;
  p[3] = 4.f;

  std::unordered_map<std::string, Tensor> values;
  values.emplace("X", std::move(x).value());

  OpKernelContext ctx(n, values, cpu.GetAllocator());
  Status st = k.value()->Compute(ctx);
  ASSERT_TRUE(st.ok()) << st.ToString();
  ASSERT_EQ(values.count("Y"), 1u);
  const Tensor& y = values.at("Y");
  const Tensor& x_ref = values.at("X");
  EXPECT_EQ(y.nbytes(), 4 * sizeof(float));
  EXPECT_FLOAT_EQ(y.data_f32()[3], 4.f);
  // Distinct buffer (copy, not alias)
  EXPECT_NE(y.data(), x_ref.data());
}

TEST(PlannerTest, BindsIdentityOnCpu) {
  Graph g = MakeIdentityGraph();
  KernelRegistry reg;
  RegisterCpuKernels(reg);
  CpuExecutionProvider cpu(&reg);
  std::vector<IExecutionProvider*> eps = {&cpu};

  StatusOr<std::vector<NodeBinding>> bindings = BindKernels(g, eps);
  ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();
  ASSERT_EQ(bindings->size(), 1u);
  EXPECT_EQ(bindings->at(0).ep_name, "CPU");
  EXPECT_EQ(bindings->at(0).node_index, 0u);
  EXPECT_NE(bindings->at(0).kernel, nullptr);
}

TEST(PlannerTest, UnsupportedOpFails) {
  Graph g;
  g.opset_version = 13;
  g.graph_inputs = {"X"};
  g.graph_outputs = {"Y"};
  Node n;
  n.op_type = "Conv";
  n.inputs = {"X"};
  n.outputs = {"Y"};
  g.nodes.push_back(std::move(n));
  ASSERT_TRUE(PrepareGraphStructure(g).ok());

  KernelRegistry reg;
  RegisterCpuKernels(reg);
  CpuExecutionProvider cpu(&reg);
  std::vector<IExecutionProvider*> eps = {&cpu};
  auto bindings = BindKernels(g, eps);
  EXPECT_FALSE(bindings.ok());
  EXPECT_EQ(bindings.status().code(), ErrorCode::kUnsupportedOperator);
}

TEST(PlannerTest, RequiresTopoOrder) {
  Graph g;
  g.opset_version = 13;
  Node n;
  n.op_type = "Identity";
  n.inputs = {"X"};
  n.outputs = {"Y"};
  g.nodes.push_back(std::move(n));
  g.graph_inputs = {"X"};
  g.graph_outputs = {"Y"};
  // no PrepareGraphStructure → empty topo_order
  KernelRegistry reg;
  RegisterCpuKernels(reg);
  CpuExecutionProvider cpu(&reg);
  std::vector<IExecutionProvider*> eps = {&cpu};
  auto bindings = BindKernels(g, eps);
  EXPECT_FALSE(bindings.ok());
  EXPECT_EQ(bindings.status().code(), ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace eduort
