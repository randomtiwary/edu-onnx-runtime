// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: CPU EP + registration of host kernels. PR5 ships Identity only.

#include "eduort/cpu_provider.h"

#include "identity_kernel.h"

#include <utility>

namespace eduort {

CpuExecutionProvider::CpuExecutionProvider(const KernelRegistry* registry)
    : registry_(registry) {}

const char* CpuExecutionProvider::Name() const { return "CPU"; }

bool CpuExecutionProvider::CanProduceKernel(const Node& node,
                                            int64_t model_opset) const {
  if (registry_ == nullptr) {
    return false;
  }
  return registry_->HasKernel(node, model_opset, Name());
}

StatusOr<std::unique_ptr<IKernel>> CpuExecutionProvider::CreateKernel(
    const Node& node, int64_t model_opset) {
  if (registry_ == nullptr) {
    return Status::Error(ErrorCode::kRuntime,
                         "CpuExecutionProvider has null registry");
  }
  return registry_->Create(node, model_opset, Name());
}

IAllocator* CpuExecutionProvider::GetAllocator() {
  return DefaultCpuAllocator();
}

void RegisterCpuKernels(KernelRegistry& registry) {
  // LEARNER: since_version=1 → valid for any model_opset ≥ 1 (covers [11,17]).
  KernelRegistration reg;
  reg.domain = "";
  reg.op_type = "Identity";
  reg.since_version = 1;
  reg.ep_name = "CPU";
  reg.factory = [](const Node& node) -> StatusOr<std::unique_ptr<IKernel>> {
    return CreateIdentityKernel(node);
  };
  registry.Register(std::move(reg));
}

}  // namespace eduort
