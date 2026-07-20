// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: CpuExecutionProvider always exists. It creates host kernels and
// uses DefaultCpuAllocator().

#pragma once

#include "eduort/execution_provider.h"
#include "eduort/registry.h"

namespace eduort {

class CpuExecutionProvider final : public IExecutionProvider {
 public:
  explicit CpuExecutionProvider(const KernelRegistry* registry);

  const char* Name() const override;
  bool CanProduceKernel(const Node& node, int64_t model_opset) const override;
  StatusOr<std::unique_ptr<IKernel>> CreateKernel(
      const Node& node, int64_t model_opset) override;
  IAllocator* GetAllocator() override;

 private:
  const KernelRegistry* registry_;  // not owned; must outlive provider
};

// Register all CPU kernels into registry (PR5: Identity only).
void RegisterCpuKernels(KernelRegistry& registry);

}  // namespace eduort
