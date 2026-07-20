// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: An *Execution Provider* (EP) is a backend that can create kernels
// and allocate memory for a device class. Production ONNX Runtime has many
// EPs (CUDA, TensorRT, …). eduort starts with CPU only; CUDA arrives in PR10.
//
// Spec: design § Execution provider interface; onnxruntime.ai EP docs

#pragma once

#include <memory>
#include <string>

#include "eduort/allocator.h"
#include "eduort/graph.h"
#include "eduort/kernel.h"
#include "eduort/status.h"

namespace eduort {

class IExecutionProvider {
 public:
  virtual ~IExecutionProvider() = default;

  // Stable name used in logs and registry keys: "CPU", "CUDA", …
  virtual const char* Name() const = 0;

  // Cheap query: can this EP create a kernel for node under model_opset?
  virtual bool CanProduceKernel(const Node& node, int64_t model_opset) const = 0;

  // Build a kernel instance for this node (or error).
  virtual StatusOr<std::unique_ptr<IKernel>> CreateKernel(
      const Node& node, int64_t model_opset) = 0;

  // Device allocator for Output() allocations on this EP.
  virtual IAllocator* GetAllocator() = 0;
};

}  // namespace eduort
