// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: A *kernel* is the executable implementation of one ONNX operator
// on one device (CPU today, CUDA later). Session does not switch on op_type
// strings in a giant if/else — it asks the registry/EP for an IKernel, then
// calls Compute().
//
// Spec: design § Kernel dispatch; ONNX Operators.md (per-op semantics)

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "eduort/allocator.h"
#include "eduort/graph.h"
#include "eduort/status.h"
#include "eduort/tensor.h"

namespace eduort {

// Forward — full Session arrives in PR9a; context only needs a value map.
class OpKernelContext;

// ---------------------------------------------------------------------------
// IKernel — one bound instance of an op for a specific Node
// ---------------------------------------------------------------------------
class IKernel {
 public:
  virtual ~IKernel() = default;
  virtual Status Compute(OpKernelContext& ctx) = 0;
};

// ---------------------------------------------------------------------------
// OpKernelContext — what a kernel sees during Compute
// ---------------------------------------------------------------------------
// LEARNER: Inputs are **const** (seed immutability contract). Outputs are
// fresh buffers allocated through the EP allocator and inserted into the
// runtime name→Tensor map that Session owns.
class OpKernelContext {
 public:
  OpKernelContext(const Node& node,
                  std::unordered_map<std::string, Tensor>& values,
                  IAllocator* allocator);

  const Node& node() const noexcept { return node_; }

  std::size_t NumInputs() const noexcept { return node_.inputs.size(); }
  std::size_t NumOutputs() const noexcept { return node_.outputs.size(); }

  // Read-only input tensor (already present in the value map).
  // Empty optional input name → error (caller must skip optional slots).
  StatusOr<const Tensor*> Input(std::size_t i) const;

  // Allocate a new output tensor and register it under node.outputs[i].
  StatusOr<Tensor*> Output(std::size_t i, DataType dtype,
                           const TensorShape& shape);

  // Look up an attribute by name; nullptr if missing.
  const Attribute* GetAttr(std::string_view name) const;

 private:
  const Node& node_;
  std::unordered_map<std::string, Tensor>& values_;
  IAllocator* allocator_;  // not owned
};

}  // namespace eduort
