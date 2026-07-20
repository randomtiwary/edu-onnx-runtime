// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/kernel.h"

#include "eduort/macros.h"

#include <utility>

namespace eduort {

OpKernelContext::OpKernelContext(const Node& node,
                                 std::unordered_map<std::string, Tensor>& values,
                                 IAllocator* allocator)
    : node_(node), values_(values), allocator_(allocator) {}

StatusOr<const Tensor*> OpKernelContext::Input(std::size_t i) const {
  if (i >= node_.inputs.size()) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "Input index out of range for op " + node_.op_type);
  }
  const std::string& name = node_.inputs[i];
  if (name.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "optional input slot " + std::to_string(i) +
                             " is empty for op " + node_.op_type);
  }
  auto it = values_.find(name);
  if (it == values_.end()) {
    return Status::Error(ErrorCode::kRuntime,
                         "missing value '" + name + "' for input of " +
                             node_.op_type);
  }
  return &it->second;
}

StatusOr<Tensor*> OpKernelContext::Output(std::size_t i, DataType dtype,
                                          const TensorShape& shape) {
  if (i >= node_.outputs.size()) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "Output index out of range for op " + node_.op_type);
  }
  const std::string& name = node_.outputs[i];
  if (name.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "empty output name for op " + node_.op_type);
  }
  if (allocator_ == nullptr) {
    return Status::Error(ErrorCode::kRuntime, "OpKernelContext has null allocator");
  }

  // LEARNER: Allocate via Tensor::Create (uses host allocator today). Later
  // CUDA EP will pass a device allocator / device kind.
  EDUORT_ASSIGN_OR_RETURN(Tensor t,
                          Tensor::Create(dtype, shape, allocator_->device()));
  auto [it, inserted] = values_.insert_or_assign(name, std::move(t));
  (void)inserted;
  return &it->second;
}

const Attribute* OpKernelContext::GetAttr(std::string_view name) const {
  for (const Attribute& a : node_.attributes) {
    if (a.name == name) {
      return &a;
    }
  }
  return nullptr;
}

}  // namespace eduort
