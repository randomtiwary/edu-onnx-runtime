// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: The kernel registry maps
//   (domain, op_type, ep_name, since_version) → factory
// Lookup for a model uses the highest since_version that is still
// ≤ model_opset (design Key Decision K14).
//
// Prefer explicit RegisterCpuKernels(registry) over static constructors
// (static init order across TUs is a classic footgun).

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "eduort/graph.h"
#include "eduort/kernel.h"
#include "eduort/status.h"

namespace eduort {

// Factory builds a kernel for a concrete Node (may capture EP state later).
using KernelFactory =
    std::function<StatusOr<std::unique_ptr<IKernel>>(const Node& node)>;

struct KernelRegistration {
  std::string domain;     // canonical "" for ai.onnx
  std::string op_type;    // e.g. "Identity"
  int64_t since_version;  // earliest opset this implementation supports
  std::string ep_name;    // "CPU", "CUDA", …
  KernelFactory factory;
};

class KernelRegistry {
 public:
  // domain is canonicalized (ai.onnx → "").
  void Register(KernelRegistration reg);

  // Highest since_version ≤ model_opset for (domain, op_type, ep_name).
  StatusOr<std::unique_ptr<IKernel>> Create(const Node& node,
                                            int64_t model_opset,
                                            const std::string& ep_name) const;

  // True if some registration could match (same lookup without building).
  bool HasKernel(const Node& node, int64_t model_opset,
                 const std::string& ep_name) const;

  std::size_t size() const noexcept { return regs_.size(); }

 private:
  // Returns index into regs_ or npos.
  std::size_t FindBest(const std::string& domain, const std::string& op_type,
                       int64_t model_opset, const std::string& ep_name) const;

  std::vector<KernelRegistration> regs_;
};

// Documented MVP op names (human list; registry is source of truth for bind).
const std::vector<std::string>& MvpOpTypeList();

}  // namespace eduort
