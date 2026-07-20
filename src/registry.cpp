// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Registration is explicit (RegisterCpuKernels). Lookup picks the
// newest kernel whose since_version is still valid for the model's opset.

#include "eduort/registry.h"

#include "eduort/graph.h"

#include <limits>

namespace eduort {

void KernelRegistry::Register(KernelRegistration reg) {
  reg.domain = CanonicalizeDomain(reg.domain);
  regs_.push_back(std::move(reg));
}

std::size_t KernelRegistry::FindBest(const std::string& domain,
                                     const std::string& op_type,
                                     int64_t model_opset,
                                     const std::string& ep_name) const {
  const std::string dom = CanonicalizeDomain(domain);
  std::size_t best = std::numeric_limits<std::size_t>::max();
  int64_t best_since = -1;
  for (std::size_t i = 0; i < regs_.size(); ++i) {
    const KernelRegistration& r = regs_[i];
    if (r.domain != dom || r.op_type != op_type || r.ep_name != ep_name) {
      continue;
    }
    if (r.since_version > model_opset) {
      continue;
    }
    if (r.since_version >= best_since) {
      best_since = r.since_version;
      best = i;
    }
  }
  return best;
}

bool KernelRegistry::HasKernel(const Node& node, int64_t model_opset,
                               const std::string& ep_name) const {
  return FindBest(node.domain, node.op_type, model_opset, ep_name) !=
         std::numeric_limits<std::size_t>::max();
}

StatusOr<std::unique_ptr<IKernel>> KernelRegistry::Create(
    const Node& node, int64_t model_opset, const std::string& ep_name) const {
  const std::size_t idx =
      FindBest(node.domain, node.op_type, model_opset, ep_name);
  if (idx == std::numeric_limits<std::size_t>::max()) {
    return Status::Error(
        ErrorCode::kUnsupportedOperator,
        "no kernel for op '" + node.op_type + "' domain='" + node.domain +
            "' ep='" + ep_name + "' model_opset=" + std::to_string(model_opset));
  }
  return regs_[idx].factory(node);
}

const std::vector<std::string>& MvpOpTypeList() {
  // LEARNER: Documented allowlist (check_model_ops.py grows in lockstep).
  // Constant is folded later — still listed as a legal graph op.
  static const std::vector<std::string> kOps = {
      "Constant", "Identity", "Add",     "Mul",     "MatMul", "Gemm",
      "Relu",     "Sigmoid",  "Softmax", "Reshape", "Flatten"};
  return kOps;
}

}  // namespace eduort
