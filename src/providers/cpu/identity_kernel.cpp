// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Identity is the simplest op — y = x (element copy). First kernel to
// prove registry → EP → Compute wiring without numeric math.
// Spec: ONNX Operators.md — Identity

#include "eduort/kernel.h"
#include "eduort/macros.h"

#include <cstring>
#include <memory>

namespace eduort {
namespace {

class IdentityKernel final : public IKernel {
 public:
  Status Compute(OpKernelContext& ctx) override {
    EDUORT_ASSIGN_OR_RETURN(const Tensor* in, ctx.Input(0));
    EDUORT_ASSIGN_OR_RETURN(Tensor* out, ctx.Output(0, in->dtype(), in->shape()));
    if (in->nbytes() != out->nbytes()) {
      return Status::Error(ErrorCode::kRuntime,
                           "Identity: input/output byte size mismatch");
    }
    if (in->nbytes() > 0) {
      // LEARNER: MVP always copies (no buffer aliasing). Clear ownership story.
      std::memcpy(out->mutable_data(), in->data(), in->nbytes());
    }
    return Status::OK();
  }
};

}  // namespace

std::unique_ptr<IKernel> CreateIdentityKernel(const Node& /*node*/) {
  return std::make_unique<IdentityKernel>();
}

}  // namespace eduort
