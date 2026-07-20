// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// Internal CPU Identity factory (not part of the public include/ tree).

#pragma once

#include <memory>

#include "eduort/graph.h"
#include "eduort/kernel.h"

namespace eduort {

std::unique_ptr<IKernel> CreateIdentityKernel(const Node& node);

}  // namespace eduort
