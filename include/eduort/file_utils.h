// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Small I/O helpers shared by loaders (ONNX models today; goldens /
// plan dumps later). Keep file-system details out of domain code like
// model_proto.cpp so each layer stays focused.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "eduort/status.h"

namespace eduort {

// Read an entire file into memory, failing if the file is larger than
// `max_bytes` (protects student machines / untrusted inputs).
//
// Errors:
//   kInvalidArgument — empty path, or file larger than max_bytes
//   kModelLoad       — open / seek / read failure (used by model loading paths)
StatusOr<std::vector<char>> ReadFileLimited(const std::string& path,
                                            std::size_t max_bytes);

}  // namespace eduort
