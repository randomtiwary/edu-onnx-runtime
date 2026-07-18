// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Version symbols are compiled into the static library so linkers
// pull a real object file even when the library is still almost empty.
// That proves the CMake target graph works before we add heavier code.

#include "eduort/version.h"

namespace eduort {

const char* VersionString() noexcept { return EDUORT_VERSION_STRING; }

const char* BuildInfo() noexcept {
  // NOTE: CUDA flag is baked at configure time via a compile definition.
  // When PR10 lands, EDUORT_ENABLE_CUDA may be ON; the string will reflect it.
#if defined(EDUORT_BUILT_WITH_CUDA)
  return "eduort " EDUORT_VERSION_STRING " (C++17, CUDA=ON)";
#else
  return "eduort " EDUORT_VERSION_STRING " (C++17, CUDA=OFF)";
#endif
}

}  // namespace eduort
