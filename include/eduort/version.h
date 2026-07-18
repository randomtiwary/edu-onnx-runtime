// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Every C++ library needs a version surface so apps and tests can
// check "which eduort am I linked against?". We expose both compile-time
// macros (for #if checks) and a runtime string (for logging / CLI --version).
//
// This is intentionally tiny in PR1 — no ONNX, no Session yet.

#pragma once

// Semantic version for the educational runtime (see docs/milestones.md).
// Bump these when tagging releases (v0.1.0 after Session E2E, etc.).
#define EDUORT_VERSION_MAJOR 0
#define EDUORT_VERSION_MINOR 0
#define EDUORT_VERSION_PATCH 1

// Single string "0.0.1" — useful for logging.
#define EDUORT_VERSION_STRING "0.0.1"

namespace eduort {

// Returns EDUORT_VERSION_STRING. Defined in src/version.cpp so the symbol
// lives in the library binary (not only in headers).
const char* VersionString() noexcept;

// Returns a short human-readable build identity, e.g.
// "eduort 0.0.1 (C++17, CUDA=OFF)". Safe to print at process start.
const char* BuildInfo() noexcept;

}  // namespace eduort
