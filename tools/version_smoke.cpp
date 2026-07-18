// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: Smallest possible "app" that links against eduort.
// Run after building:
//   ./build/eduort_version_smoke
// Expected: prints version + build info, exits 0.

#include <cstdio>

#include "eduort/version.h"

int main() {
  std::printf("%s\n", eduort::BuildInfo());
  std::printf("VersionString() = %s\n", eduort::VersionString());
  return 0;
}
