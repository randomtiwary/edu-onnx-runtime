// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/file_utils.h"

#include <fstream>
#include <string>

namespace eduort {

StatusOr<std::vector<char>> ReadFileLimited(const std::string& path,
                                            std::size_t max_bytes) {
  if (path.empty()) {
    return Status::Error(ErrorCode::kInvalidArgument, "path is empty");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::Error(ErrorCode::kModelLoad,
                         "failed to open file: " + path);
  }

  in.seekg(0, std::ios::end);
  const std::streamoff end = in.tellg();
  if (end < 0) {
    return Status::Error(ErrorCode::kModelLoad,
                         "failed to stat file: " + path);
  }
  const auto file_size = static_cast<std::uint64_t>(end);
  if (file_size > max_bytes) {
    return Status::Error(
        ErrorCode::kInvalidArgument,
        "file exceeds max_bytes (" + std::to_string(file_size) + " > " +
            std::to_string(max_bytes) + "): " + path);
  }

  in.seekg(0, std::ios::beg);
  std::vector<char> buf(static_cast<std::size_t>(file_size));
  if (file_size > 0) {
    in.read(buf.data(), static_cast<std::streamsize>(file_size));
    if (!in) {
      return Status::Error(ErrorCode::kModelLoad,
                           "failed to read file: " + path);
    }
  }
  return buf;
}

}  // namespace eduort
