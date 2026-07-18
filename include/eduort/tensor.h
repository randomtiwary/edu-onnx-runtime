// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0
//
// LEARNER: A Tensor is a multi-dimensional array: shape + dtype + device +
// contiguous row-major buffer. ONNX models pass tensors between operators;
// the runtime's job is largely "move / transform tensors correctly".
//
// Layout (K12): row-major contiguous. For shape [2, 3] of float32:
//   index (i, j) → offset = (i * 3 + j) * sizeof(float)
//
// Spec: ONNX IR.md — Tensors; docs/design.md § Memory model

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "eduort/allocator.h"
#include "eduort/status.h"

namespace eduort {

// LEARNER: MVP dtypes only (K4 / non-goals). float32 for weights/activations;
// int64 for shapes, axes, and Reshape's shape input later.
enum class DataType {
  kFloat32 = 1,
  kInt64 = 2,
};

const char* DataTypeName(DataType dt) noexcept;

// Size of one element in bytes, or 0 if unknown.
std::size_t SizeOfDataType(DataType dt) noexcept;

// ---------------------------------------------------------------------------
// TensorShape — list of dimension sizes
// ---------------------------------------------------------------------------
// LEARNER: ONNX shapes are sequences of int64 dims. Rank = number of dims.
// A *scalar* has rank 0 (empty dims) and NumElements() == 1.
// A dim of 0 is legal (empty tensor); negative dims are invalid for concrete
// runtime shapes (symbolic dims are a future / non-goal topic).
class TensorShape {
 public:
  TensorShape() = default;
  explicit TensorShape(std::vector<int64_t> dims);

  int rank() const noexcept { return static_cast<int>(dims_.size()); }
  const std::vector<int64_t>& Dims() const noexcept { return dims_; }

  // Product of dims. Rank-0 → 1. Returns false on overflow or negative dim.
  StatusOr<int64_t> NumElementsChecked() const;
  // Convenience: aborts not; returns -1 on error (prefer NumElementsChecked).
  // Actually design says NumElements() const — we'll compute and for invalid
  // shapes Create already failed; NumElements assumes valid shape.
  int64_t NumElements() const;

  bool operator==(const TensorShape& other) const noexcept {
    return dims_ == other.dims_;
  }
  bool operator!=(const TensorShape& other) const noexcept {
    return !(*this == other);
  }

  std::string ToString() const;

 private:
  std::vector<int64_t> dims_;
};

// ---------------------------------------------------------------------------
// Tensor — dtype + shape + device + buffer
// ---------------------------------------------------------------------------
class Tensor {
 public:
  // Allocate a new buffer on `device` (MVP: only kCPU is implemented).
  // Fails with kInvalidArgument for bad shape/dtype/device, kRuntime on OOM.
  static StatusOr<Tensor> Create(DataType dt, TensorShape shape,
                                 DeviceKind device = DeviceKind::kCPU);

  // Wrap an existing host buffer **without taking ownership**.
  // LEARNER (FromHostBlob): the shared_ptr uses a no-op deleter. The caller
  // must keep `data` alive for as long as any Tensor alias is used (e.g. as a
  // Session feed). `bytes` must equal shape.NumElements() * SizeOfDataType(dt).
  static StatusOr<Tensor> FromHostBlob(DataType dt, TensorShape shape,
                                       void* data, std::size_t bytes);

  DataType dtype() const noexcept { return dtype_; }
  const TensorShape& shape() const noexcept { return shape_; }
  DeviceKind device() const noexcept { return device_; }

  // Raw buffer access. Empty tensors (0 elements) may return nullptr.
  void* mutable_data() noexcept { return buffer_.get(); }
  const void* data() const noexcept { return buffer_.get(); }

  // Total payload size in bytes (elements * sizeof(dtype)).
  std::size_t nbytes() const noexcept { return nbytes_; }

  // True if this tensor does not free its buffer (FromHostBlob).
  bool owns_data() const noexcept { return owns_data_; }

  // Typed helpers (return nullptr if dtype mismatches).
  float* mutable_data_f32() noexcept;
  const float* data_f32() const noexcept;
  int64_t* mutable_data_i64() noexcept;
  const int64_t* data_i64() const noexcept;

 private:
  Tensor(DataType dt, TensorShape shape, DeviceKind device, BufferPtr buffer,
         std::size_t nbytes, bool owns_data) noexcept;

  DataType dtype_;
  TensorShape shape_;
  DeviceKind device_;
  BufferPtr buffer_;
  std::size_t nbytes_;
  bool owns_data_;
};

}  // namespace eduort
