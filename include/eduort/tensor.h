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
#include <initializer_list>
#include <string>
#include <vector>

#include "eduort/allocator.h"
#include "eduort/status.h"

namespace eduort {

// LEARNER: MVP dtypes only (design K4 / non-goals).
//   float32 — weights and activations
//   int64   — shape tensors / axes (e.g. Reshape's second input)
// When loading ONNX TensorProto we accept these same two types, from either
// raw_data bytes or the typed repeated fields (float_data / int64_data).
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
// LEARNER: ONNX *files* store dims as int64 (and may use negatives as symbolic
// markers during export). Once a shape is *concrete* at runtime, every dim is
// non-negative — so we store `uint64_t` and make illegal negatives unrepresentable.
//
// Bridge from ONNX signed dims: TensorShape::FromSignedDims (rejects d < 0).
//
// A *scalar* has rank 0 (empty dims) and NumElements() == 1.
// A dim of 0 is legal (empty tensor).
class TensorShape {
 public:
  TensorShape() = default;
  explicit TensorShape(std::vector<uint64_t> dims);
  TensorShape(std::initializer_list<uint64_t> dims);

  // LEARNER: Convert ONNX-style signed dims into a runtime shape.
  // Fails with kInvalidArgument if any dim is negative.
  static StatusOr<TensorShape> FromSignedDims(std::vector<int64_t> dims);

  int rank() const noexcept { return static_cast<int>(dims_.size()); }
  const std::vector<uint64_t>& Dims() const noexcept { return dims_; }

  // Product of dims. Rank-0 → 1. Fails only on multiply overflow.
  StatusOr<uint64_t> NumElementsChecked() const;

  // Same product; aborts if the product overflows.
  // LEARNER: Call this only on shapes that already passed Create/Validate
  // (or after a successful NumElementsChecked). Soft errors use Checked().
  uint64_t NumElements() const;

  bool operator==(const TensorShape& other) const noexcept {
    return dims_ == other.dims_;
  }
  bool operator!=(const TensorShape& other) const noexcept {
    return !(*this == other);
  }

  std::string ToString() const;

 private:
  std::vector<uint64_t> dims_;
};

// ---------------------------------------------------------------------------
// Tensor — dtype + shape + device + buffer
// ---------------------------------------------------------------------------
class Tensor {
 public:
  // Allocate a new buffer on `device` (MVP: only kCPU is implemented).
  // Fails with kInvalidArgument for bad shape/dtype/device, kRuntime on OOM.
  // Uses DefaultCpuAllocator() for host tensors.
  static StatusOr<Tensor> Create(DataType dt, TensorShape shape,
                                 DeviceKind device = DeviceKind::kCPU);

  // Allocate via a specific allocator (EP path). Device is taken from alloc.
  // LEARNER: OpKernelContext::Output must use this so kernels respect the EP
  // memory contract (not always DefaultCpuAllocator).
  static StatusOr<Tensor> Create(DataType dt, TensorShape shape,
                                 IAllocator* allocator);

  // Wrap an existing host buffer **without taking ownership**.
  // LEARNER (FromHostBlob): the shared_ptr uses a no-op deleter. The caller
  // must keep `data` alive for as long as any Tensor alias is used (e.g. as a
  // Session feed). `bytes` must equal shape.NumElements() * SizeOfDataType(dt)
  // (element count must not overflow — enforced via NumElementsChecked).
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

  // Typed helpers. LEARNER: check dtype() first; a mismatch is a programmer
  // error and **aborts** (we do not return nullptr to "soft-fail" type bugs).
  float* mutable_data_f32();
  const float* data_f32() const;
  int64_t* mutable_data_i64();
  const int64_t* data_i64() const;

 private:
  Tensor(DataType dt, TensorShape shape, DeviceKind device, BufferPtr buffer,
         std::size_t nbytes, bool owns_data) noexcept;

  void CheckDtype(DataType expected, const char* accessor) const;

  DataType dtype_;
  TensorShape shape_;
  DeviceKind device_;
  BufferPtr buffer_;
  std::size_t nbytes_;
  bool owns_data_;
};

}  // namespace eduort
