// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/tensor.h"

#include "eduort/macros.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>

namespace eduort {
namespace {

Status ValidateShape(const TensorShape& shape) {
  // Dims are uint64_t (non-negative by type). Only overflow can fail.
  EDUORT_RETURN_IF_ERROR(shape.NumElementsChecked().status());
  return Status::OK();
}

StatusOr<std::size_t> ComputeNBytes(DataType dt, const TensorShape& shape) {
  EDUORT_ASSIGN_OR_RETURN(const uint64_t n, shape.NumElementsChecked());
  const std::size_t elem = SizeOfDataType(dt);
  if (elem == 0) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         std::string("unsupported DataType: ") + DataTypeName(dt));
  }
  // Overflow-safe multiply: n * elem fits in size_t?
  if (n > 0 && n > (std::numeric_limits<std::size_t>::max() / elem)) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "tensor byte size overflows size_t");
  }
  return static_cast<std::size_t>(n) * elem;
}

}  // namespace

const char* DataTypeName(DataType dt) noexcept {
  switch (dt) {
    case DataType::kFloat32:
      return "float32";
    case DataType::kInt64:
      return "int64";
  }
  return "unknown";
}

std::size_t SizeOfDataType(DataType dt) noexcept {
  switch (dt) {
    case DataType::kFloat32:
      return 4;
    case DataType::kInt64:
      return 8;
  }
  return 0;
}

// ---- TensorShape -----------------------------------------------------------

TensorShape::TensorShape(std::vector<uint64_t> dims) : dims_(std::move(dims)) {}

TensorShape::TensorShape(std::initializer_list<uint64_t> dims) : dims_(dims) {}

StatusOr<TensorShape> TensorShape::FromSignedDims(std::vector<int64_t> dims) {
  // LEARNER: ONNX TensorProto / ValueInfo use int64 dims. Convert at the
  // boundary so the rest of the runtime only sees non-negative sizes.
  std::vector<uint64_t> out;
  out.reserve(dims.size());
  for (int64_t d : dims) {
    if (d < 0) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           "TensorShape dim must be >= 0 (got " +
                               std::to_string(d) +
                               "); symbolic/negative dims are not supported");
    }
    out.push_back(static_cast<uint64_t>(d));
  }
  return TensorShape(std::move(out));
}

StatusOr<uint64_t> TensorShape::NumElementsChecked() const {
  // LEARNER: Rank-0 (scalar) has one element by ONNX convention.
  uint64_t product = 1;
  for (uint64_t d : dims_) {
    if (d == 0) {
      return static_cast<uint64_t>(0);
    }
    if (product > std::numeric_limits<uint64_t>::max() / d) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           "TensorShape NumElements overflows uint64");
    }
    product *= d;
  }
  return product;
}

uint64_t TensorShape::NumElements() const {
  StatusOr<uint64_t> n = NumElementsChecked();
  if (!n.ok()) {
    // LEARNER: Do not return a sentinel like -1 — that invites silent bugs.
    // Invalid shapes should fail at FromSignedDims / Create via StatusOr.
    std::fprintf(stderr,
                 "eduort::TensorShape::NumElements() on invalid shape: %s\n",
                 n.status().ToString().c_str());
    std::abort();
  }
  return n.value();
}

std::string TensorShape::ToString() const {
  std::ostringstream os;
  os << '[';
  for (std::size_t i = 0; i < dims_.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    os << dims_[i];
  }
  os << ']';
  return os.str();
}

// ---- Tensor ----------------------------------------------------------------

Tensor::Tensor(DataType dt, TensorShape shape, DeviceKind device,
               BufferPtr buffer, std::size_t nbytes, bool owns_data) noexcept
    : dtype_(dt),
      shape_(std::move(shape)),
      device_(device),
      buffer_(std::move(buffer)),
      nbytes_(nbytes),
      owns_data_(owns_data) {}

void Tensor::CheckDtype(DataType expected, const char* accessor) const {
  if (dtype_ != expected) {
    std::fprintf(stderr,
                 "eduort::Tensor::%s: dtype mismatch (tensor is %s, accessor "
                 "expects %s). Check dtype() before calling typed accessors.\n",
                 accessor, DataTypeName(dtype_), DataTypeName(expected));
    std::abort();
  }
}

StatusOr<Tensor> Tensor::Create(DataType dt, TensorShape shape,
                                DeviceKind device) {
  if (SizeOfDataType(dt) == 0) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         std::string("unsupported DataType in Tensor::Create: ") +
                             DataTypeName(dt));
  }
  if (device != DeviceKind::kCPU) {
    // LEARNER: CUDA device tensors arrive in PR10 with a CUDA allocator.
    return Status::Error(ErrorCode::kInvalidArgument,
                         "Tensor::Create only supports DeviceKind::kCPU in PR2 "
                         "(CUDA allocation lands with the CUDA EP)");
  }

  EDUORT_RETURN_IF_ERROR(ValidateShape(shape));
  EDUORT_ASSIGN_OR_RETURN(const std::size_t nbytes, ComputeNBytes(dt, shape));

  IAllocator* alloc = DefaultCpuAllocator();
  void* ptr = nullptr;
  if (nbytes > 0) {
    ptr = alloc->Allocate(nbytes);
    if (ptr == nullptr) {
      return Status::Error(ErrorCode::kRuntime,
                           "CpuAllocator failed to allocate " +
                               std::to_string(nbytes) + " bytes");
    }
  }

  BufferPtr buf = MakeAllocatedBuffer(ptr, alloc);
  return Tensor(dt, std::move(shape), device, std::move(buf), nbytes,
                /*owns_data=*/true);
}

StatusOr<Tensor> Tensor::Create(DataType dt, TensorShape shape,
                                IAllocator* allocator) {
  if (allocator == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "Tensor::Create: allocator is null");
  }
  if (SizeOfDataType(dt) == 0) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         std::string("unsupported DataType in Tensor::Create: ") +
                             DataTypeName(dt));
  }
  // LEARNER: MVP host-only. CUDA allocators arrive with the CUDA EP (PR10).
  if (allocator->device() != DeviceKind::kCPU) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "Tensor::Create via allocator only supports CPU "
                         "allocators until the CUDA EP lands");
  }

  EDUORT_RETURN_IF_ERROR(ValidateShape(shape));
  EDUORT_ASSIGN_OR_RETURN(const std::size_t nbytes, ComputeNBytes(dt, shape));

  void* ptr = nullptr;
  if (nbytes > 0) {
    ptr = allocator->Allocate(nbytes);
    if (ptr == nullptr) {
      return Status::Error(ErrorCode::kRuntime,
                           std::string(allocator->Name()) +
                               " failed to allocate " + std::to_string(nbytes) +
                               " bytes");
    }
  }

  BufferPtr buf = MakeAllocatedBuffer(ptr, allocator);
  return Tensor(dt, std::move(shape), allocator->device(), std::move(buf),
                nbytes, /*owns_data=*/true);
}

StatusOr<Tensor> Tensor::FromHostBlob(DataType dt, TensorShape shape,
                                      void* data, std::size_t bytes) {
  if (SizeOfDataType(dt) == 0) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "unsupported DataType in Tensor::FromHostBlob");
  }

  EDUORT_RETURN_IF_ERROR(ValidateShape(shape));
  EDUORT_ASSIGN_OR_RETURN(const std::size_t nbytes, ComputeNBytes(dt, shape));

  if (bytes != nbytes) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "FromHostBlob bytes (" + std::to_string(bytes) +
                             ") != shape*dtype (" + std::to_string(nbytes) +
                             ")");
  }
  if (nbytes > 0 && data == nullptr) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "FromHostBlob: data is null but nbytes > 0");
  }

  // LEARNER: no-op deleter — caller owns `data`. See docs/03-tensors-and-memory.md.
  BufferPtr buf = MakeNonOwningBuffer(data);
  return Tensor(dt, std::move(shape), DeviceKind::kCPU, std::move(buf), nbytes,
                /*owns_data=*/false);
}

float* Tensor::mutable_data_f32() {
  CheckDtype(DataType::kFloat32, "mutable_data_f32");
  return static_cast<float*>(mutable_data());
}
const float* Tensor::data_f32() const {
  CheckDtype(DataType::kFloat32, "data_f32");
  return static_cast<const float*>(data());
}
int64_t* Tensor::mutable_data_i64() {
  CheckDtype(DataType::kInt64, "mutable_data_i64");
  return static_cast<int64_t*>(mutable_data());
}
const int64_t* Tensor::data_i64() const {
  CheckDtype(DataType::kInt64, "data_i64");
  return static_cast<const int64_t*>(data());
}

}  // namespace eduort
