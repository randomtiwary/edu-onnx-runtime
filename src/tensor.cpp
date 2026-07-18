// Copyright 2026 randomtiwary
// SPDX-License-Identifier: Apache-2.0

#include "eduort/tensor.h"

#include <limits>
#include <sstream>
#include <string>

namespace eduort {
namespace {

Status ValidateShape(const TensorShape& shape) {
  for (int64_t d : shape.Dims()) {
    if (d < 0) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           "TensorShape dims must be >= 0 (got " +
                               std::to_string(d) + ")");
    }
  }
  // Touch NumElementsChecked for overflow.
  StatusOr<int64_t> n = shape.NumElementsChecked();
  if (!n.ok()) {
    return n.status();
  }
  return Status::OK();
}

StatusOr<std::size_t> ComputeNBytes(DataType dt, const TensorShape& shape) {
  StatusOr<int64_t> ne = shape.NumElementsChecked();
  if (!ne.ok()) {
    return ne.status();
  }
  const std::size_t elem = SizeOfDataType(dt);
  if (elem == 0) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         std::string("unsupported DataType: ") + DataTypeName(dt));
  }
  // Overflow-safe multiply: ne * elem fits in size_t?
  const int64_t n = ne.value();
  if (n < 0) {
    return Status::Error(ErrorCode::kInvalidArgument, "negative element count");
  }
  if (n > 0 &&
      static_cast<uint64_t>(n) >
          (std::numeric_limits<std::size_t>::max() / elem)) {
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

TensorShape::TensorShape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

StatusOr<int64_t> TensorShape::NumElementsChecked() const {
  // LEARNER: Rank-0 (scalar) has one element by ONNX convention.
  int64_t product = 1;
  for (int64_t d : dims_) {
    if (d < 0) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           "negative dimension in TensorShape");
    }
    if (d == 0) {
      return static_cast<int64_t>(0);
    }
    // product * d overflow?
    if (product > std::numeric_limits<int64_t>::max() / d) {
      return Status::Error(ErrorCode::kInvalidArgument,
                           "TensorShape NumElements overflows int64");
    }
    product *= d;
  }
  return product;
}

int64_t TensorShape::NumElements() const {
  StatusOr<int64_t> n = NumElementsChecked();
  if (!n.ok()) {
    return -1;
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

  Status st = ValidateShape(shape);
  if (!st.ok()) {
    return st;
  }

  StatusOr<std::size_t> nbytes_or = ComputeNBytes(dt, shape);
  if (!nbytes_or.ok()) {
    return nbytes_or.status();
  }
  const std::size_t nbytes = nbytes_or.value();

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

StatusOr<Tensor> Tensor::FromHostBlob(DataType dt, TensorShape shape,
                                      void* data, std::size_t bytes) {
  if (SizeOfDataType(dt) == 0) {
    return Status::Error(ErrorCode::kInvalidArgument,
                         "unsupported DataType in Tensor::FromHostBlob");
  }

  Status st = ValidateShape(shape);
  if (!st.ok()) {
    return st;
  }

  StatusOr<std::size_t> nbytes_or = ComputeNBytes(dt, shape);
  if (!nbytes_or.ok()) {
    return nbytes_or.status();
  }
  const std::size_t nbytes = nbytes_or.value();

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

float* Tensor::mutable_data_f32() noexcept {
  return dtype_ == DataType::kFloat32 ? static_cast<float*>(mutable_data())
                                      : nullptr;
}
const float* Tensor::data_f32() const noexcept {
  return dtype_ == DataType::kFloat32 ? static_cast<const float*>(data())
                                      : nullptr;
}
int64_t* Tensor::mutable_data_i64() noexcept {
  return dtype_ == DataType::kInt64 ? static_cast<int64_t*>(mutable_data())
                                    : nullptr;
}
const int64_t* Tensor::data_i64() const noexcept {
  return dtype_ == DataType::kInt64 ? static_cast<const int64_t*>(data())
                                    : nullptr;
}

}  // namespace eduort
