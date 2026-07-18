# 03 — Tensors and memory

**Lands with:** PR2  
**Headers:** `include/eduort/status.h`, `allocator.h`, `tensor.h`

## Mental model

An ONNX model is a graph of operators. Edges in that graph are **named tensors** — multi-dimensional arrays with a data type and a shape.

In eduort a `Tensor` is:

| Piece | Meaning |
|-------|---------|
| `DataType` | Element type (`float32`, `int64` in MVP) |
| `TensorShape` | List of dimension sizes (`[2,3]` → rank 2) |
| `DeviceKind` | Where the bytes live (`CPU` now; `CUDA` later) |
| Buffer | Contiguous **row-major** storage |

```text
shape [2, 3] float32, row-major:

  index (i, j)  →  offset_bytes = (i * 3 + j) * 4
```

## Status and StatusOr (errors without exceptions)

Public factories return `StatusOr<T>` (design **K16**):

```cpp
eduort::StatusOr<eduort::Tensor> t =
    eduort::Tensor::Create(eduort::DataType::kFloat32,
                           eduort::TensorShape({2, 3}));
if (!t.ok()) {
  // t.status().code(), t.status().message()
  return;
}
eduort::Tensor tensor = std::move(t).value();
```

| Type | Role |
|------|------|
| `Status` | OK or (ErrorCode + message) |
| `StatusOr<T>` | Success value **or** Status |

Do **not** call `value()` / `ValueOrDie()` without checking `ok()` in library code. `ValueOrDie` aborts — tests and demos only.

## Creating tensors

### `Tensor::Create` — runtime owns the buffer

```cpp
auto t = eduort::Tensor::Create(eduort::DataType::kFloat32,
                                eduort::TensorShape({2, 3}));
// t->owns_data() == true
// freed when last Tensor shared_ptr alias is destroyed
```

Uses `DefaultCpuAllocator()` (`new[]` / `delete[]`). CUDA device creation is rejected until PR10.

### `Tensor::FromHostBlob` — caller owns the buffer

```cpp
std::vector<float> host = {1, 2, 3, 4};
auto t = eduort::Tensor::FromHostBlob(
    eduort::DataType::kFloat32, eduort::TensorShape({2, 2}),
    host.data(), host.size() * sizeof(float));
// t->owns_data() == false  — no-op deleter
// host must outlive any use of t (e.g. as a Session feed)
```

| Provenance | Owner | `owns_data()` |
|------------|-------|----------------|
| `Create` | Tensor / shared_ptr | true |
| `FromHostBlob` | Caller | false |
| Initializer (later Session) | Session | true |
| Fetch from `Run` (later) | Caller after return | true |

## Shallow copies and Session (preview)

Copying a `Tensor` copies the `shared_ptr` — **same buffer**, refcount +1.  
Session::Run will shallow-copy its value map at the start of each run so multiple names can alias the same initializer storage safely **as long as kernels treat inputs as const** (design seed immutability rule).

## Allocators

```text
IAllocator  (device, Allocate, Free)
    └── CpuAllocator   ← PR2
    └── CudaAllocator  ← PR10
```

`MakeAllocatedBuffer(ptr, alloc)` → `shared_ptr` that calls `alloc->Free`.  
`MakeNonOwningBuffer(ptr)` → no-op deleter for blobs.

## Shape edge cases

| Shape | `NumElements()` |
|-------|-----------------|
| `[]` (rank 0 scalar) | `1` |
| `[0]` or `[2,0,3]` | `0` |
| negative dim | invalid (`kInvalidArgument`) |

## What is *not* here yet

- Device (CUDA) tensors  
- Memory arenas / pools  
- Non-contiguous / strided tensors  
- Full ONNX dtype set  

## Next

- Testing conventions: [09-testing.md](09-testing.md)  
- Graph IR: PR3  
