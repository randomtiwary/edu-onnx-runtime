# 09 — Testing

**Lands with:** PR2 (framework); goldens grow with ops PRs.

## Philosophy

| Goal | How |
|------|-----|
| Correctness on CPU without a GPU | Default tests never need CUDA |
| Teach contracts early | Unit tests for Status, Tensor ownership |
| Golden numerical checks later | `.f32` dumps + JSON shapes (ops PRs) |
| Skip, don’t fail, without GPU | CUDA tests use GTEST_SKIP (PR10+) |

## Running tests

```bash
cmake -G Ninja -B build -DEDUORT_ENABLE_CUDA=OFF -DEDUORT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Or run one binary:

```bash
./build/tests/tensor_test
./build/tests/status_test
./build/tests/allocator_test
```

First configure downloads **GoogleTest v1.14.0** via CMake `FetchContent` (needs network once).

## Layout

```text
tests/
  CMakeLists.txt      # eduort_add_test helper
  status_test.cpp
  allocator_test.cpp
  tensor_test.cpp
  # later: loader_test, kernel tests, e2e/mlp_test.cpp
```

## Conventions

1. **One concern per TEST** — shape overflow separate from FromHostBlob.
2. **Assert Status codes**, not only `!ok()`, when the code is part of the contract.
3. **No GPU required** for tests in this PR.
4. Prefer `ASSERT_TRUE(x.ok()) << x.status().ToString()` so failures print the message.
5. Golden files (later): commit binary `.f32` + JSON sidecar; generate with numpy scripts; never require ORT in C++ tests.

## Tolerances (preview for ops PRs)

| Comparison | Default idea |
|------------|--------------|
| Exact int64 | bitwise equal |
| float32 elementwise | `atol=1e-5`, `rtol=1e-5` (tune per op) |
| Softmax | also check sum ≈ 1 on axis |

## Disabling tests

```bash
cmake -G Ninja -B build -DEDUORT_BUILD_TESTS=OFF
```

## Next

- Tensor ownership: [03-tensors-and-memory.md](03-tensors-and-memory.md)  
