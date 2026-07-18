# edu-onnx-runtime (`eduort`)

**Educational** ONNX inference runtime in modern C++.

This project teaches how a real inference runtime is structured — graph IR, topological execution, kernel registries, execution providers (CPU, later CUDA) — with **heavily documented** code aimed at learners who are not yet experts in compilers or ML systems.

> This is **not** a production competitor to [Microsoft ONNX Runtime](https://onnxruntime.ai/). Clarity beats performance.

| | |
|--|--|
| **Library** | `eduort` (`eduort::` namespace) |
| **Language** | C++17 |
| **Build** | CMake ≥ 3.24 + Ninja |
| **License** | Apache-2.0 |
| **Design** | [docs/design.md](docs/design.md) |
| **Milestones / PRs** | [docs/milestones.md](docs/milestones.md) |

## Quick start (PR1 skeleton)

```bash
# Optional: capture your machine toolchain / GPU state
./scripts/env_report.sh | tee docs/env-report-local.txt

cmake -G Ninja -B build -DEDUORT_ENABLE_CUDA=OFF -DEDUORT_BUILD_TESTS=ON
cmake --build build
./build/eduort_version_smoke
ctest --test-dir build --output-on-failure
```

Expected output includes something like:

```text
eduort 0.0.1 (C++17, CUDA=OFF)
VersionString() = 0.0.1
```

## What works today vs next

| Milestone | Status | What you get |
|-----------|--------|--------------|
| PR1 — skeleton | done | CMake, version API, docs scaffolding |
| **PR2** — tensors | **you are here** | `Status` / `StatusOr`, `Tensor`, host allocator, GoogleTest |
| PR3a–3b | planned | protobuf + ONNX `ModelProto` load → Graph IR |
| PR4–8 | planned | validation, topo sort, registry, CPU kernels |
| PR9a | planned | `Session::Create` / `Run` end-to-end on CPU → **v0.1.0** |
| PR9b | planned | `eduort-run` CLI |
| PR10–11 | planned | optional CUDA EP + mixed CPU/GPU graphs → **v0.2.0** |
| PR13 | planned after v0.2 | optional educational AVX MatMul → **v0.3.0** |

## Hardware notes (author machine)

- **CPU**: Intel i9-11900H (AVX-512 capable) — always supported.
- **GPU**: NVIDIA RTX 3050 Mobile + CUDA toolkit may be present, but drivers can intermittently report no device. The design **requires CPU fallback** and never makes a GPU mandatory for default builds.

## Learning path

1. Read [docs/00-overview.md](docs/00-overview.md) — big picture.
2. Read [docs/01-building.md](docs/01-building.md) — build & options.
3. Read [docs/03-tensors-and-memory.md](docs/03-tensors-and-memory.md) — Status, Tensor, ownership.
4. Skim [docs/design.md](docs/design.md) — architecture and PR plan.
5. Follow [docs/comment-style.md](docs/comment-style.md) when reading or writing code (`LEARNER:` tags).

## Comment style

Code that teaches a concept is tagged:

```cpp
// LEARNER: ... plain-language explanation ...
// Spec: ONNX Operators.md — Relu
```

See [docs/comment-style.md](docs/comment-style.md).

## Contributing / PR process

We implement the design in **small, reviewable PRs** that each leave the tree green (configure + build succeed). See [docs/milestones.md](docs/milestones.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
