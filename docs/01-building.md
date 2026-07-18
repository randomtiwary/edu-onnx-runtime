# 01 — Building eduort

## Prerequisites

| Tool | Minimum | Notes |
|------|---------|-------|
| CMake | **3.24** | Author machine has 4.x; 3.24+ for modern FetchContent/CUDA UX |
| C++ compiler | g++ 10+ or clang 12+ | Project defaults to **C++17** |
| Ninja | recommended | `cmake -G Ninja` |
| CUDA toolkit | optional | Only when `-DEDUORT_ENABLE_CUDA=ON` (PR10+) |
| protobuf / protoc | later | PR3a |
| Python 3 | later | Fixture export scripts |

## Capture your environment

```bash
./scripts/env_report.sh | tee docs/env-report-local.txt
```

(`docs/env-report-local.txt` is gitignored — it is for *you*, not the repo.)

## Configure and build

```bash
cmake -G Ninja -B build \
  -DEDUORT_ENABLE_CUDA=OFF \
  -DEDUORT_BUILD_TESTS=ON \
  -DEDUORT_WARNINGS_AS_ERRORS=OFF

cmake --build build
./build/eduort_version_smoke
ctest --test-dir build --output-on-failure   # PR2+
```

### CMake options (PR1)

| Option | Default | Meaning |
|--------|---------|---------|
| `EDUORT_ENABLE_CUDA` | `OFF` | Record CUDA intent; real kernels arrive in PR10 |
| `EDUORT_BUILD_TESTS` | `ON` | Placeholder until GoogleTest lands in PR2 |
| `EDUORT_BUILD_TOOLS` | `OFF` | CLI `eduort-run` in PR9b |
| `EDUORT_WARNINGS_AS_ERRORS` | `OFF` | `-Werror` for strict local builds |
| `EDUORT_CXX_STANDARD` | `17` | Set `20` only if you are experimenting |

## Definition of “green”

A change is mergeable when:

1. `cmake` **configure** succeeds  
2. `cmake --build` succeeds  
3. Enabled tests pass (none required beyond smoke in PR1)  
4. Default builds do **not** require a GPU  

## Troubleshooting

### `cmake_minimum_required` fails

Upgrade CMake ≥ 3.24 (Kitware packages or `pip install cmake` in a venv).

### NVIDIA: `nvidia-smi` says “No devices were found”

On some laptops (including the author’s) the driver can be flaky even when `/dev/nvidia0` exists. **This is expected.** Keep `EDUORT_ENABLE_CUDA=OFF` until PR10, and when CUDA is enabled the runtime must **probe** and fall back to CPU (design K5 / K13).

### Want compile_commands.json for clangd?

CMake already sets `CMAKE_EXPORT_COMPILE_COMMANDS ON`. Symlink if your editor expects it at the repo root:

```bash
ln -sf build/compile_commands.json compile_commands.json
```

## Next

- Comment conventions: [comment-style.md](comment-style.md)
- Roadmap: [milestones.md](milestones.md)
