# Milestones and PR plan

Full design: [design.md](design.md). This page is the **learner-facing** checklist.

**Product choices (2026-07-18):**

- Repo: public [`randomtiwary/edu-onnx-runtime`](https://github.com/randomtiwary/edu-onnx-runtime)
- GitHub Actions CI: deferred (local green is enough for now)
- Optional SIMD MatMul (**PR13**): yes, after CUDA **v0.2.0**

## Learning outcomes (by milestone)

| ID | PR | Learning outcome | Release |
|----|-----|------------------|---------|
| M0 | PR1 | CMake library layout, version surface, doc spine | 0.0.1 |
| M1 | PR2 | Errors (`Status`/`StatusOr`), tensors, host memory | |
| M2 | PR3a | Protobuf codegen; parse `ModelProto` | |
| M3 | **PR3b** | Map ONNX → in-memory Graph IR | |
| M4 | PR4 | Structural validation + topological sort | |
| M5 | PR5 | EP interface, kernel registry, CPU shell | |
| M6 | PR6 | Elementwise CPU kernels + broadcast | |
| M7 | PR7 | MatMul / Gemm (naive) | |
| M8 | PR8 | Reshape, Flatten, Softmax | |
| M9a | PR9a | Full `Session::Run` on CPU | **v0.1.0** |
| M9b | PR9b | CLI demo | v0.1.1 optional |
| M10 | PR10 | CUDA probe + provider skeleton | |
| M11 | PR11 | CUDA kernels + mixed-graph copies | **v0.2.0** |
| M12 | PR12 | Docs polish | |
| M13 | PR13 | Educational AVX MatMul (optional) | **v0.3.0** |

## PR titles (implementation order)

1. `chore: initial repository skeleton and CMake build` ✓
2. `feat: Status, StatusOr, Tensor, TensorShape, and host allocator` ✓
3. `build: protobuf + onnx.proto generation and ModelProto smoke parse` (3a) ✓
4. `feat: map ModelProto to Graph IR` (3b) ← **this PR**
5. `feat: graph structural validation and topological sort`
6. `feat: execution provider interface, registry, and CPU provider shell`
7. `feat(cpu): elementwise kernels and shape_inference extension point`
8. `feat(cpu): MatMul and Gemm kernels`
9. `feat(cpu): Reshape, Flatten, Softmax`
10. `feat: Session::Create/Run end-to-end on CPU` → tag **v0.1.0**
11. `feat: eduort-run CLI and README quickstart`
12. `feat(cuda): probe state machine, provider skeleton, device allocator`
13. `feat(cuda): MatMul/Gemm/Add/Relu + mixed-graph D2H/H2D tests` → **v0.2.0**
14. `docs: polish architecture narrative and operator catalog`
15. `feat(cpu): educational AVX2/AVX-512 MatMul` (optional) → **v0.3.0**

## Green definition (every PR)

1. Configure succeeds  
2. Build succeeds  
3. Enabled tests pass  
4. GPU not required unless explicitly configuring CUDA **and** a device is Ready  

## MVP operators (end of PR8 / PR9a)

`Constant`, `Identity`, `Add`, `Mul`, `MatMul`, `Gemm`, `Relu`, `Sigmoid`, `Softmax`, `Reshape`, `Flatten`
