# Vendored ONNX schema (pin)

| Item | Value |
|------|--------|
| File | `onnx.proto` (full official schema, **not** hand-trimmed) |
| Upstream | https://github.com/onnx/onnx |
| Tag / commit | **v1.14.1** |
| Source URL | `https://raw.githubusercontent.com/onnx/onnx/v1.14.1/onnx/onnx.proto` |

## LEARNER

This is the **wire format** definition (Protocol Buffers) for `.onnx` files.
CMake runs `protoc` at build time to generate C++ classes such as `onnx::ModelProto`.

We **interpret only a subset** of messages in later PRs (design K3); the schema itself stays complete so we stay compatible with real files.

Do not edit `onnx.proto` by hand — bump the pin and re-copy from upstream.
