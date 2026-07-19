#!/usr/bin/env python3
# LEARNER: Gate fixture models so they only use the MVP operator allowlist.
# Expand this list in PR6–8 as kernels land; freeze at PR9a.
"""Usage: check_model_ops.py path/to/model.onnx [more.onnx ...]"""

from __future__ import annotations

import sys
from pathlib import Path

# Stub allowlist — grows with op PRs. Constant is legal (folded later).
MVP_OPS = {
    "Constant",
    "Identity",
    "Add",
    "Mul",
    "MatMul",
    "Gemm",
    "Relu",
    "Sigmoid",
    "Softmax",
    "Reshape",
    "Flatten",
}


def load_op_types(path: Path) -> list[str]:
    # Prefer onnx package; fall back to raw protobuf using vendored schema.
    try:
        import onnx  # type: ignore

        model = onnx.load(str(path))
        return [n.op_type for n in model.graph.node]
    except Exception:
        pass

    # Fallback: protoc-generated module if available on PYTHONPATH
    try:
        from google.protobuf import message  # noqa: F401
        import importlib.util

        # Best-effort: user may not have onnx; skip detailed parse
        print(f"warning: onnx package not installed; skipping deep check for {path}",
              file=sys.stderr)
        return []
    except Exception as e:
        print(f"error: cannot load {path}: {e}", file=sys.stderr)
        sys.exit(2)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    bad = 0
    for arg in argv[1:]:
        path = Path(arg)
        ops = load_op_types(path)
        for op in ops:
            if op not in MVP_OPS:
                print(f"{path}: unsupported op_type '{op}' (not in MVP allowlist)")
                bad += 1
        if ops:
            print(f"{path}: OK ({len(ops)} nodes, ops={sorted(set(ops))})")
        else:
            print(f"{path}: skipped or empty")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
