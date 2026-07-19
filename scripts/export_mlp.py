#!/usr/bin/env python3
# LEARNER: Build a tiny hand-made ONNX graph with onnx.helper (no torch export).
# Requires: pip install onnx
"""Export a minimal Add graph used as a development fixture."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        print("error: pip install onnx  (needed for export_mlp.py)", file=sys.stderr)
        return 2

    import numpy as np

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    W = numpy_helper.from_array(np.array([[1.0, 2.0]], dtype=np.float32), name="W")

    node = helper.make_node("Add", inputs=["X", "W"], outputs=["Y"], name="add")
    graph = helper.make_graph([node], "export_add", [X], [Y], initializer=[W])
    model = helper.make_model(
        graph,
        producer_name="eduort-export_mlp",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8

    out = Path(__file__).resolve().parents[1] / "testdata" / "models" / "export_add.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    print("wrote", out)

    # Optional allowlist check
    checker = Path(__file__).with_name("check_model_ops.py")
    if checker.exists():
        import subprocess

        r = subprocess.call([sys.executable, str(checker), str(out)])
        return r
    return 0


if __name__ == "__main__":
    sys.exit(main())
