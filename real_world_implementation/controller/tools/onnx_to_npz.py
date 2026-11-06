"""Policy ONNX files -> numpy weight files (.npz), for a robot without onnxruntime.

The chain policies are small MLPs: optional normalizer (x - mean) / div, then
Gemm / Elu layers. Each ``<name>.onnx`` in the bundle gets a ``<name>.npz`` with
the same numbers; chain_controller.py prefers the .npz when it exists. Also
writes ``selftest.npz``: random observations and onnxruntime's outputs for each
policy, which ``check_robot.py`` replays on the robot.

Run with any env that has onnx + onnxruntime (e.g. the training repo's):
  python controller/tools/onnx_to_npz.py bundles/<bundle>
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper



def convert(path: Path) -> dict[str, np.ndarray]:
  g = onnx.load(path).graph
  init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
  out: dict[str, np.ndarray] = {}
  layer = 0
  for node in g.node:
    ins = list(node.input)
    if node.op_type == "Sub":
      out["mean"] = init[ins[1]].astype(np.float64).reshape(-1)
    elif node.op_type == "Div":
      out["div"] = init[ins[1]].astype(np.float64).reshape(-1)
    elif node.op_type == "Gemm":
      attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
      w = init[ins[1]].astype(np.float64)
      if not attrs.get("transB", 0):
        w = w.T  # stored as (out, in)
      assert attrs.get("alpha", 1.0) == 1.0 and attrs.get("beta", 1.0) == 1.0
      out[f"w{layer}"] = w
      out[f"b{layer}"] = init[ins[2]].astype(np.float64).reshape(-1)
      layer += 1
    elif node.op_type == "Elu":
      alpha = {a.name: a.f for a in node.attribute}.get("alpha", 1.0)
      assert alpha == 1.0, alpha
    else:
      raise ValueError(f"{path.name}: unsupported op {node.op_type}")
  out["layers"] = np.array(layer)
  return out


def main() -> None:
  bundle = Path(sys.argv[1])
  sys.path.insert(0, str(bundle))
  from chain_controller import NumpyMLP  # the class the robot runs
  rng = np.random.default_rng(0)
  tests: dict[str, np.ndarray] = {}
  for onnx_path in sorted(bundle.glob("*.onnx")):
    weights = convert(onnx_path)
    npz = onnx_path.with_suffix(".npz")
    np.savez(npz, **weights)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dim = sess.get_inputs()[0].shape[-1]
    obs = rng.normal(size=(64, dim)).astype(np.float32)
    ref = np.concatenate([sess.run(None, {"obs": o[None]})[0] for o in obs])
    mlp = NumpyMLP(npz)
    got = np.array([mlp(o) for o in obs])
    err = float(np.abs(got - ref).max())
    assert err < 1e-4, (onnx_path.name, err)
    tests[f"{onnx_path.stem}_obs"] = obs
    tests[f"{onnx_path.stem}_act"] = ref
    print(f"{onnx_path.name} -> {npz.name}: {int(weights['layers'])} layers, "
          f"max |numpy - onnxruntime| {err:.1e}")  # fmt: skip
  np.savez(bundle / "selftest.npz", **tests)
  print(f"wrote {bundle / 'selftest.npz'}")


if __name__ == "__main__":
  main()
