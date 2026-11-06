"""Read-only check of a computer for running the chain controller (the G1's own
computer or the laptop). Publishes nothing, moves nothing.

  python3 check_robot.py            # from the g1_onboard folder

  python3 check_robot.py            # laptop: from controller/ (any env with numpy)

Checks: Python >= 3.8, numpy, Unitree's SDK (unitree_sdk2py + cyclonedds), the
network interface on the robot's 192.168.123.x network, that the numpy policies
give the stored reference outputs, and how long one 50 Hz control step takes.
"""

from __future__ import annotations

import os

# One math thread: the policies are small matrix-vector products, and a
# multi-threaded BLAS stalls the 500 Hz loop when other processes are busy
# (measured: 5.7 ms policy steps and hundreds of late ticks). Before numpy loads.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_var, "1")

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Robot's own computer (g1_onboard/): Unitree's SDK and the CycloneDDS Python
# bindings live in ./vendor (nothing installed system-wide); the CycloneDDS C
# library is the one Unitree built in ~/cyclonedds_ws.
_VENDOR = HERE / "vendor"
if _VENDOR.is_dir():
  sys.path.insert(0, str(_VENDOR))
  _DDS = Path.home() / "cyclonedds_ws" / "install" / "cyclonedds"
  if _DDS.is_dir():
    os.environ.setdefault("CYCLONEDDS_HOME", str(_DDS))
    try:  # make libddsc findable for the bindings' C extension
      import ctypes

      ctypes.CDLL(str(_DDS / "lib" / "libddsc.so.0"), mode=ctypes.RTLD_GLOBAL)
    except OSError:
      pass
OK, BAD = "OK  ", "FAIL"
problems = []


def line(ok: bool, what: str) -> None:
  print(f"[{OK if ok else BAD}] {what}")
  if not ok:
    problems.append(what)


def main() -> None:
  v = sys.version_info
  line(v >= (3, 8), f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
  try:
    import numpy as np

    line(True, f"numpy {np.__version__}")
  except ImportError:
    line(False, "numpy missing (pip3 install numpy)")
    return
  try:
    import onnxruntime  # noqa: F401

    print(f"[info] onnxruntime {onnxruntime.__version__} (not needed: numpy weights)")
  except ImportError:
    print("[info] no onnxruntime (fine: the policies run in numpy)")
  try:
    import cyclonedds  # noqa: F401
    import unitree_sdk2py  # noqa: F401
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: F401
    from unitree_sdk2py.utils.crc import CRC

    line(True, f"unitree_sdk2py ({Path(unitree_sdk2py.__file__).parent})")
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

    cmd = unitree_hg_msg_dds__LowCmd_()
    crc = CRC()
    t0 = time.perf_counter()
    for _ in range(200):
      crc.Crc(cmd)
    print(f"[info] LowCmd CRC: {1e3 * (time.perf_counter() - t0) / 200:.3f} ms each")
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_

    raw = unitree_hg_msg_dds__LowState_().serialize()
    t0 = time.perf_counter()
    for _ in range(100):
      LowState_.deserialize(raw)
    dec = 1e3 * (time.perf_counter() - t0) / 100
    print(
      f"[info] LowState decode: {dec:.3f} ms each -> every message (~1000/s): "
      f"{dec * 100:.0f}% of a core; --poll-state at 100 Hz: {dec * 10:.0f}%"
    )
  except Exception as e:  # noqa: BLE001
    line(False, f"unitree_sdk2py / cyclonedds not importable: {e!r}")

  try:
    out = subprocess.run(
      ["ip", "-4", "-o", "addr"], capture_output=True, text=True, check=False
    ).stdout
    nets = [ln.split()[1] for ln in out.splitlines() if " 192.168.123." in ln]
    addrs = [ln.split()[3] for ln in out.splitlines() if " 192.168.123." in ln]
    line(bool(nets), f"robot network: {', '.join(f'{n} {a}' for n, a in zip(nets, addrs)) or 'none on 192.168.123.x'}")
    if nets:
      print(f"[info] start with: --net {nets[0]}")
  except FileNotFoundError:
    print("[info] 'ip' not found; check the interface on 192.168.123.x by hand")

  # g1_onboard/bundle on the robot; ../bundles/g1_chain_v3 in this repo.
  bundle = next(
    (b for b in (HERE / "bundle", HERE.parent / "bundles" / "g1_chain_v3")
     if (b / "chain.json").exists()),
    None,
  )  # fmt: skip
  if bundle is None:
    line(False, "no bundle/ next to this script (or ../bundles/g1_chain_v3)")
    return
  sys.path.insert(0, str(bundle))
  try:
    import chain_controller as cc

    ctrl = cc.ChainController(bundle, transition_s=8.0)
    line(True, f"bundle loads: crawl styles {ctrl.crawl_styles}, policies in {ctrl.backend}")
  except Exception as e:  # noqa: BLE001
    line(False, f"bundle does not load: {e!r}")
    return
  ref = np.load(bundle / "selftest.npz")
  nets = {"walk": ctrl.walk_net, "stand_crawl": ctrl.trans_net}
  nets.update({f"crawl_{k}": v for k, v in ctrl.crawl_nets.items()})
  for name, net in nets.items():
    if f"{name}_obs" not in ref.files:
      continue
    got = np.array([net(o) for o in ref[f"{name}_obs"]])
    err = float(np.abs(got - ref[f"{name}_act"]).max())
    line(err < 1e-4, f"policy {name}: max error vs reference {err:.1e}")

  # One control step: every policy (shadow mode) + gates + forward kinematics.
  q = ctrl.q_stand.copy()
  quat = np.array([1.0, 0, 0, 0])
  ctrl.reset(q)
  for _ in range(20):
    ctrl.step(quat, np.zeros(3), q, np.zeros(25))
  n = 200
  t0 = time.perf_counter()
  for _ in range(n):
    ctrl.step(quat, np.zeros(3), q, np.zeros(25))
  ms = 1e3 * (time.perf_counter() - t0) / n
  line(ms < 8.0, f"one 50 Hz control step: {ms:.2f} ms (budget 20 ms; the 500 Hz "
       f"loop needs the step + I/O well under 20 ms)")  # fmt: skip

  print()
  if problems:
    print("NOT READY:\n  - " + "\n  - ".join(problems))
    sys.exit(1)
  print("READY. Next (read-only, publishes nothing):")
  print("  python3 g1_chain_real.py --real --net <interface above> --monitor --remote")
  print("Run with ./run_robot.sh (100 Hz loop, state polled): on this computer the")
  print("default 500 Hz + every message made the policy late and the walk unstable.")


if __name__ == "__main__":
  main()
