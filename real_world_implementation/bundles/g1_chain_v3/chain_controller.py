"""Walk -> crawl -> walk controller for the braced 29-DoF G1 (numpy + onnxruntime).

The state machine, observation builders and targets of ``scripts/play_chain.py``,
using only what the real robot measures: the pelvis IMU (quaternion, gyro) and
the joint encoders. No mjlab, no torch, no MuJoCo. Call it once per 50 Hz tick::

  ctrl = ChainController("deploy/g1_chain_v2")
  ctrl.reset()
  while running:
    targets = ctrl.step(quat_wxyz, gyro, q, qd)  # 25 PD targets, policy order
    # send them with ctrl.kp / ctrl.kd (chain.json "pd")

Inputs are in **policy joint order** (``ctrl.joints``; motor indices in
``chain.json["motor_index"]``). ``gyro`` is the pelvis IMU angular velocity
(rad/s, pelvis frame), ``quat_wxyz`` the pelvis IMU orientation.

States (chain.json "switcher")::

  WALK --C--> STOPPING --still--> DOWN --phi 1--> HOLD --crawl gate--> CRAWL
  CRAWL --U--> CRAWL_STOP --still on 4 limbs--> HOLD --crawl gate--> UP
  UP --phi 0--> STAND_HOLD --upright, still--> WALK

The sim gates use contacts, pelvis height and base velocity. Here they use what
the robot has: joint speeds, the gyro, gravity from the IMU, and forward
kinematics of ``mujoco/scene.xml`` (in numpy) for the pelvis height over the
limbs and for which limbs are down (lowest point of each foot sole / extension
tip within ``limb_down_m`` of the lowest one; flat ground).

v3 bundles (``chain.json`` ``crawl_variants``) carry several crawl styles with
the same observation and action (creepy, scuttle). All of them run every tick
in shadow; ``request_style()`` switches, while crawling only once the crawl
command is 0 and the robot is still, then blends over ``blend_s``. HOLD gives
up after ``hold_timeout_s`` without the crawl gate and stands up (the 10-29
stuck nose-down hold).

Keys: ``press_down()`` (C), ``press_up()`` (U), ``set_walk_command(vx, vy, wz)``,
``set_crawl_command(fwd, lat, turn)``, ``set_step_height(h)``, ``stop()``,
``force_gate()`` (operator override of the gate the robot waits on).
``gate_why`` says what that gate is still waiting for.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
  import onnxruntime as ort
except ImportError:  # e.g. the robot's own computer: numpy weights (.npz) only
  ort = None

(WALK, STOPPING, DOWN, HOLD, CRAWL, CRAWL_STOP, UP, STAND_HOLD) = range(8)
MODE_NAMES = (
  "WALK",
  "STOPPING",
  "DOWN",
  "HOLD",
  "CRAWL",
  "CRAWL_STOP",
  "UP",
  "STAND_HOLD",
)
WALK_MODES = (WALK, STOPPING)
CRAWL_MODES = (CRAWL, CRAWL_STOP)
LIMBS = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_forearm_extension",
  "right_forearm_extension",
)
LIMB_TAGS = ("left foot", "right foot", "left hand", "right hand")
STEP_HEIGHTS = (0.03, 0.08, 0.12)  # play_switcher.py; crawl starts at the first


def _bare(name: str) -> str:
  """MJCF name without mjlab's 'robot/' prefix (str.removeprefix is py3.9+)."""
  return name[len("robot/") :] if name.startswith("robot/") else name


def smoothstep(x):
  x = np.clip(x, 0.0, 1.0)
  return x * x * (3.0 - 2.0 * x)


def quat_to_mat(q) -> np.ndarray:
  w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
  )


def gravity_in_body(quat_wxyz) -> np.ndarray:
  """World gravity direction (0, 0, -1) in the body frame: R^T @ (0, 0, -1)."""
  return -quat_to_mat(quat_wxyz)[2]


def _axis_angle(axis: np.ndarray, a: float) -> np.ndarray:
  x, y, z = axis
  c, s, t = math.cos(a), math.sin(a), 1.0 - math.cos(a)
  return np.array(
    [
      [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
      [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
      [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
  )


class Schedule:
  """ref(phi): per-joint smoothstep between consecutive keyframes; gravity(phi)
  from the keyframe base pitch (``TransitionSchedule`` in mjlab)."""

  def __init__(self, keyframes: list[dict], joints: list[str]) -> None:
    self.phis = np.array([k["phi"] for k in keyframes])
    self.q = np.array([[k["joint_pos"][j] for j in joints] for k in keyframes])
    self.pitch = np.radians([k["pitch_deg"] for k in keyframes])

  def _segment(self, phi: float) -> tuple[int, float]:
    i = int(
      np.clip(np.searchsorted(self.phis, phi, side="right") - 1, 0, len(self.phis) - 2)
    )
    return i, float(
      smoothstep((phi - self.phis[i]) / (self.phis[i + 1] - self.phis[i]))
    )

  def __call__(self, phi: float) -> np.ndarray:
    i, s = self._segment(phi)
    return self.q[i] + (self.q[i + 1] - self.q[i]) * s

  def gravity(self, phi: float) -> np.ndarray:
    i, s = self._segment(phi)
    p = self.pitch[i] + (self.pitch[i + 1] - self.pitch[i]) * s
    return np.array([math.sin(p), 0.0, -math.cos(p)])


@dataclass
class LimbState:
  """Forward kinematics in the pelvis-centred, gravity-aligned frame (z up)."""

  origin_z: np.ndarray  # the 4 LIMBS body origins, below the pelvis (< 0)
  low_z: np.ndarray  # lowest point of each foot sole / extension tip
  above_limbs: float  # pelvis over the mean limb origin (crawl gate)
  down: np.ndarray  # limb within limb_down_m of the lowest one (flat ground)
  pelvis_height: float  # pelvis over the lower foot sole (standing)
  feet_gap: float  # |left sole - right sole|


class LimbKinematics:
  """Numpy forward kinematics of the braced G1 from ``mujoco/scene.xml``: only
  the chains from the pelvis to the feet and the forearm extensions."""

  def __init__(self, xml_path: Path, joints: list[str]) -> None:
    root = ET.parse(xml_path).getroot()
    pelvis = root.find(".//body[@name='robot/pelvis']")
    assert pelvis is not None, xml_path
    f = lambda e, k, d: np.array([float(v) for v in e.get(k, d).split()])  # noqa: E731
    # Every body on a path pelvis -> limb, parents first.
    self.names: list[str] = []
    self.parent: list[int] = []
    self.pos: list[np.ndarray] = []
    self.rot: list[np.ndarray] = []
    self.joint: list[int] = []
    self.axis: list[np.ndarray] = []
    self.capsules: list[list[tuple[np.ndarray, np.ndarray, float]]] = []
    index: dict[int, int] = {}  # id(element) -> body index

    def visit(body: ET.Element, path: list[ET.Element]) -> None:
      if _bare(body.get("name", "")) in LIMBS:
        parent = -1  # the pelvis
        for b in path + [body]:
          if id(b) not in index:
            index[id(b)] = len(self.names)
            self._add(b, parent, f, joints)
          parent = index[id(b)]
      for child in body.findall("body"):
        visit(child, path + [body])  # path: ancestors below the pelvis

    for child in pelvis.findall("body"):
      visit(child, [])
    self.limb_idx = [self.names.index(n) for n in LIMBS]
    for n in LIMBS:
      assert self.capsules[self.names.index(n)], f"no contact capsules on {n}"

  def _add(self, b: ET.Element, parent: int, f, joints: list[str]) -> None:
    self.parent.append(parent)
    self.names.append(_bare(b.get("name", "")))
    self.pos.append(f(b, "pos", "0 0 0"))
    self.rot.append(quat_to_mat(f(b, "quat", "1 0 0 0")))
    js = [j for j in b.findall("joint") if j.get("type", "hinge") == "hinge"]
    assert len(js) <= 1
    if js:
      assert np.allclose(f(js[0], "pos", "0 0 0"), 0.0), "hinge off the body origin"
      self.joint.append(joints.index(_bare(js[0].get("name", ""))))
      ax = f(js[0], "axis", "0 0 1")
      self.axis.append(ax / np.linalg.norm(ax))
    else:
      self.joint.append(-1)
      self.axis.append(np.zeros(3))
    caps = []
    for g in b.findall("geom"):
      gname = g.get("name", "")
      if gname.endswith("_collision") and ("foot" in gname or "forearm_tip" in gname):
        size = f(g, "size", "0 0 0")
        c = f(g, "pos", "0 0 0")
        half = quat_to_mat(f(g, "quat", "1 0 0 0")) @ np.array([0.0, 0.0, size[1]])
        caps.append((c - half, c + half, float(size[0])))
    self.capsules.append(caps)

  def __call__(self, quat_wxyz, q: np.ndarray, limb_down_m: float) -> LimbState:
    r0 = quat_to_mat(quat_wxyz)
    xs: list[np.ndarray] = []
    rs: list[np.ndarray] = []
    for i in range(len(self.names)):
      p = self.parent[i]
      xp, rp = (np.zeros(3), r0) if p < 0 else (xs[p], rs[p])
      r = rp @ self.rot[i]
      if self.joint[i] >= 0:
        r = r @ _axis_angle(self.axis[i], float(q[self.joint[i]]))
      xs.append(xp + rp @ self.pos[i])
      rs.append(r)
    origin_z = np.array([xs[i][2] for i in self.limb_idx])
    low_z = np.array(
      [
        min(
          min((xs[i] + rs[i] @ a)[2], (xs[i] + rs[i] @ b)[2]) - rad
          for a, b, rad in self.capsules[i]
        )
        for i in self.limb_idx
      ]
    )
    ground = low_z.min()
    return LimbState(
      origin_z=origin_z,
      low_z=low_z,
      above_limbs=float(-origin_z.mean()),
      down=(low_z - ground) < limb_down_m,
      pelvis_height=float(-low_z[:2].min()),
      feet_gap=float(abs(low_z[0] - low_z[1])),
    )


class NumpyMLP:
  """A policy in plain numpy: (x - mean) / div, then Linear + ELU layers, from
  the .npz written next to its .onnx (controller/tools/onnx_to_npz.py).
  Matches onnxruntime to ~3e-5."""

  def __init__(self, path: Path) -> None:
    d = np.load(path)
    self.mean = d["mean"] if "mean" in d.files else None
    self.div = d["div"] if "div" in d.files else None
    self.layers = [(d["w%d" % i], d["b%d" % i]) for i in range(int(d["layers"]))]
    self.input_dim = self.layers[0][0].shape[1]

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float64).reshape(-1)
    if self.mean is not None:
      x = x - self.mean
    if self.div is not None:
      x = x / self.div
    last = len(self.layers) - 1
    for i, (w, b) in enumerate(self.layers):
      x = w @ x + b
      if i < last:
        x = np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))  # ELU, alpha 1
    return x


def load_policy(path: Path, input_dim: int):
  """The numpy weights if ``<name>.npz`` exists, else the ONNX file."""
  npz = path.with_suffix(".npz")
  if npz.exists():
    net = NumpyMLP(npz)
    assert net.input_dim == input_dim, (npz, net.input_dim, input_dim)
    return net
  if ort is None:
    raise ImportError(f"no {npz.name} and no onnxruntime to run {path.name}")
  return _Onnx(path, input_dim)


class _Onnx:
  def __init__(self, path: Path, input_dim: int) -> None:
    opts = ort.SessionOptions()  # ty: ignore[possibly-missing-attribute]
    opts.intra_op_num_threads = 1
    self.sess = ort.InferenceSession(
      str(path), opts, providers=["CPUExecutionProvider"]
    )
    shape = self.sess.get_inputs()[0].shape
    assert shape[-1] == input_dim, (path, shape, input_dim)

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    out = self.sess.run(None, {"obs": obs[None].astype(np.float32)})[0]
    return np.asarray(out, dtype=np.float64)[0]


@dataclass
class Gates:
  """Thresholds of the switcher gates (chain.json "switcher", robot version)."""

  stand_joint_speed: float = 0.6  # rad/s, max over joints
  crawl_joint_speed: float = 0.5
  calm_gyro: float = 0.3  # rad/s, |gyro| (sim: base speed < 0.1 m/s)
  calm_yaw: float = 0.2  # rad/s
  feet_gap_m: float = 0.02  # both soles down
  upright_gz: float = -0.97
  stand_height_m: float = 0.70  # STAND_HOLD: pelvis over the soles
  crawl_height_tol_m: float = 0.03  # |pelvis over limbs - 0.151|
  crawl_orient: float = 0.15  # |g - g_crawl|, crawl gate
  settle_orient: float = 0.25  # |g - g_crawl|, CRAWL_STOP
  limb_down_m: float = 0.04


class ChainController:
  def __init__(
    self,
    bundle_dir: str | Path,
    transition_s: float = 6.0,
    blend_s: float = 0.5,
    dwell_s: float = 1.0,
    gates: Gates | None = None,
    hold_timeout_s: float = 8.0,
  ) -> None:
    d = Path(bundle_dir)
    cfg = json.loads((d / "chain.json").read_text())
    assert cfg["version"] == 2, "needs a v2+ bundle ([phi, dphi/dt])"
    self.cfg = cfg
    self.joints: list[str] = cfg["joints"]
    J = self.joints
    self.dt = 1.0 / cfg["control_hz"]
    self.kp = np.array([cfg["pd"][j]["kp"] for j in J])
    self.kd = np.array([cfg["pd"][j]["kd"] for j in J])
    self.peak_nm = np.array([cfg["pd"][j]["peak_nm"] for j in J])
    self.q_stand = np.array([cfg["stand_pose"][j] for j in J])
    self.q_crawl = np.array([cfg["crawl_pose"][j] for j in J])
    self.crawl_above_limbs = float(cfg["crawl_pelvis_above_limbs_m"])
    pw, ps, pc = (cfg["policies"][k] for k in ("walk", "stand_crawl", "crawl"))
    self.walk_scale = np.array([pw["scale"][j] for j in J])
    self.walk_clip = float(pw["action_clip"])
    self.trans_clip = float(ps["action_clip"])
    self.trans_scale = float(ps["action_scale"])
    self.crawl_clip = float(pc["action_clip"])
    self.walk_net = load_policy(d / pw["file"], pw["input_dim"])
    self.trans_net = load_policy(d / ps["file"], ps["input_dim"])
    styles = {
      k: v for k, v in cfg.get("crawl_variants", {}).items() if not k.startswith("_")
    } or {"crawl": {"file": pc["file"], "command_ranges": pc["command_ranges"]}}
    self.crawl_styles = list(styles)  # the first one is the default (v3: creepy)
    self.crawl_nets = {
      k: load_policy(d / v["file"], pc["input_dim"]) for k, v in styles.items()
    }
    self.backend = type(self.walk_net).__name__  # NumpyMLP or _Onnx
    self.style_ranges = {k: v["command_ranges"] for k, v in styles.items()}
    self.walk_ranges = pw["command_ranges"]
    self.schedule = Schedule(ps["keyframes"], J)
    assert np.allclose(self.schedule(0.0), self.q_stand, atol=1e-5)
    assert np.allclose(self.schedule(1.0), self.q_crawl, atol=1e-5)
    self.g_crawl = self.schedule.gravity(1.0)
    self.fk = LimbKinematics(d / "mujoco" / "scene.xml", J)
    # Targets never outside the joint ranges (v3 policies trained with this
    # clamp; v2 never clamped in training, only penalized the soft ranges).
    lim = cfg["joint_limits"]
    self.q_lo = np.array([lim[j][0] for j in J])
    self.q_hi = np.array([lim[j][1] for j in J])
    self.transition_s, self.blend_s, self.dwell_s = transition_s, blend_s, dwell_s
    self.hold_timeout_s = hold_timeout_s
    self.gates = gates or Gates()
    self.reset()

  # -- state ------------------------------------------------------------------

  def reset(self, q: np.ndarray | None = None, start_crawl: bool = False) -> None:
    """``start_crawl``: the robot lies in the crawl pose; start in HOLD at
    phi 1 (the crawl gate then hands over to CRAWL)."""
    n = len(self.joints)
    self.mode = HOLD if start_crawl else WALK
    self.time = 0.0
    self.dwell = 0.0
    self.queued = -1
    self.phase = 1.0 if start_crawl else 0.0
    self.rate = 0.0
    self.walk_cmd = np.zeros(3)
    self.crawl_cmd = np.zeros(7)
    self.crawl_cmd[3] = STEP_HEIGHTS[0]
    self.walk_last = np.zeros(n)
    self.trans_last = np.zeros(n)
    self.crawl_last = {k: np.zeros(n) for k in self.crawl_styles}
    self.style = self.crawl_styles[0]
    self.style_pending: str | None = None
    self.style_dwell = 0.0
    self.style_why = ""
    self.mode_t0 = 0.0
    self.hold_timed_out = False
    self.blend_t0 = -1e9
    rest = self.q_crawl if start_crawl else self.q_stand
    self.applied = rest.copy() if q is None else np.asarray(q, float).copy()
    self.blend_from = self.applied.copy()
    self.limb: LimbState | None = None
    self.gate_ok = False
    self.gate_why = ""
    self._force = False

  @property
  def mode_name(self) -> str:
    return MODE_NAMES[self.mode]

  @property
  def active_action(self) -> np.ndarray:
    if self.mode in WALK_MODES:
      return self.walk_last
    return self.crawl_last[self.style] if self.mode in CRAWL_MODES else self.trans_last

  @property
  def crawl_ranges(self) -> dict:
    """Command ranges of the active crawl style."""
    return self.style_ranges[self.style]

  def _set(self, mode: int, blend: bool) -> None:
    self.mode = mode
    self.mode_t0 = self.time
    self.dwell = 0.0
    if blend:
      self.blend_t0 = self.time
      self.blend_from = self.applied.copy()

  # -- keys -------------------------------------------------------------------

  def set_walk_command(self, vx: float, vy: float, wz: float) -> None:
    """Used in WALK only; clipped to the trained ranges."""
    r = self.walk_ranges
    self.walk_cmd = np.array(
      [np.clip(vx, *r["vx"]), np.clip(vy, *r["vy"]), np.clip(wz, *r["wz"])]
    )

  def set_crawl_command(self, fwd: float, lat: float, turn: float) -> None:
    """Used in CRAWL only. ``turn`` > 0 is clockwise seen from above (the
    opposite of the walk's wz)."""
    r = self.crawl_ranges
    self.crawl_cmd[:3] = [
      np.clip(fwd, *r["fwd"]),
      np.clip(lat, *r["lat"]),
      np.clip(turn, *r["turn"]),
    ]

  def set_step_height(self, h: float) -> None:
    self.crawl_cmd[3] = float(np.clip(h, *self.crawl_ranges["step_height"]))

  def stop(self) -> None:
    self.walk_cmd[:] = 0.0
    self.crawl_cmd[:3] = 0.0

  def press_down(self) -> None:
    """C: walk -> crawl. While standing up, goes back down once standing."""
    if self.mode == WALK:
      self.walk_cmd[:] = 0.0
      self._set(STOPPING, blend=False)
    elif self.mode == STAND_HOLD:
      self._set(DOWN, blend=False)
    elif self.mode == UP:
      self.queued = DOWN

  def press_up(self) -> None:
    """U: crawl -> walk. While going down / holding, stands up from the pose."""
    if self.mode == CRAWL:
      self.crawl_cmd[:3] = 0.0
      self._set(CRAWL_STOP, blend=False)
    elif self.mode in (HOLD, DOWN):
      self.queued = UP

  def request_style(self, name: str | None = None) -> str:
    """Switch the crawl style (``None``: the next one). While crawling it waits
    until the crawl command is 0 and the robot is still on four limbs (a
    mid-stride switch jolts the robot), then blends over ``blend_s``; anywhere
    else it applies at once. Returns the style it will switch to."""
    if name is None:
      i = self.crawl_styles.index(self.style_pending or self.style)
      name = self.crawl_styles[(i + 1) % len(self.crawl_styles)]
    if name not in self.crawl_styles:
      raise ValueError(f"crawl style {name!r} not in {self.crawl_styles}")
    self.style_pending = None if name == self.style else name
    self.style_dwell = 0.0
    return name

  def force_gate(self) -> None:
    """Operator override: the gate the current mode waits on counts as passed
    on the next tick (STOPPING, HOLD, CRAWL_STOP, STAND_HOLD)."""
    self._force = self.mode in (STOPPING, HOLD, CRAWL_STOP, STAND_HOLD)

  # -- policies ---------------------------------------------------------------

  def _walk_target(self, g, w, q, qd) -> np.ndarray:
    cmd = self.walk_cmd if self.mode == WALK else np.zeros(3)
    obs = np.concatenate([g, w, cmd, q - self.q_stand, qd, self.walk_last])
    self.walk_last = np.clip(self.walk_net(obs), -self.walk_clip, self.walk_clip)
    return self.q_stand + self.walk_scale * self.walk_last

  def _trans_target(self, g, w, q, qd) -> np.ndarray:
    ref = self.schedule(self.phase)
    obs = np.concatenate(
      [g, w, q - self.q_stand, qd, self.trans_last, [self.phase, self.rate], ref - q]
    )
    self.trans_last = np.clip(self.trans_net(obs), -self.trans_clip, self.trans_clip)
    return ref + self.trans_scale * self.trans_last

  def _crawl_target(self, g, w, q, qd) -> np.ndarray:
    """Every crawl style on the same observation (each with its own last
    action); the active one's target."""
    cmd = self.crawl_cmd.copy()
    if self.mode != CRAWL:
      cmd[:3] = 0.0
    for k, net in self.crawl_nets.items():
      obs = np.concatenate([g, w, cmd, q - self.q_crawl, qd, self.crawl_last[k]])
      self.crawl_last[k] = np.clip(net(obs), -self.crawl_clip, self.crawl_clip)
    return self.q_crawl + 0.5 * self.crawl_last[self.style]

  # -- gates ------------------------------------------------------------------

  def _gate(self, m: int, g, w, qd, limb: LimbState) -> tuple[bool, str]:
    """Is mode ``m``'s exit condition met now? Also says what is missing."""
    G = self.gates
    speed = float(np.abs(qd).max())
    why: list[str] = []
    if m in (STOPPING, STAND_HOLD):
      if speed >= G.stand_joint_speed:
        why.append(f"|qd| {speed:.2f}")
      if np.linalg.norm(w) >= G.calm_gyro or abs(w[2]) >= G.calm_yaw:
        why.append(f"gyro {np.linalg.norm(w):.2f}")
      if limb.feet_gap >= G.feet_gap_m:
        why.append(f"feet gap {100 * limb.feet_gap:.1f} cm")
      if m == STAND_HOLD:
        if g[2] >= G.upright_gz:
          why.append(f"g_z {g[2]:.3f}")
        if limb.pelvis_height <= G.stand_height_m:
          why.append(f"height {limb.pelvis_height:.2f}")
    elif m in (HOLD, CRAWL_STOP):
      if speed >= G.crawl_joint_speed:
        why.append(f"|qd| {speed:.2f}")
      dg = float(np.linalg.norm(g - self.g_crawl))
      if dg >= (G.crawl_orient if m == HOLD else G.settle_orient):
        why.append(f"|g-g_crawl| {dg:.2f}")
      if not limb.down.all():
        up = [t for t, d in zip(LIMB_TAGS, limb.down) if not d]
        why.append("up: " + ",".join(up))
      if m == HOLD:
        dh = limb.above_limbs - self.crawl_above_limbs
        if abs(dh) >= G.crawl_height_tol_m:
          why.append(f"height {100 * dh:+.1f} cm")
    else:
      return False, ""
    return not why, " ".join(why)

  # -- one 50 Hz tick ---------------------------------------------------------

  def step(self, quat_wxyz, gyro, q, qd) -> np.ndarray:
    g = gravity_in_body(quat_wxyz)
    w = np.asarray(gyro, float)
    q = np.asarray(q, float)
    qd = np.asarray(qd, float)
    self.time += self.dt
    m = self.mode
    inv_t = 1.0 / self.transition_s
    self.rate = inv_t if m == DOWN else (-inv_t if m == UP else 0.0)

    # Shadow mode: every policy runs every tick on the live state, so each last
    # action is its own.
    t_walk = self._walk_target(g, w, q, qd)
    t_trans = self._trans_target(g, w, q, qd)
    t_crawl = self._crawl_target(g, w, q, qd)
    if m in WALK_MODES:
      target = t_walk
    elif m in CRAWL_MODES:
      target = t_crawl
    else:
      target = t_trans
    s = smoothstep((self.time - self.blend_t0) / self.blend_s)
    self.applied = self.blend_from + (target - self.blend_from) * s

    # Phase and modes (play_chain.py ChainSwitcher.__call__).
    self.phase = float(np.clip(self.phase + self.rate * self.dt, 0.0, 1.0))
    if m == STOPPING:
      self.phase = 0.0
    elif m == CRAWL_STOP:
      self.phase = 1.0
    self.limb = self.fk(quat_wxyz, q, self.gates.limb_down_m)
    self.gate_ok, self.gate_why = self._gate(m, g, w, qd, self.limb)
    self.dwell = self.dwell + self.dt if self.gate_ok else 0.0
    need = self.dwell_s if m == HOLD else 0.5
    if self._force:
      self.dwell = max(self.dwell, need)
      self._force = False
    ready = self.dwell >= need
    going_up = self.queued == UP
    if m == STOPPING and ready:
      self._set(DOWN, blend=True)
    elif m == DOWN and self.phase >= 1.0:
      self._set(HOLD, blend=False)
    elif m == HOLD and ready and not going_up:
      self._set(CRAWL, blend=True)
    elif m == CRAWL_STOP and ready:
      self._set(HOLD, blend=True)
      self.queued = UP
    elif m == HOLD and ready and going_up:
      self._set(UP, blend=False)
      self.queued = -1
    elif m == UP and self.phase <= 0.0:
      self._set(STAND_HOLD, blend=False)
    elif m == STAND_HOLD and ready and self.queued != DOWN:
      self._set(WALK, blend=True)
    if self.mode == STAND_HOLD and self.queued == DOWN:
      self._set(DOWN, blend=False)
      self.queued = -1
    if (
      self.mode == HOLD
      and m == HOLD
      and self.hold_timeout_s > 0
      and not self.gate_ok
      and self.time - self.mode_t0 >= self.hold_timeout_s
    ):
      # No crawl gate after hold_timeout_s: stand up rather than sit in a bad
      # (hot) pose.
      self._set(UP, blend=False)
      self.queued = -1
      self.hold_timed_out = True
    self._switch_style(g, qd)
    if self.mode == WALK and m != WALK:
      self.walk_cmd[:] = 0.0
    if self.mode == CRAWL and m != CRAWL:
      self.crawl_cmd[:3] = 0.0

    self.applied = np.clip(self.applied, self.q_lo, self.q_hi)
    return self.applied.copy()

  def _switch_style(self, g, qd) -> None:
    if self.style_pending is None:
      self.style_why = ""
      return
    if self.mode not in CRAWL_MODES:  # the crawl is not driving: switch now
      self.style, self.style_pending, self.style_why = self.style_pending, None, ""
      return
    if self.mode != CRAWL:  # CRAWL_STOP: switch once in HOLD
      self.style_why = "after the stop"
      return
    G = self.gates
    if np.any(self.crawl_cmd[:3]):
      self.style_why = "set the crawl command to 0 (x)"
      still = False
    else:
      speed = float(np.abs(qd).max())
      dg = float(np.linalg.norm(g - self.g_crawl))
      down = self.limb is not None and bool(self.limb.down.all())
      still = speed < G.crawl_joint_speed and dg < G.settle_orient and down
      self.style_why = "" if still else f"settling (|qd| {speed:.2f})"
    self.style_dwell = self.style_dwell + self.dt if still else 0.0
    if self.style_dwell >= 0.5:
      self.style, self.style_pending, self.style_why = self.style_pending, None, ""
      self.style_dwell = 0.0
      self.blend_t0 = self.time
      self.blend_from = self.applied.copy()

  def expected_gravity(self) -> np.ndarray:
    """Where gravity should point in the pelvis frame in the current mode (for
    a tilt e-stop that also works while prone)."""
    if self.mode in (WALK, STOPPING, STAND_HOLD):
      return np.array([0.0, 0.0, -1.0])
    if self.mode in (HOLD, CRAWL, CRAWL_STOP):
      return self.g_crawl
    return self.schedule.gravity(self.phase)
