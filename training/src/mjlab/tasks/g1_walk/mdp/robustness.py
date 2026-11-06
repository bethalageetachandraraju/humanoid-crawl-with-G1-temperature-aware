"""The crawl task's robustness curriculum, gated on walking performance.

``crawl_robustness`` (see ``g1_crawl/mdp/robustness.py``) owns every
disturbance the policy is hardened against -- pushes, physics randomization,
sensor noise, command delay, torque-limit penalties -- behind one global level
that only rises while the policy copes. Its gate scores finished episodes by
distance crawled over distance commanded, read from the crawl command term.
``walk_robustness`` keeps the ramp and the gate logic and scores the velocity
command instead: planar displacement in the commanded direction over commanded
displacement (``UniformVelocityCommand`` integrates the command over the
episode), plus the fall rate from the terminations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.g1_crawl.mdp.robustness import RobustnessRamp, crawl_robustness
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg

WALK_RAMP = RobustnessRamp(
  # Level 0 is already a deployable floor (the stock G1 recipes' ranges), so a
  # policy exported before the curriculum has climbed is not a sim-only one;
  # small pushes from the start also keep the gait from settling into a stand.
  push_xy_mps=(0.3, 1.0),
  push_yaw_radps=(0.1, 0.6),
  friction_low=(0.4, 0.25),
  friction_high=(1.0, 1.25),
  mass_scale=(0.10, 0.15),
  payload_low_kg=(-0.5, -2.0),
  payload_high_kg=(0.5, 3.0),
  com_offset_m=(0.01, 0.03),
  joint_friction_nm=(0.1, 0.6),
  armature_scale=(0.05, 0.15),
  pd_gain_scale=(0.05, 0.15),
  effort_scale_low=(0.9, 0.85),
  obs_noise_scale=(1.0, 1.5),
  delay_max_steps=(2.0, 4.0),
  delay_min_steps=(0.0, 1.0),
  motor_penalty_weight=(0.0, -5.0, 0.2, 0.6),
  braced_penalty_weight=(0.0, -1.0, 0.2, 0.8),
)
"""Level-0 to level-1 values for walking; same end points as the crawl."""

WALK_RAMP_HARD = RobustnessRamp(
  # Level 0 is WALK_RAMP's level 1 (what the deployed walk already copes with);
  # level 1 widens everything a robot without system identification can be off
  # by: heavier/lighter, weaker motors, gains and rotor inertia off by a
  # quarter, sticky joints, slippery floors, harder kicks, noisier sensors.
  push_xy_mps=(1.0, 1.2),
  push_yaw_radps=(0.6, 0.8),
  friction_low=(0.25, 0.2),
  friction_high=(1.25, 1.4),
  mass_scale=(0.15, 0.2),
  payload_low_kg=(-2.0, -3.0),
  payload_high_kg=(3.0, 5.0),
  com_offset_m=(0.03, 0.05),
  joint_friction_nm=(0.6, 1.0),
  armature_scale=(0.15, 0.3),
  pd_gain_scale=(0.15, 0.25),
  effort_scale_low=(0.85, 0.75),
  obs_noise_scale=(1.5, 2.0),
  delay_max_steps=(4.0, 4.0),
  delay_min_steps=(1.0, 1.0),
  motor_penalty_weight=(-5.0, -5.0),
  braced_penalty_weight=(-1.0, -1.0),
)
"""The hardened walk: starts where ``WALK_RAMP`` ends."""


class projected_gravity_tilted:  # noqa: N801 -- MJLab terms are lowercase.
  """Observation: base-frame gravity as an IMU mounted (or estimating) a few
  degrees off would report it: a fixed tilt, drawn per episode, about a random
  horizontal axis. The actor sees this; the critic keeps the true gravity."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv) -> None:
    self.max_rad = math.radians(cfg.params.get("max_deg", 2.0))
    self.rot = torch.eye(3, device=env.device).repeat(env.num_envs, 1, 1)
    self.device = env.device
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(self.rot.shape[0], device=self.device)
    n = len(env_ids)
    if n == 0:
      return
    angle = self.max_rad * torch.rand(n, device=self.device)
    heading = 2.0 * math.pi * torch.rand(n, device=self.device)
    k = torch.stack([heading.cos(), heading.sin(), torch.zeros_like(heading)], 1)
    kx = torch.zeros(n, 3, 3, device=self.device)  # Cross-product matrix of k.
    kx[:, 0, 2], kx[:, 1, 2] = k[:, 1], -k[:, 0]
    kx[:, 2, 0], kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = angle.sin()[:, None, None], angle.cos()[:, None, None]
    eye = torch.eye(3, device=self.device).expand(n, 3, 3)
    self.rot[env_ids] = eye + s * kx + (1.0 - c) * (kx @ kx)  # Rodrigues.

  def __call__(self, env: ManagerBasedRlEnv, max_deg: float = 2.0) -> torch.Tensor:
    del max_deg  # Read once at construction.
    g = env.scene["robot"].data.projected_gravity_b
    return torch.einsum("nij,nj->ni", self.rot, g)


class walk_robustness(crawl_robustness):  # noqa: N801 -- MJLab terms are lowercase.
  """``crawl_robustness`` scored on the velocity command. Same params."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    ramp: RobustnessRamp,
    events: dict[str, str],
    noisy_obs: tuple[str, ...],
    motor_reward: str | None = None,
    braced_reward: str | None = None,
    start_level: float = 0.0,
    step: float = 0.1,
    warmup_iterations: int = 500,
    min_iterations_per_level: int = 150,
    promote_progress: float = 0.75,
    promote_max_flip_rate: float = 0.05,
    demote_progress: float = 0.45,
    demote_flip_rate: float = 0.25,
    min_commanded_m: float = 1.0,
    ema_decay: float = 0.995,
    steps_per_iteration: int = 24,
  ) -> dict[str, torch.Tensor]:
    if len(env_ids) > 0:
      cmd = env.command_manager.get_term(command_name)
      assert isinstance(cmd, UniformVelocityCommand)
      commanded = cmd.commanded_displacement_w[env_ids]
      robot = cmd.robot
      actual = (
        robot.data.root_link_pos_w[env_ids, :2] - cmd.episode_start_pos_w[env_ids]
      )
      length = torch.linalg.vector_norm(commanded, dim=1)
      scored = length >= max(min_commanded_m, 1e-3)
      # Distance made good along the commanded direction, over distance commanded.
      along = (actual[scored] * commanded[scored]).sum(dim=1) / length[scored]
      ratio = (along / length[scored]).clamp(max=1.2)
      fell = env.termination_manager.terminated[env_ids].float()
      self.prog_sum = self.prog_sum * ema_decay + ratio.sum().item()
      self.prog_n = self.prog_n * ema_decay + float(ratio.numel())
      self.flip_sum = self.flip_sum * ema_decay + fell.sum().item()
      self.flip_n = self.flip_n * ema_decay + float(fell.numel())
    progress = self.prog_sum / self.prog_n if self.prog_n > 50 else 0.0
    fall_rate = self.flip_sum / self.flip_n if self.flip_n > 50 else 1.0

    iteration = env.common_step_counter // steps_per_iteration
    settled = iteration - self.last_change_iter >= min_iterations_per_level
    if iteration >= warmup_iterations and settled:
      old = self.level
      if progress >= promote_progress and fall_rate <= promote_max_flip_rate:
        self.level = min(1.0, round(self.level + self.step, 6))
      elif progress < demote_progress or fall_rate > demote_flip_rate:
        self.level = max(0.0, round(self.level - self.step, 6))
      if self.level != old:
        self.last_change_iter = iteration
        self._apply()

    r = self.ramp
    values = {
      "level": self.level,
      "gate_progress_ratio": progress,
      "gate_fall_rate": fall_rate,
      "push_mps": self._at(r.push_xy_mps),
      "friction_min": self._at(r.friction_low),
      "mass_scale_pm": self._at(r.mass_scale),
      "payload_max_kg": self._at(r.payload_high_kg),
      "com_offset_cm": self._at(r.com_offset_m) * 100,
      "joint_friction_max_nm": self._at(r.joint_friction_nm),
      "motor_gain_scale_pm": self._at(r.pd_gain_scale),
      "motor_strength_min": self._at(r.effort_scale_low),
      "obs_noise_scale": self._at(r.obs_noise_scale),
      "delay_max_ms": round(self._at(r.delay_max_steps)) * env.physics_dt * 1000,
      "motor_penalty_weight": self._at(r.motor_penalty_weight),
      "braced_penalty_weight": self._at(r.braced_penalty_weight),
    }
    return {k: torch.tensor(v, device=env.device) for k, v in values.items()}


_ROBOT = SceneEntityCfg("robot")


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def pd_gains_correlated(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  joint_spread: float = 0.1,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Motor stiffness and damping scaled per robot by one draw for every motor
  (``kp_range``, ``kd_range``: the robustness curriculum's knob) times a small
  per-motor draw (``1 +- joint_spread``). A real gain mismatch (firmware, a
  softer drive train) is shared by the motors; independent draws per motor
  almost never make the whole robot soft at once.

  For the position actuators of the G1 (kp in gainprm[0] and -biasprm[1], kd in
  -biasprm[2]), scaled from the model defaults."""
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)
  n = len(env_ids)
  ctrl = torch.cat([a.global_ctrl_ids for a in asset.actuators]).to(env.device)

  def draw(lo: float, hi: float) -> torch.Tensor:
    shared = torch.empty(n, 1, device=env.device).uniform_(lo, hi)
    each = torch.empty(n, len(ctrl), device=env.device).uniform_(
      1.0 - joint_spread, 1.0 + joint_spread
    )
    return shared * each

  kp, kd = draw(*kp_range), draw(*kd_range)
  gain = env.sim.get_default_field("actuator_gainprm")
  bias = env.sim.get_default_field("actuator_biasprm")
  rows, cols = env_ids[:, None], ctrl[None, :]
  env.sim.model.actuator_gainprm[rows, cols, 0] = gain[ctrl, 0] * kp
  env.sim.model.actuator_biasprm[rows, cols, 1] = bias[ctrl, 1] * kp
  env.sim.model.actuator_biasprm[rows, cols, 2] = bias[ctrl, 2] * kd
