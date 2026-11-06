"""Performance-gated robustness curriculum for the crawl task.

One global ``level`` in [0, 1] drives everything the policy is hardened
against, in stages:

* level 0: learn to crawl -- mild physics randomization, half sensor noise, no
  pushes, no delay, and no torque-limit penalties, so the gait can form freely;
* from 0.1: pushes, wider randomization, more noise, command delay, ramping to
  their full values at level 1;
* from 0.2: the torque-limit penalties fade in (motors fully on at 0.6, the
  braced wrists at 0.8), once the robot already crawls well.

Level 1 is well past the source project's settings, aimed at the real robot.

The level only rises once the robot can crawl. Every finished episode is scored
by the command term's own accumulators -- distance crawled in the commanded
direction over distance commanded -- and by whether it ended in a roll-over.
Both are kept as exponential averages over recent episodes. After a warm-up,
the level steps up when the policy tracks well and rarely flips, and has had
``min_iterations_per_level`` iterations to adapt to the current level; it steps
back down if tracking or stability collapse, so a disturbance the policy cannot
yet handle never takes over the run.

Each knob in ``RobustnessRamp`` is ``(value at level 0, value at level 1)``,
linear in between, or ``(start value, end value, start level, end level)``: held
at the start value until ``start level``, linear to the end value at ``end
level``. The current value of every knob is logged under
``Curriculum/<term name>/...``.

The level is not stored in checkpoints. A resumed run starts again at
``start_level`` (default 0) and climbs back at one step per
``min_iterations_per_level`` while the policy keeps meeting the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.utils.buffers.delay_buffer import DelayBuffer
from mjlab.utils.noise import UniformNoiseCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


Knob = tuple[float, float] | tuple[float, float, float, float]


@dataclass(frozen=True)
class RobustnessRamp:
  """Every knob as (v0, v1) over levels [0, 1] or (v0, v1, start, end)."""

  push_xy_mps: Knob = (0.0, 1.0)
  """Velocity kick added to the base, each horizontal axis, every 4-8 s."""
  push_yaw_radps: Knob = (0.0, 0.6)
  """Spin kick about the vertical axis."""
  friction_low: Knob = (0.5, 0.25)
  friction_high: Knob = (1.0, 1.25)
  """Robot geom friction, sampled per episode in [low, high]."""
  mass_scale: Knob = (0.05, 0.15)
  """Every body's mass scaled by 1 +/- this."""
  payload_low_kg: Knob = (-0.5, -2.0)
  payload_high_kg: Knob = (0.5, 3.0)
  """Mass added to the torso: battery, compute, cables."""
  com_offset_m: Knob = (0.0, 0.03)
  """Torso centre-of-mass shift, each axis."""
  joint_friction_nm: Knob = (0.1, 0.6)
  """Joint dry friction, sampled in [0, this]. The stock model has none."""
  armature_scale: Knob = (0.02, 0.15)
  """Rotor inertia scaled by 1 +/- this."""
  pd_gain_scale: Knob = (0.0, 0.15)
  """Motor stiffness and damping scaled by 1 +/- this."""
  effort_scale_low: Knob = (1.0, 0.85)
  """Motor peak torque scaled into [this, 1]: battery sag, warm motors."""
  obs_noise_scale: Knob = (0.5, 1.5)
  """Multiplier on each actor term's base noise."""
  delay_max_steps: Knob = (0.0, 4.0)
  delay_min_steps: Knob = (0.0, 1.0)
  """Motor command delay in 5 ms physics steps (level 1: 5-20 ms)."""
  motor_penalty_weight: Knob = (0.0, -5.0, 0.2, 0.6)
  """Weight of the motor torque-margin penalty: off until the crawl is learned."""
  braced_penalty_weight: Knob = (0.0, -1.0, 0.2, 0.8)
  """Weight of the braced-wrist torque penalty: off until the crawl is learned."""


class crawl_robustness:  # noqa: N801 -- MJLab curriculum terms are lowercase.
  """Curriculum term; see the module docstring.

  Params:
    command_name: The crawl velocity command (provides the progress accumulators).
    ramp: ``RobustnessRamp``.
    events: Event term names for ``push``, ``friction``, ``mass``, ``payload``,
      ``com``, ``joint_friction``, ``armature``, ``pd_gains``, ``effort``.
    noisy_obs: Actor observation term names whose noise is scaled.
    motor_reward: Reward term name for the motor torque-margin penalty, or None.
    braced_reward: Reward term name for the braced-wrist penalty, or None.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    p = cfg.params
    self.ramp: RobustnessRamp = p["ramp"]
    self.step: float = p.get("step", 0.1)
    self.level: float = p.get("start_level", 0.0)
    self.last_change_iter = 0
    self.prog_sum = self.prog_n = 0.0
    self.flip_sum = self.flip_n = 0.0

    self.events = {k: env.event_manager.get_term_cfg(v) for k, v in p["events"].items()}
    self.noise_terms = []
    for name in p["noisy_obs"]:
      term = env.observation_manager.get_term_cfg("actor", name)
      noise = term.noise
      if noise is None:  # Corruption disabled for this group: nothing to scale.
        continue
      assert isinstance(noise, UniformNoiseCfg), name
      assert isinstance(noise.n_min, float) and isinstance(noise.n_max, float), name
      self.noise_terms.append((term, noise.n_min, noise.n_max))
    rewards = env.reward_manager
    motor, braced = p.get("motor_reward"), p.get("braced_reward")
    self.motor = rewards.get_term_cfg(motor) if motor else None
    self.braced = rewards.get_term_cfg(braced) if braced else None
    robot = env.scene["robot"]
    buffers = {id(a._delay_buffer): a._delay_buffer for a in robot.actuators}
    self.delay_buffers: list[DelayBuffer] = [b for b in buffers.values() if b]
    self._apply()

  def _at(self, knob: Knob) -> float:
    v0, v1, *span = knob
    start, end = span if span else (0.0, 1.0)
    t = min(max((self.level - start) / (end - start), 0.0), 1.0)
    return v0 + (v1 - v0) * t

  def _apply(self) -> None:
    r, e = self.ramp, self.events
    v, w = self._at(r.push_xy_mps), self._at(r.push_yaw_radps)
    e["push"].params["velocity_range"] = {"x": (-v, v), "y": (-v, v), "yaw": (-w, w)}
    e["friction"].params["ranges"] = (
      self._at(r.friction_low),
      self._at(r.friction_high),
    )
    m = self._at(r.mass_scale)
    e["mass"].params["ranges"] = (1.0 - m, 1.0 + m)
    e["payload"].params["ranges"] = (
      self._at(r.payload_low_kg),
      self._at(r.payload_high_kg),
    )
    c = self._at(r.com_offset_m)
    e["com"].params["ranges"] = (-c, c)
    e["joint_friction"].params["ranges"] = (0.0, self._at(r.joint_friction_nm))
    a = self._at(r.armature_scale)
    e["armature"].params["ranges"] = (1.0 - a, 1.0 + a)
    g = self._at(r.pd_gain_scale)
    e["pd_gains"].params["kp_range"] = (1.0 - g, 1.0 + g)
    e["pd_gains"].params["kd_range"] = (1.0 - g, 1.0 + g)
    e["effort"].params["effort_limit_range"] = (self._at(r.effort_scale_low), 1.0)
    # Noise configs cache their tensors, so swap in fresh ones.
    s = self._at(r.obs_noise_scale)
    for term, lo, hi in self.noise_terms:
      term.noise = UniformNoiseCfg(n_min=lo * s, n_max=hi * s)
    if self.motor is not None:
      self.motor.weight = self._at(r.motor_penalty_weight)
    if self.braced is not None:
      self.braced.weight = self._at(r.braced_penalty_weight)
    hi_lag = round(self._at(r.delay_max_steps))
    lo_lag = min(round(self._at(r.delay_min_steps)), hi_lag)
    for buffer in self.delay_buffers:
      buffer.max_lag = min(hi_lag, buffer._buffer.max_length - 1)
      buffer.min_lag = min(lo_lag, buffer.max_lag)

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
    del ramp, events, noisy_obs, motor_reward, braced_reward, start_level, step
    # Score the episodes that just ended (before the command term resets).
    if len(env_ids) > 0:
      cmd = env.command_manager.get_term(command_name)
      commanded = cmd.commanded_acc[env_ids]  # type: ignore[attr-defined]
      crawled = cmd.progress_acc[env_ids]  # type: ignore[attr-defined]
      # Never score a zero-distance command: 0/0 would poison the average.
      scored = commanded >= max(min_commanded_m, 1e-3)
      ratio = (crawled[scored] / commanded[scored]).clamp(max=1.2)
      flipped = env.termination_manager.terminated[env_ids].float()
      self.prog_sum = self.prog_sum * ema_decay + ratio.sum().item()
      self.prog_n = self.prog_n * ema_decay + float(ratio.numel())
      self.flip_sum = self.flip_sum * ema_decay + flipped.sum().item()
      self.flip_n = self.flip_n * ema_decay + float(flipped.numel())
    progress = self.prog_sum / self.prog_n if self.prog_n > 50 else 0.0
    flip_rate = self.flip_sum / self.flip_n if self.flip_n > 50 else 1.0

    iteration = env.common_step_counter // steps_per_iteration
    settled = iteration - self.last_change_iter >= min_iterations_per_level
    if iteration >= warmup_iterations and settled:
      old = self.level
      if progress >= promote_progress and flip_rate <= promote_max_flip_rate:
        self.level = min(1.0, round(self.level + self.step, 6))
      elif progress < demote_progress or flip_rate > demote_flip_rate:
        self.level = max(0.0, round(self.level - self.step, 6))
      if self.level != old:
        self.last_change_iter = iteration
        self._apply()

    r = self.ramp
    device = env.device
    values = {
      "level": self.level,
      "gate_progress_ratio": progress,
      "gate_flip_rate": flip_rate,
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
    return {k: torch.tensor(v, device=device) for k, v in values.items()}
