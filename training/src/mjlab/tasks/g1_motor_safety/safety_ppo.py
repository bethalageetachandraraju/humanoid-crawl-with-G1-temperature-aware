"""Hamilton-Jacobi safety learning for the G1 motors: Safety PPO on rsl_rl.

THE PROBLEM. Learn the motor-safety value function V(s) and a fallback policy that
maximizes it. The failure set is motor overload, read from the margin

    g(s) = min_j (T_LIMIT - T_w,j) / G_SCALE        (clamped to +-3)

(``mdp.motor_margin``): g < 0 as soon as any winding is past its rating. The
value is the avoid (safety) Bellman fixed point of Fisac et al. (ICRA 2019), as
implemented by safety_sb3 (``safety_sb3.backups.avoid_target``, Safe Robotics
Lab):

    V(s) = (1 - gamma) g(s) + gamma min( g(s), V(s') ),   V(terminal) = g.

V(s) >= 0 certifies (up to learning error) that the fallback can keep every
motor under its rating from s on. There is no reward to shape: the env's reward
channel IS g (``scale_rewards_by_dt=False``), and timeouts are not bootstrapped
(they are horizon cut-offs, valued at their own margin, safety_sb3's default).

WHY RSL_RL AND NOT SB3. The fallback has to start from a policy that already
walks, i.e. our rsl_rl G1 walker; safety_sb3's learners are SB3 (different
networks, no warm start). So the PPO update is rsl_rl's and only the return
computation is replaced: the advantages are safety_sb3's tensor-buffer recursion
(``TensorSafetyRolloutBuffer.compute_returns_and_advantage``) with the avoid
target from ``safety_sb3.backups``, imported, not re-implemented.

HANDOVER STATES (``PilotEnvWrapper``). The fallback takes over from the task
policy at deployment, mid-gait and with hot motors. To train on that
distribution, each episode starts with a frozen pilot (the task policy) driving
for a random time; the transitions the pilot drove carry no gradient (their
advantage is zeroed and their value target equals the prediction), so the
learner is only scored on what it did after the handover.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class SafetyPPO(PPO):
  """rsl_rl PPO with the avoid (HJ safety) backup. Extra kwargs:

  ``actor_warmup_updates``  updates during which only the critic learns (a
      warm-started actor is kept while the fresh value function fits).
  """

  def __init__(self, *args, actor_warmup_updates: int = 0, **kwargs) -> None:
    # Imported here so mjlab's task discovery does not need safety_sb3.
    from safety_sb3 import backups  # type: ignore[import-not-found]

    self._backups = backups
    super().__init__(*args, **kwargs)
    self.actor_warmup_updates = int(actor_warmup_updates)
    self._n_updates = 0
    st = self.storage
    self._pilot = torch.zeros(
      st.num_transitions_per_env, st.num_envs, dtype=torch.bool, device=self.device
    )

  def process_env_step(
    self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
  ) -> None:
    # The reward is the margin g(s): no intrinsic reward, no timeout bootstrap.
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    pilot = extras.get("pilot")
    self._pilot[self.storage.step] = (
      pilot.to(self.device) if pilot is not None else False
    )
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  @torch.no_grad()
  def compute_returns(self, obs: TensorDict) -> None:
    st = self.storage
    hidden = self.critic.get_hidden_state()
    last_values = self.critic(obs).detach().reshape(-1)
    self.critic.reset(hidden_state=hidden)
    g = st.rewards.reshape(st.num_transitions_per_env, -1)
    values = st.values.reshape(st.num_transitions_per_env, -1)
    dones = st.dones.reshape(st.num_transitions_per_env, -1).float()
    adv = torch.zeros_like(last_values)
    advantages = torch.zeros_like(values)
    for step in reversed(range(st.num_transitions_per_env)):
      v_next = (
        last_values if step == st.num_transitions_per_env - 1 else values[step + 1]
      )
      not_done = 1.0 - dones[step]
      target = self._backups.avoid_target(g[step], v_next, not_done, self.gamma)
      delta = target - values[step]
      adv = delta + self.gamma * self.lam * not_done * adv
      advantages[step] = adv
    returns = advantages + values
    # Pilot-driven transitions: no policy gradient, no value error.
    learner = ~self._pilot
    advantages = torch.where(learner, advantages, torch.zeros_like(advantages))
    returns = torch.where(learner, returns, values)
    st.returns = returns.reshape(st.returns.shape)
    if not self.normalize_advantage_per_mini_batch:
      a = advantages[learner]
      mean, std = (a.mean(), a.std()) if a.numel() > 1 else (0.0, 1.0)
      advantages = torch.where(learner, (advantages - mean) / (std + 1e-8), advantages)
    st.advantages = advantages.reshape(st.values.shape)

  def update(self) -> dict[str, float]:
    warm = self._n_updates < self.actor_warmup_updates
    schedule = self.schedule
    if warm:
      for p in self.actor.parameters():
        p.requires_grad_(False)
      # A frozen actor shows a float-noise KL (~1e-9 > 0), which the adaptive
      # schedule reads as "too cautious" and ramps the rate to its 1e-2 cap;
      # the first actor step after warm-up then wrecks the walker. Hold it.
      self.schedule = "fixed"
    try:
      out = super().update()
    finally:
      self.schedule = schedule
      if warm:
        for p in self.actor.parameters():
          p.requires_grad_(True)
    self._n_updates += 1
    out["pilot_frac"] = float(self._pilot.float().mean())
    out["value_mean"] = float(self.storage.values.mean())
    out["margin_mean"] = float(self.storage.rewards.mean())
    return out


def load_actor_from_checkpoint(path: str, obs: TensorDict, cfg: dict, device: str):
  """The actor (obs -> mean action) of an rsl_rl checkpoint."""
  sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
  act_dim = sd["distribution.std_param"].shape[0]
  a = cfg["actor"]
  actor = MLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    act_dim,
    hidden_dims=a["hidden_dims"],
    activation=a["activation"],
    obs_normalization=a["obs_normalization"],
    distribution_cfg=a.get("distribution_cfg"),
  ).to(device)
  actor.load_state_dict(sd, strict=True)
  actor.eval()
  return actor


class PilotEnvWrapper:
  """Lets a frozen pilot policy drive the first part of each episode.

  At every reset an env gets a pilot time: with probability ``prob`` uniform in
  [0, ``max_s``], else 0. While it runs, the pilot's action replaces the
  learner's; ``extras["pilot"]`` marks those steps for ``SafetyPPO``.
  """

  def __init__(self, env, pilot, prob: float, max_s: float) -> None:
    self.env = env
    self.pilot = pilot
    self.prob = float(prob)
    self.max_steps = int(max_s / env.unwrapped.step_dt)
    self.left = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self._resample(torch.ones_like(self.left, dtype=torch.bool))
    self._obs = env.get_observations()

  def _resample(self, mask: torch.Tensor) -> None:
    n = int(mask.sum())
    if n == 0:
      return
    dev = self.left.device
    on = torch.rand(n, device=dev) < self.prob
    steps = torch.randint(0, max(self.max_steps, 1), (n,), device=dev)
    self.left[mask] = torch.where(on, steps, torch.zeros_like(steps))

  def __getattr__(self, name: str) -> Any:
    return getattr(self.env, name)

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.env.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:
    self.env.episode_length_buf = value

  def get_observations(self) -> TensorDict:
    self._obs = self.env.get_observations()
    return self._obs

  def step(self, actions: torch.Tensor):
    active = self.left > 0
    if bool(active.any()):
      with torch.no_grad():
        pa = self.pilot(
          TensorDict({"actor": self._obs["actor"]}, batch_size=[actions.shape[0]])
        )
      actions = torch.where(active[:, None], pa.to(actions.device), actions)
    obs, rew, dones, extras = self.env.step(actions)
    self.left = (self.left - 1).clamp(min=0)
    self._resample(dones.bool().reshape(-1))
    extras["pilot"] = active
    self._obs = obs
    return obs, rew, dones, extras


class SafetyRunner(VelocityOnPolicyRunner):
  """Runner for the motor-safety value + fallback.

  Train-config keys (beyond rsl_rl's): ``pilot_checkpoint`` (frozen task policy
  for the handover states; env var ``MOTOR_SAFETY_PILOT`` overrides),
  ``pilot_prob``, ``pilot_max_s``, and ``init_actor_only`` (``--agent.resume``
  loads only the actor of the given checkpoint: the value net starts fresh,
  since a reward critic is not a safety value).
  """

  def __init__(
    self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
  ) -> None:
    self._init_actor_only = bool(train_cfg.pop("init_actor_only", True))
    path = os.environ.get("MOTOR_SAFETY_PILOT") or train_cfg.pop("pilot_checkpoint", "")
    train_cfg.pop("pilot_checkpoint", None)
    prob = float(train_cfg.pop("pilot_prob", 0.7))
    max_s = float(train_cfg.pop("pilot_max_s", 15.0))
    if path:
      obs = env.get_observations().to(device)
      pilot = load_actor_from_checkpoint(path, obs, train_cfg, device)
      env = PilotEnvWrapper(env, pilot, prob, max_s)
      print(f"[safety] pilot {path}: p={prob}, up to {max_s} s per episode")
    super().__init__(env, train_cfg, log_dir, device)  # type: ignore[arg-type]

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    if self._init_actor_only and load_cfg is None:
      load_cfg = {
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
      }
      print(f"[safety] warm start: actor only from {path} (fresh value function)")
    return super().load(path, load_cfg, strict, map_location)
