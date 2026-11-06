"""PPO for fine-tuning a deployed policy without forgetting it.

Three additions to rsl_rl's PPO, all aimed at changing one behaviour (here: how
much torque the robot holds) while keeping everything else the policy does:

* **Anchor.** ``anchor_coef * mean ||mu(s) - mu_old(s)||^2`` on every rollout
  state, where ``mu_old`` is a frozen copy of the actor as it was loaded. The
  coefficient decays linearly to ``anchor_coef_final`` over
  ``anchor_decay_updates`` updates, so the policy is held close at first and
  given room once the critic and the new reward terms have settled.
* **Critic warm-up.** For ``actor_freeze_updates`` updates only the value loss
  is trained: the loaded critic has never seen the new reward terms.
* **Frozen actor normalizer.** Its statistics are baked into the ONNX on the
  robot; with ``freeze_actor_normalizer`` they stay exactly as deployed.

``reset_learning_rate`` replaces the learning rate restored from the
checkpoint's optimizer with the configured one (the adaptive schedule still
moves it from there).

Non-recurrent, single-GPU, no RND: the update below is rsl_rl's PPO.update
with those branches removed and the additions marked.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class AnchoredPPO(PPO):
  def __init__(
    self,
    *args,
    anchor_coef: float = 1.0,
    anchor_coef_final: float = 0.1,
    anchor_decay_updates: int = 2000,
    actor_freeze_updates: int = 50,
    freeze_actor_normalizer: bool = True,
    reset_learning_rate: bool = True,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    assert self.rnd is None and not self.is_multi_gpu
    assert not (self.actor.is_recurrent or self.critic.is_recurrent)
    self.anchor_coef = anchor_coef
    self.anchor_coef_final = anchor_coef_final
    self.anchor_decay_updates = anchor_decay_updates
    self.actor_freeze_updates = actor_freeze_updates
    self.freeze_actor_normalizer = freeze_actor_normalizer
    self.reset_learning_rate = reset_learning_rate
    self.configured_learning_rate = self.learning_rate
    self.anchor: nn.Module | None = None
    self.num_updates_done = 0

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    out = super().load(loaded_dict, load_cfg, strict)
    if self.reset_learning_rate:
      self.learning_rate = self.configured_learning_rate
      for group in self.optimizer.param_groups:
        group["lr"] = self.learning_rate
    return out

  def _current_anchor_coef(self) -> float:
    t = min(self.num_updates_done / max(self.anchor_decay_updates, 1), 1.0)
    return self.anchor_coef + (self.anchor_coef_final - self.anchor_coef) * t

  def update(self) -> dict[str, float]:
    # Addition: freeze a copy of the actor as loaded (the first update runs
    # after the checkpoint is restored).
    if self.anchor is None:
      self.anchor = copy.deepcopy(self._raw_actor).eval()
      for p in self.anchor.parameters():
        p.requires_grad_(False)
    freeze_actor = self.num_updates_done < self.actor_freeze_updates
    anchor_coef = self._current_anchor_coef()

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_symmetry_loss = 0.0 if self.symmetry else None
    mean_anchor_loss = 0.0

    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in generator:
      assert batch.observations is not None and batch.advantages is not None
      assert batch.old_distribution_params is not None
      original_batch_size = batch.observations.batch_size[0]
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          adv = batch.advantages
          batch.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      with torch.amp.autocast(
        device_type=torch.device(self.device).type,
        enabled=self.use_mixed_precision,
        dtype=torch.bfloat16,
      ):
        self.actor(batch.observations, masks=batch.masks, stochastic_output=True)
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
        values = self.critic(batch.observations, masks=batch.masks)
        distribution_params = tuple(
          p[:original_batch_size] for p in self.actor.output_distribution_params
        )
        entropy = self.actor.output_entropy[:original_batch_size]
        mean = self.actor.output_mean[:original_batch_size]

        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            old_params = tuple(batch.old_distribution_params)
            kl = self.actor.get_kl_divergence(old_params, distribution_params)
            kl_mean = torch.mean(kl)
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for param_group in self.optimizer.param_groups:
              param_group["lr"] = self.learning_rate

        ratio = torch.exp(
          actions_log_prob - torch.squeeze(batch.old_actions_log_prob)  # type: ignore
        )
        advantages = torch.squeeze(batch.advantages)  # type: ignore
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(
            -self.clip_param, self.clip_param
          )
          value_losses = (values - batch.returns).pow(2)
          value_losses_clipped = (value_clipped - batch.returns).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = (batch.returns - values).pow(2).mean()

        # Addition: anchor to the loaded actor on the original (unmirrored) states.
        with torch.no_grad():
          old_mean = self.anchor(batch.observations[:original_batch_size])
        anchor_loss = (mean - old_mean).square().mean()

        if freeze_actor:
          # Addition: critic warm-up, the actor is not trained.
          loss = self.value_loss_coef * value_loss
        else:
          loss = (
            surrogate_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy.mean()
            + anchor_coef * anchor_loss
          )
          if self.symmetry:
            symmetry_loss = self.symmetry.compute_loss(
              self.actor, batch, original_batch_size
            )
            if self.symmetry.use_mirror_loss:
              loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss
        if self.symmetry and freeze_actor:
          with torch.no_grad():
            symmetry_loss = self.symmetry.compute_loss(
              self.actor, batch, original_batch_size
            )

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_anchor_loss += anchor_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    obs: Any = self.storage.observations.flatten(0, 1)
    if not self.freeze_actor_normalizer:
      self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "anchor": mean_anchor_loss / num_updates,
      "anchor_coef": anchor_coef,
      "actor_frozen": float(freeze_actor),
    }
    if mean_symmetry_loss is not None:
      loss_dict["symmetry"] = mean_symmetry_loss / num_updates
    self.storage.clear()
    self.num_updates_done += 1
    return loss_dict
