"""PPO settings copied from the checked-in Isaac Lab crawl runner config."""

from dataclasses import dataclass, fields

import tyro

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.g1_crawl.mdp.symmetry import make_crawl_symmetry_augmentation
from mjlab.tasks.g1_crawl.robot import CRAWL_JOINTS
from mjlab.tasks.g1_crawl.robot29 import CRAWL_JOINTS_29


@dataclass
class CrawlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO plus the source task's mirrored data and mirror loss."""

  # Hidden from the command line: it holds a function, which tyro cannot print,
  # so ``--help`` (and the message for any mistyped option) used to crash on it.
  symmetry_cfg: tyro.conf.Suppress[dict | None] = None


def unitree_g1_crawl_ppo_runner_cfg(
  num_cmd: int = 3, terrain: bool = False, dof29: bool = False
) -> RslRlOnPolicyRunnerCfg:
  """PPO runner config.

  ``terrain=True`` uses MJLab's velocity-task PPO settings (clip 0.2) and a lower
  entropy coefficient: with the source's clip of 0.05 the adaptive KL schedule
  collapsed the learning rate to its 1e-5 floor as soon as the terrain
  curriculum changed the data distribution, and the policy stopped learning.
  The 29-DoF tasks use the same settings from the start: their robustness
  curriculum shifts the distribution every time it changes level. They log to
  their own experiment folder, ``logs/rsl_rl/g1_crawl_29dof``.
  """
  stable = terrain or dof29
  # Second generation (29-DoF, 7 commands): gyro in the actor inputs, long run.
  gen2 = dof29 and num_cmd == 7
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=CrawlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2 if stable else 0.05,
      entropy_coef=0.005 if stable else 0.008,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      symmetry_cfg={
        "data_augmentation_func": make_crawl_symmetry_augmentation(
          num_cmd, CRAWL_JOINTS_29 if dof29 else CRAWL_JOINTS, gyro=gen2
        ),
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.5,
      },
    ),
    experiment_name="g1_crawl_29dof" if dof29 else "g1_crawl",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50000 if gen2 else 8000 if dof29 else 1000,
    clip_actions=5.0,
  )


@dataclass
class AnchoredPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """``AnchoredPPO``: PPO for fine-tuning a deployed policy (see
  ``g1_crawl/anchored_ppo.py``) -- anchor to the loaded actor, critic warm-up,
  frozen actor normalizer."""

  symmetry_cfg: tyro.conf.Suppress[dict | None] = None
  class_name: str = "mjlab.tasks.g1_crawl.anchored_ppo:AnchoredPPO"
  anchor_coef: float = 1.0
  """Weight of mean ||mu - mu_loaded||^2 at the start of the fine-tune."""
  anchor_coef_final: float = 0.1
  anchor_decay_updates: int = 2000
  actor_freeze_updates: int = 50
  freeze_actor_normalizer: bool = True
  reset_learning_rate: bool = True


def anchored_finetune(
  cfg: RslRlOnPolicyRunnerCfg,
  learning_rate: float = 3e-4,
  desired_kl: float = 0.008,
  max_iterations: int = 3000,
  actor_freeze_updates: int = 50,
  anchor_coef: float = 1.0,
  anchor_decay_updates: int = 2000,
) -> RslRlOnPolicyRunnerCfg:
  """Swap ``cfg``'s PPO for ``AnchoredPPO`` (same settings otherwise), with a
  lower learning rate and KL target, and checkpoints every 100 iterations."""
  old = cfg.algorithm
  algorithm = AnchoredPpoAlgorithmCfg()
  for f in fields(old):
    if f.name != "class_name":
      setattr(algorithm, f.name, getattr(old, f.name))
  algorithm.learning_rate = learning_rate
  algorithm.desired_kl = desired_kl
  algorithm.actor_freeze_updates = actor_freeze_updates
  algorithm.anchor_coef = anchor_coef
  algorithm.anchor_decay_updates = anchor_decay_updates
  cfg.algorithm = algorithm
  cfg.save_interval = 100
  cfg.max_iterations = max_iterations
  return cfg
