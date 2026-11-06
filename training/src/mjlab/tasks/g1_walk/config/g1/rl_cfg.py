"""PPO settings for the walk and stand-to-crawl tasks of the braced 29-DoF G1."""

from dataclasses import dataclass

import tyro

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.g1_walk.mdp.symmetry import make_walk_symmetry_augmentation


@dataclass
class WalkPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO plus the left-right mirrored data and mirror loss (optional)."""

  # Hidden from the command line: it holds a function tyro cannot print.
  symmetry_cfg: tyro.conf.Suppress[dict | None] = None


def unitree_g1_walk_ppo_runner_cfg(
  experiment_name: str = "g1_walk_29dof",
  max_iterations: int = 20_000,
  clip_actions: float = 5.0,
  symmetry: bool = False,
) -> RslRlOnPolicyRunnerCfg:
  """MJLab's velocity-task PPO (observation normalization on, exported to ONNX).

  Clip 0.2 and a low entropy coefficient, as the crawl tasks settled on: the
  robustness curriculum shifts the data distribution at every level change, and
  the adaptive KL schedule copes with that at clip 0.2 but not at 0.05.
  ``symmetry`` adds the crawl task's mirrored-data augmentation and mirror loss
  (walk task layout), for a gait that turns equally well both ways.
  """
  symmetry_cfg = (
    {
      "data_augmentation_func": make_walk_symmetry_augmentation(),
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 0.5,
    }
    if symmetry
    else None
  )
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=WalkPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      symmetry_cfg=symmetry_cfg,
    ),
    experiment_name=experiment_name,
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    clip_actions=clip_actions,
  )
