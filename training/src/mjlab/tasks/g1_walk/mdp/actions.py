"""Joint position action as a bounded residual on a scheduled reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ScheduledJointPositionActionCfg(JointPositionActionCfg):
  """Target = reference joint pose of a ``TransitionPhaseCommand`` + scale * action.

  The slow reference carries the motion; the policy only adds a residual to
  keep its balance and choose its contacts. With the runner clipping raw
  actions, the residual is bounded, which bounds how far and how fast the
  policy can depart from the schedule (the idea of HumanUP's slowed-down
  tracking stage and of "Learning to Get Up").
  """

  command_name: str = "phase"
  use_default_offset: bool = False

  def build(self, env: ManagerBasedRlEnv) -> ScheduledJointPositionAction:
    return ScheduledJointPositionAction(self, env)


class ScheduledJointPositionAction(JointPositionAction):
  cfg: ScheduledJointPositionActionCfg

  def __init__(self, cfg: ScheduledJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command: Any = None  # Built after actions.

  def _reference(self) -> torch.Tensor:
    if self._command is None:
      # Any phase command with a scheduled joint pose: the stand-to-crawl
      # ``TransitionPhaseCommand`` or the kneel's ``KneelPhaseCommand``.
      term = cast(Any, self._env.command_manager.get_term(self.cfg.command_name))
      ids = term.joint_ids.tolist()
      assert ids == list(self._target_ids), "action and schedule joint order differ"
      self._command = term
    reference: torch.Tensor = self._command.ref_joint_pos
    return reference

  def process_actions(self, actions: torch.Tensor):
    self._offset = self._reference()
    super().process_actions(actions)
