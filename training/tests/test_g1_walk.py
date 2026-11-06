"""Targeted checks of the g1_walk task pieces that need no simulation."""

import math

import torch
from tensordict import TensorDict

from mjlab.tasks.g1_walk.mdp import make_walk_symmetry_augmentation
from mjlab.tasks.g1_walk.mdp.keyframes import KEYFRAME_TABLE
from mjlab.tasks.g1_walk.mdp.transition import KEYFRAMES, TransitionSchedule
from mjlab.tasks.g1_walk.robot import (
  CRAWL_HEIGHT,
  CRAWL_POSE_29,
  JOINTS_29,
  STAND_HEIGHT,
  STAND_POSE_29,
)


def test_keyframe_table_is_complete_and_ordered():
  phis = [k["phi"] for k in KEYFRAME_TABLE]
  assert phis[0] == 0.0 and phis[-1] == 1.0
  assert all(a < b for a, b in zip(phis, phis[1:], strict=False))
  for k in KEYFRAME_TABLE:
    assert set(k["joint_pos"]) == set(JOINTS_29)
    assert 0.2 <= k["height"] <= 0.8
    assert 0.0 <= k["pitch_deg"] <= 90.0
    assert k["hand_height"] >= 0.0


def test_schedule_endpoints_are_the_stand_and_crawl_poses():
  sched = TransitionSchedule("cpu")
  phi = torch.tensor([0.0, 1.0])
  q = sched.joint_pos(phi)
  stand = torch.tensor([STAND_POSE_29[j] for j in JOINTS_29])
  crawl = torch.tensor([CRAWL_POSE_29[j] for j in JOINTS_29])
  assert torch.allclose(q[0], stand, atol=1e-5)
  assert torch.allclose(q[1], crawl, atol=1e-5)
  assert math.isclose(float(sched.height(phi)[0]), STAND_HEIGHT, abs_tol=1e-4)
  assert math.isclose(float(sched.height(phi)[1]), CRAWL_HEIGHT, abs_tol=1e-4)
  # Upright at the start, prone (+X down) at the end.
  g = sched.gravity(phi)
  assert torch.allclose(g[0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
  assert g[1, 0] > 0.99 and abs(float(g[1, 2])) < 0.05
  assert len(KEYFRAMES) == len(KEYFRAME_TABLE)


def test_schedule_is_continuous_and_pauses_at_keyframes():
  sched = TransitionSchedule("cpu")
  phi = torch.linspace(0.0, 1.0, 2001)
  q = sched.joint_pos(phi)
  step = (q[1:] - q[:-1]).abs().amax(dim=1)
  assert float(step.max()) < 0.02  # No jumps along the path.
  for k in KEYFRAME_TABLE[1:-1]:
    i = int(round(k["phi"] * 2000))
    assert float(step[i - 1]) < 1e-3 and float(step[i]) < 1e-3  # Eased in and out.


def test_walk_symmetry_augmentation_is_an_involution():
  aug = make_walk_symmetry_augmentation()
  n = 25
  obs = TensorDict(
    {"actor": torch.randn(5, 9 + 3 * n), "critic": torch.randn(5, 12 + 3 * n + 10)},
    batch_size=[5],
  )
  actions = torch.randn(5, n)
  once_obs, once_act = aug(None, obs, actions)
  assert once_obs is not None and once_act is not None
  assert once_obs.batch_size[0] == 10 and once_act.shape[0] == 10
  twice_obs, twice_act = aug(None, once_obs[5:], once_act[5:])
  assert twice_obs is not None and twice_act is not None
  assert torch.allclose(twice_obs["actor"][5:], obs["actor"])
  assert torch.allclose(twice_obs["critic"][5:], obs["critic"])
  assert torch.allclose(twice_act[5:], actions)
  # The lateral and turn commands flip, forward does not.
  mirrored = once_obs["actor"][5:]
  assert torch.allclose(mirrored[:, 6], obs["actor"][:, 6])
  assert torch.allclose(mirrored[:, 7], -obs["actor"][:, 7])
  assert torch.allclose(mirrored[:, 8], -obs["actor"][:, 8])


def test_kneel_schedule_ends_and_swing_window():
  from mjlab.tasks.g1_walk.mdp.kneel import (
    KNEEL_KEYFRAMES,
    KNEEL_POSE_29,
    SWING_WINDOW,
    KneelSchedule,
  )

  sched = KneelSchedule("cpu")
  phi = torch.tensor([0.0, 1.0])
  q = sched.joint_pos(phi)
  stand = torch.tensor([STAND_POSE_29[j] for j in JOINTS_29])
  kneel = torch.tensor([KNEEL_POSE_29[j] for j in JOINTS_29])
  assert torch.allclose(q[0], stand, atol=1e-5)
  assert torch.allclose(q[1], kneel, atol=1e-5)
  assert math.isclose(float(sched.height(phi)[0]), STAND_HEIGHT, abs_tol=1e-4)
  assert float(sched.knee_height(phi)[1]) < 0.07  # Knee on the floor.
  a, b = SWING_WINDOW
  assert 0.0 < a < b < 1.0
  # The stepping foot is up somewhere inside the swing window only.
  inside = torch.linspace(a, b, 50)
  assert float(sched.rfoot_height(inside).max()) > 0.08
  assert float(sched.rfoot_height(torch.tensor([0.0, a]))[0]) < 0.01
  assert len(KNEEL_KEYFRAMES) >= 5
  steps = (q[1:] - q[:-1]).abs()
  assert torch.isfinite(steps).all()


def test_hard_walk_ramp_starts_where_walk_ramp_ends():
  from dataclasses import fields

  from mjlab.tasks.g1_walk.mdp import WALK_RAMP, WALK_RAMP_HARD

  for f in fields(WALK_RAMP):
    end = getattr(WALK_RAMP, f.name)[1]
    start = getattr(WALK_RAMP_HARD, f.name)[0]
    assert math.isclose(start, end), (f.name, start, end)
