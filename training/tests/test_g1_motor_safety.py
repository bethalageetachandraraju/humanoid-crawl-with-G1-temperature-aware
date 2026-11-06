"""CPU tests of the G1 motor model, the motor safety filter and the HJ returns."""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab.tasks.g1_motor_safety.motor_model import (
  ALPHA_CU,
  MOTOR_5020_DUAL,
  MOTOR_7520_22,
  PLANT_SPREAD,
  T_LIMIT,
  T_REF,
  TEMP_QUANT,
  budget_tables,
  cluster_adjacency,
  envelope_bounds,
  horizon_power_budget,
  motor_power,
  thermal_step,
  thermal_torque_bound,
)
from mjlab.tasks.g1_motor_safety.safety_ppo import SafetyPPO

_FIELDS = ("r25", "kt", "c_w", "r_wh", "c_h", "r_ha", "k_iron")


def _params(motor, n: int, spread: float = 0.0, seed: int = 0):
  g = torch.Generator().manual_seed(seed)
  p = {}
  for k in _FIELDS:
    u = torch.rand(n, 1, generator=g) * 2 - 1
    w = PLANT_SPREAD[k] * spread
    p[k] = getattr(motor, k) * (1 + w * u)
  p["n"] = torch.full((n, 1), float(motor.n_motors))
  p["gear"] = torch.full((n, 1), motor.gear)
  return p


def _filter_params(motor, n: int):
  s = PLANT_SPREAD
  one = torch.ones(n, 1)
  return {
    "n": one * motor.n_motors,
    "gear": one * motor.gear,
    "kt": one * motor.kt * (1 - s["kt"]),
    "r25": one * motor.r25 * (1 + s["r25"]),
    "c_w": one * motor.c_w * (1 - s["c_w"]),
    "r_wh_hi": one * motor.r_wh * (1 + s["r_wh"]),
    "r_wh_lo": one * motor.r_wh * (1 - s["r_wh"]),
    "k_iron": one * motor.k_iron * (1 + s["k_iron"]),
  }


def test_steady_state_matches_continuous_rating():
  """Holding the continuous torque settles the winding at the rating."""
  m = MOTOR_7520_22
  p = _params(m, 1)
  t_w = torch.full((1, 1), T_REF)
  t_h = torch.full((1, 1), T_REF)
  amb = torch.full((1, 1), T_REF)
  tau = torch.full((1, 1), m.continuous_torque())
  for _ in range(int(3000 / 0.01)):
    t_w, t_h = thermal_step(t_w, t_h, amb, motor_power(tau, 0 * tau, t_w, p), p, 0.01)
  assert abs(float(t_w) - T_LIMIT) < 0.5


def test_barrier_keeps_winding_below_limit_under_model_mismatch():
  """Greedy torque requests through the filter never push any plant in the
  identified uncertainty box over T_LIMIT, even starting at the limit."""
  m = MOTOR_7520_22
  n, dt = 2048, 0.005
  plant = _params(m, n, spread=1.0, seed=1)
  fp = _filter_params(m, n)
  g = torch.Generator().manual_seed(2)
  amb = 20 + 10 * torch.rand(n, 1, generator=g)
  t_h = amb + 50 * torch.rand(n, 1, generator=g)
  t_w = torch.minimum(
    t_h + 30 * torch.rand(n, 1, generator=g), torch.full_like(t_h, T_LIMIT - 0.5)
  )
  worst = 0.0
  for _ in range(20_000):  # 100 s of maximum demand
    q = TEMP_QUANT
    dq = 10 * (torch.rand(n, 1, generator=g) * 2 - 1)
    bound = thermal_torque_bound(
      torch.round(t_w / q) * q, torch.round(t_h / q) * q, dq, fp
    )
    tau = torch.minimum(bound, torch.full_like(bound, m.peak_torque))
    t_w, t_h = thermal_step(t_w, t_h, amb, motor_power(tau, dq, t_w, plant), plant, dt)
    worst = max(worst, float(t_w.max()))
  assert worst <= T_LIMIT


def test_cold_motor_gets_full_peak_and_hot_motor_its_continuous_torque():
  m = MOTOR_7520_22
  fp = _filter_params(m, 1)
  zero = torch.zeros(1, 1)
  cold = thermal_torque_bound(zero + T_REF, zero + T_REF, zero, fp)
  assert float(cold) > m.peak_torque
  # At the limit, with the housing at its steady state for the continuous rating,
  # the bound is the (conservatively estimated) continuous torque.
  p_cont = (T_LIMIT - T_REF) / (m.r_wh + m.r_ha)
  t_h = T_REF + p_cont * m.r_ha
  hot = thermal_torque_bound(zero + T_LIMIT - 1, zero + t_h, zero, fp)
  assert 0.6 * m.continuous_torque() < float(hot) < m.continuous_torque()
  # A cold housing soaks up more heat, so a hot winding may still give more.
  soak = thermal_torque_bound(zero + T_LIMIT - 1, zero + 30.0, zero, fp)
  assert float(soak) > float(hot)


def test_envelope_bounds_follow_torque_speed_curve():
  m = MOTOR_7520_22
  p = {
    "stall": torch.tensor([m.stall_torque]),
    "no_load": torch.tensor([m.no_load_speed]),
    "peak": torch.tensor([m.peak_torque]),
  }
  lo, hi = envelope_bounds(
    torch.tensor([[0.0, m.no_load_speed / 2, -m.no_load_speed]]), p
  )
  assert torch.allclose(hi[0, 0], torch.tensor(m.peak_torque))
  assert torch.allclose(hi[0, 1], torch.tensor(m.stall_torque / 2))
  assert torch.allclose(lo[0, 1], torch.tensor(-m.peak_torque))
  assert float(hi[0, 2]) == m.peak_torque  # driving against the motion: full torque


def test_copper_loss_grows_with_temperature():
  m = MOTOR_7520_22
  p = _params(m, 1)
  tau = torch.full((1, 1), 50.0)
  cold = motor_power(tau, 0 * tau, torch.full((1, 1), 25.0), p)
  hot = motor_power(tau, 0 * tau, torch.full((1, 1), 90.0), p)
  assert torch.allclose(hot / cold, torch.tensor(1 + ALPHA_CU * 65.0))


def test_fan_and_forced_convection_cool_the_motor():
  """Same torque, full fan and walking speed: the winding settles cooler, and the
  steady state matches the cooled continuous-rating formula."""
  m = MOTOR_7520_22
  p = _params(m, 1)
  tau = torch.full((1, 1), m.continuous_torque())
  amb = torch.full((1, 1), T_REF)

  def settle(gain: float) -> float:
    t_w = torch.full((1, 1), T_REF)
    t_h = torch.full((1, 1), T_REF)
    for _ in range(int(3000 / 0.01)):
      pw = motor_power(tau, 0 * tau, t_w, p)
      t_w, t_h = thermal_step(t_w, t_h, amb, pw, p, 0.01, ha_gain=gain)
    return float(t_w)

  passive, cooled = settle(1.0), settle(1.6)
  assert abs(passive - T_LIMIT) < 0.5
  assert cooled < passive - 10.0
  hot = m.continuous_torque(cooling=1.6)
  assert hot > m.continuous_torque()


def test_motor_coupling_conserves_heat():
  """Heat exchanged between neighbouring housings sums to zero."""
  names = [
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_knee_joint",
    "waist_yaw_joint",
  ]
  a = cluster_adjacency(names, "cpu")
  assert a[0, 1] == 1 and a[0, 3] == 0 and a[4].sum() == 0
  t_h = torch.tensor([[60.0, 40.0, 30.0, 80.0, 50.0]])
  q = 0.15 * (t_h @ a.T - t_h * a.sum(-1))
  assert abs(float(q.sum())) < 1e-5
  assert float(q[0, 0]) < 0 < float(q[0, 2])


class _Critic:
  """Stands in for the rsl_rl value net: returns a fixed bootstrap value."""

  def __init__(self, value: float) -> None:
    self.value = value

  def __call__(self, obs) -> torch.Tensor:
    return torch.tensor([[self.value]])

  def get_hidden_state(self):
    return None

  def reset(self, hidden_state=None) -> None:
    pass


def test_safety_ppo_returns_are_avoid_targets():
  backups = pytest.importorskip("safety_sb3.backups")
  g = [1.0, 0.5, -0.2, 0.8, 0.3]
  v = [0.4, 0.3, 0.6, 0.1, 0.2]
  dones = [0.0, 0.0, 1.0, 0.0, 0.0]
  gamma, last = 0.9, 0.25
  t = len(g)
  # The state SafetyPPO.compute_returns reads, for one env, lam = 0.
  algo = SimpleNamespace(
    storage=SimpleNamespace(
      num_transitions_per_env=t,
      rewards=torch.tensor(g).reshape(t, 1, 1),
      values=torch.tensor(v).reshape(t, 1, 1),
      dones=torch.tensor(dones).reshape(t, 1, 1),
      returns=torch.zeros(t, 1, 1),
      advantages=torch.zeros(t, 1, 1),
    ),
    critic=_Critic(last),
    _backups=backups,
    gamma=gamma,
    lam=0.0,
    _pilot=torch.tensor([True, False, False, False, False]).reshape(t, 1),
    normalize_advantage_per_mini_batch=True,
  )
  SafetyPPO.compute_returns(algo, obs=None)  # type: ignore[arg-type]
  # One-step avoid targets; a terminal step is worth its own margin; the
  # pilot-driven step keeps its prediction and gets no advantage.
  expect = [
    v[0],
    (1 - gamma) * g[1] + gamma * min(g[1], v[2]),
    g[2],
    (1 - gamma) * g[3] + gamma * min(g[3], v[4]),
    (1 - gamma) * g[4] + gamma * min(g[4], last),
  ]
  assert torch.allclose(algo.storage.returns.reshape(-1), torch.tensor(expect))
  assert algo.storage.advantages.reshape(-1)[0] == 0.0


def test_horizon_budget_keeps_winding_below_limit():
  m = MOTOR_5020_DUAL
  p = {k: torch.tensor([[getattr(m, k)]], dtype=torch.float64) for k in _FIELDS}
  p["c_w"] = p["c_w"] * (1 - PLANT_SPREAD["c_w"])
  p["c_h"] = p["c_h"] * (1 - PLANT_SPREAD["c_h"])
  p["r_wh"] = p["r_wh"] * (1 + PLANT_SPREAD["r_wh"])
  p["r_ha"] = p["r_ha"] * (1 + PLANT_SPREAD["r_ha"])
  horizons = (2.0, 30.0)
  times = torch.logspace(-1.3, math.log10(horizons[-1]), 64, dtype=torch.float64)
  tables = budget_tables(p, times)
  masks = torch.stack([times <= h * (1 + 1e-6) for h in horizons])
  g = torch.Generator().manual_seed(0)
  n = 64
  t_amb = 20 + 20 * torch.rand(n, 1, generator=g, dtype=torch.float64)
  t_h = t_amb + (60 - t_amb) * torch.rand(n, 1, generator=g, dtype=torch.float64)
  t_w = t_h + (88 - t_h) * torch.rand(n, 1, generator=g, dtype=torch.float64)
  q_drv = torch.full((n, 1), m.q_drv, dtype=torch.float64)
  budget = horizon_power_budget(t_w, t_h, t_amb, q_drv, p["r_ha"], tables, masks)
  dt = 1e-3
  for k, h in enumerate(horizons):
    # Any profile at or below the budget: here random fractions of it.
    tw, th = t_w.clone(), t_h.clone()
    peak = tw.clone()
    for _ in range(int(h / dt)):
      pw = budget[:, k] * torch.rand(n, 1, generator=g, dtype=torch.float64).sqrt()
      dtw = (pw - (tw - th) / p["r_wh"]) / p["c_w"]
      dth = ((tw - th) / p["r_wh"] - (th - t_amb) / p["r_ha"] + q_drv) / p["c_h"]
      tw, th = tw + dt * dtw, th + dt * dth
      peak = torch.maximum(peak, tw)
    assert (peak <= T_LIMIT + 0.05).all(), (h, float(peak.max()))
  # Longer horizons can only be tighter.
  assert (budget[:, 1] <= budget[:, 0] + 1e-9).all()
