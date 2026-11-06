"""Motor temperature guard for the G1 chain controller (numpy only).

Drop-in for the laptop's chain loop (``g1_chain_real.py``): call ``update``
once per tick with every motor's temperature readings and the chain state, and
act on the answer. It keeps the 85 C damping stop the 2026-10-29 test used,
and adds an earlier, gentler exit: stand up from the crawl side while the
motors can still do it.

  guard = ThermalGuard()
  ...
  action = guard.update(t, temps, in_crawl_side=state in (HOLD, CRAWL, CRAWL_STOP))
  if action == "damp":      # any motor >= 85 C: all motors to damping
  elif action == "stand_up":  # crawl side and hot: press U (crawl -> stand -> walk)
  allow_down = guard.allow_down  # gate the C key on it

``temps``: shape (29,) or (29, 2), one row per Unitree motor (LowState
``motor_state[i].temperature``, both readings); the hottest reading counts.
Units C. Thresholds:

- ``stand_up_c`` 72 C, or a motor heating faster than ``rate_c_per_s`` (1 C/s
  over ``rate_window_s``, which was the stuck nose-down hold on 10-29 at 1.4
  C/s) while above ``rate_floor_c``: stand up.
- ``stop_c`` 85 C: damping, latched until ``reset()``.
- ``allow_down`` stays False after a stand-up until every motor is back under
  ``resume_c`` (60 C); walking cools (-0.05 C/s measured on 10-29).
"""

from __future__ import annotations

from collections import deque

import numpy as np

OK, WARN, STAND_UP, DAMP = "ok", "warn", "stand_up", "damp"


class ThermalGuard:
  def __init__(
    self,
    warn_c: float = 65.0,
    stand_up_c: float = 72.0,
    stop_c: float = 85.0,
    resume_c: float = 60.0,
    rate_c_per_s: float = 1.0,
    rate_window_s: float = 5.0,
    rate_floor_c: float = 55.0,
  ) -> None:
    self.warn_c, self.stand_up_c, self.stop_c = warn_c, stand_up_c, stop_c
    self.resume_c = resume_c
    self.rate_c_per_s, self.rate_window_s = rate_c_per_s, rate_window_s
    self.rate_floor_c = rate_floor_c
    self.reset()

  def reset(self) -> None:
    self.history: deque[tuple[float, np.ndarray]] = deque()
    self.latched_damp = False
    self.allow_down = True
    self.reason = ""
    self.hottest_motor = -1
    self.hottest_c = float("nan")

  def update(self, t: float, temps: np.ndarray, in_crawl_side: bool) -> str:
    """``t`` in seconds (any monotonic clock); ``in_crawl_side``: the chain is
    in HOLD, CRAWL or CRAWL_STOP. Pass False during DOWN (finish the descent,
    the stand-up follows in HOLD), UP, STAND_HOLD and walking."""
    temps = np.asarray(temps, dtype=np.float64)
    per_motor = temps.max(axis=1) if temps.ndim == 2 else temps
    self.hottest_motor = int(np.argmax(per_motor))
    self.hottest_c = float(per_motor[self.hottest_motor])

    self.history.append((t, per_motor.copy()))
    while self.history and t - self.history[0][0] > self.rate_window_s:
      self.history.popleft()
    t0, old = self.history[0]
    rate = (per_motor - old) / (t - t0) if t - t0 > 0.5 * self.rate_window_s else None

    if self.latched_damp or self.hottest_c >= self.stop_c:
      self.latched_damp = True
      self.reason = f"motor {self.hottest_motor} at {self.hottest_c:.0f} C"
      return DAMP
    if self.hottest_c < self.resume_c:
      self.allow_down = True
    heating = None
    if rate is not None:
      fast = (rate > self.rate_c_per_s) & (per_motor > self.rate_floor_c)
      if fast.any():
        heating = int(np.argmax(np.where(fast, rate, -np.inf)))
    if self.hottest_c >= self.stand_up_c or heating is not None:
      self.allow_down = False
      if heating is not None and self.hottest_c < self.stand_up_c:
        self.reason = (
          f"motor {heating} heating {rate[heating]:.1f} C/s at {per_motor[heating]:.0f} C"
        )
      else:
        self.reason = f"motor {self.hottest_motor} at {self.hottest_c:.0f} C"
      return STAND_UP if in_crawl_side else WARN
    if self.hottest_c >= self.warn_c:
      self.reason = f"motor {self.hottest_motor} at {self.hottest_c:.0f} C"
      return WARN
    self.reason = ""
    return OK


def _self_test() -> None:
  g = ThermalGuard()
  temps = np.full((29, 2), 45.0)
  assert g.update(0.0, temps, True) == OK
  temps[23] = 66.0
  assert g.update(1.0, temps, True) == WARN
  temps[23] = 73.0
  assert g.update(2.0, temps, True) == STAND_UP and not g.allow_down
  assert g.update(3.0, temps, False) == WARN  # standing up / walking: keep going
  temps[23] = 61.0
  g.update(60.0, temps, False)
  assert not g.allow_down  # above resume_c: no crawl yet
  temps[23] = 59.0
  assert g.update(120.0, temps, False) == OK and g.allow_down
  # Stuck hold: 57 -> 64 C in 5 s (1.4 C/s), under 72 C.
  g = ThermalGuard()
  temps = np.full(29, 50.0)
  for k in range(11):
    temps[3] = 57.0 + 0.7 * k
    a = g.update(0.5 * k, temps, True)
  assert a == STAND_UP and "heating" in g.reason, (a, g.reason)
  temps[3] = 85.0
  assert g.update(6.0, temps, False) == DAMP
  temps[3] = 50.0
  assert g.update(7.0, temps, False) == DAMP  # latched
  print("thermal_guard self-test passed")


if __name__ == "__main__":
  _self_test()
