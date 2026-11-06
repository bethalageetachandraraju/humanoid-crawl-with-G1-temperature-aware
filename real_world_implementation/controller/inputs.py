"""Operator inputs: terminal keyboard and the Unitree wireless remote."""

from __future__ import annotations

import os
import queue
import select
import struct
import sys
import termios
import threading
import tty


class TerminalKeys:
  """Non-blocking single-key reader for the terminal the controller runs in."""

  def __init__(self) -> None:
    self.keys: queue.Queue[str] = queue.Queue()
    self._fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    self._old = None
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)

  def __enter__(self) -> TerminalKeys:
    if self._fd is not None:
      self._old = termios.tcgetattr(self._fd)
      tty.setcbreak(self._fd)
      self._thread.start()
    return self

  def __exit__(self, *exc) -> None:
    self._stop.set()
    if self._fd is not None and self._old is not None:
      termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

  def _run(self) -> None:
    assert self._fd is not None
    while not self._stop.is_set():
      r, _, _ = select.select([self._fd], [], [], 0.05)
      if r:
        ch = os.read(self._fd, 1).decode(errors="ignore")
        if ch:
          self.keys.put(ch)

  def get_all(self) -> list[str]:
    out = []
    while not self.keys.empty():
      out.append(self.keys.get())
    return out


class Remote:
  """Unitree wireless remote, decoded from LowState_.wireless_remote (40 bytes).

  Same layout as unitree_sdk2_python/example/wireless_controller.
  """

  BUTTONS = (
    ("R1", 2, 0), ("L1", 2, 1), ("Start", 2, 2), ("Select", 2, 3),
    ("R2", 2, 4), ("L2", 2, 5), ("F1", 2, 6), ("F3", 2, 7),
    ("A", 3, 0), ("B", 3, 1), ("X", 3, 2), ("Y", 3, 3),
    ("Up", 3, 4), ("Right", 3, 5), ("Down", 3, 6), ("Left", 3, 7),
  )  # fmt: skip

  def __init__(self) -> None:
    self.lx = self.rx = self.ry = self.ly = 0.0
    self.pressed: dict[str, bool] = {name: False for name, _, _ in self.BUTTONS}
    self._prev = dict(self.pressed)

  def update(self, data) -> list[str]:
    """Parse one message; returns buttons that went down since the last call."""
    raw = bytes(data)
    self.lx = struct.unpack("<f", raw[4:8])[0]
    self.rx = struct.unpack("<f", raw[8:12])[0]
    self.ry = struct.unpack("<f", raw[12:16])[0]
    self.ly = struct.unpack("<f", raw[20:24])[0]
    self._prev = dict(self.pressed)
    for name, byte, bit in self.BUTTONS:
      self.pressed[name] = bool((raw[byte] >> bit) & 1)
    return [n for n in self.pressed if self.pressed[n] and not self._prev[n]]
