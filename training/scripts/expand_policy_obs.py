"""Widen a crawl checkpoint's command block from 3 to 7 values (zero columns).

The actor observation is [gravity(3), command(3), ...] and the critic's is
[lin vel(3), ang vel(3), gravity(3), command(3), ...]. Four zero-initialised
input columns are inserted right after the command block in the first layer of
both networks, and in the Adam moments for those weights, so the expanded policy
behaves exactly like the original while the new slots are 0.0. Writes the result
to ``<out-dir>/<checkpoint name>`` so ``--agent.resume`` can load it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ACTOR_INSERT_AT = 3 + 3  # After gravity(3) + command(3).
CRITIC_INSERT_AT = 9 + 3  # After lin vel(3) + ang vel(3) + gravity(3) + command(3).
EXTRA = 4


def widen(weight: torch.Tensor, at: int) -> torch.Tensor:
  zeros = torch.zeros(weight.shape[0], EXTRA, dtype=weight.dtype)
  return torch.cat((weight[:, :at], zeros, weight[:, at:]), dim=1)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out-dir", required=True, help="new run folder for the result")
  args = ap.parse_args()

  ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  actor_w = ck["actor_state_dict"]["mlp.0.weight"]
  critic_w = ck["critic_state_dict"]["mlp.0.weight"]
  assert actor_w.shape[1] == 75 and critic_w.shape[1] == 81, (
    f"expected 75/81 inputs, got {actor_w.shape[1]}/{critic_w.shape[1]}"
  )
  ck["actor_state_dict"]["mlp.0.weight"] = widen(actor_w, ACTOR_INSERT_AT)
  ck["critic_state_dict"]["mlp.0.weight"] = widen(critic_w, CRITIC_INSERT_AT)

  # Adam moments for the two first-layer weights (matched by shape).
  for state in ck.get("optimizer_state_dict", {}).get("state", {}).values():
    for key in ("exp_avg", "exp_avg_sq"):
      t = state.get(key)
      if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == 256:
        if t.shape[1] == 75:
          state[key] = widen(t, ACTOR_INSERT_AT)
        elif t.shape[1] == 81:
          state[key] = widen(t, CRITIC_INSERT_AT)

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out = out_dir / Path(args.checkpoint).name
  torch.save(ck, out)
  print(
    f"wrote {out}  (actor in {actor_w.shape[1]}->79, critic in {critic_w.shape[1]}->85)"
  )


if __name__ == "__main__":
  main()
