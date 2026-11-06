"""Reset the optimizer learning rate stored in an RSL-RL checkpoint.

MJLab's runner restores the optimizer state on resume, including the learning
rate the adaptive KL schedule had reached (possibly its 1e-5 floor). Writes a
copy with every param group's ``lr`` set to the given value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--min-std", type=float, default=None)
  args = ap.parse_args()
  ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  groups = ck.get("optimizer_state_dict", {}).get("param_groups", [])
  old = [g.get("lr") for g in groups]
  for g in groups:
    g["lr"] = args.lr
  if args.min_std is not None:
    std = ck["actor_state_dict"]["distribution.std_param"]
    print(f"std mean {float(std.mean()):.3f} -> ", end="")
    std.clamp_(min=args.min_std)
    print(f"{float(std.mean()):.3f}")
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out = out_dir / Path(args.checkpoint).name
  torch.save(ck, out)
  print(f"wrote {out}: lr {old} -> {args.lr} (iter {ck.get('iter')})")


if __name__ == "__main__":
  main()
