# Checkpoints (RSL-RL, mjlab)

The PyTorch checkpoints behind the ONNX policies in `deploy/g1_chain_v3_1/`, for
fine-tuning or playing in mjlab (see `training/`).

| File | Policy | mjlab task | Run |
|---|---|---|---|
| `walk_hard_track_55499.pt` | walk | `Mjlab-Walk-Rough-Sym-Hard-Unitree-G1-29dof` (yaw-tracking fine-tune) | `g1_walk_29dof/2026-10-23_06-41-09_walk_hard_track` |
| `stand_crawl_robust_v5_35197.pt` | stand <-> crawl | `Mjlab-StandCrawl-Robust5-Unitree-G1-29dof` | `g1_stand_to_crawl_29dof/2026-10-31_09-27-40_robust_v5` |
| `crawl_creepy_hard_59499.pt` | crawl, creepy (default) | `Mjlab-Crawl-Flat-Cmd7-CreepyHard-Unitree-G1-29dof` | `g1_crawl_29dof/2026-10-30_03-24-17_creepy_hard` |
| `crawl_scuttle_hard_59299.pt` | crawl, scuttle | `Mjlab-Crawl-Flat-Cmd7-ScuttleHard-Unitree-G1-29dof` | `g1_crawl_29dof/2026-10-30_03-25-17_scuttle_hard` |
