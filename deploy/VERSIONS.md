# Deploy bundles and policy versions

Every bundle folder is self-contained (ONNX files, `chain.json`, `mujoco/`) and
is never overwritten: to deploy an older version, point the controller at its
folder. Checkpoints live in `mjlab/logs/rsl_rl/<experiment>/<run>/model_N.pt`
(not in git: back up `logs/` with the bundles). Code: git branch
`g1-thermal-creepy-crawl` (local).

| Bundle | Date | Walk | Stand <-> crawl | Crawl | Status |
|---|---|---|---|---|---|
| `g1_chain/` | 10-18 | `walk_sym_spin2/25000` | one-way `stand_to_crawl_v2` | trot `gen2_flat_16k/49999` | superseded |
| `g1_chain_v2/` | 10-23 | `walk_hard_track/55499` | `stand_crawl_v3/24999` | trot `gen2_flat_16k/49999` | **on the robot 10-29**: motors reached 85 C holding the crawl; foot caught on carpet |
| `g1_chain_v3/` | 10-30 | `walk_hard_track/55499` | `push_v1/30200` (thermal, carpet, shove-hardened) | creepy `creepy_hard/59499` (default) + scuttle `scuttle_hard/59299` | on the laptop 10-30: the stand-up needed a hand |
| `g1_chain_v3_1/` | 10-31 | as v3 | `robust_v5/35197` (lag to 40 ms, weak/heavy/soft robots, carpet twist, real hand-over starts) | as v3 | **new**: use with the relaxed stand hand-over and the 8 s HOLD timeout; see its README |
| `g1_kneel/` | 10-22 | walk | kneel | - | separate |

## Alternate checkpoints (not bundled; export with `scripts/export_chain_v2.py`)

| Role | Checkpoint | Why kept |
|---|---|---|
| stand <-> crawl | `g1_stand_to_crawl_29dof/2026-10-30_22-01-41_robust_v1/model_32199.pt` | first wide-margin run (StandCrawl-Robust): falls far less than v3 under lag and weak/heavy worlds but drifts at the top of the stand-up; 5.5 % hard+push loop failures |
| stand <-> crawl | `.../2026-10-31_00-04-59_robust_v2/model_33698.pt` | Robust2: as v3.1's without the hand-over starts or per-robot stiffness; 4.7 % hard+push loop failures (27 brace touches in HOLD of 1024) |
| stand <-> crawl | `.../2026-10-31_05-23-55_robust_v3/model_35197.pt` | trained from recorded creepy/scuttle hand-over poses: best hard+push loop (2.4 %), but sits the pelvis down in the hold when every motor is 30 % softer (79 %), and sags going down in the heaviest worlds |
| stand <-> crawl | `.../2026-10-31_07-35-35_robust_v4/model_36196.pt` | robust_v3 with more descent practice: fixes the heavy descent, keeps the soft-motor sag (74 %) |
| stand <-> crawl | `g1_stand_to_crawl_29dof/2026-10-29_20-04-05_cool_v2/model_29497.pt` | thermal + carpet fine-tune before the shove hardening; 3.9-9.4 % hard+push loop failures |
| stand <-> crawl | `.../2026-10-29_22-21-52_cool_v3/model_30600.pt` | cooler holds (0.79x), touches the brace under hard randomization (20 % loop failures) |
| stand <-> crawl | `.../2026-10-29_23-27-24_carpet_v1/model_31900.pt` | coolest (0.66x), trained with carpet twist friction; 23 % hard-loop failures |
| crawl (creepy) | `g1_crawl_29dof/2026-10-30_00-57-43_creepy_v4/model_57900.pt` | before hardening: slightly cooler, 15.6 % hard-loop failures |
| crawl (scuttle) | `g1_crawl_29dof/2026-10-30_02-19-38_scuttle_v3/model_57700.pt` | before hardening: 18 % hard-loop failures |
| crawl (fast, trot-like) | `g1_crawl_29dof/2026-10-30_00-19-11_fast_v3/model_55200.pt` | 0.92 m/s at 1 m/s commands, waist yaw 1.22x: hot |
| crawl (trot, cool rest) | `g1_crawl_29dof/2026-10-29_21-01-18_cool_v3/model_54000.pt` | the trot with rest load 1.5x -> 0.8x; stands through 0.1 m/s commands |
| crawl (scuttle) | `g1_crawl_29dof/2026-10-30_06-33-19_scuttle_cool/model_60498.pt` | rejected: waist-pitch heat term; 0.87x at 0.75 m/s (vs 0.92x) but hotter at rest, and 10.5 % hard-loop failures (vs 5.5 %), mostly the brace touching in the hold after crawling |

Export a bundle from any combination, e.g.:

```bash
uv run python scripts/export_chain_v2.py --stand-crawl-checkpoint <pt> \
  --crawl-checkpoint <pt> --crawl-variants name=<pt>:fwd_lo,fwd_hi,lat,turn \
  --out deploy/<new folder> --note "<what and why>"
```

| stand <-> crawl | `.../2026-10-30_05-29-27_push_v1/model_30996.pt` | end of the shove run; 7 % hard+push loop failures (30200 is the pick) |
